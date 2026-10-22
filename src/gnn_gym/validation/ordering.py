"""Ordering energies of the site-energy model against the uMLIP.

The site-energy model is trained on vacancy jumps: differences of the
site-energy sum between two configurations that differ in two sites, in cells
that contain a vacancy. Here it is asked for the energy of ordered
arrangements relative to the random alloy, configurations that differ in
every site and contain no defect, and the answer is compared with direct
uMLIP energies of the same candidates (gnn_gym.validation.ground_state).

The absolute site-energy sum carries a gauge, a constant per species; it
cancels between configurations of the same composition, and all candidates
here are equiatomic. Only E - E_random per atom is reported.

CANDIDATES (size^3 cell of the system's lattice)

    random     equiatomic decorations, averaged
    B2         the elements split into two equal groups over the two
               sublattices, random within each group, averaged; every split
               with the first element in the first group
    layered    the planes perpendicular to z occupied in the cyclic
               `layer_sequence` (only for the element set of that sequence)

T_ord = |dE| / (k_B dS) is the ideal-entropy estimate of the ordering
temperature, with dS the configurational entropy the candidate removes from
the random alloy per atom: ln 2 for an even split over two sublattices, ln m
for complete order of m elements. It is an upper bound for a transition into
the candidate, which competes with the short-range-ordered alloy rather than
the random one.

    python -m gnn_gym.validation.ordering configs/validation/ordering.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..lattice import Supercell, locality_report
from ..metropolis.observables import K_B
from ..model import LatticeGraph, load_model
from ..paths import load_yaml, resolve
from ..systems import get_system
from .ground_state import partitions


def fill_equiatomic(sites, elements, rng):
    """Deal the elements out in equal blocks over `sites` in random order."""
    out = {}
    for el, chunk in zip(elements, np.array_split(rng.permutation(sites),
                                                  len(elements))):
        for j in chunk:
            out[int(j)] = el
    return out


def candidates(cell, elements, n_random, random_seed, b2_seed, seed=0,
               layer_sequence=None):
    """{name: ([symbol list per decoration], dS / k_B per atom)}."""
    n = cell.n_sites
    sub = cell.sublattice()
    out = {}
    rows = []
    for s in range(n_random):
        rng = np.random.default_rng(random_seed + seed + s)
        a = fill_equiatomic(np.arange(n), list(elements), rng)
        rows.append([a[i] for i in range(n)])
    out["random"] = (rows, 0.0)
    for ga, gb in partitions(list(elements)):
        rows = []
        for s in range(n_random):
            rng = np.random.default_rng(b2_seed + seed + s)
            a = fill_equiatomic(np.flatnonzero(sub == 0), list(ga), rng)
            a.update(fill_equiatomic(np.flatnonzero(sub == 1), list(gb), rng))
            rows.append([a[i] for i in range(n)])
        out["B2 {%s}|{%s}" % (",".join(ga), ",".join(gb))] = (rows, np.log(2.0))
    if layer_sequence and set(layer_sequence) == set(elements):
        if cell.box % len(layer_sequence):
            raise ValueError(f"{cell.size}^3: the box does not hold whole "
                             f"periods of the layer sequence")
        plane = cell.coords[:, 2] % len(layer_sequence)
        out["layered"] = ([[layer_sequence[p] for p in plane]],
                          np.log(float(len(set(layer_sequence)))))
    return out


@torch.no_grad()
def total_energies(net, model_elements, decorations, device, batch=4):
    """Site-energy sum of each decoration (eV, gauge-dependent)."""
    code = {e: i for i, e in enumerate(model_elements)}
    z = torch.from_numpy(np.array([[code[s] for s in d] for d in decorations],
                                  dtype=np.int64)).to(device)
    out = []
    for k in range(0, len(z), batch):
        out.append(net.site_energies(z[k:k + batch]).sum(-1).cpu().numpy())
    return np.concatenate(out)


def read_reference(path, relax):
    """{candidate: dE meV/atom} of a uMLIP ground-state file, or {}."""
    path = resolve(path)
    if not Path(path).exists():
        print(f"[ordering] no uMLIP reference at {path}")
        return {}
    data = json.loads(Path(path).read_text())
    return {k: float(v) for k, v in
            data["dE_meV_per_atom_vs_random"][relax].items()}


def t_ord(dE_meV, dS):
    return abs(dE_meV) * 1e-3 / (K_B * dS) if dE_meV < 0 and dS > 0 else None


def run_system(name, sc, cfg, net, model_cfg, reach, device):
    system = get_system(name)
    elements = list(sc["elements"])
    ref = read_reference(sc["reference"], cfg["relax"])
    result = dict(system=name, elements=elements, reach=reach,
                  reference=str(sc["reference"]), relax=cfg["relax"], sizes={})
    for size in sc["sizes"]:
        cell = Supercell(system.lattice_obj, int(size))
        rep = locality_report(system.lattice_obj, cell.size, reach,
                              tuple(model_cfg.shells),
                              a=system.lattice_parameter_A)
        print(f"\n[ordering] {name} {cell.size}^3 = {cell.n_sites} atoms, site "
              f"locality {'ok' if rep['local'] else 'WRAPS'} "
              f"({rep['sites_in_ball']} of {rep['sites_unbounded']} sites)")
        net.set_graph(LatticeGraph.for_supercell(cell, model_cfg.shells,
                                                 model_cfg.de_line_graph,
                                                 device=device))
        cand = candidates(cell, elements, int(cfg["n_random"]),
                          int(cfg["random_seed"]), int(cfg["b2_seed"]),
                          int(cfg["seed"]), cfg.get("layer_sequence"))
        means, stds = {}, {}
        for cname, (decs, _) in cand.items():
            e = total_energies(net, model_cfg.elements, decs, device) / cell.n_sites
            means[cname], stds[cname] = float(e.mean()), float(e.std())
        rows = {}
        print(f"  {'candidate':22s} {'model':>14s} {'uMLIP':>8s} "
              f"{'T_ord model':>12s} {'T_ord uMLIP':>12s}")
        for cname in sorted(means, key=means.get):
            if cname == "random":
                continue
            dS = cand[cname][1]
            dE = (means[cname] - means["random"]) * 1e3
            r = ref.get(cname)
            rows[cname] = dict(model_meV_per_atom=dE,
                               model_std_meV=stds[cname] * 1e3,
                               uma_meV_per_atom=r, entropy_drop_kB=float(dS),
                               T_ord_K=t_ord(dE, dS),
                               T_ord_uma_K=None if r is None else t_ord(r, dS))
            fmt = lambda v: "-" if v is None else f"{v:.0f} K"   # noqa: E731
            print(f"  {cname:22s} {dE:+8.2f}+-{stds[cname] * 1e3:4.1f} "
                  f"{'-' if r is None else f'{r:+8.2f}':>8s} "
                  f"{fmt(rows[cname]['T_ord_K']):>12s} "
                  f"{fmt(rows[cname]['T_ord_uma_K']):>12s}")
        result["sizes"][str(cell.size)] = dict(n_atoms=cell.n_sites,
                                               local=bool(rep["local"]), rows=rows)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    device = args.device or cfg.get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    net, model_cfg, ck = load_model(resolve(cfg["model"]), device, bind=False)
    if model_cfg.target != "de":
        raise SystemExit(f"{cfg['model']} is not a site-energy model")
    reach = int(ck["reach"])
    out_dir = Path(resolve(args.out_dir or cfg["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sc in cfg["systems"].items():
        result = run_system(name, sc, cfg, net, model_cfg, reach, device)
        path = out_dir / f"ordering_{name}.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"[ordering] -> {path}")


if __name__ == "__main__":
    main()
