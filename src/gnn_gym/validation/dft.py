"""The uMLIP oracle and the lattice model against DFT-NEB barriers.

The reference is an independent set of DFT-NEB vacancy jumps in Mo-Nb-Ta-W
(VASP, 4x4x4 cells of 127 atoms plus the vacancy): per configuration the
forward and backward DFT barriers and the relaxed and unrelaxed endpoint
structures.

uMLIP
    The forward uMLIP barrier of each configuration is a uMLIP NEB on that
    configuration (data/dft, see the config). Its backward barrier needs only
    the reaction energy, because both barriers share the saddle:
        E_bwd = E_fwd - dE,   dE = E(final) - E(initial),
    with single-point uMLIP energies of the two relaxed DFT endpoints.

Lattice model
    The model reads only the occupation of the ideal lattice. The 4x4x4
    configuration is recovered from the unrelaxed endpoints (atoms mapped
    onto the nearest ideal site; the empty site is the vacancy) and replicated
    2x2x2, so that every local environment, including the periodic images
    the DFT cell saw, is that of the small cell while the model is local at
    that size. The seven copies of the vacancy outside the first block are
    filled with the jumping species; they lie far outside the read-out of
    the jump. Forward and backward barriers are both evaluated.

Statistics, all in the backward convention: for every pair of methods the
mean offset, the MAE after removing it, the raw MAE and Pearson r; per
element; and the offset of the reaction energy against DFT.

    python -m gnn_gym.validation.dft configs/validation/dft.yaml umlip
    python -m gnn_gym.validation.dft configs/validation/dft.yaml model
    python -m gnn_gym.validation.dft configs/validation/dft.yaml stats
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..lattice import Supercell, get_lattice
from ..paths import load_yaml, resolve


def structure(cfg, folder, name):
    from ase.io import read
    return read(str(resolve(cfg["structures"]) / folder / f"{name}.xyz"))


# ---------------------------------------------------------------------------
# uMLIP
# ---------------------------------------------------------------------------

def umlip(cfg, log=print):
    """Single-point reaction energies and backward barriers per uMLIP model."""
    from ..oracle.uma import Uma
    df = pd.read_csv(resolve(cfg["dft_csv"]))
    fwd = pd.read_csv(resolve(cfg["umlip_forward_csv"])).set_index("structure_folder")
    out = pd.DataFrame({"structure_folder": df["structure_folder"]})
    for key, model in cfg["umlip_models"].items():
        uma = Uma(model, cfg["umlip_task"], log=log)
        Ei = np.array([uma.energy(structure(cfg, f, "initial")) for f in df.structure_folder])
        Ef = np.array([uma.energy(structure(cfg, f, "final")) for f in df.structure_folder])
        out[f"umlip_{key}_E_initial_eV"] = Ei
        out[f"umlip_{key}_E_final_eV"] = Ef
        out[f"umlip_{key}_dE_eV"] = Ef - Ei
        out[f"umlip_{key}_fwd_eV"] = fwd.loc[df.structure_folder, f"umlip_{key}_fwd_eV"].to_numpy()
        out[f"umlip_{key}_bwd_eV"] = out[f"umlip_{key}_fwd_eV"] - out[f"umlip_{key}_dE_eV"]
        log(f"[dft] uMLIP-{key} ({model}): {len(df)} configurations")
    return out


# ---------------------------------------------------------------------------
# Lattice model
# ---------------------------------------------------------------------------

def sites_of(atoms, cell, tol=1e-2):
    """Ideal site of every atom (nearest in fractional coordinates)."""
    ideal = cell.coords / cell.box
    frac = atoms.get_scaled_positions(wrap=True)
    out = np.empty(len(frac), dtype=np.int64)
    worst = 0.0
    for i, f in enumerate(frac):
        d = ideal - f
        d -= np.round(d)
        dist = np.linalg.norm(d, axis=1)
        out[i] = int(np.argmin(dist))
        worst = max(worst, float(dist[out[i]]))
    if worst > tol:
        raise RuntimeError(f"atoms not on ideal sites (max error {worst:.2e})")
    return out


def lattice_configuration(initial, final, lattice, elements):
    """Species (-1 = vacancy), vacancy site V and jumping site K of a jump."""
    n_sites = len(initial) + 1
    size = round((n_sites / len(lattice.basis)) ** (1 / 3))
    cell = Supercell(lattice, size)
    if cell.n_sites != n_sites:
        raise RuntimeError(f"{n_sites} sites do not form a cubic supercell")
    s_i, s_f = sites_of(initial, cell), sites_of(final, cell)
    V = int(np.setdiff1d(np.arange(n_sites), s_i)[0])
    K = int(np.setdiff1d(np.arange(n_sites), s_f)[0])
    if K not in set(cell.neighbours(lattice.jump_shell)[V].tolist()):
        raise RuntimeError("the jumping site is not a first neighbour of the vacancy")
    code = {e: i for i, e in enumerate(elements)}
    species = np.full(n_sites, -1, dtype=np.int64)
    species[s_i] = [code[s] for s in initial.get_chemical_symbols()]
    return cell, species, V, K


def replicate(cell, species, V, K, factor):
    """factor^3 replica; one vacancy (first block), other copies -> species[K]."""
    big = Supercell(cell.lattice, cell.size * factor)
    small_of = cell.replication_map(factor)
    sp = species[small_of].copy()
    V8 = big.index(cell.coords[V])
    sp[(small_of == V) & (np.arange(big.n_sites) != V8)] = species[K]
    K8 = [j for j in big.neighbours(cell.lattice.jump_shell)[V8] if small_of[j] == K]
    if len(K8) != 1:
        raise RuntimeError("jumping site not unique in the replica")
    return big, sp, V8, int(K8[0])


def model(cfg, log=print):
    from ..model.predictor import JumpPredictor
    df = pd.read_csv(resolve(cfg["dft_csv"]))
    lattice = get_lattice(cfg["lattice"])
    els = list(cfg["elements"])
    pred = None
    fwd, bwd = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for i, f in enumerate(df.structure_folder):
        cell, sp, V, K = lattice_configuration(structure(cfg, f, "initial_unrelaxed"),
                                               structure(cfg, f, "final_unrelaxed"),
                                               lattice, els)
        big, sp8, V8, K8 = replicate(cell, sp, V, K, int(cfg["replica"]))
        if pred is None or pred.cell.size != big.size:
            pred = JumpPredictor(resolve(cfg["models"]["de"]),
                                 resolve(cfg["models"]["bsym"]), big, els,
                                 halo=False, verbose=False)
        fwd[i] = pred.barriers(sp8, V8, [K8])[0]
        sp8[V8], sp8[K8] = sp8[K8], -1          # after the jump
        bwd[i] = pred.barriers(sp8, K8, [V8])[0]
    log(f"[dft] lattice model: {np.isfinite(fwd).sum()} configurations")
    return pd.DataFrame({"structure_folder": df.structure_folder,
                         "model_fwd_eV": fwd, "model_bwd_eV": bwd})


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def pair_stats(a, b):
    """a against b: offset mean(a - b), MAE after removing it, raw MAE, r."""
    d = a - b
    off = float(d.mean())
    return dict(n=int(len(a)), offset_meV=off * 1e3,
                mae_meV=float(np.abs(d - off).mean() * 1e3),
                raw_mae_meV=float(np.abs(d).mean() * 1e3),
                pearson_r=float(np.corrcoef(a, b)[0, 1]))


def stats(cfg, table, log=print):
    df = table
    methods = {"DFT": df["dft_bwd_eV"]}
    for key in cfg["umlip_models"]:
        methods[f"uMLIP-{key}"] = df[f"umlip_{key}_bwd_eV"]
    if "model_bwd_eV" in df:
        methods["model"] = df["model_bwd_eV"]
    names = list(methods)
    out = {"n": int(len(df)), "convention": "backward barriers", "pairs": {},
           "per_element": {}, "forward_offset_meV": {}, "reaction_energy_offset_meV": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s = pair_stats(methods[b].to_numpy(float), methods[a].to_numpy(float))
            out["pairs"][f"{b} vs {a}"] = s
            log(f"[dft] {b:8s} vs {a:8s}: offset {s['offset_meV']:+6.1f} meV, MAE "
                f"(offset removed) {s['mae_meV']:6.1f} meV, raw MAE "
                f"{s['raw_mae_meV']:6.1f} meV, r = {s['pearson_r']:.3f}")
    dft_dE = (df["dft_fwd_eV"] - df["dft_bwd_eV"]).to_numpy(float)
    for key in cfg["umlip_models"]:
        f_off = (df[f"umlip_{key}_fwd_eV"] - df["dft_fwd_eV"]).mean() * 1e3
        r_off = (df[f"umlip_{key}_dE_eV"].to_numpy(float) - dft_dE).mean() * 1e3
        out["forward_offset_meV"][f"uMLIP-{key}"] = float(f_off)
        out["reaction_energy_offset_meV"][f"uMLIP-{key}"] = float(r_off)
        log(f"[dft] uMLIP-{key}: forward-barrier offset {f_off:+.1f} meV, reaction-"
            f"energy offset {r_off:+.1f} meV")
    el = df["diffusing_element"].to_numpy()
    for a, b in [(f"uMLIP-{k}", "DFT") for k in cfg["umlip_models"]] + \
            ([("model", "DFT"), ("model", f"uMLIP-{next(iter(cfg['umlip_models']))}")]
             if "model" in methods else []):
        out["per_element"][f"{a} vs {b}"] = {
            e: pair_stats(methods[a].to_numpy(float)[el == e],
                          methods[b].to_numpy(float)[el == e])["mae_meV"]
            for e in sorted(set(el))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("action", choices=["umlip", "model", "stats"])
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    table_path = resolve(cfg["table"])
    if args.action in ("umlip", "model"):
        new = umlip(cfg) if args.action == "umlip" else model(cfg)
        if table_path.exists():
            old = pd.read_csv(table_path)
            keep = [c for c in old.columns if c not in new.columns or c == "structure_folder"]
            new = old[keep].merge(new, on="structure_folder", how="left")
        else:
            dft = pd.read_csv(resolve(cfg["dft_csv"]))
            base = dft[["structure_folder", "composition_string", "diffusing_element",
                        "forward_barrier_eV", "backward_barrier_eV"]].rename(
                columns={"forward_barrier_eV": "dft_fwd_eV",
                         "backward_barrier_eV": "dft_bwd_eV"})
            new = base.merge(new, on="structure_folder", how="left")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(table_path, index=False)
        print(f"[dft] -> {table_path}")
    else:
        res = stats(cfg, pd.read_csv(table_path))
        out = resolve(cfg["summary"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"[dft] -> {out}")


if __name__ == "__main__":
    main()
