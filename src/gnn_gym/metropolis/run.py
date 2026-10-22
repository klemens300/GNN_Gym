"""Metropolis chains through a list of temperatures (cooling or heating).

    python -m gnn_gym.metropolis.run configs/metropolis/tord_MoNbTaW.yaml
    python -m gnn_gym.metropolis.run configs/metropolis/tord_MoNbTaW.yaml \
        --chain MoNbTaW_cool_s0

A chain samples the canonical ensemble of the site-energy model on a
defect-free supercell at fixed composition with the swap sampler
(gnn_gym.metropolis.sampler). It visits the temperatures in falling order
(mode cool) or rising order (mode heat) and carries the configuration from
one temperature to the next. At every temperature n_eq sweeps are discarded
and n_samp sweeps are sampled; one sweep is one decided proposal per site.
`batch` proposals are evaluated per call of the model and decided with the
walk `walk` (gnn_gym.metropolis.sampler: until_conflict, the default, or
the exact single-swap chain until_accept).

START STATES

    random    the composition dealt in the element order of the config and
              shuffled
    b2        first half of the element list on sublattice 0, second half on
              sublattice 1, dealt at random within each half
    layered   the planes perpendicular to z occupied in the cyclic sequence
              `layer_sequence` (integer plane index modulo its length)
    snapshot  a configuration stored by an earlier chain at temperature T

    The element order therefore fixes which species a given random
    permutation puts on which site; it also fixes the pair order of the
    output columns. It does not change any random stream.

RANDOM REFERENCE

    Energies are reported per atom relative to E_rand, the mean energy of
    n_reference random decorations drawn from a generator with a fixed seed,
    so that E - E_rand of different chains is directly comparable.

OUTPUTS, per chain (stem = chain name, in out_dir)

    <stem>.csv            one row per temperature and pair: alpha_1 with
                          block error, eta of the two species and the mean
                          eta, E - E_rand with block error, C_v from the
                          fluctuations and from dE/dT along the branch,
                          acceptance, run lengths
    <stem>.snapshots.npz  final configuration of every temperature (int8
                          model species codes, keys "T<temperature>")
    <stem>.state.npz      current configuration, for resuming
    <stem>.trace.csv      per sweep: E - E_rand, acceptance and alpha_1 of
                          every pair (with trace: true)

RESUME

    A finished temperature is appended to the CSV at once and the
    configuration is checkpointed; a restart skips the finished temperatures
    and continues from the stored configuration with a fresh random stream
    of the same seed. A chain run without interruption is reproducible bit
    for bit on the same hardware; a resumed one is a valid continuation but
    not the same random sequence.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from ..lattice import Supercell, get_lattice, locality_report
from ..model import load_model
from ..paths import load_yaml, resolve
from ..systems import get_system
from .observables import (K_B, block_stats, eta_per_species,
                          heat_capacity_fluctuation, heat_capacity_slope,
                          warren_cowley)
from .sampler import SwapSampler

FIELDS = ["host", "other", "T_K", "mode", "seed", "alpha_1", "alpha_1_err",
          "eta_host", "eta_other", "eta_mean", "E_rel_meV_per_atom",
          "E_err_meV_per_atom", "cv_fluct_kB", "cv_dEdT_kB", "accept_rate",
          "n_atoms", "n_eq", "n_samp", "sweep_s"]

DEFAULTS = dict(size=12, n_eq=2000, n_samp=4000, batch=8,
                walk="until_conflict", trace=True, device="cuda",
                reference_seed=1000, n_reference=8, start_seed_offset=1000,
                layer_sequence=None)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

SHARED = set(DEFAULTS) | {"model", "system", "elements", "temperatures",
                          "out_dir"}
PER_CHAIN = {"name", "mode", "seed", "start"}


def load_config(path):
    """The chains of a file: shared settings, overridden per chain."""
    data = load_yaml(path)
    chains = data.pop("chains")
    unknown = set(data) - SHARED
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    settings = {**DEFAULTS, **data}
    out = []
    for ch in chains:
        unknown = set(ch) - SHARED - PER_CHAIN
        if unknown or not PER_CHAIN <= set(ch):
            raise ValueError(f"{path}: chain {ch.get('name')}: needs "
                             f"{sorted(PER_CHAIN)}, unknown {sorted(unknown)}")
        c = {**settings, **ch}
        if c["mode"] not in ("cool", "heat"):
            raise ValueError(f"chain {c['name']}: mode must be cool or heat")
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# start configurations
# ---------------------------------------------------------------------------

def start_configuration(kind, cell, elements, code, rng, layer_sequence=None):
    """Species codes of a random, b2 or layered start (module docstring)."""
    n = cell.n_sites
    if kind == "random":
        sym = np.array([code[e] for e in elements])[np.arange(n) % len(elements)]
        rng.shuffle(sym)
        return sym.astype(np.int64)
    if kind == "layered":
        seq = list(layer_sequence or [])
        if not seq or set(elements) != set(seq):
            raise ValueError("layered start needs a layer_sequence over the "
                             "elements of the run")
        if cell.box % len(seq):
            raise ValueError(f"layered start needs the box ({cell.box} "
                             f"planes) to hold whole periods of {len(seq)}")
        plane = cell.coords[:, 2] % len(seq)
        return np.array([code[seq[p]] for p in plane], dtype=np.int64)
    if kind == "b2":
        sub = cell.sublattice()
        m = len(elements)
        if m % 2:
            raise ValueError("b2 start needs an even number of elements")
        z = np.empty(n, dtype=np.int64)
        for s, group in ((0, elements[: m // 2]), (1, elements[m // 2:])):
            sites = np.flatnonzero(sub == s)
            sym = np.array([code[e] for e in group])[np.arange(len(sites)) % len(group)]
            rng.shuffle(sym)
            z[sites] = sym
        return z
    raise ValueError(f"unknown start {kind!r}")


def load_snapshot(path, T, n_sites):
    """Configuration stored at temperature T in a <stem>.snapshots.npz."""
    path = resolve(path)
    with np.load(path) as d:
        snaps = {float(k[1:]): d[k].astype(np.int64) for k in d.files}
    T = float(T)
    if T not in snaps:
        raise ValueError(f"{path}: no snapshot at {T:g} K "
                         f"(stored: {sorted(snaps)})")
    z = snaps[T]
    if z.size != n_sites:
        raise ValueError(f"{path}: snapshot has {z.size} sites, the cell "
                         f"has {n_sites}")
    return z, path


# ---------------------------------------------------------------------------
# CSV with resume
# ---------------------------------------------------------------------------

def read_done(path):
    done = {}
    if Path(path).exists():
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                done.setdefault(float(row["T_K"]), []).append(row)
    return done


def append_rows(path, rows, new_file):
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)


def add_dEdT(path):
    """Rewrite the CSV with C_v from the slope of E(T) along the branch."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    per_T = {float(r["T_K"]): float(r["E_rel_meV_per_atom"]) for r in rows}
    Ts = np.array(sorted(per_T))
    if len(Ts) >= 2:
        cv_of = dict(zip(Ts, heat_capacity_slope(Ts, [per_T[t] for t in Ts])))
        for r in rows:
            r["cv_dEdT_kB"] = f"{cv_of[float(r['T_K'])]:.6g}"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _fmt(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------------------
# one chain
# ---------------------------------------------------------------------------

def run_chain(c, out_dir=None):
    device = c["device"]
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    net, cfg, ck = load_model(resolve(c["model"]), device, bind=False)
    if cfg.target != "de":
        raise ValueError(f"{c['model']} is not a site-energy model")
    reach = int(ck["reach"])
    system = get_system(c["system"])
    lattice = get_lattice(system.lattice)
    if lattice.name != cfg.lattice:
        raise ValueError(f"model lattice {cfg.lattice} != system lattice "
                         f"{lattice.name}")
    elements = list(c.get("elements") or system.elements)
    missing = set(elements) - set(cfg.elements)
    if missing:
        raise ValueError(f"{sorted(missing)} not in the model's {cfg.elements}")
    code = {e: i for i, e in enumerate(cfg.elements)}
    n_species = len(cfg.elements)
    seed, mode = int(c["seed"]), c["mode"]
    temps = sorted((float(t) for t in c["temperatures"]), reverse=(mode == "cool"))
    cell = Supercell(lattice, int(c["size"]))
    n = cell.n_sites

    rep = locality_report(lattice, cell.size, reach, tuple(cfg.shells),
                          a=system.lattice_parameter_A)
    print(f"[mc] {c['name']}: {Path(c['model']).name}, reach {reach} hops; "
          f"{cell.size}^3 = {n} sites, {''.join(elements)}, mode {mode}, "
          f"seed {seed}; site locality {'ok' if rep['local'] else 'WRAPS'} "
          f"(r_max {rep['r_max_A']:.2f} A, half box {rep['half_box_A']:.2f} A)")
    if not rep["local"]:
        raise ValueError("cell too small for the receptive field of the model")

    out = Path(resolve(out_dir or c["out_dir"])) / f"{c['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(".state.npz")
    snap_path = out.with_suffix(".snapshots.npz")
    done = read_done(out)
    snapshots = {}
    if snap_path.exists():
        with np.load(snap_path) as d:
            snapshots = {k: d[k] for k in d.files}

    rng_start = np.random.default_rng(int(c["start_seed_offset"]) + seed)
    start = c["start"]
    # the start label goes into the trace of every temperature of the chain
    label = (f"snapshot:{Path(start['snapshot']).name}@{float(start['T']):g}K"
             if isinstance(start, dict) else str(start))
    if state_path.exists() and done:
        with np.load(state_path) as d:
            z = d["z"].astype(np.int64)
        print(f"[mc] resuming: {len(done)} temperatures done, configuration "
              f"from {state_path.name}")
    elif isinstance(start, dict):
        z, src = load_snapshot(start["snapshot"], start["T"], n)
        present = set(int(v) for v in np.unique(z))
        if present != {code[e] for e in elements}:
            raise ValueError(f"{src}: snapshot holds species {sorted(present)}, "
                             f"the run has {[code[e] for e in elements]}")
    else:
        z = start_configuration(start, cell, elements, code, rng_start,
                                c.get("layer_sequence"))

    sampler = SwapSampler(net, cell, cfg.shells, reach, z, device,
                          batch=int(c["batch"]), seed=seed, walk=c["walk"])
    rng_ref = np.random.default_rng(int(c["reference_seed"]))
    rand = np.stack([start_configuration("random", cell, elements, code, rng_ref)
                     for _ in range(int(c["n_reference"]))])
    e_rand = sampler.site_energies(rand).sum(-1).mean() / n
    e0 = (sampler.eps_cpu().sum() / n - e_rand) * 1e3
    geo = sampler.geometry
    print(f"[mc] start {label}: E - E_rand = {e0:+.2f} meV/atom; halo "
          f"{geo.n_h} sites per swapped site, read-out ball "
          f"{len(geo.read_off)}, layer levels {geo.level_sizes}, batch "
          f"{sampler.B} ({sampler.walk}), "
          f"{'CUDA graph' if sampler.graph is not None else 'eager'}",
          flush=True)

    pairs = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
    nn_table = cell.neighbours(lattice.jump_shell)
    sub = cell.sublattice()
    n_eq, n_samp = int(c["n_eq"]), int(c["n_samp"])
    per_T = n_eq + n_samp
    todo = [T for T in temps if T not in done]
    new_file = not out.exists()

    trace_fh = trace_w = None
    if c["trace"]:
        trace_path = out.with_suffix(".trace.csv")
        was_there = trace_path.exists()
        trace_fh = open(trace_path, "a", newline="")
        trace_w = csv.writer(trace_fh)
        if not was_there:
            trace_w.writerow(["T_K", "mode", "seed", "start", "phase", "sweep",
                              "E_rel_meV_per_atom", "accept"]
                             + [f"alpha_{x}-{y}" for x, y in pairs])

    print(f"{'T[K]':>6} {'acc':>6} {'E-E_rand':>10} {'C_v':>6} {'eta':>6} "
          + " ".join(f"{a}-{b}".rjust(8) for a, b in pairs), flush=True)
    t_start = time.time()
    try:
        for T in todo:
            kT = K_B * T
            t_T0 = time.time()
            E, A, ETA, acc = [], [], [], []
            drift = 0.0
            for k in range(per_T):
                a, _ = sampler.sweep(kT)
                z = sampler.z
                # One full-cell evaluation per sweep: the running check of the
                # halo, and the exact site energies for the observables.
                drift = max(drift, sampler.refresh())
                eps = sampler.eps_cpu()
                al = None
                if k >= n_eq:
                    al = warren_cowley(z, nn_table, n_species)
                    E.append(eps.sum() / n)
                    A.append(al)
                    ETA.append(eta_per_species(z, sub, n_species))
                    acc.append(a)
                if trace_w is not None:
                    if al is None:
                        al = warren_cowley(z, nn_table, n_species)
                    trace_w.writerow(
                        [f"{T:g}", mode, seed, label,
                         "eq" if k < n_eq else "samp", k + 1,
                         f"{(eps.sum() / n - e_rand) * 1e3:.4f}", f"{a:.5f}"]
                        + [f"{al[code[x], code[y]]:.6f}" for x, y in pairs])
            E = np.asarray(E)
            e_mean, e_err = block_stats(E)
            a_mean, a_err = block_stats(np.asarray(A))
            eta_mean, _ = block_stats(np.asarray(ETA))
            eta_all = float(np.mean([eta_mean[code[e]] for e in elements]))
            cv = heat_capacity_fluctuation(E, n, T)
            sweep_s = (time.time() - t_T0) / per_T
            rows = [dict(host=a, other=b, T_K=T, mode=mode, seed=seed,
                         alpha_1=f"{a_mean[code[a], code[b]]:.6f}",
                         alpha_1_err=f"{a_err[code[a], code[b]]:.6f}",
                         eta_host=f"{eta_mean[code[a]]:.5f}",
                         eta_other=f"{eta_mean[code[b]]:.5f}",
                         eta_mean=f"{eta_all:.5f}",
                         E_rel_meV_per_atom=f"{(e_mean - e_rand) * 1e3:.4f}",
                         E_err_meV_per_atom=f"{e_err * 1e3:.4f}",
                         cv_fluct_kB=f"{cv:.5f}", cv_dEdT_kB="",
                         accept_rate=f"{np.mean(acc):.5f}", n_atoms=n,
                         n_eq=n_eq, n_samp=n_samp, sweep_s=f"{sweep_s:.2f}")
                    for a, b in pairs]
            append_rows(out, rows, new_file)
            new_file = False
            if trace_fh is not None:
                trace_fh.flush()
            snapshots[f"T{T:.0f}"] = z.astype(np.int8)
            np.savez_compressed(snap_path, **snapshots)
            np.savez_compressed(state_path, z=z, T=T)
            print(f"{T:6.0f} {np.mean(acc):6.3f} {(e_mean - e_rand) * 1e3:+10.2f} "
                  f"{cv:6.3f} {eta_all:6.3f} "
                  + " ".join(f"{a_mean[code[a], code[b]]:+8.3f}" for a, b in pairs)
                  + f"   [{_fmt(time.time() - t_start)}, {sweep_s:.2f} s/sweep, "
                  f"halo deviation {drift * 1e3:.1e} meV]", flush=True)
    finally:
        if trace_fh is not None:
            trace_fh.close()
    add_dEdT(out)
    print(f"[mc] -> {out} (+ .snapshots.npz, .state.npz"
          f"{', .trace.csv' if c['trace'] else ''})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="chain file (YAML)")
    ap.add_argument("--chain", nargs="+", default=None,
                    help="names of the chains to run (default: all, in order)")
    ap.add_argument("--out-dir", default=None, help="override out_dir")
    ap.add_argument("--device", default=None, help="override device")
    args = ap.parse_args(argv)
    chains = load_config(args.config)
    if args.chain:
        names = {c["name"] for c in chains}
        unknown = set(args.chain) - names
        if unknown:
            raise SystemExit(f"unknown chains {sorted(unknown)}; "
                             f"defined: {sorted(names)}")
        chains = [c for c in chains if c["name"] in args.chain]
    for c in chains:
        if args.device:
            c["device"] = args.device
        run_chain(c, args.out_dir)


if __name__ == "__main__":
    main()
