"""Cost of a KMC step: the model alone and the whole engine.

    python -m gnn_gym.validation.speed configs/validation/speed.yaml [--device cuda]

A step evaluates the barriers of all candidate jumps of the vacancy, so all
timings are per step (divide by the coordination for the cost per barrier).
Three levels are timed on a random configuration of the system:

heads      the two models on the whole cell, the candidates batched, the
           B_sym encoder fed the initial configuration once per candidate:
           the cost of the plain formulation
predictor  gnn_gym.model.predictor.JumpPredictor on the whole cell (initial
           configuration encoded once) and on the halo around the vacancy
           (production); the halo cost does not depend on the cell size
engine     complete KMC runs through gnn_gym.kmc.runner at several
           temperatures, with escape layer, snapshots and checkpoints as in
           the campaigns, split into predictor time and the rest
campaign   the steps of the listed campaign configs at the mean engine cost
           per step, against the same number of barriers from uMLIP-NEB at
           the given cost per barrier

Results are written as JSON.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..kmc.config import RunConfig, load_campaign
from ..kmc.runner import initial_state, make_cell, make_predictor, run_point
from ..paths import load_yaml, resolve


def _timed(fn, n, device, warmup=10):
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def model_timings(cfg, de, bsym, device, n_calls):
    """ms per step: heads on the full cell, predictor full cell and halo."""
    full = make_predictor(cfg, de, bsym, device=device, halo=False)
    halo = make_predictor(cfg, de, bsym, device=device, halo=True,
                          cell=full.cell)
    state = initial_state(cfg, full.cell)
    js = state.targets()
    zi, zf, de_mask, bs_mask = full._cell_inputs(
        state.vacancy, js, full.model_species(state.species, state.vacancy))

    @torch.no_grad()
    def de_head():
        eps = full.de_net.site_energies(torch.cat([zi, zf]))
        return ((eps[1:] - eps[:1]) * de_mask).sum(-1)

    @torch.no_grad()
    def bsym_head():
        return full.bs_net(zi.expand(len(js), -1), zf, bs_mask)

    def both():
        de_head()
        bsym_head()

    out = dict(
        n_sites=full.cell.n_sites,
        halo_sites=int(halo.n_sub - 1) if halo.use_halo else full.cell.n_sites,
        heads_ms=dict(dE=_timed(de_head, n_calls, device),
                      B_sym=_timed(bsym_head, n_calls, device),
                      together=_timed(both, n_calls, device)),
        predictor_ms=dict(
            full_cell=_timed(lambda: full.barriers(state.species, state.vacancy,
                                                   js), n_calls, device),
            halo=_timed(lambda: halo.barriers(state.species, state.vacancy, js),
                        n_calls, device)))
    out["candidates"] = int(len(js))
    return out


def engine_timings(cfg, de, bsym, device, temperatures):
    """ms per step of complete runs, predictor share separated."""
    pred = make_predictor(cfg, de, bsym, device=device, cell=make_cell(cfg))
    rows = {}
    for T in temperatures:
        point = RunConfig.from_dict({**cfg.to_dict(), "T_K": float(T)})
        pred.reset_timers()
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            run_point(point, pred, Path(tmp), log=lambda *a: None)
            wall = time.perf_counter() - t0
        st = pred.stats()
        per_step = wall / point.n_steps * 1e3
        rows[str(int(T))] = dict(ms_per_step=per_step,
                                 ms_predictor=st["seconds"] / point.n_steps * 1e3,
                                 predictor_calls=st["calls"],
                                 clamped=st["clamped"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="configs/validation/speed.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    de, bsym = str(resolve(cfg["de_model"])), str(resolve(cfg["bsym_model"]))
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"
    result = dict(device=device, gpu=gpu, system=cfg["system"], model={},
                  engine={})

    for size in cfg["cell_sizes"]:
        run = RunConfig(system=cfg["system"], T_K=1000.0, seed=cfg["seed"],
                        n_steps=1, supercell_size=int(size))
        m = model_timings(run, de, bsym, device, int(cfg["predictor_calls"]))
        result["model"][str(size)] = m
        print(f"[speed] {size}^3 = {m['n_sites']} sites, halo "
              f"{m['halo_sites']} sites, {m['candidates']} candidates per step")
        for k, v in m["heads_ms"].items():
            print(f"  head {k:9s} {v:8.2f} ms/step (full cell)")
        for k, v in m["predictor_ms"].items():
            print(f"  predictor {k:9s} {v:8.2f} ms/step   "
                  f"{v / m['candidates']:.2f} ms/barrier")

    e = cfg["engine"]
    run = RunConfig.from_dict({"system": cfg["system"], "T_K": 1000.0,
                               "seed": cfg["seed"], **e["run"]})
    rows = engine_timings(run, de, bsym, device, e["temperatures_K"])
    result["engine"] = dict(settings=run.to_dict(), rows=rows)
    print(f"[speed] engine, {run.supercell_size}^3, {run.n_steps} steps per "
          f"temperature, escape {run.escape}")
    for T, r in rows.items():
        print(f"  {T:>5s} K  {r['ms_per_step']:8.2f} ms/step  predictor "
              f"{r['ms_predictor']:8.2f} ms  rest "
              f"{r['ms_per_step'] - r['ms_predictor']:6.2f} ms")

    camp = cfg["campaign"]
    points = [p for c in camp["configs"] for p in load_campaign(c).points]
    steps = sum(p.n_steps for p in points)
    ms = float(np.mean([r["ms_per_step"] for r in rows.values()]))
    z = result["model"][str(run.supercell_size)]["candidates"]
    neb = float(camp["umlip_neb_s_per_barrier"])
    result["campaign"] = dict(
        configs=list(camp["configs"]), runs=len(points), steps=steps,
        ms_per_step=ms, gpu_days=steps * ms / 1e3 / 86400.0,
        umlip_neb_s_per_barrier=neb,
        umlip_years=steps * z * neb / (365.25 * 86400.0))
    c = result["campaign"]
    print(f"[speed] campaign: {c['runs']} runs, {steps:.3g} steps, "
          f"{c['gpu_days']:.1f} GPU-days at {ms:.1f} ms/step; with uMLIP-NEB "
          f"({neb:.0f} s per barrier) {c['umlip_years']:.0f} years")

    out = resolve(args.out or cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[speed] -> {out}")
    return 0


if __name__ == "__main__":
    main()
