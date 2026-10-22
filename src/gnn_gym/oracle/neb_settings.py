"""Image count and force tolerance of the NEB.

The endpoints of a jump are relaxed once per (environment, window) and every
NEB setting starts from the same two structures, so dE is identical across
the sweep and a setting can only move the saddle energy. Each setting of an
image-count x force-tolerance grid is compared with a deliberately expensive
reference band (15 images, fmax 0.01 eV/A). Two window sizes show whether
the NEB error depends on the cell size. Windows and environments are those of
the cell-size study (gnn_gym.oracle.window_study).

    python -m gnn_gym.oracle.neb_settings configs/oracle/neb_settings.yaml run
    python -m gnn_gym.oracle.neb_settings configs/oracle/neb_settings.yaml analyse
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

from ..paths import load_yaml, resolve
from ..systems import get_system
from .window_study import window_endpoints

COLUMNS = ["env", "n", "n_atoms", "axis", "value", "n_images", "fmax", "method",
           "k", "climb", "E_initial_eV", "E_final_eV", "dE_eV", "E_saddle_eV",
           "E_fwd_eV", "E_bwd_eV", "B_sym_eV", "saddle_image", "neb_steps",
           "neb_conv", "t_s", "seed", "timestamp"]


def settings(cfg):
    """[(axis, value, setting)]: the reference first, then the grid."""
    base = dict(cfg["base"])
    out = [("reference", "-", dict(base, **cfg["reference"]))]
    for ni in cfg["grid"]["n_images"]:
        for fm in cfg["grid"]["fmax"]:
            out.append(("grid", f"{ni}img_fmax{fm}", dict(base, n_images=ni, fmax=fm)))
    return out


def run(cfg, out):
    from .uma import Uma
    recipe = load_yaml(cfg["recipe"])
    wcfg = load_yaml(cfg["window_study"])
    system = get_system(wcfg["system"])
    done = set()
    if out.exists():
        with open(out, newline="") as fh:
            done = {(int(r["env"]), int(r["n"]), r["axis"], r["value"])
                    for r in csv.DictReader(fh)}
    uma = Uma.from_recipe(recipe)
    for env in range(int(cfg["n_env"])):
        for n in cfg["windows"]:
            (init_id, fin_id), _vs, _js = window_endpoints(wcfg, system, env, n)
            init_rel, E_i, c1, _ = uma.relax(init_id, wcfg["relax_fmax"])
            fin_rel, E_f, c2, _ = uma.relax(fin_id, wcfg["relax_fmax"])
            for axis, val, s in settings(cfg):
                if (env, n, axis, str(val)) in done:
                    continue
                t0 = time.time()
                _im, en, conv, steps = uma.neb(init_rel, fin_rel,
                                               n_images=s["n_images"], fmax=s["fmax"],
                                               k=s["k"], climb=s["climb"],
                                               method=s["method"], steps=s["steps"])
                E_sad = max(en)
                fwd, bwd = E_sad - en[0], E_sad - en[-1]
                row = {"env": env, "n": n, "n_atoms": len(init_id), "axis": axis,
                       "value": val, "n_images": s["n_images"], "fmax": s["fmax"],
                       "method": s["method"], "k": s["k"], "climb": s["climb"],
                       "E_initial_eV": f"{E_i:.6f}", "E_final_eV": f"{E_f:.6f}",
                       "dE_eV": f"{E_f - E_i:.6f}", "E_saddle_eV": f"{E_sad:.6f}",
                       "E_fwd_eV": f"{fwd:.6f}", "E_bwd_eV": f"{bwd:.6f}",
                       "B_sym_eV": f"{0.5 * (fwd + bwd):.6f}",
                       "saddle_image": int(np.argmax(en)), "neb_steps": steps,
                       "neb_conv": conv, "t_s": f"{time.time() - t0:.1f}",
                       "seed": int(wcfg["seed_base"]) + env,
                       "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                new = not out.exists()
                with open(out, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=COLUMNS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"env {env} n={n:2d} {axis:9s} {str(val):16s} E_fwd {fwd:.4f} "
                      f"B_sym {0.5 * (fwd + bwd):.4f} eV  {steps} steps "
                      f"{time.time() - t0:5.0f} s", flush=True)


def analyse(cfg, log=print):
    with open(resolve(cfg["data"]), newline="") as fh:
        rows = list(csv.DictReader(fh))
    ref = {(r["env"], r["n"]): r for r in rows if r["axis"] == "reference"}
    dev = defaultdict(lambda: defaultdict(list))
    cost = defaultdict(list)
    for r in rows:
        if r["axis"] == "reference":
            continue
        o = ref[(r["env"], r["n"])]
        d = (float(r["B_sym_eV"]) - float(o["B_sym_eV"])) * 1e3
        dev[r["value"]][int(r["n"])].append(d)
        cost[r["value"]].append(float(r["t_s"]))
    t_ref = np.mean([float(r["t_s"]) for r in ref.values()])
    out = {"reference": dict(cfg["reference"]), "settings": {}}
    sizes = sorted({int(n) for _, n in ref})
    log(f"[neb] B_sym - B_sym(reference) in meV per environment; windows {sizes}")
    for val in dev:
        out["settings"][val] = {
            "dB_sym_meV": {str(n): [float(x) for x in v] for n, v in dev[val].items()},
            "max_abs_meV": {str(n): float(np.max(np.abs(v)))
                            for n, v in dev[val].items()},
            "time_rel_reference": float(np.mean(cost[val]) / t_ref)}
        log(f"  {val:16s} " + "  ".join(
            f"n={n}: " + " ".join(f"{x:+5.1f}" for x in dev[val][n]) for n in sizes)
            + f"   time {np.mean(cost[val]) / t_ref:.2f} x reference")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("action", choices=["run", "analyse"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.action == "run":
        out = resolve(args.out or cfg["data"])
        out.parent.mkdir(parents=True, exist_ok=True)
        run(cfg, out)
    else:
        res = analyse(cfg)
        out = resolve(args.out or cfg["results"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"[neb] -> {out}")


if __name__ == "__main__":
    main()
