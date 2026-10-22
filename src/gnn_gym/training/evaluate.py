"""Error of the assembled barrier E_a = B_sym + 1/2 dE on a jump dataset.

    python -m gnn_gym.training.evaluate \
        --de models/MoNbTaW/de_L2_s0.pt --bsym models/MoNbTaW/bsym_site_s0.pt \
        --csv data/jumps/MoNbTaW_7cube_referee.csv \
        --out results/model/barrier_eval.json \
        --predictions results/model/referee_predictions.csv

The two models are trained and reported separately, but a rate depends on
the assembled barrier, so its error is what the simulation feels. Both labels
come from the same NEB calculation; whether the two errors add up or partly
cancel is measured (their correlation). The error is also reported for the
fastest quartile of jumps, which carries the KMC time scale, and translated
into the factor by which a single rate can be off at a few temperatures.
"""
import argparse
import csv
import json

import numpy as np
import torch

from ..model.checkpoint import load_model
from ..model.locality import readout_masks, training_cell
from ..paths import resolve
from .data import build_arrays

K_B = 8.617333262e-5          # eV/K
RATE_TEMPERATURES = (300, 450, 700, 1000, 1500)


@torch.no_grad()
def predict(net, arrays, mask, device, batch=64):
    z_i = torch.from_numpy(arrays["z_init"]).long().to(device)
    z_f = torch.from_numpy(arrays["z_final"]).long().to(device)
    m = torch.from_numpy(np.asarray(mask)).to(device)
    out = np.empty(len(arrays["dE"]), dtype=np.float64)
    for s in range(0, len(out), batch):
        sl = slice(s, s + batch)
        out[sl] = net(z_i[sl], z_f[sl], m).cpu().numpy()
    return out


def error_stats(err, what):
    e = err * 1e3
    return {"what": what, "n": int(len(e)), "mae_meV": float(np.abs(e).mean()),
            "rmse_meV": float(np.sqrt((e ** 2).mean())),
            "bias_meV": float(e.mean())}


def evaluate_pair(de_path, bsym_path, csv_path, device):
    """Predictions and labels of both models on one dataset."""
    de_net, de_cfg, de_ck = load_model(de_path, device)
    bs_net, bs_cfg, _ = load_model(bsym_path, device)
    if de_cfg.target != "de" or bs_cfg.target != "bsym":
        raise SystemExit("--de must be a site-energy and --bsym a barrier model")
    if de_cfg.elements != bs_cfg.elements or de_cfg.size != bs_cfg.size:
        raise SystemExit("the two models were trained on different systems")

    cell = training_cell(de_cfg)
    arrays = build_arrays(csv_path, de_cfg, cell)
    de_mask = readout_masks(de_cfg, de_ck["reach"], cell)[0]
    bs_mask = readout_masks(bs_cfg, de_ck["reach"], cell)[1]
    pred = {"dE": predict(de_net, arrays, de_mask, device),
            "bsym": predict(bs_net, arrays, bs_mask, device)}
    pred["Ea"] = pred["bsym"] + 0.5 * pred["dE"]
    true = {"dE": arrays["dE"], "bsym": arrays["bsym"]}
    true["Ea"] = true["bsym"] + 0.5 * true["dE"]
    return arrays, de_cfg, true, pred


def summarise(arrays, true, pred):
    e = {k: pred[k] - true[k] for k in ("dE", "bsym", "Ea")}
    out = {"dE": error_stats(e["dE"], "dE"),
           "B_sym": error_stats(e["bsym"], "B_sym"),
           "E_a": error_stats(e["Ea"], "E_a")}
    for key, q in (("dE", "dE"), ("B_sym", "bsym"), ("E_a", "Ea")):
        out[key]["pearson_r"] = float(np.corrcoef(true[q], pred[q])[0, 1])
    out["error_correlation"] = float(np.corrcoef(e["bsym"], 0.5 * e["dE"])[0, 1])
    out["sigma_if_independent_meV"] = float(
        np.sqrt(np.var(e["bsym"]) + np.var(0.5 * e["dE"])) * 1e3)

    q = float(np.quantile(true["Ea"], 0.25))
    fast = true["Ea"] <= q
    out["E_a_fastest_quartile"] = error_stats(e["Ea"][fast],
                                              "E_a (fastest quartile)")
    out["fastest_quartile_cut_eV"] = q
    out["rate_factor"] = {
        str(T): float(np.exp(out["E_a"]["mae_meV"] * 1e-3 / (K_B * T)))
        for T in RATE_TEMPERATURES}

    sub = arrays["subsystem"]
    out["per_subsystem"] = {
        str(s): {"n": int(m.sum()),
                 "dE_meV": float(np.abs(e["dE"][m]).mean() * 1e3),
                 "B_sym_meV": float(np.abs(e["bsym"][m]).mean() * 1e3),
                 "E_a_meV": float(np.abs(e["Ea"][m]).mean() * 1e3),
                 "std_label_meV": float(true["Ea"][m].std() * 1e3)}
        for s in sorted(set(sub.tolist())) for m in [sub == s]}
    return out


def write_predictions(path, arrays, elements, true, pred):
    """One row per jump: the input of the parity figure."""
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "subsystem", "jump_species",
                    "dE_true_eV", "dE_pred_eV", "bsym_true_eV", "bsym_pred_eV",
                    "Ea_true_eV", "Ea_pred_eV"])
        for k in range(len(true["dE"])):
            w.writerow([int(arrays["idx"][k]), arrays["subsystem"][k],
                        elements[int(arrays["jump_idx"][k])]]
                       + [f"{d[q][k]:.6f}" for q in ("dE", "bsym", "Ea")
                          for d in (true, pred)])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--de", required=True, help="site-energy checkpoint")
    ap.add_argument("--bsym", required=True, help="barrier checkpoint")
    ap.add_argument("--csv", default="data/jumps/MoNbTaW_7cube_referee.csv")
    ap.add_argument("--out", default="results/model/barrier_eval.json")
    ap.add_argument("--predictions", default=None,
                    help="per-jump CSV (parity figure input)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    arrays, cfg, true, pred = evaluate_pair(args.de, args.bsym, args.csv,
                                            device)
    out = summarise(arrays, true, pred)
    out["inputs"] = {"de": args.de, "bsym": args.bsym, "csv": args.csv}

    print(f"[evaluate] {out['E_a']['n']} jumps")
    for k in ("dE", "B_sym", "E_a"):
        r = out[k]
        print(f"  {k:6s} MAE {r['mae_meV']:7.2f} meV   RMSE "
              f"{r['rmse_meV']:7.2f}   bias {r['bias_meV']:+6.2f}   "
              f"r {r['pearson_r']:.4f}")
    print(f"  correlation of the two errors: {out['error_correlation']:+.3f}")
    print(f"  fastest quartile (E_a <= {out['fastest_quartile_cut_eV']:.3f} eV)"
          f": MAE {out['E_a_fastest_quartile']['mae_meV']:.2f} meV")
    for T, f in out["rate_factor"].items():
        print(f"  {int(T):5d} K  a single rate is off by up to a factor {f:.2f}")
    print(f"\n{'subsystem':10s} {'n':>5s} {'dE':>8s} {'B_sym':>8s} {'E_a':>8s}")
    for s, v in sorted(out["per_subsystem"].items(),
                       key=lambda kv: -kv[1]["E_a_meV"]):
        print(f"{s:10s} {v['n']:5d} {v['dE_meV']:8.2f} {v['B_sym_meV']:8.2f} "
              f"{v['E_a_meV']:8.2f}")

    path = resolve(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\n[evaluate] -> {path}")
    if args.predictions:
        write_predictions(args.predictions, arrays, cfg.elements, true, pred)
        print(f"[evaluate] -> {resolve(args.predictions)}")


if __name__ == "__main__":
    main()
