"""Referee error against training-set size.

    python -m gnn_gym.training.learning_curve configs/training/de_L2.yaml \
        --limits 1000 2500 5000 10000 --out results/model/learning_curve_de.csv

Each point is a full training run on the first `limit` rows of the training
set (90 % fit, 10 % early stopping, as for the full model); the referee set
is always complete, so all points are judged on the same jumps.
"""
import argparse
import csv
import json

from ..paths import resolve
from .config import load_run
from .train import main as train_main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--limits", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/learning_curve")
    ap.add_argument("--out", required=True, help="CSV with one row per point")
    args = ap.parse_args(argv)

    mcfg, tcfg = load_run(args.config)
    base = tcfg.run_name or mcfg.target
    rows = []
    for limit in args.limits:
        name = f"{base}_n{limit}"
        train_main([args.config, "--limit", str(limit), "--seed", str(args.seed),
                    "--run-name", name, "--out-dir", args.out_dir])
        m = json.loads((resolve(args.out_dir) / f"{name}.metrics.json").read_text())
        rows.append({"limit": limit, "n_train": m["n_train"], "model": name,
                     "seed": args.seed,
                     "referee_mae_meV": round(m["referee"]["mae_meV"], 3),
                     "val_mae_meV": round(m["val_mae_meV"], 3)})

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[learning curve] -> {out}")


if __name__ == "__main__":
    main()
