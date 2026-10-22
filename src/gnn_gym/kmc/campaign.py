"""Run every point of a KMC campaign.

    python -m gnn_gym.kmc.campaign configs/kmc/escape_local.yaml
    python -m gnn_gym.kmc.campaign configs/kmc/escape_local.yaml --only kmc_MoNbTaW_T700
    python -m gnn_gym.kmc.campaign configs/kmc/escape_local.yaml --dry-run

Points are run one after another. A point whose checkpoint holds its step
target is skipped, an interrupted one continues from its checkpoint, so an
interrupted campaign is finished by running the same command again. Raising
n_steps of a point in the config extends its trajectory.
"""
import argparse
import sys
import time

from . import checkpoint as ck
from .config import load_campaign
from .runner import make_cell, make_predictor, run_point


def status(campaign, point):
    path = ck.checkpoint_path(campaign.run_dir(point), point.T_K)
    return ck.stored_steps(path) if path.exists() else 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="campaign file, configs/kmc/*.yaml")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME",
                    help="run only these points (run directory names)")
    ap.add_argument("--device", default="auto", help="cuda, cpu or auto")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the points and their state, run nothing")
    args = ap.parse_args(argv)

    camp = load_campaign(args.config)
    points = camp.points
    if args.only:
        points = [p for p in points if p.name in args.only]
        missing = set(args.only) - {p.name for p in points}
        if missing:
            ap.error(f"no such points: {sorted(missing)}")

    print(f"[campaign] {args.config}: {len(points)} points -> {camp.runs_dir}")
    todo = []
    for p in points:
        done = status(camp, p)
        state = "complete" if done >= p.n_steps else f"{done}/{p.n_steps}"
        print(f"  {p.name:22s} seed {p.seed:>6d}  escape {str(p.escape):5s}"
              f"  {state}")
        if done < p.n_steps:
            todo.append(p)
    if args.dry_run or not todo:
        return 0

    for k, p in enumerate(todo, 1):
        print(f"\n[campaign] ({k}/{len(todo)}) {p.name}  {time.strftime('%F %T')}",
              flush=True)
        t0 = time.time()
        cell = make_cell(p)
        predictor = make_predictor(p, camp.de_model, camp.bsym_model,
                                   device=args.device, cell=cell)
        run_point(p, predictor, camp.run_dir(p))
        print(f"[campaign] {p.name} done in {(time.time() - t0) / 3600:.2f} h",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
