"""Train one jump model (site-energy or barrier) on a jump dataset.

    python -m gnn_gym.training.train configs/training/de_L2.yaml
    python -m gnn_gym.training.train configs/training/bsym_site.yaml --seed 1

The whole dataset is two int8 arrays of n_rows x n_sites entries. It is moved
to the device once and batched by slicing a permutation; there is no data
loader and no graph cache.

Before the fit the receptive field of the encoder is measured. For the
site-energy model it is a gate: if the n-hop ball wraps around the training
cell, the site energies are not functions of a local environment and the run
is refused. After the fit the same jumps are evaluated in a 2x2x2 replica of
the training cell (size transfer) and the referee error is broken down by
subsystem.

Outputs in <out_dir>: <run>.pt (best validation epoch) and <run>.metrics.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..lattice import locality_report
from ..model.checkpoint import save_checkpoint
from ..model.graph import LatticeGraph
from ..model.locality import (measure_reach, readout_masks,
                              size_transfer_error, training_cell)
from ..model.networks import build_network, count_parameters
from ..paths import resolve
from .config import load_run
from .data import build_arrays, split_train_val


def to_device(arrays, mask, device, target):
    z_i = torch.from_numpy(arrays["z_init"]).long().to(device)
    z_f = torch.from_numpy(arrays["z_final"]).long().to(device)
    y = arrays["dE"] if target == "de" else arrays["bsym"]
    y = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    m = torch.from_numpy(np.asarray(mask)).to(device)
    return z_i, z_f, y, m


@torch.no_grad()
def evaluate(net, z_i, z_f, y, mask, batch, subsystem=None):
    """MAE, RMSE and bias in meV; optionally per subsystem with the label
    spread of that subsystem as the scale the error is to be read against."""
    net.eval()
    err = torch.empty_like(y)
    for s in range(0, len(y), batch):
        sl = slice(s, s + batch)
        err[sl] = net(z_i[sl], z_f[sl], mask) - y[sl]
    err = err.cpu().numpy()
    out = {"n": int(len(err)), "mae_meV": float(np.abs(err).mean() * 1e3),
           "rmse_meV": float(np.sqrt((err ** 2).mean()) * 1e3),
           "bias_meV": float(err.mean() * 1e3)}
    if subsystem is not None:
        y_np = y.cpu().numpy()
        out["per_subsystem"] = {
            str(s): {"n": int(m.sum()),
                     "mae_meV": float(np.abs(err[m]).mean() * 1e3),
                     "bias_meV": float(err[m].mean() * 1e3),
                     "std_label_meV": float(y_np[m].std() * 1e3)}
            for s in sorted(set(subsystem.tolist()))
            for m in [subsystem == s]}
    return out, err


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="run file (YAML)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--limit", type=int, help="first N training rows only")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--run-name")
    ap.add_argument("--out-dir")
    ap.add_argument("--device")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    mcfg, tcfg = load_run(args.config)
    for k in ("seed", "limit", "epochs", "run_name", "out_dir", "device"):
        if getattr(args, k) is not None:
            setattr(tcfg, k, getattr(args, k))
    run_name = tcfg.run_name or f"{mcfg.target}_L{mcfg.layers}_s{tcfg.seed}"

    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    device = tcfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("[train] no CUDA, using the CPU")
    torch.backends.cuda.matmul.allow_tf32 = tcfg.tf32
    torch.backends.cudnn.allow_tf32 = tcfg.tf32

    cell = training_cell(mcfg)
    graph = LatticeGraph.for_supercell(cell, mcfg.shells, mcfg.line_graph,
                                       device=device)
    net = build_network(mcfg, n_angle_classes=len(cell.lattice.angle_dots()))
    net.set_graph(graph)
    net = net.to(device)
    print(f"[train] {mcfg.target} model: {count_parameters(net)} parameters, "
          f"line graph {'on' if mcfg.line_graph else 'off'}")

    reach = measure_reach(net, mcfg, cell)
    loc = locality_report(cell.lattice, mcfg.size, reach, mcfg.shells,
                          a=tcfg.lattice_parameter_A)
    print(f"[reach] measured {reach} hops -> {loc['sites_in_ball']} sites "
          f"({loc['sites_unbounded']} unbounded), r_max {loc['r_max_A']:.2f} A, "
          f"half box {loc['half_box_A']:.2f} A, local = {loc['local']}")
    if mcfg.target == "de" and not loc["local"]:
        raise SystemExit(
            f"refusing to train: the site energies reach {reach} hops and wrap "
            f"around the {mcfg.size}^3 cell, so their sum is not a state "
            f"function. Reduce de_layers or enlarge the training cell.")
    if mcfg.target == "bsym" and not loc["local"]:
        print("[reach] the barrier model reaches around the training cell; "
              "the size-transfer check below quantifies the effect")

    changed, pooled = readout_masks(mcfg, reach, cell)
    mask_np = changed if mcfg.target == "de" else pooled
    print(f"[train] read-out mask: {int(mask_np.sum())} of {cell.n_sites} sites")

    train_all = build_arrays(tcfg.train_csv, mcfg, cell)
    referee = build_arrays(tcfg.referee_csv, mcfg, cell)
    if tcfg.limit:
        train_all = {k: v[:tcfg.limit] for k, v in train_all.items()}
    tr, va = split_train_val(train_all, tcfg.val_frac, tcfg.seed)
    print(f"[train] {len(tr['dE'])} train / {len(va['dE'])} validation / "
          f"{len(referee['dE'])} referee")

    zi_tr, zf_tr, y_tr, mask = to_device(tr, mask_np, device, mcfg.target)
    zi_va, zf_va, y_va, _ = to_device(va, mask_np, device, mcfg.target)
    zi_re, zf_re, y_re, _ = to_device(referee, mask_np, device, mcfg.target)

    opt = torch.optim.AdamW(net.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=tcfg.lr_factor, patience=tcfg.lr_patience)
    delta = tcfg.huber_delta_meV * 1e-3
    loss_fn = torch.nn.HuberLoss(delta=delta) if delta > 0 \
        else torch.nn.MSELoss()

    out_dir = resolve(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{run_name}.pt"
    metrics_path = out_dir / f"{run_name}.metrics.json"

    n = len(y_tr)
    best, best_epoch, history = float("inf"), -1, []
    t0 = time.time()
    for epoch in range(tcfg.epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, tcfg.batch_size):
            b = perm[s:s + tcfg.batch_size]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(zi_tr[b], zf_tr[b], mask), y_tr[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), tcfg.grad_clip)
            opt.step()
            tot += float(loss.detach()) * len(b)
        val, _ = evaluate(net, zi_va, zf_va, y_va, mask, tcfg.batch_size)
        sched.step(val["mae_meV"])
        history.append(dict(epoch=epoch, train_loss=tot / n,
                            val_mae_meV=val["mae_meV"],
                            lr=opt.param_groups[0]["lr"]))

        improved = val["mae_meV"] < best - 1e-6
        if improved:
            best, best_epoch = val["mae_meV"], epoch
            ref, _ = evaluate(net, zi_re, zf_re, y_re, mask, tcfg.batch_size,
                              subsystem=referee["subsystem"])
            save_checkpoint(ckpt_path, net, mcfg, reach, epoch=epoch,
                            val_mae_meV=best, referee_mae_meV=ref["mae_meV"],
                            training=tcfg.to_dict())
        if epoch % 5 == 0 or improved:
            print(f"  epoch {epoch:4d}  loss {tot / n:.3e}  "
                  f"val MAE {val['mae_meV']:7.2f} meV  "
                  f"[{(time.time() - t0) / (epoch + 1):.1f} s/epoch]"
                  f"{'  *' if improved else ''}", flush=True)
        if epoch - best_epoch >= tcfg.patience:
            print(f"[train] early stop: no improvement for {tcfg.patience} "
                  f"epochs")
            break

    print(f"\n[train] best validation MAE {best:.2f} meV at epoch {best_epoch} "
          f"({time.time() - t0:.0f} s)")
    net.load_state_dict(torch.load(ckpt_path, map_location=device,
                                   weights_only=True)["state_dict"])
    ref, _ = evaluate(net, zi_re, zf_re, y_re, mask, tcfg.batch_size,
                      subsystem=referee["subsystem"])
    transfer = None
    if tcfg.transfer_samples:
        transfer = size_transfer_error(net, mcfg, reach,
                                       n_samples=tcfg.transfer_samples,
                                       device=device)
        print(f"[transfer] same jump in a 2x2x2 replica: mean "
              f"{transfer['mean_meV']:.3f} meV, max {transfer['max_meV']:.3f} "
              f"meV over {transfer['n']} jumps")

    metrics = dict(run=run_name, model=mcfg.to_dict(), training=tcfg.to_dict(),
                   reach=reach, locality=loc, epoch=best_epoch,
                   val_mae_meV=best, referee=ref, n_train=int(n),
                   size_transfer=transfer, history=history)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    print(f"[train] referee MAE {ref['mae_meV']:.2f} meV "
          f"(bias {ref['bias_meV']:+.2f})")
    for s, v in sorted(ref["per_subsystem"].items(),
                       key=lambda kv: -kv[1]["mae_meV"]):
        print(f"    {s:10s} n={v['n']:4d}  MAE {v['mae_meV']:7.2f} meV   "
              f"label std {v['std_label_meV']:8.1f} meV")
    print(f"[train] -> {ckpt_path}")


if __name__ == "__main__":
    main()
