"""Jump datasets: one CSV row -> two species vectors and two labels.

A row stores scalars only: the realised composition, the seed of the
decoration and the labels of one relaxed NEB calculation. The structure is
rebuilt from them with the same `decorate` the generator used, and the
vacancy and the jumping atom always sit on the reference sites of the
training cell (`Supercell.reference_jump`). The rebuild is checked: the
species it puts on those two sites must equal the row's `vac_species` and
`jump_species`, otherwise the generator and the loader disagree about the
construction convention.

The model input is the ideal lattice, the label the relaxed NEB result: the
KMC only knows on-lattice configurations, so this is the mapping it later
evaluates.

    initial   vacancy on V, jumping species on J
    final     jumping species on V, vacancy on J
    dE        E(final) - E(initial)
    B_sym     (E_fwd + E_bwd) / 2
"""
import csv
from pathlib import Path

import numpy as np

from ..lattice import counts_from_fractions, decorate
from ..paths import resolve


def load_rows(csv_path, size):
    """Rows of a jump CSV with converged relaxation and NEB."""
    rows, skipped = [], {"unconverged": 0, "wrong_size": 0}
    with open(resolve(csv_path), newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["size"]) != size:
                skipped["wrong_size"] += 1
                continue
            if r.get("relax_conv", "True") != "True" or \
                    r.get("neb_conv", "True") != "True":
                skipped["unconverged"] += 1
                continue
            rows.append(r)
    return rows, skipped


def build_arrays(csv_path, cfg, cell, log=print):
    """Species arrays and labels of every usable row.

    z_init, z_final   (n_rows, n_sites) int8 species indices into
                      cfg.elements, cfg.vac_index marks the vacancy
    dE, bsym          labels in eV
    jump_idx          species index of the migrating atom
    subsystem, idx    bookkeeping columns of the CSV
    """
    n = cell.n_sites
    v, j = cell.reference_jump()
    code = {e: i for i, e in enumerate(cfg.elements)}
    vac = cfg.vac_index

    rows, skipped = load_rows(csv_path, cell.size)
    if not rows:
        raise RuntimeError(f"no usable rows in {csv_path}")

    z_i = np.empty((len(rows), n), dtype=np.int8)
    z_f = np.empty((len(rows), n), dtype=np.int8)
    dE = np.empty(len(rows), dtype=np.float64)
    bsym = np.empty(len(rows), dtype=np.float64)
    jump_idx = np.empty(len(rows), dtype=np.int8)
    idx = np.empty(len(rows), dtype=np.int64)
    subsystem = []

    for k, r in enumerate(rows):
        fracs = {e: float(r[f"frac_{e}"]) for e in cfg.elements}
        symbols = decorate(counts_from_fractions(fracs, n), int(r["seed"]), n)
        if symbols[v] != r["vac_species"] or symbols[j] != r["jump_species"]:
            raise RuntimeError(
                f"{csv_path} row idx={r['idx']}: rebuilt decoration has "
                f"({symbols[v]}, {symbols[j]}) on (V, J), the row says "
                f"({r['vac_species']}, {r['jump_species']})")

        z = np.fromiter((code[s] for s in symbols), dtype=np.int8, count=n)
        z_i[k] = z
        z_i[k, v] = vac
        z_f[k] = z
        z_f[k, v] = code[r["jump_species"]]
        z_f[k, j] = vac

        dE[k] = float(r["dE_eV"])
        bsym[k] = float(r["B_sym_eV"])
        jump_idx[k] = code[r["jump_species"]]
        subsystem.append(r["subsystem"])
        idx[k] = int(r["idx"])

    log(f"[data] {Path(csv_path).name}: {len(rows)} rows (skipped "
        f"{skipped['unconverged']} unconverged, {skipped['wrong_size']} of "
        f"another cell size), reconstruction verified")
    return dict(z_init=z_i, z_final=z_f, dE=dE, bsym=bsym, idx=idx,
                jump_idx=jump_idx, subsystem=np.array(subsystem))


def split_train_val(arrays, val_frac, seed):
    """Deterministic train/validation split (the referee set is separate)."""
    n = len(arrays["dE"])
    perm = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val, train = perm[:n_val], perm[n_val:]

    def take(i):
        return {k: v[i] for k, v in arrays.items()}

    return take(train), take(val)
