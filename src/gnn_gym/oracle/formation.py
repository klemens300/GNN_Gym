"""Vacancy formation energies from the reference jump dataset.

Every row of the jump dataset holds the relaxed energy of the vacancy-free
cell, E_perfect, and of the same cell with the vacancy, E_initial (the
initial endpoint of the jump), together with the removed species X. The
formation energy of that vacancy in that environment is

    E_f = E_initial - E_perfect + mu_X.

In a concentrated solution the removed atom rejoins the alloy, not its pure
metal, so mu_X is the partial molar energy of X in the alloy at the target
composition. It follows from a regular-solution fit

    e(x) = sum_X x_X mu0_X + sum_{X<Y} W_XY x_X x_Y

to the energy per atom of all perfect cells of the dataset (every subsystem,
every composition): mu_X = e + de/dx_X - sum_Y x_Y de/dx_Y. The pure-element
reference mu_X = energy per atom of the relaxed pure element (unary rows) is
computed alongside for comparison; the two differ by the mixing energy of X.

The dataset samples the whole composition simplex, so the cells of an alloy
are weighted by their distance from the target composition with a Gaussian
kernel and the E_f distribution is resampled with these weights (fixed
seed). The pure elements get one E_f each, the mean over their unary cells.

Outputs (read by gnn_gym.realtime): one spectrum file per alloy and one file
with the pure-element formation energies.

    python -m gnn_gym.oracle.formation configs/oracle/formation.yaml
"""
import argparse
import csv
import json
from itertools import combinations

import numpy as np

from ..paths import load_yaml, resolve
from ..realtime.vacancy import K_B_EV_PER_K, c_v_spectrum
from ..systems import get_system


def read_rows(paths):
    """Rows of the jump datasets with a converged relaxation."""
    rows = []
    for p in paths:
        with open(resolve(p), newline="") as fh:
            rows += [r for r in csv.DictReader(fh)
                     if r.get("relax_conv", "True") == "True"]
    return rows


def fractions(row, elements):
    return np.array([float(row[f"frac_{e}"]) for e in elements])


def fit_regular_solution(rows, elements):
    """mu0 (n,) and W (n, n) of e(x) fitted to E_perfect / n_sites."""
    n = len(elements)
    pairs = list(combinations(range(n), 2))
    X, y = [], []
    for r in rows:
        x = fractions(r, elements)
        X.append(list(x) + [x[i] * x[j] for i, j in pairs])
        y.append(float(r["E_perfect_eV"]) / float(r["n_sites"]))
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    W = np.zeros((n, n))
    for k, (i, j) in enumerate(pairs):
        W[i, j] = W[j, i] = coef[n + k]
    return coef[:n], W, float(resid.std())


def partial_energies(mu0, W, x):
    """mu_X = e + de/dx_X - sum_Y x_Y de/dx_Y at composition x."""
    e = float(mu0 @ x + 0.5 * x @ W @ x)
    grad = mu0 + W @ x
    return e + grad - float(x @ grad)


def subsystem_name(present):
    """Dataset label of a subsystem: its element symbols, sorted."""
    return "".join(sorted(present))


def kernel_resample(sub, elements, target, width, n_samples, seed):
    """Rows of `sub` drawn with Gaussian weights around `target`."""
    xs = np.array([fractions(r, elements) for r in sub])
    w = np.exp(-np.sum((xs - target) ** 2, axis=1) / (2 * width ** 2))
    w /= w.sum()
    n_eff = 1.0 / np.sum(w ** 2)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(sub), size=n_samples, replace=True, p=w)
    return [sub[k] for k in pick], float(n_eff)


def formation_energies(rows, mu, elements):
    """E_f = E_initial - E_perfect + mu[X] for each row."""
    out = []
    for r in rows:
        dE = float(r["E_initial_eV"]) - float(r["E_perfect_eV"])
        out.append(dE + mu[elements.index(r["vac_species"])])
    return np.array(out)


def alloy_spectrum(rows, system, elements, mu0, W, mu_pure, cfg, width):
    target = np.array([system.composition.get(e, 0.0) for e in elements])
    mu_alloy = partial_energies(mu0, W, target)
    label = subsystem_name(system.elements)
    sub = [r for r in rows if r["subsystem"] == label]
    sel, n_eff = kernel_resample(sub, elements, target, width,
                                 int(cfg["n_samples"]), int(cfg["seed"]))
    E_alloy = formation_energies(sel, mu_alloy, elements)
    E_pure = formation_energies(sel, [mu_pure[e] for e in elements], elements)
    return dict(target=target, mu_alloy=mu_alloy, n_rows=len(sub), n_eff=n_eff,
                sel=sel, E_alloy=E_alloy, E_pure=E_pure)


def arrhenius_energy(T1, c1, T2, c2):
    """Effective formation energy -k_B d ln c_v / d(1/T) between two points."""
    return -K_B_EV_PER_K * np.log(c2 / c1) / (1.0 / T2 - 1.0 / T1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="configs/oracle/formation.yaml")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    S_f = float(load_yaml(cfg["realtime"])["S_f_kB"])
    elements = list(cfg["elements"])
    temps = [float(t) for t in cfg["temperatures"]]

    rows = read_rows(cfg["datasets"])
    print(f"[E_f] {len(rows)} rows from {', '.join(map(str, cfg['datasets']))}")
    mu0, W, resid = fit_regular_solution(rows, elements)
    print(f"[E_f] regular-solution fit on {len(rows)} perfect cells: "
          f"residual {1e3 * resid:.1f} meV/atom (std)")

    # pure elements: mu = energy per atom of the relaxed unary cells
    unary = {e: [r for r in rows if r["subsystem"] == e] for e in elements}
    mu_pure = {e: float(np.mean([float(r["E_perfect_eV"]) / float(r["n_sites"])
                                 for r in unary[e]])) for e in elements}
    pure = {}
    for e in elements:
        un = unary[e]
        Ef = formation_energies(un, [mu_pure[x] for x in elements], elements)
        pure[e] = dict(n=len(un), mean=float(Ef.mean()), std=float(Ef.std()),
                       min=float(Ef.min()), max=float(Ef.max()))
        print(f"[E_f] pure {e}: mu {mu_pure[e]:.4f} eV/atom, E_f = "
              f"{Ef.mean():.3f} +- {Ef.std():.3f} eV (n={len(un)}, "
              f"min {Ef.min():.3f}, max {Ef.max():.3f})")
    out = resolve(cfg["pure_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    digits = int(cfg["pure_digits"])
    with open(out, "w") as fh:
        json.dump(dict(
            description="vacancy formation energy in the pure elements: mean "
                        "over the unary cells of the jump dataset, mu = energy "
                        "per atom of the relaxed pure element",
            E_f_eV={e: round(pure[e]["mean"], digits) for e in elements},
            E_f_stats_eV=pure, mu_pure_eV=mu_pure,
            source=[str(p) for p in cfg["datasets"]]), fh, indent=2)
    print(f"[E_f] -> {out}")

    for name, path in cfg["alloys"].items():
        system = get_system(name)
        width = float(cfg["kernel_width"])
        s = alloy_spectrum(rows, system, elements, mu0, W, mu_pure, cfg, width)
        print(f"\n[E_f] {name}: kernel width {width} over {s['n_rows']} cells, "
              f"effective sample size {s['n_eff']:.0f}, resampled "
              f"{len(s['sel'])}")
        for e in system.elements:
            i = elements.index(e)
            print(f"[E_f]   mu_alloy({e}) = {s['mu_alloy'][i]:.4f} eV, "
                  f"mu_alloy - mu_pure = "
                  f"{1e3 * (s['mu_alloy'][i] - mu_pure[e]):+.0f} meV")
        for tag, Ef in (("mu_alloy", s["E_alloy"]), ("mu_pure", s["E_pure"])):
            print(f"[E_f]   E_f ({tag}): mean {Ef.mean():.3f} eV, std "
                  f"{Ef.std():.3f}, min {Ef.min():.3f}, max {Ef.max():.3f}")
        per_species = {}
        vac = np.array([r["vac_species"] for r in s["sel"]])
        for e in system.elements:
            m = vac == e
            if m.any():
                per_species[e] = dict(n=int(m.sum()),
                                      mean=float(s["E_alloy"][m].mean()),
                                      std=float(s["E_alloy"][m].std()))

        wide = alloy_spectrum(rows, system, elements, mu0, W, mu_pure, cfg,
                              float(cfg["kernel_width_check"]))
        print(f"[E_f]   c_v = exp({S_f}) <exp(-E_f/kT)>; columns: mu_alloy, "
              f"mu_pure, at mean E_f, kernel {cfg['kernel_width_check']}")
        table = []
        for T in temps:
            ca = c_v_spectrum(s["E_alloy"], T, S_f)
            cp = c_v_spectrum(s["E_pure"], T, S_f)
            cm = c_v_spectrum(np.array([s["E_alloy"].mean()]), T, S_f)
            cw = c_v_spectrum(wide["E_alloy"], T, S_f)
            table.append(dict(T_K=T, c_v=ca, c_v_mu_pure=cp, c_v_mean_E_f=cm,
                              c_v_kernel_check=cw))
            print(f"    T {T:6.0f} K  {ca:10.3e}  {cp:10.3e}  {cm:10.3e}  "
                  f"{cw:10.3e}   alloy/pure {ca / cp:5.2f}  "
                  f"Jensen/mean {ca / cm:9.3g}  wide/narrow {cw / ca:5.2f}")
        lo, hi = cfg["arrhenius_range"]
        c_lo = c_v_spectrum(s["E_alloy"], float(lo), S_f)
        c_hi = c_v_spectrum(s["E_alloy"], float(hi), S_f)
        print(f"[E_f]   effective formation energy {lo}-{hi} K: "
              f"{arrhenius_energy(float(lo), c_lo, float(hi), c_hi):.3f} eV")

        out = resolve(path)
        with open(out, "w") as fh:
            json.dump(dict(
                system=name,
                composition={e: float(system.composition[e]) for e in system.elements},
                source=[str(p) for p in cfg["datasets"]],
                n_rows=s["n_rows"], kernel_width=width,
                n_effective=s["n_eff"], n_samples=len(s["sel"]),
                seed=int(cfg["seed"]),
                mu_alloy_eV={e: float(s["mu_alloy"][elements.index(e)])
                             for e in elements},
                mu_pure_eV=mu_pure,
                regular_solution=dict(
                    mu0_eV={e: float(v) for e, v in zip(elements, mu0)},
                    W_eV={f"{elements[i]}-{elements[j]}": float(W[i, j])
                          for i, j in combinations(range(len(elements)), 2)},
                    residual_std_eV=resid),
                E_f_per_species=per_species,
                c_v_table=dict(S_f_kB=S_f, rows=table),
                E_f_samples_eV=[float(v) for v in s["E_alloy"]],
                E_f_samples_mu_pure_eV=[float(v) for v in s["E_pure"]],
            ), fh, indent=2)
        print(f"[E_f] -> {out}")


if __name__ == "__main__":
    main()
