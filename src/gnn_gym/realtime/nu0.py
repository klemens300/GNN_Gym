"""Attempt frequency nu0 of a composition: Debye frequency from elastic constants.

nu0 is the Debye frequency of the crystal at that composition (Anderson, J.
Phys. Chem. Solids 24, 909 (1963)). The cubic elastic constants C11, C12,
C44 give the bulk modulus and the Voigt-Reuss-Hill shear modulus, these the
longitudinal and transverse sound velocities and their mean v_m, and

    nu_D = v_m / (2 pi) * (6 pi^2 n / V)^(1/3),

with n/V the number density of the ideal crystal at lattice parameter a. It
is the harmonic upper estimate of the jump attempt frequency; harmonic
transition-state prefactors of bcc refractory metals lie about a factor of
three below it. The same convention is used for every composition, so the
ratios between compositions are the robust part.

The elastic constants are uMLIP values of relaxed cells on a regular grid of
the composition simplex (gnn_gym.oracle.elastic). `build` averages them per
grid composition and fits each of C11, C12, C44 and a as a homogeneous cubic
polynomial in the atomic fractions (all monomials x_i x_j x_k; with
sum x = 1 the lower orders are contained in it). A composition on the grid
gets the grid averages, any other composition the fit; nu0 is then computed
from those constants, the lattice parameter and the density of the ideal
crystal at that lattice parameter.

    python -m gnn_gym.realtime.nu0 configs/realtime.yaml          # print table
    python -m gnn_gym.realtime.nu0 configs/realtime.yaml --build  # grid + fit
"""
import argparse
import csv
import itertools
import json
from functools import lru_cache

import numpy as np

from ..lattice import get_lattice
from ..paths import load_yaml, resolve

ATOMIC_MASS_KG = 1.66053907e-27


def density(x, a, masses, atoms_per_cell):
    """Mass density (kg/m^3) of the ideal crystal, mean mass at fractions x."""
    mbar = np.asarray(x) @ np.asarray(masses)
    return atoms_per_cell * mbar * ATOMIC_MASS_KG / (np.asarray(a) * 1e-10) ** 3


def debye_frequency(C11, C12, C44, rho, a, atoms_per_cell):
    """Debye frequency (Hz) from cubic C_ij (GPa), density and a (Angstrom)."""
    C11, C12, C44 = (np.asarray(c, float) * 1e9 for c in (C11, C12, C44))
    B = (C11 + 2 * C12) / 3
    Gv = (C11 - C12 + 3 * C44) / 5
    Gr = 5 * (C11 - C12) * C44 / (4 * C44 + 3 * (C11 - C12))
    G = (Gv + Gr) / 2
    vl = np.sqrt((B + 4 * G / 3) / rho)
    vt = np.sqrt(G / rho)
    vm = (1 / 3 * (1 / vl ** 3 + 2 / vt ** 3)) ** (-1 / 3)
    return vm / (2 * np.pi) * (6 * np.pi ** 2 * atoms_per_cell
                               / (np.asarray(a, float) * 1e-10) ** 3) ** (1 / 3)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

class Nu0Table:
    """Grid averages and cubic fit of the elastic constants."""

    QUANTITIES = ("C11", "C12", "C44", "a")

    def __init__(self, grid_csv, fit_json, lattice):
        with open(resolve(fit_json)) as fh:
            self.fit = json.load(fh)
        self.elements = list(self.fit["elements"])
        self.masses = np.array([self.fit["masses_u"][e] for e in self.elements])
        self.atoms_per_cell = len(get_lattice(lattice).basis)
        with open(resolve(grid_csv), newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.grid_x = np.array([[float(r[e]) for e in self.elements] for r in rows])
        self.grid = {q: np.array([float(r[q]) for r in rows])
                     for q in self.QUANTITIES}
        self.monomials = [[self.elements.index(e) for e in m]
                          for m in self.fit["monomials"]]

    def fractions(self, composition):
        unknown = set(composition) - set(self.elements)
        if unknown:
            raise ValueError(f"no elastic data for {sorted(unknown)}")
        x = np.array([float(composition.get(e, 0.0)) for e in self.elements])
        if x.sum() <= 0:
            raise ValueError("empty composition")
        return x / x.sum()

    def elastic(self, composition, tol=1e-6):
        """C11, C12, C44 (GPa), a (Angstrom), rho (kg/m^3), nu0 (Hz), source."""
        x = self.fractions(composition)
        hit = np.where(np.abs(self.grid_x - x).sum(axis=1) < tol)[0]
        if hit.size:
            i = hit[0]
            out = {q: self.grid[q][i] for q in self.QUANTITIES}
            out["source"] = "grid"
        else:
            D = np.array([np.prod(x[m]) for m in self.monomials])
            out = {q: float(D @ np.array(self.fit["coefficients"][q]))
                   for q in self.QUANTITIES}
            out["source"] = "fit"
        out["rho"] = float(density(x, out["a"], self.masses, self.atoms_per_cell))
        out["nu0"] = float(debye_frequency(out["C11"], out["C12"], out["C44"],
                                           out["rho"], out["a"],
                                           self.atoms_per_cell))
        return out

    def nu0(self, composition):
        return self.elastic(composition)["nu0"]


@lru_cache(maxsize=None)
def load_table(grid_csv, fit_json, lattice="bcc"):
    return Nu0Table(grid_csv, fit_json, lattice)


# ---------------------------------------------------------------------------
# Build: elastic-constant sweep -> grid table + cubic fit
# ---------------------------------------------------------------------------

def build(cfg, log=print):
    """Write the grid table and the fit of cfg['nu0'] from the sweep CSV."""
    import pandas as pd

    c = cfg["nu0"]
    els = list(c["elements"])
    masses = np.array([c["masses_u"][e] for e in els])
    atoms_per_cell = len(get_lattice(c["lattice"]).basis)
    degree = int(c["degree"])

    df = pd.read_csv(resolve(c["source"]))
    # the sweep stores the edge of its size^3 supercell
    df["a_A"] = df.lattice_parameter_A / c["source_cell_size"]
    g = (df.groupby(["composition_string"] + els)
           .agg(n=("C11_GPa", "size"),
                C11=("C11_GPa", "mean"), C11_sd=("C11_GPa", "std"),
                C12=("C12_GPa", "mean"), C12_sd=("C12_GPa", "std"),
                C44=("C44_GPa", "mean"), C44_sd=("C44_GPa", "std"),
                a=("a_A", "mean"), a_sd=("a_A", "std"))
           .reset_index())
    X = g[els].values
    g["rho"] = density(X, g.a.values, masses, atoms_per_cell)
    g["nu_D_Hz"] = debye_frequency(g.C11, g.C12, g.C44, g.rho, g.a,
                                   atoms_per_cell)

    monos = list(itertools.combinations_with_replacement(range(len(els)), degree))

    def design(x):
        return np.column_stack([np.prod(x[:, list(m)], axis=1) for m in monos])

    A = design(X)
    coef, loo = {}, {}
    for q in Nu0Table.QUANTITIES:
        y = g[q].values
        coef[q], *_ = np.linalg.lstsq(A, y, rcond=None)
        res = np.empty(len(y))
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            ci, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
            res[i] = y[i] - A[i] @ ci
        loo[q] = dict(rmse=float(np.sqrt(np.mean(res ** 2))),
                      max_abs=float(np.abs(res).max()))

    def fitted(x):
        x = np.atleast_2d(np.asarray(x, float))
        D = design(x)
        C11, C12, C44, a = (D @ coef[q] for q in Nu0Table.QUANTITIES)
        nu = debye_frequency(C11, C12, C44, density(x, a, masses, atoms_per_cell),
                             a, atoms_per_cell)
        return nu, dict(C11=C11, C12=C12, C44=C44, a=a)

    nu_fit_grid, _ = fitted(X)
    rel = nu_fit_grid / g.nu_D_Hz.values - 1
    centre = np.full(len(els), 1.0 / len(els))
    nu_c, props_c = fitted(centre)

    out = {
        "description": "Debye frequency (Anderson 1963, Voigt-Reuss-Hill) from "
                       "uMLIP elastic constants, fitted as homogeneous "
                       "polynomials in the atomic fractions",
        "source": str(c["source"]),
        "elements": els,
        "degree": degree,
        "monomials": [[els[i] for i in m] for m in monos],
        "coefficients": {q: coef[q].tolist() for q in coef},
        "units": {"C11": "GPa", "C12": "GPa", "C44": "GPa", "a": "Angstrom",
                  "nu0": "Hz"},
        "masses_u": {e: float(m) for e, m in zip(els, masses)},
        "loo": loo,
        "nu0_fit_vs_grid_rel": {"rmse": float(np.sqrt(np.mean(rel ** 2))),
                                "max_abs": float(np.abs(rel).max())},
        "equiatomic": {"nu0_Hz": float(nu_c[0]),
                       **{k: float(v[0]) for k, v in props_c.items()}},
        "pure_elements_grid": {
            e: {"nu0_Hz": float(g.loc[g[e] == 1, "nu_D_Hz"].iloc[0]),
                "a_A": float(g.loc[g[e] == 1, "a"].iloc[0]),
                "C44_GPa": float(g.loc[g[e] == 1, "C44"].iloc[0])}
            for e in els if (g[e] == 1).any()},
    }
    fit_path, grid_path = resolve(c["fit"]), resolve(c["grid"])
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fit_path, "w") as fh:
        json.dump(out, fh, indent=1)
    g.to_csv(grid_path, index=False, float_format="%.6g")

    log(f"[nu0] {len(df)} cells, {len(g)} grid compositions")
    log("[nu0] leave-one-out: " + ", ".join(
        f"{q} {v['rmse']:.3g} (max {v['max_abs']:.3g})" for q, v in loo.items()))
    log(f"[nu0] nu0 fit vs grid: rmse {out['nu0_fit_vs_grid_rel']['rmse']*100:.2f} %, "
        f"max {out['nu0_fit_vs_grid_rel']['max_abs']*100:.2f} %")
    log(f"[nu0] -> {grid_path}\n[nu0] -> {fit_path}")
    load_table.cache_clear()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="configs/realtime.yaml")
    ap.add_argument("--build", action="store_true",
                    help="rebuild grid table and fit from the elastic sweep")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.build:
        build(cfg)
    c = cfg["nu0"]
    table = load_table(c["grid"], c["fit"], c["lattice"])
    from ..systems import load_systems
    for name, s in load_systems().items():
        p = table.elastic(s.composition)
        print(f"{name:8s} nu0 {p['nu0'] / 1e12:6.3f} THz  a {p['a']:.4f} A  "
              f"C11/C12/C44 {p['C11']:6.1f}/{p['C12']:6.1f}/{p['C44']:6.1f} GPa  "
              f"[{p['source']}]")


if __name__ == "__main__":
    main()
