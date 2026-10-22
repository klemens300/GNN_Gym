"""Neighbour fractions in the observable of a two-group EXAFS fit.

An EXAFS fit with two averaged neighbour types, a group of elements against
the rest (4d against 5d for Mo-Nb-Ta-W), reports for every absorber e and
shell s the fraction of neighbours that belong to the group; it is the sum
of the group concentrations for the random alloy. From a configuration it
follows from the Warren-Cowley parameters of that shell,

    f(e, s) = sum over j in the group of P_s(j | e) = sum_j c_j (1 - alpha_s(e, j)).

Every stored configuration at the requested temperature (one per chain)
gives one value; mean and spread over the chains are reported next to the
measured value.
"""
import csv

import numpy as np

from ..paths import resolve
from .observables import warren_cowley_shells


def read_measurement(path):
    """{(absorber, shell): (fraction, error)} from a CSV with the columns
    absorber, shell, fraction, error ('#' lines are comments)."""
    with open(resolve(path), newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return {(r["absorber"], int(r["shell"])): (float(r["fraction"]),
                                               float(r["error"]))
            for r in csv.DictReader(lines)}


def group_fractions(z, cell, model_elements, group, shells=(1, 2)):
    """{(absorber, shell): fraction of group neighbours} of one configuration."""
    code = {e: i for i, e in enumerate(model_elements)}
    comp = np.bincount(z, minlength=len(model_elements)) / z.size
    al = warren_cowley_shells(z, cell, len(model_elements), shells)
    present = [e for e in model_elements if comp[code[e]] > 0]
    return {(e, s): float(sum(comp[code[j]] * (1.0 - al[s][code[e], code[j]])
                              for j in group))
            for e in present for s in shells}


def compare(configurations, cell, model_elements, group, measured,
            shells=(1, 2)):
    """Rows absorber, shell, model mean/sd/min/max over the configurations,
    measured value, error and (model - measured) / error."""
    vals = {}
    for z in configurations:
        for key, v in group_fractions(z, cell, model_elements, group,
                                      shells).items():
            vals.setdefault(key, []).append(v)
    rows = []
    for (e, s), v in vals.items():
        v = np.asarray(v)
        meas, err = measured.get((e, s), (np.nan, np.nan))
        mean = float(v.mean())
        rows.append(dict(absorber=e, shell=s, n_configurations=len(v),
                         model_mean=mean,
                         model_sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                         model_min=float(v.min()), model_max=float(v.max()),
                         measured=meas, error=err,
                         z_score=(mean - meas) / err if np.isfinite(err) else np.nan))
    return rows
