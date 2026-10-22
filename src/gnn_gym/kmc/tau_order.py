"""Ordering time from the relaxation of a Warren-Cowley parameter.

    alpha(t) = alpha_inf + (alpha_0 - alpha_inf) exp(-t / tau)

is fitted to alpha(t) of a run, either with alpha_inf free or pinned to the
equilibrium value from Metropolis sampling. Time is normalised to [0, 1] for
the fit and rescaled afterwards; the start value of tau comes from the
half-way point of the visible change. alpha_0 and alpha_inf are bounded to
[-1, 1], and tau to at most 1e4 times the covered span: beyond that the curve
is a straight line and tau is not identifiable.

The status says what the number is worth:

    converged     the curve covers at least `converged_spans` tau
    step_limited  it covers less: alpha_inf is an extrapolation; tau is at
                  least of order the span and at most tau_upper_bound_s
    no_trend      the fitted change is below three times the residual scatter

An exponential relaxation is concave, so the change observed over the span
is at most the initial slope times the span; with |alpha_inf| <= 1 this
bounds tau from above independently of the fit (tau_upper_bound_s).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TauOrderFit:
    tau_order_s: float
    alpha_0: float
    alpha_inf: float
    n_points_used: int
    rmse: float
    status: str
    t_span_s: float
    fraction_completed: float     # 1 - exp(-t_span / tau)
    tau_upper_bound_s: float
    alpha_inf_fixed: bool

    @property
    def converged(self):
        return self.status == "converged"


def fit_tau_order(times_s, alpha_t, min_points=5, alpha_inf_fixed=None,
                  converged_spans=2.0):
    """TauOrderFit, or None if there are too few finite points or the fit fails."""
    from scipy.optimize import curve_fit

    t = np.asarray(times_s, dtype=np.float64)
    a = np.asarray(alpha_t, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(a)
    t, a = t[mask], a[mask]
    if t.shape[0] < min_points:
        return None
    t_min = float(t[0])
    t_scale = float(t[-1] - t[0])
    if t_scale <= 0:
        return None
    t_norm = (t - t_min) / t_scale

    def model(tt, tau_norm, alpha_0, alpha_inf):
        return alpha_inf + (alpha_0 - alpha_inf) * np.exp(-tt / tau_norm)

    a0_init = float(np.clip(a[0], -1.0, 1.0))
    ainf_init = (float(alpha_inf_fixed) if alpha_inf_fixed is not None
                 else float(np.clip(a[-1], -1.0, 1.0)))
    midpoint = 0.5 * (a0_init + ainf_init)
    t_half = float(t_norm[int(np.argmin(np.abs(a - midpoint)))])
    tau_init = max(t_half / float(np.log(2.0)), 1e-6) if t_half > 1e-6 \
        else 1.0 / 3.0

    tau_max = 1e4
    try:
        if alpha_inf_fixed is None:
            popt, _ = curve_fit(
                model, t_norm, a, p0=(tau_init, a0_init, ainf_init),
                bounds=([1e-12, -1.0, -1.0], [tau_max, 1.0, 1.0]),
                maxfev=10000, ftol=1e-12, xtol=1e-12)
            tau_norm, a0, ainf = (float(popt[0]), float(popt[1]),
                                  float(popt[2]))
        else:
            ainf = float(alpha_inf_fixed)
            popt, _ = curve_fit(
                lambda tt, tau_norm, alpha_0: model(tt, tau_norm, alpha_0, ainf),
                t_norm, a, p0=(tau_init, a0_init),
                bounds=([1e-12, -1.0], [tau_max, 1.0]),
                maxfev=10000, ftol=1e-12, xtol=1e-12)
            tau_norm, a0 = float(popt[0]), float(popt[1])
    except Exception:
        return None

    residuals = a - model(t_norm, tau_norm, a0, ainf)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    spans = 1.0 / tau_norm
    if abs(ainf - a0) < 3.0 * max(rmse, 1e-12):
        status = "no_trend"
    elif spans >= converged_spans:
        status = "converged"
    else:
        status = "step_limited"
    d_obs = abs(float(a[-1]) - a0)
    tau_ub = (abs(np.sign(ainf - a0) * 1.0 - a0) * t_scale / d_obs
              if d_obs > 0 else np.nan)
    return TauOrderFit(
        tau_order_s=tau_norm * t_scale, alpha_0=a0, alpha_inf=ainf,
        n_points_used=int(t.shape[0]), rmse=rmse, status=status,
        t_span_s=t_scale, fraction_completed=float(1.0 - np.exp(-spans)),
        tau_upper_bound_s=float(tau_ub),
        alpha_inf_fixed=alpha_inf_fixed is not None)
