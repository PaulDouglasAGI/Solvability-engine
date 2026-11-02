"""'What actually happened to cases like this one?'

Finds real reported cases in the training data that match the answers given, and
summarises their outcomes directly: how many there were, and the observed share
cleared by each horizon (Kaplan-Meier, so cases still open count correctly). This
sits beside the model's estimate so every number can be traced to real records.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REF_COLS = ["source", "jurisdiction", "item_category", "location_type", "victim_type",
            "loss_value", "days_to_report", "pattern_count", "offender_count", "insider", "has_footage", "has_witness",
            "has_plate_read", "incident_date", "cleared", "duration_days", "followup_days",
            "timing_known"]

# Order in which match criteria are relaxed when too few cases match
RELAX_ORDER = ["days_to_report", "victim_type", "jurisdiction", "loss_value", "location_type", "item_category"]
MIN_MATCHES = 100


def compact_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only what's needed to look up similar cases, in small dtypes."""
    ref = df[[c for c in REF_COLS if c in df.columns]].copy()
    for c in ["source", "jurisdiction", "item_category", "location_type", "victim_type"]:
        ref[c] = ref[c].astype("category")
    for c in ["loss_value", "days_to_report", "pattern_count", "offender_count", "insider", "has_footage", "has_witness",
              "has_plate_read", "duration_days", "followup_days"]:
        ref[c] = ref[c].astype("float32")
    ref["cleared"] = ref["cleared"].astype("int8")
    return ref.reset_index(drop=True)


def kaplan_meier(durations, events, times, return_ci: bool = False):
    """Observed P(cleared by t) = 1 - KM survival, NaN past the point where fewer than
    20 cases were still being followed. With `return_ci`, also a 95% Greenwood band."""
    d = np.asarray(durations, float)
    e = np.asarray(events, int)
    times = np.asarray(times, float)
    nan = np.full(len(times), np.nan)
    if len(d) == 0:
        return (nan, nan, nan) if return_ci else nan
    ut = np.unique(d[e == 1])
    S, V, s, g = [], [], 1.0, 0.0
    for t in ut:
        n = (d >= t).sum()
        k = ((d == t) & (e == 1)).sum()
        s *= 1 - k / n
        g += k / (n * (n - k)) if n > k else 0.0
        S.append(s)
        V.append(g)
    S, V = np.array(S), np.array(V)
    idx = np.searchsorted(ut, times, side="right") - 1
    has = idx >= 0
    surv = np.ones(len(times))
    gw = np.zeros(len(times))
    if len(S):
        surv[has] = S[idx[has]]
        gw[has] = V[idx[has]]
    F = 1 - surv
    cutoff = np.sort(d)[-min(20, len(d))]
    F[times > cutoff] = np.nan
    if not return_ci:
        return F
    se = surv * np.sqrt(gw)
    lo = np.clip(F - 1.96 * se, 0, 1)
    hi = np.clip(F + 1.96 * se, 0, 1)
    lo[np.isnan(F)] = np.nan
    hi[np.isnan(F)] = np.nan
    return F, lo, hi


def _criteria(case: pd.Series, asked: set[str]) -> dict:
    crit = {}
    for c in ["item_category", "location_type", "victim_type", "jurisdiction"]:
        v = case.get(c)
        if c in asked and isinstance(v, str) and v not in ("unknown", "other"):
            crit[c] = v
    v = case.get("loss_value")
    if "loss_value" in asked and pd.notna(v) and v > 0:
        crit["loss_value"] = (v / 2, v * 2)
    v = case.get("days_to_report")
    if "days_to_report" in asked and pd.notna(v):
        crit["days_to_report"] = (0, 1) if v <= 1 else (v / 2, v * 2)
    return crit


def _mask(ref: pd.DataFrame, crit: dict) -> np.ndarray:
    m = np.ones(len(ref), bool)
    for c, v in crit.items():
        if isinstance(v, tuple):
            col = ref[c].to_numpy(dtype=float)
            m &= (col >= v[0]) & (col <= v[1])
        else:
            m &= (ref[c].astype(str) == v).to_numpy()
    return m


def describe_criteria(crit: dict) -> list[str]:
    out = []
    for c, v in crit.items():
        if c == "loss_value":
            out.append(f"value ${v[0]:,.0f}–${v[1]:,.0f}")
        elif c == "days_to_report":
            out.append("reported within a day" if v == (0, 1) else f"reported after {v[0]:g}–{v[1]:g} days")
        else:
            out.append(f"{c.replace('_', ' ')} = {v.replace('_', ' ')}")
    return out


def similar_cases(ref: pd.DataFrame, case: pd.DataFrame, asked_fields: set[str], horizons,
                  long_run: float, min_matches: int = MIN_MATCHES) -> dict:
    row = case.iloc[0]
    crit = _criteria(row, asked_fields)
    dropped = []
    m = _mask(ref, crit)
    for c in RELAX_ORDER:
        if m.sum() >= min_matches or not crit:
            break
        if c in crit:
            crit.pop(c)
            dropped.append(c)
            m = _mask(ref, crit)
    sub = ref[m]
    tk = sub["timing_known"].to_numpy(bool)
    timed = sub[tk]
    km = kaplan_meier(timed["duration_days"], timed["cleared"], horizons) if len(timed) else np.full(len(horizons), np.nan)

    # long-run share from every source that followed the case a full horizon
    full = sub[(sub["followup_days"] >= long_run) | ((sub["cleared"] == 1) & sub["timing_known"]
                                                     & (sub["duration_days"] <= long_run))]
    if len(timed):
        full_rate = float(km[-1]) if not np.isnan(km[-1]) else np.nan
    else:
        full_rate = float(full["cleared"].mean()) if len(full) else np.nan
    cleared_t = timed.loc[timed["cleared"] == 1, "duration_days"]
    grid = np.arange(0, int(long_run) + 1, dtype=float)
    if len(timed):
        cF, cLo, cHi = kaplan_meier(timed["duration_days"], timed["cleared"], grid, return_ci=True)
        curve = {"days": grid.tolist(), "p": cF.tolist(), "lo": cLo.tolist(), "hi": cHi.tolist()}
    else:
        curve = None
    by_source = sub.groupby("source", observed=True).size().to_dict()
    years = sub["incident_date"].dt.year
    return {
        "n": int(len(sub)),
        "n_with_dates": int(tk.sum()),
        "criteria": describe_criteria(crit),
        "relaxed": dropped,
        "observed_by_horizon": {int(h): (None if np.isnan(v) else float(v)) for h, v in zip(horizons, km)},
        "observed_long_run": None if np.isnan(full_rate) else full_rate,
        "observed_curve": curve,
        "median_days_to_clear": float(cleared_t.median()) if len(cleared_t) else None,
        "sources": {str(k): int(v) for k, v in by_source.items()},
        "years": [int(years.min()), int(years.max())] if len(sub) else None,
    }
