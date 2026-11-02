"""Validation and cleaning of canonical case tables, plus derived fields."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from .schema import CANONICAL_COLS, FLAG_FIELDS, NEWS_PATTERN, VOCAB

MAX_LOSS = 5_000_000          # caps typos like 1e9; the model works on log(value)
MAX_REPORT_LAG = 3650
PATTERN_WINDOW_DAYS = 14


class NewsSourceError(ValueError):
    """Raised when news-derived cases would be used for validation."""


def find_news_rows(df: pd.DataFrame) -> pd.Series:
    """Rows whose source/origin metadata looks like a news or media collection."""
    rx = re.compile(NEWS_PATTERN)
    mask = pd.Series(False, index=df.index)
    for col in ("source", "data_origin", "origin", "collection"):
        if col in df.columns:
            mask |= df[col].astype(str).str.contains(rx, na=False)
    return mask


def assert_no_news(df: pd.DataFrame, purpose: str = "evaluation") -> None:
    bad = find_news_rows(df)
    if bad.any():
        raise NewsSourceError(
            f"{int(bad.sum())} rows look like they come from news or media ({purpose} refused). "
            "Solved cases get far more coverage than unsolved ones, so validating on news "
            "would make the model look better than it is. Use official records "
            "(NIBRS, open-data portals, agency exports).")


def _to_flag(s: pd.Series) -> pd.Series:
    m = {"1": 1.0, "true": 1.0, "t": 1.0, "yes": 1.0, "y": 1.0,
         "0": 0.0, "false": 0.0, "f": 0.0, "no": 0.0, "n": 0.0, "1.0": 1.0, "0.0": 0.0}
    return s.map(lambda v: m.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan).astype(float)


def derive_pattern_count(df: pd.DataFrame, window: int = PATTERN_WINDOW_DAYS) -> pd.Series:
    """Similar thefts (same area and item category) in the `window` days BEFORE each case.

    Only earlier incidents are counted, so the feature is something an investigator
    could have known when the case was reported.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    ok = df["incident_date"].notna() & df["area"].notna()
    sub = df.loc[ok, ["area", "item_category", "incident_date"]]
    for _, g in sub.groupby(["area", "item_category"], sort=False):
        t = g["incident_date"].values.astype("datetime64[D]").astype(np.int64)
        order = np.argsort(t, kind="stable")
        ts = t[order]
        lo = np.searchsorted(ts, ts - window, side="left")
        hi = np.searchsorted(ts, ts, side="left")          # strictly earlier days
        cnt = (hi - lo).astype(float)
        out.loc[g.index[order]] = cnt
    return out


def clean(df: pd.DataFrame, *, report: dict | None = None) -> pd.DataFrame:
    """Return a cleaned copy. `report` (if given) is filled with counts of what changed."""
    rep = report if report is not None else {}
    df = df.copy()
    for c in CANONICAL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    n0 = len(df)

    df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df[df["incident_date"].notna()]
    rep["dropped_no_date"] = n0 - len(df)

    n1 = len(df)
    df = df.drop_duplicates("case_id", keep="last")
    rep["dropped_duplicates"] = n1 - len(df)

    # --- numerics -------------------------------------------------------------
    df["loss_value"] = pd.to_numeric(df["loss_value"], errors="coerce")
    bad_loss = df["loss_value"] <= 0
    rep["loss_nonpositive_to_missing"] = int(bad_loss.sum())
    df.loc[bad_loss, "loss_value"] = np.nan
    df["loss_value"] = df["loss_value"].clip(upper=MAX_LOSS)

    df["days_to_report"] = pd.to_numeric(df["days_to_report"], errors="coerce")
    lag_from_dates = (df["report_date"] - df["incident_date"]).dt.days
    df["days_to_report"] = df["days_to_report"].fillna(lag_from_dates)
    neg = df["days_to_report"] < 0
    rep["negative_report_lag_to_missing"] = int(neg.sum())
    df.loc[neg, "days_to_report"] = np.nan
    df["days_to_report"] = df["days_to_report"].clip(upper=MAX_REPORT_LAG)

    for c in ["pattern_count", "offender_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0)

    # --- categoricals ------------------------------------------------------------
    for c, vocab in VOCAB.items():
        s = df[c].astype("string").str.strip().str.lower().fillna("unknown")
        df[c] = s.where(s.isin(vocab), "other" if "other" in vocab else "unknown").astype(object)
    df["jurisdiction"] = df["jurisdiction"].astype("string").fillna("unknown").str.lower().astype(object)
    df["area"] = df["area"].astype("string").fillna(df["jurisdiction"]).astype(object)

    for c in FLAG_FIELDS:
        df[c] = _to_flag(df[c])

    # --- outcomes ---------------------------------------------------------------
    df["cleared"] = pd.to_numeric(df["cleared"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    df["followup_days"] = pd.to_numeric(df["followup_days"], errors="coerce")
    df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")
    df["timing_known"] = df["timing_known"].fillna(False).astype(bool)
    # an event cannot happen after we stopped watching
    over = df["timing_known"] & (df["duration_days"] > df["followup_days"])
    df.loc[over, "duration_days"] = df.loc[over, "followup_days"]
    df.loc[df["timing_known"] & df["duration_days"].isna(), "timing_known"] = False
    df["clearance_type"] = df["clearance_type"].fillna("").astype(str)
    df = df[df["followup_days"].notna() & (df["followup_days"] >= 0)]
    rep["dropped_bad_followup"] = n1 - rep["dropped_duplicates"] - len(df)

    # --- derived ------------------------------------------------------------------
    need = df["pattern_count"].isna()
    if need.any():
        derived = derive_pattern_count(df)
        df.loc[need, "pattern_count"] = derived[need]

    news = find_news_rows(df)
    if news.any():
        warnings.warn(f"{int(news.sum())} rows look news-derived; they cannot be used for evaluation.")
    rep["rows_out"] = len(df)
    return df.reset_index(drop=True)


def horizon_labels(df: pd.DataFrame, horizon: float, long_run_horizon: float) -> pd.Series:
    """Cleared-by-`horizon` label per row; NaN where the data cannot say.

    Rows with known timing are labelled only if they were followed for the whole horizon
    (or cleared within it AND were followed that long). Counting quick clearances from
    cases followed for less than the horizon, while dropping their uncleared
    neighbours, would inflate the rate.
    Rows where the source records only *whether* a case cleared (timing unknown) can only
    be labelled at the long-run horizon, and only if we watched them at least that long.
    Labelling them at shorter horizons would quietly turn late clearances into "not
    cleared".
    """
    y = pd.Series(np.nan, index=df.index, dtype=float)
    tk = df["timing_known"].astype(bool)
    ev, dur, fu = df["cleared"], df["duration_days"], df["followup_days"]

    full = tk & (fu >= horizon)
    y[full] = ((ev == 1) & (dur <= horizon))[full].astype(float)

    if horizon >= long_run_horizon:
        m = ~tk & (fu >= long_run_horizon)
        y[m] = ev[m].astype(float)
    return y
