"""Hypothetical cases, what-if comparisons, and 'how long to wait' analysis.

All levers here describe the *reported case*: what was taken, where, how fast it was
reported, what evidence exists. They are the questions a victim, an insurer, or a
curious reader can ask ("does reporting the same day matter?", "how much does footage
help?", "it's been 3 weeks, what are the odds now?").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import clean
from .features import CATS, LOG_NUMERIC
from .models.engine import FIELD_LABELS, SolvabilityEngine
from .schema import CANONICAL_COLS, FLAG_FIELDS, INPUT_FIELDS, VOCAB

ALIASES = {
    "value": "loss_value", "loss": "loss_value", "item": "item_category", "category": "item_category",
    "location": "location_type", "where": "location_type", "victim": "victim_type",
    "report_delay": "days_to_report", "report_lag": "days_to_report", "days_to_report": "days_to_report",
    "footage": "has_footage", "video": "has_footage", "cctv": "has_footage",
    "witness": "has_witness", "witnesses": "has_witness",
    "plate": "has_plate_read", "plate_read": "has_plate_read", "alpr": "has_plate_read",
    "insider": "insider", "pattern": "pattern_count", "similar_recent": "pattern_count",
    "offenders": "offender_count", "jurisdiction": "jurisdiction", "city": "jurisdiction",
}


class CaseInputError(ValueError):
    pass


def normalise_fields(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = ALIASES.get(k.strip().lower(), k.strip().lower())
        if key not in INPUT_FIELDS:
            raise CaseInputError(
                f"unknown case field {k!r}. The engine only takes facts about a reported case: "
                + ", ".join(sorted(INPUT_FIELDS)))
        out[key] = v
    return out


def make_case(fields: dict, engine: SolvabilityEngine | None = None) -> pd.DataFrame:
    """One-row canonical frame from user fields (missing = unknown).

    With `engine`, evidence flags the training data never recorded both ways are reset to
    unknown. Otherwise a flag seen only as "yes" (e.g. LA's insider flag, which only
    marks embezzlement) would stand in for something else entirely.
    """
    f = normalise_fields(fields)
    row = {c: np.nan for c in CANONICAL_COLS}
    row.update({"case_id": "hypothetical", "source": "hypothetical",
                "incident_date": pd.Timestamp("2000-01-01"), "followup_days": 0,
                "cleared": 0, "timing_known": False, "pattern_count": 0})
    row.update(f)
    if "jurisdiction" not in f:
        row["jurisdiction"] = "other"
    df = clean(pd.DataFrame([row]))
    if "pattern_count" not in f:
        df["pattern_count"] = np.nan
    if engine is not None:
        for c in FLAG_FIELDS:
            if not engine.builder.flag_observed_.get(c, False):
                df[c] = np.nan
    return df


def check_case(engine: SolvabilityEngine, case: pd.DataFrame, asked: dict | None = None) -> list[str]:
    """Plain-language warnings about inputs the model can't speak to."""
    b = engine.builder
    warns = []
    r = case.iloc[0]
    asked_n = normalise_fields(asked) if asked else {}
    for c in FLAG_FIELDS:
        if asked_n.get(c) is not None and not b.flag_observed_.get(c, False):
            warns.append(f"The training data has (almost) no records of {FIELD_LABELS[c]} either way, "
                         f"so the model can't learn its effect. This input was ignored.")
    for c in LOG_NUMERIC:
        lo, hi = b.ranges_.get(c, (np.nan, np.nan))
        v = r[c]
        if pd.notna(v) and pd.notna(lo) and (v < lo or v > hi):
            warns.append(f"{FIELD_LABELS[c]}={v:g} is outside the range of most training cases "
                         f"({lo:g} to {hi:g}). Treat the estimate as extrapolation.")
        if pd.notna(v) and b.missing_rate_.get(c, 0) > 0.98:
            warns.append(f"{FIELD_LABELS[c]} was almost never recorded in training, so this input has little effect.")
    for c in CATS:
        v = str(r[c])
        if c == "jurisdiction":
            if asked and "jurisdiction" in normalise_fields(asked) and v not in b.levels_[c]:
                warns.append(f"Jurisdiction {v!r} isn't in the training data. Using a pooled baseline.")
        elif v not in b.levels_[c] and v != "unknown":
            warns.append(f"{FIELD_LABELS[c]}={v!r} never appears in training; it is treated like an average case.")
    if asked:
        for c in ("item_category", "location_type", "victim_type"):
            raw = normalise_fields(asked).get(c)
            if raw is not None and str(raw).lower() not in VOCAB[c]:
                warns.append(f"{FIELD_LABELS[c]} {raw!r} isn't a known value. Recorded as {r[c]!r}. "
                             f"Options: {', '.join(VOCAB[c])}")
    return warns


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

def compare(engine: SolvabilityEngine, base_fields: dict, scenarios: dict[str, dict]) -> pd.DataFrame:
    """Each scenario = base fields with some overrides. Returns P(cleared by h) per scenario."""
    names = ["as given"] + list(scenarios)
    rows = [make_case(base_fields, engine)] + [make_case({**base_fields, **ov}, engine) for ov in scenarios.values()]
    P = engine.predict_proba(pd.concat(rows, ignore_index=True))
    out = pd.DataFrame(P, columns=[f"by_{h}d" for h in engine.horizons])
    out.insert(0, "scenario", names)
    return out


def sweep(engine: SolvabilityEngine, base_fields: dict, field: str, values: list) -> pd.DataFrame:
    field = normalise_fields({field: None}).popitem()[0]
    rows = [make_case({**base_fields, field: v}, engine) for v in values]
    P = engine.predict_proba(pd.concat(rows, ignore_index=True))
    out = pd.DataFrame(P, columns=[f"by_{h}d" for h in engine.horizons])
    out.insert(0, field, values)
    return out


def wait_analysis(engine: SolvabilityEngine, fields: dict, waited=(0, 1, 3, 7, 14, 30, 60, 90, 180),
                  window: int = 30, give_up_below: float = 0.02) -> dict:
    """Odds the case still clears, given it is still open after `w` days.

    P(clear in next `window` days | open at w) = (F(w+window) - F(w)) / (1 - F(w))
    P(clear by long-run horizon | open at w)   = (F(L) - F(w)) / (1 - F(w))

    Beyond the longest horizon the data says nothing, so it is never extrapolated.
    """
    case = make_case(fields, engine)
    if not engine.has_timing:
        engine.curve(case, [0])          # raises with an explanation
    L = engine.long_run
    waited = [w for w in waited if w < L]
    grid = sorted(set(list(waited) + [min(w + window, L) for w in waited] + [L]))
    F = dict(zip(grid, engine.curve(case, grid)[0]))
    rows = []
    for w in waited:
        surv = 1.0 - F[w]
        nxt = (F[min(w + window, L)] - F[w]) / surv if surv > 0 else 0.0
        rest = (F[L] - F[w]) / surv if surv > 0 else 0.0
        rows.append({"days_open": w, f"p_next_{window}d": nxt, f"p_by_{int(L)}d": rest})
    table = pd.DataFrame(rows)

    # first day on a fine grid where the remaining long-run chance drops below the bar
    fine = np.arange(0, int(L))
    Ff = engine.curve(case, np.append(fine, L))[0]
    remaining = (Ff[-1] - Ff[:-1]) / np.clip(1 - Ff[:-1], 1e-9, None)
    below = np.nonzero(remaining < give_up_below)[0]
    return {"table": table, "window": window, "long_run": L,
            "diminishing_returns_day": int(fine[below[0]]) if len(below) else None,
            "threshold": give_up_below, "warnings": check_case(engine, case, fields)}
