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
from .evidence import county_mask
from .schema import CANONICAL_COLS, FILTER_FIELDS, FLAG_FIELDS, INPUT_FIELDS, VOCAB

ALIASES = {
    "value": "loss_value", "loss": "loss_value", "item": "item_category", "category": "item_category",
    "location": "location_type", "where": "location_type", "victim": "victim_type",
    "report_delay": "days_to_report", "report_lag": "days_to_report", "days_to_report": "days_to_report",
    "footage": "has_footage", "video": "has_footage", "cctv": "has_footage",
    "witness": "has_witness", "witnesses": "has_witness",
    "plate": "has_plate_read", "plate_read": "has_plate_read", "alpr": "has_plate_read",
    "insider": "insider", "pattern": "pattern_count", "similar_recent": "pattern_count",
    "offenders": "offender_count", "jurisdiction": "jurisdiction", "city": "jurisdiction",
    "agency": "jurisdiction", "county": "county",
}


class CaseInputError(ValueError):
    pass


def normalise_fields(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = ALIASES.get(k.strip().lower(), k.strip().lower())
        if key not in INPUT_FIELDS and key not in FILTER_FIELDS:
            raise CaseInputError(
                f"unknown case field {k!r}. The engine only takes facts about a reported case: "
                + ", ".join(sorted(INPUT_FIELDS + FILTER_FIELDS)))
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

# --------------------------------------------------------------------------- #
# "Don't know" = average over real cases
# --------------------------------------------------------------------------- #
POP_SIZE = 400
_MATCH_CATS = ["item_category", "location_type", "victim_type", "jurisdiction"]


def population(engine: SolvabilityEngine, fields: dict, k: int = POP_SIZE, seed: int = 0) -> pd.DataFrame:
    """Real training cases standing in for whatever wasn't answered.

    Draws up to `k` real cases that match the categorical answers (falling back to all
    cases if too few match), then overwrites them with every answer given. Averaging
    predictions over this sample means a skipped question counts as "don't know",
    not as the rare recorded category "unknown".
    """
    f = normalise_fields(fields)
    ref = engine.reference_cases
    if ref is None:                       # older model files: single hypothetical row
        return make_case(f, engine)
    case = make_case(f, engine)           # cleaned / validated answers
    pool = ref
    if f.get("county"):
        cm = county_mask(pool, f["county"])
        if not cm.any():
            raise CaseInputError(f"no training cases from a county matching {f['county']!r}. "
                                 f"Known counties include: {', '.join(engine.counties()[:25])} ...")
        pool = pool[cm]
    juris = case.iloc[0]["jurisdiction"] if "jurisdiction" in f else None
    if engine.has_timing and (juris is None or juris not in engine.untimed_jurisdictions_):
        pool = pool[pool["timing_known"].to_numpy(bool)]   # agencies with a time curve
    m = np.ones(len(pool), bool)
    for c in _MATCH_CATS:
        if c in f:
            m &= (pool[c].astype(str) == str(case.iloc[0][c])).to_numpy()
    if m.sum() >= 50:
        pool = pool[m]
    sample = pool.sample(min(k, len(pool)), random_state=seed).copy()
    for c in sample.columns:
        if isinstance(sample[c].dtype, pd.CategoricalDtype):
            sample[c] = sample[c].astype(object)
    for c in f:
        if c in FILTER_FIELDS:
            continue
        sample[c] = case.iloc[0][c]
    for c in FLAG_FIELDS:                 # flags the model never learned stay out entirely
        if not engine.builder.flag_observed_.get(c, False):
            sample[c] = np.nan
    for c in CANONICAL_COLS:
        if c not in sample.columns:
            sample[c] = np.nan
    return sample.reset_index(drop=True)


def predict_fields(engine: SolvabilityEngine, fields: dict) -> dict:
    """Mean P(cleared by h) over the population for these answers."""
    pop = population(engine, fields)
    allp = engine.predict_all(pop)
    out = {"headline": _nanmean(allp["headline"])}
    if "cox" in allp:
        out["cox"] = _nanmean(allp["cox"])
    out["n_population"] = len(pop)
    return out


def _nanmean(P):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(P, axis=0)


def explain_fields(engine: SolvabilityEngine, fields: dict, horizon=None, top: int = 6) -> list[dict]:
    """For each answer: how far it moves the estimate compared with not knowing it."""
    f = normalise_fields(fields)
    horizon = horizon or engine._default_horizon()
    j = list(engine.horizons).index(horizon)
    base = predict_fields(engine, f)["headline"][j]
    rows = []
    for k, v in f.items():
        if v is None or k in FILTER_FIELDS:
            continue
        without = {kk: vv for kk, vv in f.items() if kk != k}
        p = predict_fields(engine, without)["headline"][j]
        if np.isnan(base) or np.isnan(p):
            continue
        rows.append({"field": k, "label": FIELD_LABELS.get(k, k), "value": _show(k, v),
                     "delta_pp": 100.0 * (base - p)})
    rows.sort(key=lambda r: -abs(r["delta_pp"]))
    return rows[:top]


def _show(field, v):
    if field in FLAG_FIELDS:
        return "yes" if v in (1, 1.0, True) else "no" if v in (0, 0.0, False) else str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def compare(engine: SolvabilityEngine, base_fields: dict, scenarios: dict[str, dict]) -> pd.DataFrame:
    """Each scenario = base fields with some overrides. Returns P(cleared by h) per scenario."""
    names = ["as given"] + list(scenarios)
    variants = [base_fields] + [{**base_fields, **ov} for ov in scenarios.values()]
    P = np.vstack([predict_fields(engine, v)["headline"] for v in variants])
    out = pd.DataFrame(P, columns=[f"by_{h}d" for h in engine.horizons])
    out.insert(0, "scenario", names)
    return out


def sweep(engine: SolvabilityEngine, base_fields: dict, field: str, values: list) -> pd.DataFrame:
    field = normalise_fields({field: None}).popitem()[0]
    P = np.vstack([predict_fields(engine, {**base_fields, field: v})["headline"] for v in values])
    out = pd.DataFrame(P, columns=[f"by_{h}d" for h in engine.horizons])
    out.insert(0, field, values)
    return out


def mean_curve(engine: SolvabilityEngine, fields: dict, t) -> np.ndarray:
    pop = population(engine, fields)
    return _nanmean(engine.curve(pop, t))


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
    if np.isnan(predict_fields(engine, fields)["headline"]).any():
        raise ValueError("This agency's records say whether a case cleared but not when, so there "
                         "is no time curve for it. Leave the jurisdiction blank to use agencies "
                         "that record clearance dates.")
    L = engine.long_run
    waited = [w for w in waited if w < L]
    fine = np.arange(0, int(L) + 1, dtype=float)
    Ff = mean_curve(engine, fields, fine)
    F = dict(zip(fine.astype(int), Ff))
    rows = []
    for w in waited:
        surv = 1.0 - F[w]
        nxt = (F[int(min(w + window, L))] - F[w]) / surv if surv > 0 else 0.0
        rest = (F[int(L)] - F[w]) / surv if surv > 0 else 0.0
        rows.append({"days_open": w, f"p_next_{window}d": nxt, f"p_by_{int(L)}d": rest})
    table = pd.DataFrame(rows)

    # first day where the remaining long-run chance drops below the bar
    remaining = (Ff[-1] - Ff[:-1]) / np.clip(1 - Ff[:-1], 1e-9, None)
    below = np.nonzero(remaining < give_up_below)[0]
    return {"table": table, "window": window, "long_run": L,
            "diminishing_returns_day": int(fine[below[0]]) if len(below) else None,
            "threshold": give_up_below, "warnings": check_case(engine, case, fields)}
