"""The solvability engine: feature builder + Cox survival model + per-horizon gradient
boosting, each calibrated on held-out data, plus explanations."""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..cleaning import horizon_labels
from ..features import FeatureBuilder
from ..schema import FLAG_FIELDS, DEFAULT_HORIZONS, SOURCE_CAVEATS, STANDING_CAVEATS
from .cox import CoxPH

MIN_POSITIVES = 30
MIN_COX_EVENTS = 50
ISOTONIC_MIN_ROWS = 20000   # isotonic steps are too coarse below this; Platt is smoother
ISOTONIC_MIN_POS = 500

FIELD_LABELS = {
    "loss_value": "loss value",
    "days_to_report": "time to report",
    "pattern_count": "similar recent thefts nearby",
    "offender_count": "number of offenders",
    "item_category": "item category",
    "location_type": "location type",
    "victim_type": "victim type",
    "jurisdiction": "jurisdiction",
    "has_footage": "video footage",
    "has_witness": "witnesses",
    "has_plate_read": "licence-plate read",
    "insider": "insider involvement",
}


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

@dataclass
class Calibrator:
    method: str = "identity"
    model: object = None

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, p, y) -> "Calibrator":
        p, y = np.asarray(p, float), np.asarray(y, float)
        if len(y) < 50 or y.sum() < 5 or y.sum() > len(y) - 5:
            self.method = "identity"
            return self
        if len(y) >= ISOTONIC_MIN_ROWS and y.sum() >= ISOTONIC_MIN_POS:
            self.method = "isotonic"
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        else:
            self.method = "platt"
            self.model = LogisticRegression(C=1e4).fit(self._logit(p)[:, None], y)
        return self

    def transform(self, p):
        p = np.asarray(p, float)
        if self.method == "isotonic":
            return self.model.predict(p)
        if self.method == "platt":
            return self.model.predict_proba(self._logit(p)[:, None])[:, 1]
        return p


def _monotone(P: np.ndarray) -> np.ndarray:
    """Cumulative probabilities must not fall as the horizon grows. Horizons with no
    model (NaN) stay NaN."""
    P = np.clip(P, 0.0, 1.0)
    out = np.fmax.accumulate(P, axis=1)
    out[np.isnan(P)] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

@dataclass
class SolvabilityEngine:
    horizons: tuple = DEFAULT_HORIZONS
    headline: str = "blend"               # "blend" (mean of the two) | "gbm" | "cox"
    cox_penalizer: float = 0.1
    stratify_cox: bool = True
    random_state: int = 0

    builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    cox: CoxPH | None = None
    gbms: dict = field(default_factory=dict)
    cal_cox: dict = field(default_factory=dict)
    cal_gbm: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def long_run(self) -> float:
        return float(max(self.horizons))

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, calib: pd.DataFrame) -> "SolvabilityEngine":
        Xtr = self.builder.fit_transform(train)
        Xca = self.builder.transform(calib)

        # Cox on rows whose clearance timing is known
        tk = train["timing_known"].to_numpy(bool)
        if tk.sum() and train.loc[tk, "cleared"].sum() >= MIN_COX_EVENTS:
            strata = train.loc[tk, "jurisdiction"].to_numpy() if self.stratify_cox else None
            self.cox = CoxPH(penalizer=self.cox_penalizer).fit(
                Xtr[tk], train.loc[tk, "duration_days"].to_numpy(), train.loc[tk, "cleared"].to_numpy(), strata)
        else:
            self.cox = None
            warnings.warn("Not enough clearance dates to fit the Cox model. Only "
                          "per-horizon models will be used.")

        # One gradient-boosting classifier per horizon, trained on rows we can label
        self.gbms = {}
        for h in self.horizons:
            y = horizon_labels(train, h, self.long_run)
            m = y.notna().to_numpy()
            if m.sum() < 200 or y[m].sum() < MIN_POSITIVES or y[m].sum() > m.sum() - MIN_POSITIVES:
                continue
            clf = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=25, random_state=self.random_state)
            clf.fit(Xtr[m], y[m].to_numpy())
            self.gbms[h] = clf

        # Calibrate both on held-out rows
        raw_cox = self._raw_cox(Xca, calib) if self.cox is not None else None
        raw_gbm = self._raw_gbm(Xca)
        self.cal_cox, self.cal_gbm = {}, {}
        for j, h in enumerate(self.horizons):
            y = horizon_labels(calib, h, self.long_run)
            m = y.notna().to_numpy()
            if raw_cox is not None:
                self.cal_cox[h] = Calibrator().fit(raw_cox[m, j], y[m])
            if h in self.gbms:
                self.cal_gbm[h] = Calibrator().fit(raw_gbm[m, j], y[m])

        self.meta = {
            "n_train": int(len(train)), "n_calib": int(len(calib)),
            "sources": sorted(set(train["source"].astype(str))),
            "jurisdictions": sorted(set(train["jurisdiction"].astype(str))),
            "date_range": [str(train["incident_date"].min().date()), str(train["incident_date"].max().date())],
            "train_clearance_rate_long_run": float(np.nanmean(horizon_labels(train, self.long_run, self.long_run))),
            "timing_known_share": float(train["timing_known"].mean()),
            "gbm_horizons": sorted(self.gbms),
            "cox_fitted": self.cox is not None,
            "calibration": {"cox": {h: c.method for h, c in self.cal_cox.items()},
                            "gbm": {h: c.method for h, c in self.cal_gbm.items()}},
        }
        return self

    # ---------------------------------------------------------- raw scores
    def _strata(self, df):
        if self.cox is None or not self.stratify_cox:
            return None
        return df["jurisdiction"].astype(str).to_numpy()

    def _raw_cox(self, X, df) -> np.ndarray:
        return self.cox.predict_cdf(X, self.horizons, self._strata(df))

    def _raw_gbm(self, X) -> np.ndarray:
        out = np.full((X.shape[0], len(self.horizons)), np.nan)
        for j, h in enumerate(self.horizons):
            if h in self.gbms:
                out[:, j] = self.gbms[h].predict_proba(X)[:, 1]
        return out

    # ----------------------------------------------------------- predict
    def predict_all(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Raw and calibrated P(cleared by h) from each model -> arrays (n, n_horizons)."""
        X = self.builder.transform(df)
        n, H = X.shape[0], len(self.horizons)
        out = {}
        if self.cox is not None:
            rc = self._raw_cox(X, df)
            cc = np.column_stack([self.cal_cox[h].transform(rc[:, j]) if h in self.cal_cox else rc[:, j]
                                  for j, h in enumerate(self.horizons)])
            out["cox_raw"], out["cox"] = rc, _monotone(cc)
        rg = self._raw_gbm(X)
        cg = np.column_stack([self.cal_gbm[h].transform(rg[:, j]) if h in self.cal_gbm else rg[:, j]
                              for j, h in enumerate(self.horizons)])
        out["gbm_raw"] = rg
        # fill horizons GBM couldn't learn from the calibrated Cox curve
        if "cox" in out:
            cg = np.where(np.isnan(cg), out["cox"], cg)
        out["gbm"] = _monotone(cg)

        if self.headline == "cox" and "cox" in out:
            out["headline"] = out["cox"]
        elif self.headline == "blend" and "cox" in out:
            out["headline"] = _monotone(0.5 * (out["cox"] + out["gbm"]))
        else:
            out["headline"] = out["gbm"] if not np.isnan(out["gbm"]).all() else out.get("cox", out["gbm"])
        return out

    @property
    def available_horizons(self) -> list:
        """Horizons this engine can answer (no clearance dates -> only the long run)."""
        if self.cox is not None:
            return list(self.horizons)
        return [h for h in self.horizons if h in self.gbms]

    @property
    def has_timing(self) -> bool:
        return self.available_horizons == list(self.horizons)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_all(df)["headline"]

    def curve(self, df: pd.DataFrame, t) -> np.ndarray:
        """Headline P(cleared by t) at any day t (up to the longest horizon).

        Passes exactly through the calibrated horizon points. Between them it follows
        the shape of the Cox survival curve (clearances bunch up early), or a straight
        line if no Cox model was fitted.
        """
        if not self.has_timing:
            raise ValueError(
                "This model was trained without clearance dates (its sources record only "
                "WHETHER a case cleared, not when), so it can't draw a curve over time. "
                "Add NIBRS or canonical data that includes clearance dates.")
        P = self.predict_proba(df)
        knots = np.concatenate([[0.0], np.asarray(self.horizons, float)])
        t = np.clip(np.atleast_1d(np.asarray(t, float)), 0, knots[-1])
        Fk = np.column_stack([np.zeros(len(P)), P])
        if self.cox is not None:
            X = self.builder.transform(df)
            C_t = self.cox.predict_cdf(X, t, self._strata(df))
            C_k = self.cox.predict_cdf(X, knots, self._strata(df))
        out = np.empty((len(P), len(t)))
        j = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, len(knots) - 2)
        a, b = knots[j], knots[j + 1]
        for i in range(len(P)):
            lin = (t - a) / (b - a)
            if self.cox is not None:
                ca, cb = C_k[i, j], C_k[i, j + 1]
                frac = np.where(cb - ca > 1e-12, (C_t[i] - ca) / np.where(cb - ca > 1e-12, cb - ca, 1), lin)
            else:
                frac = lin
            out[i] = Fk[i, j] + np.clip(frac, 0, 1) * (Fk[i, j + 1] - Fk[i, j])
        return out

    # ----------------------------------------------------------- explain
    def explain(self, case: pd.DataFrame, horizon: float | None = None, top: int = 5) -> list[dict]:
        """Top contributing factors for one case.

        Swaps one field at a time to the value of a typical reported case and reports how
        far the headline probability moves (percentage points). When the Cox model is
        available, also gives that field's hazard multiplier against the training average.
        """
        horizon = horizon or self._default_horizon()
        j = list(self.horizons).index(horizon)
        base = float(self.predict_proba(case)[0, j])
        ref = self.builder.reference_

        cox_by_field = {}
        if self.cox is not None:
            contrib = self.cox.contributions(self.builder.transform(case))[0]
            groups = self.builder.column_groups()
            for f in set(groups):
                cox_by_field[f] = float(np.exp(contrib[groups == f].sum()))

        rows = []
        variants, fields = [], []
        for f in self.builder.raw_fields:
            cur = case.iloc[0][f]
            rv = ref.get(f, np.nan)
            same = (pd.isna(cur) and pd.isna(rv)) or (not pd.isna(cur) and not pd.isna(rv) and cur == rv)
            if same:
                continue
            alt = case.copy()
            alt[f] = rv
            variants.append(alt)
            fields.append(f)
        if variants:
            alt_p = self.predict_proba(pd.concat(variants, ignore_index=True))[:, j]
            for f, ap in zip(fields, alt_p):
                rows.append({
                    "field": f, "label": FIELD_LABELS.get(f, f),
                    "value": _fmt(case.iloc[0][f], f), "typical": _fmt(ref.get(f), f),
                    "delta_pp": 100.0 * (base - float(ap)),
                    "hazard_multiplier": cox_by_field.get(f),
                })
        rows.sort(key=lambda r: -abs(r["delta_pp"]))
        return rows[:top]

    def _default_horizon(self):
        av = self.available_horizons
        return 90 if 90 in av else av[-1]

    # ------------------------------------------------------------ caveats
    def caveats(self) -> list[str]:
        out = list(STANDING_CAVEATS)
        for s in self.meta.get("sources", []):
            if s in SOURCE_CAVEATS:
                out.append(SOURCE_CAVEATS[s])
        return out

    @property
    def is_synthetic(self) -> bool:
        return "synthetic" in self.meta.get("sources", [])

    # ---------------------------------------------------------------- io
    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "SolvabilityEngine":
        # Only load model files you created yourself: pickle can execute code.
        with open(path, "rb") as f:
            return pickle.load(f)


def _fmt(v, field=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    if field in FLAG_FIELDS:
        return "yes" if v == 1 else "no"
    if isinstance(v, (float, np.floating)):
        return f"{v:g}"
    return str(v)


def split(df: pd.DataFrame, how: str = "random", fracs=(0.6, 0.2, 0.2), seed: int = 0):
    """Train / calibration / test split. 'temporal' trains on the oldest incidents;
    later cases have shorter follow-up, which `horizon_labels` handles."""
    n = len(df)
    if how == "temporal":
        idx = np.argsort(df["incident_date"].to_numpy(), kind="stable")
    else:
        idx = np.random.default_rng(seed).permutation(n)
    a = int(fracs[0] * n)
    b = a + int(fracs[1] * n)
    return (df.iloc[idx[:a]].reset_index(drop=True), df.iloc[idx[a:b]].reset_index(drop=True),
            df.iloc[idx[b:]].reset_index(drop=True))
