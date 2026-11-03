"""Feature builder: canonical case table -> numeric design matrix.

Only facts available about the reported case are used. Outcome columns are never read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import FLAG_FIELDS, VOCAB

LOG_NUMERIC = ["loss_value", "days_to_report", "pattern_count", "offender_count"]
CATS = ["item_category", "location_type", "victim_type", "jurisdiction"]


@dataclass
class FeatureBuilder:
    min_jurisdiction_count: int = 200
    max_jurisdictions: int = 600

    medians_: dict = field(default_factory=dict)
    levels_: dict = field(default_factory=dict)
    columns_: list = field(default_factory=list)
    groups_: dict = field(default_factory=dict)          # column -> raw field
    reference_: dict = field(default_factory=dict)       # "typical case" raw values
    ranges_: dict = field(default_factory=dict)          # numeric p1/p99
    flag_observed_: dict = field(default_factory=dict)   # flag -> both 0 and 1 seen?
    missing_rate_: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        for c in LOG_NUMERIC:
            v = pd.to_numeric(df[c], errors="coerce")
            self.missing_rate_[c] = float(v.isna().mean())
            lv = np.log1p(v.dropna())
            self.medians_[c] = float(lv.median()) if len(lv) else 0.0
            self.ranges_[c] = (float(v.quantile(0.01)), float(v.quantile(0.99))) if v.notna().any() else (np.nan, np.nan)
            # "typical" = most common state, and for a mostly-unrecorded field that is unknown
            self.reference_[c] = float(v.median()) if v.notna().mean() > 0.5 else np.nan

        for c in CATS:
            s = df[c].astype(str)
            if c == "jurisdiction":
                vc = s.value_counts()
                keep = vc[vc >= self.min_jurisdiction_count].index[: self.max_jurisdictions]
                self.levels_[c] = sorted(set(keep) | {"other"})
            else:
                seen = set(s.unique())
                self.levels_[c] = [lv for lv in VOCAB[c] if lv in seen]
            self.reference_[c] = s.mode().iat[0] if len(s) else "unknown"

        for c in FLAG_FIELDS:
            v = pd.to_numeric(df[c], errors="coerce")
            self.missing_rate_[c] = float(v.isna().mean())
            self.flag_observed_[c] = bool((v == 1).sum() >= 20 and (v == 0).sum() >= 20)
            known = v.dropna()
            self.reference_[c] = float(known.mode().iat[0]) if v.notna().mean() > 0.5 else np.nan

        self.columns_ = list(self._transform(df.head(1)).columns)
        return self

    # ------------------------------------------------------------ transform
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, np.ndarray] = {}
        groups: dict[str, str] = {}
        for c in LOG_NUMERIC:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            lv = np.log1p(np.clip(v, 0, None))
            miss = np.isnan(lv)
            cols[f"log_{c}"] = np.where(miss, self.medians_[c], lv)
            groups[f"log_{c}"] = c
            if self.missing_rate_.get(c, 0) > 0:
                cols[f"{c}__missing"] = miss.astype(float)
                groups[f"{c}__missing"] = c
        for c in CATS:
            if c == "jurisdiction":
                # Agencies enter the models elsewhere: as Cox strata (own baseline
                # hazard) and as smoothed agency clearance rates for gradient boosting.
                # Hundreds of one-hot columns would not fit in memory at 1.7M rows.
                continue
            s = df[c].astype(str).to_numpy()
            levels = self.levels_[c]
            for lv in levels:
                name = f"{c}={lv}"
                cols[name] = (s == lv).astype(float)
                groups[name] = c
        for c in FLAG_FIELDS:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            cols[f"{c}=yes"] = (v == 1).astype(float)
            groups[f"{c}=yes"] = c
            if self.missing_rate_.get(c, 0) > 0:
                cols[f"{c}=unknown"] = np.isnan(v).astype(float)
                groups[f"{c}=unknown"] = c
        self.groups_ = groups
        return pd.DataFrame(cols, index=df.index)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._transform(df)
        return X.reindex(columns=self.columns_, fill_value=0.0).to_numpy(dtype=np.float32)

    def collapse_jurisdiction(self, s: pd.Series) -> np.ndarray:
        """Jurisdictions too small to model separately are pooled as 'other'."""
        s = s.astype(str).to_numpy()
        return np.where(np.isin(s, self.levels_["jurisdiction"]), s, "other")

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def raw_fields(self) -> list[str]:
        return LOG_NUMERIC + CATS + FLAG_FIELDS

    def column_groups(self) -> np.ndarray:
        return np.array([self.groups_[c] for c in self.columns_])
