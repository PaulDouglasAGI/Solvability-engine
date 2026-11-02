"""Held-out evaluation: discrimination, calibration, and a calibration plot."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .cleaning import assert_no_news, horizon_labels
from .models.cox import concordance_index
from .models.engine import SolvabilityEngine

# Validated categorical slots (light mode) - fixed order: GBM first, Cox second
COLORS = {"gbm": "#2a78d6", "cox": "#eb6834", "headline": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODEL_LABEL = {"gbm": "Gradient boosting", "cox": "Cox survival", "headline": "Headline"}


def reliability_bins(p, y, n_bins: int = 10) -> pd.DataFrame:
    """Quantile bins so every bin has data even when most predictions are small."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return pd.DataFrame(columns=["mean_pred", "obs_rate", "n"])
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([p.min(), p.max() + 1e-9])
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    g = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b")
    return pd.DataFrame({"mean_pred": g["p"].mean(), "obs_rate": g["y"].mean(), "n": g.size()}).reset_index(drop=True)


def ece(p, y, n_bins: int = 10) -> float:
    t = reliability_bins(p, y, n_bins)
    if t.empty:
        return float("nan")
    return float((t["n"] * (t["mean_pred"] - t["obs_rate"]).abs()).sum() / t["n"].sum())


def evaluate(engine: SolvabilityEngine, test: pd.DataFrame) -> dict:
    assert_no_news(test, "evaluation")
    preds = engine.predict_all(test)
    rows, bins = [], {}
    for j, h in enumerate(engine.horizons):
        y = horizon_labels(test, h, engine.long_run)
        m = y.notna().to_numpy()
        yy = y[m].to_numpy()
        for model in ("headline", "gbm", "cox"):
            if model not in preds or np.isnan(preds[model]).all():
                continue
            kinds = (("calibrated", model),) if model == "headline" else (("raw", f"{model}_raw"), ("calibrated", model))
            for kind, key in kinds:
                p = preds[key][m, j]
                ok = ~np.isnan(p)
                if ok.sum() == 0:
                    continue
                pp, yv = p[ok], yy[ok]
                both = 0 < yv.sum() < len(yv)
                rows.append({
                    "horizon_days": h, "model": model, "kind": kind, "n": int(ok.sum()),
                    "observed_rate": float(yv.mean()), "mean_pred": float(pp.mean()),
                    "brier": float(brier_score_loss(yv, pp)) if len(yv) else np.nan,
                    "auc": float(roc_auc_score(yv, pp)) if both else np.nan,
                    "ece": ece(pp, yv),
                })
                bins[(h, model, kind)] = reliability_bins(pp, yv)

    out = {"metrics": pd.DataFrame(rows), "bins": bins}

    tk = test["timing_known"].to_numpy(bool)
    if engine.cox is not None and tk.sum() > 10:
        X = engine.builder.transform(test[tk])
        out["c_index_cox"] = concordance_index(
            test.loc[tk, "duration_days"], test.loc[tk, "cleared"], engine.cox.log_partial_hazard(X))
        lr_j = len(engine.horizons) - 1
        out["c_index_headline"] = concordance_index(
            test.loc[tk, "duration_days"], test.loc[tk, "cleared"], preds["headline"][tk, lr_j])

    # Calibration-in-the-large by jurisdiction at the long-run horizon: shows where
    # differing "cleared" definitions break a pooled model
    y = horizon_labels(test, engine.long_run, engine.long_run)
    m = y.notna()
    jr = pd.DataFrame({"jurisdiction": test.loc[m, "jurisdiction"], "y": y[m],
                       "p": preds["headline"][m.to_numpy(), -1]})
    by_j = jr.groupby("jurisdiction").agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    out["by_jurisdiction"] = by_j[by_j["n"] >= 30].sort_values("n", ascending=False)
    return out


def calibration_plot(engine: SolvabilityEngine, result: dict, path, title: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = [h for h in engine.horizons if any(k[0] == h for k in result["bins"])]
    if not hs:
        raise ValueError("nothing to plot: no horizon has labelled test rows")
    ncol = min(3, len(hs))
    nrow = int(np.ceil(len(hs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(4.2 * ncol, 7.5), 4.2 * nrow + 0.6), squeeze=False,
                             facecolor=SURFACE)
    for ax in axes.flat[len(hs):]:
        ax.set_visible(False)

    for ax, h in zip(axes.flat, hs):
        ax.set_facecolor(SURFACE)
        lim = 0.0
        models = ("gbm", "cox", "headline") if engine.cox is not None else ("gbm",)
        for model in models:
            for kind, ls, lw, alpha in (("raw", (0, (3, 2)), 1.5, 0.75), ("calibrated", "-", 2.0, 1.0)):
                t = result["bins"].get((h, model, kind))
                if t is None or t.empty:
                    continue
                lim = max(lim, t["mean_pred"].max(), t["obs_rate"].max())
                ax.plot(t["mean_pred"], t["obs_rate"], linestyle=ls, linewidth=lw, color=COLORS[model],
                        alpha=alpha, marker="o" if kind == "calibrated" else None, markersize=4.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.2,
                        label=MODEL_LABEL[model] + ("" if model == "headline" else f" ({kind})"))
        lim = min(1.0, max(0.05, lim * 1.1))
        ax.plot([0, lim], [0, lim], color=INK2, linewidth=1, linestyle=":", label="Perfect calibration")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"Cleared within {h} day{'s' if h != 1 else ''}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Predicted probability", color=INK2, fontsize=9)
        ax.set_ylabel("Observed clearance rate", color=INK2, fontsize=9)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(color=GRID, linewidth=0.8)
        for s in ax.spines.values():
            s.set_color(GRID)
        m = result["metrics"]
        mm = m[(m.horizon_days == h) & (m.kind == "calibrated")]
        mm = mm[mm.model.isin(models)]
        mm = mm.sort_values("model", key=lambda s: s.map({"headline": 0, "gbm": 1, "cox": 2}))
        txt = "\n".join(f"{MODEL_LABEL[r.model]}: Brier {r.brier:.3f}, ECE {r.ece:.3f}, n={r.n:,}"
                        for r in mm.itertuples())
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=7, color=INK2)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    sup = title or "Calibration on held-out cases (quantile bins)"
    if engine.is_synthetic:
        sup += "  [SYNTHETIC DATA: not real estimates]"
    fig.suptitle(sup, color=INK, fontsize=12, x=0.02, ha="left")
    fig.text(0.02, 0.005, "Reported thefts only. 'Cleared' is defined differently by each department.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
