"""Cox proportional-hazards model for time-to-clearance, written with numpy.

- Breslow handling of tied clearance days
- Ridge (L2) penalty so one-hot blocks and sparse flags stay stable
- Optional strata: a separate baseline hazard per stratum (we stratify by jurisdiction,
  because departments define and record "cleared" differently) with shared coefficients
- Newton-Raphson with step halving

Right-censoring: cases still open at the as-of date contribute to every risk set up
to their follow-up time. They are never counted as "not cleared".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoxPH:
    penalizer: float = 0.1
    max_iter: int = 50
    tol: float = 1e-7

    coef_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    baselines_: dict = field(default_factory=dict)   # stratum -> (times, cumhaz)
    loglik_: float = np.nan
    n_iter_: int = 0
    converged_: bool = False

    # ----------------------------------------------------------- internals
    @staticmethod
    def _stratum_stats(X, T, E, beta, need_hess=True):
        """Penalty-free log partial likelihood, gradient, Hessian for one stratum."""
        order = np.argsort(-T, kind="stable")
        X, T, E = X[order], T[order], E[order]
        eta = X @ beta
        c = eta.max() if len(eta) else 0.0
        w = np.exp(eta - c)
        p = X.shape[1]

        # unique times in descending order; block boundaries
        uniq, start = np.unique(-T, return_index=True)  # -T ascending == T descending
        ends = np.append(start[1:], len(T))

        S0 = 0.0
        S1 = np.zeros(p)
        S2 = np.zeros((p, p)) if need_hess else None
        ll = 0.0
        grad = np.zeros(p)
        hess = np.zeros((p, p)) if need_hess else None
        times, d_list, s0_list = [], [], []
        for b0, b1 in zip(start, ends):
            Xb, wb, Eb = X[b0:b1], w[b0:b1], E[b0:b1]
            S0 += wb.sum()
            S1 += wb @ Xb
            if need_hess:
                S2 += (Xb * wb[:, None]).T @ Xb
            d = Eb.sum()
            if d > 0:
                xbar = S1 / S0
                ll += eta[b0:b1][Eb == 1].sum() - d * (np.log(S0) + c)
                grad += Xb[Eb == 1].sum(axis=0) - d * xbar
                if need_hess:
                    hess -= d * (S2 / S0 - np.outer(xbar, xbar))
                times.append(T[b0])
                d_list.append(d)
                s0_list.append(S0 * np.exp(c))
        return ll, grad, hess, (np.array(times), np.array(d_list), np.array(s0_list))

    def _total(self, Xs, Ts, Es, beta, need_hess=True):
        p = len(beta)
        ll, g, H = 0.0, np.zeros(p), np.zeros((p, p))
        for X, T, E in zip(Xs, Ts, Es):
            l_, g_, H_, _ = self._stratum_stats(X, T, E, beta, need_hess)
            ll += l_
            g += g_
            if need_hess:
                H += H_
        ll -= 0.5 * self.penalizer * beta @ beta
        g -= self.penalizer * beta
        if need_hess:
            H -= self.penalizer * np.eye(p)
        return ll, g, H

    def _std(self, X):
        return (X - self.mean_) / self.scale_

    # -------------------------------------------------------------- public
    def fit(self, X: np.ndarray, durations, events, strata=None) -> "CoxPH":
        X = np.asarray(X, dtype=float)
        T = np.asarray(durations, dtype=float)
        E = np.asarray(events, dtype=int)
        strata = np.zeros(len(T), dtype=object) if strata is None else np.asarray(strata, dtype=object)

        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        Z = self._std(X)

        keys = list(dict.fromkeys(strata.tolist()))
        Xs = [Z[strata == k] for k in keys]
        Ts = [T[strata == k] for k in keys]
        Es = [E[strata == k] for k in keys]

        beta = np.zeros(Z.shape[1])
        ll, g, H = self._total(Xs, Ts, Es, beta)
        for it in range(1, self.max_iter + 1):
            step = np.linalg.solve(-H, g)
            t = 1.0
            while True:
                nb = beta + t * step
                nll, ng, nH = self._total(Xs, Ts, Es, nb)
                if nll >= ll - 1e-12 or t < 1e-6:
                    break
                t *= 0.5
            improvement = nll - ll
            beta, ll, g, H = nb, nll, ng, nH
            self.n_iter_ = it
            if abs(improvement) < self.tol * (1 + abs(ll)):
                self.converged_ = True
                break
        self.coef_ = beta
        self.loglik_ = ll

        # Breslow baseline cumulative hazard per stratum (on the standardised scale)
        self.baselines_ = {}
        for k, Xk, Tk, Ek in zip(keys, Xs, Ts, Es):
            _, _, _, (times, d, s0) = self._stratum_stats(Xk, Tk, Ek, beta, need_hess=False)
            if len(times) == 0:
                self.baselines_[k] = (np.array([0.0]), np.array([0.0]))
                continue
            o = np.argsort(times)
            self.baselines_[k] = (times[o], np.cumsum((d / s0)[o]))
        return self

    def log_partial_hazard(self, X) -> np.ndarray:
        return self._std(np.asarray(X, dtype=float)) @ self.coef_

    def contributions(self, X) -> np.ndarray:
        """Per-column additive log-hazard contributions relative to the training mean."""
        return self._std(np.asarray(X, dtype=float)) * self.coef_

    def hazard_ratios(self) -> np.ndarray:
        """Hazard ratio per one-unit change of each original column."""
        return np.exp(self.coef_ / self.scale_)

    def _cumhaz0(self, stratum, t):
        if stratum not in self.baselines_:
            # unseen stratum: pool baselines by averaging cumulative hazards
            return np.mean([self._cumhaz0(k, t) for k in self.baselines_], axis=0)
        times, ch = self.baselines_[stratum]
        idx = np.searchsorted(times, np.asarray(t, dtype=float), side="right") - 1
        return np.where(idx >= 0, ch[np.clip(idx, 0, None)], 0.0)

    def predict_cdf(self, X, times, strata=None) -> np.ndarray:
        """P(cleared by t) for each row and each t -> shape (n, len(times))."""
        X = np.asarray(X, dtype=float)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        n = X.shape[0]
        strata = np.zeros(n, dtype=object) if strata is None else np.asarray(strata, dtype=object)
        risk = np.exp(self.log_partial_hazard(X))
        out = np.empty((n, len(times)))
        for k in set(strata.tolist()):
            m = strata == k
            H0 = self._cumhaz0(k, times)
            out[m] = 1.0 - np.exp(-np.outer(risk[m], H0))
        return out


def concordance_index(durations, events, risk_scores) -> float:
    """Harrell's C: higher risk score should mean earlier clearance. O(n log n)-ish."""
    T = np.asarray(durations, dtype=float)
    E = np.asarray(events, dtype=int)
    R = np.asarray(risk_scores, dtype=float)
    order = np.argsort(T, kind="stable")
    T, E, R = T[order], E[order], R[order]
    # For each event i, compare to all j with T_j > T_i
    conc = disc = ties = 0.0
    # bin risks into ranks for a Fenwick-tree count of later rows
    ranks = np.unique(R, return_inverse=True)[1] + 1
    m = ranks.max() if len(ranks) else 0
    tree = np.zeros(m + 1)

    def add(i):
        while i <= m:
            tree[i] += 1
            i += i & -i

    def query(i):  # count of ranks <= i
        s = 0.0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    n = len(T)
    i = n - 1
    total_later = 0
    while i >= 0:
        j = i
        while j > 0 and T[j - 1] == T[i]:
            j -= 1
        # rows j..i share the same time; tree holds strictly later rows
        for k in range(j, i + 1):
            if E[k]:
                r = ranks[k]
                less = query(r - 1)
                eq = query(r) - less
                conc += less
                ties += eq
                disc += total_later - less - eq
        for k in range(j, i + 1):
            add(ranks[k])
            total_later += 1
        i = j - 1
    denom = conc + disc + ties
    return float((conc + 0.5 * ties) / denom) if denom else float("nan")
