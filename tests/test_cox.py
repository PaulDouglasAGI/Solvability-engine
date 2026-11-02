import numpy as np
import pytest

from solvability.models.cox import CoxPH, concordance_index


def _simulate(n=4000, beta=(0.8, -0.5, 0.0), seed=1, strata=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta)))
    lp = X @ np.array(beta)
    base = np.where(rng.random(n) < 0.5, 0.01, 0.03) if strata else 0.02
    T = rng.exponential(1 / (base * np.exp(lp)))
    C = rng.uniform(0, 150, n)
    dur = np.ceil(np.minimum(T, C))          # integer days -> lots of ties
    ev = (T <= C).astype(int)
    s = np.where(base == 0.01, "a", "b") if strata else None
    return X, dur, ev, s


def test_recovers_coefficients():
    X, d, e, _ = _simulate()
    m = CoxPH(penalizer=0.0).fit(X, d, e)
    est = np.log(m.hazard_ratios())
    assert m.converged_
    np.testing.assert_allclose(est, [0.8, -0.5, 0.0], atol=0.08)


def test_stratified_baselines_and_shared_coefficients():
    X, d, e, s = _simulate(strata=True, seed=3)
    m = CoxPH(penalizer=0.0).fit(X, d, e, strata=s)
    np.testing.assert_allclose(np.log(m.hazard_ratios()), [0.8, -0.5, 0.0], atol=0.1)
    x0 = np.zeros((1, 3))
    pa = m.predict_cdf(x0, [30], ["a"])[0, 0]
    pb = m.predict_cdf(x0, [30], ["b"])[0, 0]
    assert pb > pa                       # stratum b has the higher baseline hazard
    # unseen stratum falls back to a pooled baseline between the two
    pu = m.predict_cdf(x0, [30], ["zzz"])[0, 0]
    assert pa < pu < pb


def test_cdf_monotone_in_time_and_risk():
    X, d, e, _ = _simulate()
    m = CoxPH().fit(X, d, e)
    P = m.predict_cdf(X[:50], [1, 7, 30, 90])
    assert np.all(np.diff(P, axis=1) >= -1e-12)
    lo, hi = np.array([[-2, 0, 0]]), np.array([[2, 0, 0]])
    assert m.predict_cdf(hi, [30])[0, 0] > m.predict_cdf(lo, [30])[0, 0]


def test_ridge_shrinks():
    X, d, e, _ = _simulate(n=500)
    a = CoxPH(penalizer=0.0).fit(X, d, e).coef_
    b = CoxPH(penalizer=50.0).fit(X, d, e).coef_
    assert np.linalg.norm(b) < np.linalg.norm(a)


def _brute_c(T, E, R):
    num = den = 0.0
    for i in range(len(T)):
        if not E[i]:
            continue
        for j in range(len(T)):
            if T[j] > T[i]:
                den += 1
                num += 1.0 if R[i] > R[j] else 0.5 if R[i] == R[j] else 0.0
    return num / den


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_concordance_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    T = rng.integers(0, 20, 120).astype(float)
    E = rng.integers(0, 2, 120)
    R = rng.integers(0, 6, 120).astype(float)
    assert concordance_index(T, E, R) == pytest.approx(_brute_c(T, E, R))


def test_matches_statsmodels_reference():
    """Cross-check against an independent Cox implementation (statsmodels PHReg,
    Breslow ties, same strata): coefficients and log-likelihood should agree."""
    sm = pytest.importorskip("statsmodels.duration.hazard_regression")
    X, d, e, s = _simulate(n=3000, strata=True, seed=11)
    ours = CoxPH(penalizer=0.0, tol=1e-10).fit(X, d, e, strata=s)
    ref = sm.PHReg(d, X, status=e, strata=s, ties="breslow").fit()
    np.testing.assert_allclose(np.log(ours.hazard_ratios()), ref.params, atol=1e-4)
    assert ours.loglik_ == pytest.approx(ref.llf, abs=1e-3)
