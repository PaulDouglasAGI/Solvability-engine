import numpy as np
import pandas as pd
import pytest

from solvability import whatif as wi
from solvability.cleaning import NewsSourceError, clean
from solvability.cli import main
from solvability.data.synthetic import generate
from solvability.evaluation import calibration_plot, evaluate
from solvability.models.engine import SolvabilityEngine, split


@pytest.fixture(scope="module")
def data():
    return clean(generate(8000, seed=7))


@pytest.fixture(scope="module")
def engine(data):
    tr, ca, _ = split(data, seed=0)
    return SolvabilityEngine().fit(tr, ca)


def test_fit_and_predict_shapes(engine, data):
    P = engine.predict_proba(data.head(50))
    assert P.shape == (50, len(engine.horizons))
    assert np.all((P >= 0) & (P <= 1))
    assert np.all(np.diff(P, axis=1) >= -1e-12)     # cumulative over time
    assert engine.cox is not None and engine.has_timing


def test_evaluation_and_plot(engine, data, tmp_path):
    _, _, te = split(data, seed=0)
    res = evaluate(engine, te)
    m = res["metrics"]
    cal = m[(m.kind == "calibrated") & (m.model == "headline")]
    assert len(cal) == len(engine.horizons)
    assert (cal["ece"] < 0.05).all()
    assert (cal["auc"] > 0.65).all()
    assert 0.6 < res["c_index_cox"] < 1
    out = tmp_path / "cal.png"
    calibration_plot(engine, res, out)
    assert out.stat().st_size > 10_000


def test_evaluation_refuses_news(engine, data):
    bad = data.head(100).copy()
    bad["source"] = "newspaper archive"
    with pytest.raises(NewsSourceError):
        evaluate(engine, bad)


def test_explain_direction(engine):
    case = wi.make_case({"item": "retail_merchandise", "location": "retail_commercial",
                         "footage": 1, "insider": 1, "report_delay": 0}, engine)
    f = {r["field"]: r for r in engine.explain(case, top=12)}
    assert f["insider"]["delta_pp"] > 0
    assert f["insider"]["hazard_multiplier"] > 1


def test_whatif_effects_have_expected_sign(engine):
    base = {"item": "electronics", "value": 800, "location": "residence", "report_delay": 3}
    t = wi.compare(engine, base, {"footage": {"footage": 1}, "late": {"report_delay": 60}})
    lr = t.columns[-1]
    p = dict(zip(t.scenario, t[lr]))
    assert p["footage"] > p["as given"] > p["late"]


def test_wait_is_conditional_and_decreasing(engine):
    res = wi.wait_analysis(engine, {"item": "bicycle", "location": "street_parking"})
    col = [c for c in res["table"].columns if c.startswith("p_by_")][0]
    v = res["table"][col].to_numpy()
    assert np.all(np.diff(v) <= 1e-9)
    assert v[0] > v[-1]


def test_curve_passes_through_horizon_points(engine):
    case = wi.make_case({"item": "vehicle"}, engine)
    P = engine.predict_proba(case)[0]
    C = engine.curve(case, list(engine.horizons))[0]
    np.testing.assert_allclose(C, P, atol=1e-9)


def test_rejects_unknown_fields():
    with pytest.raises(wi.CaseInputError):
        wi.make_case({"resale_channel": "online"})


def test_unlearned_flag_is_ignored(data):
    d = data.copy()
    d["has_plate_read"] = np.where(d["has_plate_read"] == 1, 1.0, np.nan)   # only "yes" ever recorded
    tr, ca, _ = split(d, seed=0)
    e = SolvabilityEngine(horizons=(30, 365)).fit(tr, ca)
    base = {"item": "vehicle"}
    t = wi.compare(e, base, {"plate": {"plate": 1}})
    assert t.iloc[0, 1:].tolist() == t.iloc[1, 1:].tolist()
    assert any("ignored" in w for w in wi.check_case(e, wi.make_case({"plate": 1}, e), {"plate": 1}))


def test_timing_unknown_sources_only_long_run(data):
    d = data.copy()
    d["timing_known"] = False
    d.loc[d.cleared == 1, "duration_days"] = np.nan
    tr, ca, _ = split(d, seed=0)
    e = SolvabilityEngine().fit(tr, ca)
    assert e.cox is None and e.available_horizons == [365]
    P = e.predict_proba(d.head(5))
    assert np.isnan(P[:, :-1]).all() and not np.isnan(P[:, -1]).any()
    with pytest.raises(ValueError):
        wi.wait_analysis(e, {"item": "bicycle"})


def test_cli_end_to_end(tmp_path, capsys):
    csv, model = tmp_path / "s.csv", tmp_path / "m.pkl"
    main(["synth", "--n", "4000", "--out", str(csv)])
    main(["train", "--data", f"synthetic={csv}", "--out", str(model), "--plot", str(tmp_path / "c.png")])
    main(["predict", "--model", str(model), "--set", "item=bicycle", "value=900", "footage=yes"])
    out = capsys.readouterr().out
    assert "P(case cleared within" in out and "What moves the estimate" in out and "Similar real cases" in out and "SYNTHETIC" in out
    assert "Only REPORTED thefts" in out
    main(["whatif", "--model", str(model), "--set", "item=bicycle", "--sweep", "report_delay=0,7,30"])
    main(["wait", "--model", str(model), "--set", "item=bicycle", "--json"])
    assert '"diminishing_returns_day"' in capsys.readouterr().out


def test_kaplan_meier_handles_censoring():
    from solvability.evidence import kaplan_meier
    # 4 cases: cleared day 1, censored day 2, cleared day 3, censored day 5 (x10 so the tail is reportable)
    d = np.tile([1, 2, 3, 5], 10)
    e = np.tile([1, 0, 1, 0], 10)
    F = kaplan_meier(d, e, [1, 3])
    np.testing.assert_allclose(F, [0.25, 1 - 0.75 * 0.5])


def test_similar_cases_match_and_relax(engine):
    case = wi.make_case({"item": "bicycle", "location": "street_parking", "value": 600}, engine)
    sim = engine.similar(case, {"item_category", "location_type", "loss_value"})
    ref = engine.reference_cases
    assert sim["n"] >= 100
    if not sim["relaxed"]:
        m = (ref.item_category == "bicycle") & (ref.location_type == "street_parking") & ref.loss_value.between(300, 1200)
        assert sim["n"] == int(m.sum())
    # an impossible combination forces relaxation rather than an empty answer
    rare = wi.make_case({"item": "jewelry_precious", "location": "transit", "value": 90000,
                         "report_delay": 40, "victim": "government_institution"}, engine)
    sim2 = engine.similar(rare, {"item_category", "location_type", "loss_value", "days_to_report", "victim_type"})
    assert sim2["relaxed"] and sim2["n"] >= 100


def test_ask_only_asks_learned_questions(tmp_path, capsys):
    from solvability.cli import cmd_ask
    import argparse
    csv, model = tmp_path / "s.csv", tmp_path / "m.pkl"
    df = generate(4000, seed=3)
    df["has_plate_read"] = np.nan                       # never recorded -> must not be asked
    df.to_csv(csv, index=False)
    main(["train", "--data", f"synthetic={csv}", "--out", str(model), "--plot", str(tmp_path / "c.png")])
    capsys.readouterr()
    prompts = []

    def fake_input(p):
        prompts.append(p)
        return {0: "1", 1: "1", 2: "500"}.get(len(prompts) - 1, "")

    cmd_ask(argparse.Namespace(model=str(model), top=3, plot=str(tmp_path / "case.png")), inp=fake_input)
    out = capsys.readouterr().out
    assert "licence plate" not in " ".join(prompts).lower()
    assert "Not asked" in out and "Similar real cases" in out
    assert (tmp_path / "case.png").exists()
