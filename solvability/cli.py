"""Command-line interface.

  solvability synth    --out data/synth.csv
  solvability fetch    chicago --since 2023-01-01 --out data/raw/chicago.csv
  solvability train    --data synthetic=data/synth.csv --out artifacts/model.pkl
  solvability evaluate --model artifacts/model.pkl --data canonical=holdout.csv --plot calib.png
  solvability predict  --model artifacts/model.pkl --set item=bicycle value=900 location=street_parking
  solvability whatif   --model ... --set ... --scenario "reported same day: report_delay=0"
  solvability whatif   --model ... --set ... --sweep report_delay=0,1,3,7,30
  solvability wait     --model ... --set ... --window 30
  solvability info     --model artifacts/model.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import whatif as wi
from .cleaning import NewsSourceError, assert_no_news, clean, find_news_rows
from .data.loader import PORTALS, fetch_nibrs, fetch_portal, load_many
from .data.synthetic import generate
from .evaluation import calibration_plot, case_plot, evaluate
from .models.engine import SolvabilityEngine, split
from .schema import DEFAULT_HORIZONS

WRAP = 88


def _wrap(s, indent="  "):
    return textwrap.fill(s, WRAP, initial_indent=indent, subsequent_indent=indent + "  ")


def _pct(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "  n/a"
    return f"{100 * p:5.1f}%"


def _parse_value(v: str):
    s = v.strip()
    low = s.lower()
    if low in {"yes", "true", "y"}:
        return 1
    if low in {"no", "false", "n"}:
        return 0
    if low in {"unknown", "?", "na", "none", ""}:
        return None
    try:
        return float(s.replace("$", "").replace(",", ""))
    except ValueError:
        return s


def _parse_assignments(items) -> dict:
    out = {}
    for it in items or []:
        for part in it.split(",") if "=" in it and it.count("=") > 1 else [it]:
            if "=" not in part:
                raise SystemExit(f"expected key=value, got {part!r}")
            k, v = part.split("=", 1)
            out[k.strip()] = _parse_value(v)
    return out


def _case_fields(args) -> dict:
    fields = {}
    if getattr(args, "case", None):
        src = args.case
        txt = Path(src).read_text() if Path(src).exists() else src
        fields.update(json.loads(txt))
    fields.update(_parse_assignments(getattr(args, "set", None)))
    try:
        return wi.normalise_fields({k: v for k, v in fields.items()})
    except wi.CaseInputError as e:
        raise SystemExit(str(e))


def _load_engine(path) -> SolvabilityEngine:
    return SolvabilityEngine.load(path)


def _print_caveats(engine: SolvabilityEngine, out=None):
    out = out or sys.stdout
    print("\nCaveats", file=out)
    if engine.is_synthetic:
        print(_wrap("!! This model was trained on SYNTHETIC demo data. The numbers show how the "
                    "engine works. They are not real-world estimates."), file=out)
    for c in engine.caveats():
        print(_wrap("- " + c), file=out)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_synth(args):
    df = generate(n=args.n, seed=args.seed, as_of=args.as_of)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} synthetic cases -> {args.out}  (clearance by as-of: {df.cleared.mean():.1%})")


def cmd_fetch(args):
    if args.portal == "nibrs":
        if not args.state or not args.years:
            raise SystemExit("fetch nibrs needs --state XX [YY ...] and --years YYYY [YYYY ...]")
        for st in args.state:
            for y in args.years:
                try:
                    print(f"wrote {fetch_nibrs(st, y, args.out)}")
                except FileNotFoundError as e:
                    print(f"skipped: {e}")
        return
    if not args.since:
        raise SystemExit("portal downloads need --since YYYY-MM-DD")
    p = fetch_portal(args.portal, args.out, since=args.since, until=args.until, limit=args.limit)
    print(f"wrote {p}")


def _load_clean(specs, as_of):
    raw = load_many(specs, as_of=as_of)
    rep = {}
    df = clean(raw, report=rep)
    print(f"loaded {len(raw):,} rows -> {len(df):,} after cleaning  {json.dumps(rep)}")
    return df


def _print_eval(engine, res, out=None):
    out = out or sys.stdout
    m = res["metrics"]
    if m.empty:
        print("no labelled test rows", file=out)
        return
    show = m[m.kind == "calibrated"][["horizon_days", "model", "n", "observed_rate", "mean_pred",
                                      "brier", "auc", "ece"]]
    print("\nHeld-out metrics (calibrated)", file=out)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"), file=out)
    if "c_index_cox" in res:
        print(f"\nConcordance (time-to-clearance): Cox {res['c_index_cox']:.3f}  "
              f"headline {res['c_index_headline']:.3f}", file=out)
    bj = res.get("by_jurisdiction")
    if bj is not None and len(bj):
        print(f"\nBy jurisdiction, cleared within {int(engine.long_run)} days (observed vs predicted). "
              "Gaps here often reflect different definitions of 'cleared':", file=out)
        print(bj.head(15).to_string(float_format=lambda x: f"{x:.3f}"), file=out)


def cmd_train(args):
    df = _load_clean(args.data, args.as_of)
    news = find_news_rows(df)
    if news.any():
        if not args.allow_news_training:
            raise SystemExit(f"{int(news.sum())} rows look news-derived. News coverage over-represents "
                             "solved cases. Remove them, or pass --allow-news-training (they are still "
                             "never used for evaluation).")
        df_eval_pool = df[~news]
    else:
        df_eval_pool = df
    horizons = tuple(sorted(int(h) for h in args.horizons.split(","))) if args.horizons else DEFAULT_HORIZONS
    train, calib, test = split(df, args.split, seed=args.seed)
    test = test[test["case_id"].isin(df_eval_pool["case_id"])]
    calib = calib[calib["case_id"].isin(df_eval_pool["case_id"])]
    engine = SolvabilityEngine(horizons=horizons, headline=args.headline, random_state=args.seed,
                               stratify_cox=not args.no_strata)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.fit(train, calib)
    for w in caught:
        print(f"note: {w.message}")
    engine.meta["split"] = args.split
    engine.save(args.out)
    print(f"saved model -> {args.out}   (train {len(train):,} / calib {len(calib):,} / test {len(test):,})")

    res = evaluate(engine, test)
    _print_eval(engine, res)
    if args.plot:
        calibration_plot(engine, res, args.plot)
        print(f"\ncalibration plot -> {args.plot}")
    if args.metrics_json:
        res["metrics"].to_json(args.metrics_json, orient="records", indent=2)
    _print_caveats(engine)


def cmd_evaluate(args):
    engine = _load_engine(args.model)
    df = _load_clean(args.data, args.as_of)
    try:
        assert_no_news(df, "evaluation")
    except NewsSourceError as e:
        raise SystemExit(str(e))
    res = evaluate(engine, df)
    _print_eval(engine, res)
    if args.plot:
        calibration_plot(engine, res, args.plot)
        print(f"\ncalibration plot -> {args.plot}")
    _print_caveats(engine)


def _prediction_payload(engine, fields, horizon=None, top=5):
    case = wi.make_case(fields, engine)
    pred = wi.predict_fields(engine, fields)
    horizon = horizon or engine._default_horizon()

    def _d(arr):
        return {str(h): (None if np.isnan(arr[j]) else float(arr[j])) for j, h in enumerate(engine.horizons)}

    return {
        "inputs": {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v) for k, v in fields.items()},
        "horizons_days": list(engine.horizons),
        "p_cleared": _d(pred["headline"]),
        "p_cleared_cox": _d(pred["cox"]) if "cox" in pred else None,
        "averaged_over_real_cases": pred["n_population"],
        "headline_model": engine.headline,
        "factor_horizon_days": horizon,
        "top_factors": wi.explain_fields(engine, fields, horizon=horizon, top=top),
        "warnings": wi.check_case(engine, case, fields),
        "similar_real_cases": engine.similar(case, fields.keys()),
        "synthetic_model": engine.is_synthetic,
        "caveats": engine.caveats(),
    }


def _case_phrase(k, v):
    if k == "loss_value":
        return f"${v:,.0f}"
    if k == "days_to_report":
        return f"reported after {v:g} days"
    if k == "jurisdiction":
        parts = str(v).split(" ", 1)
        if len(parts) == 2 and len(parts[0]) == 2:
            return f"{parts[1].title()} ({parts[0].upper()})"
        return str(v).replace("_", " ").title()
    if k in ("has_footage", "has_witness", "has_plate_read", "insider"):
        return f"{wi.FIELD_LABELS[k]}: {wi._show(k, v)}"
    return str(v).replace("_", " ")


def _render_prediction(engine, fields, pay, plot=None):
    print("Case: " + (", ".join(f"{k}={v}" for k, v in fields.items()) or "(all unknown)"))
    sim = pay.get("similar_real_cases")
    obs = (sim or {}).get("observed_by_horizon", {})
    print("\nP(case cleared within ...)")
    if not engine.has_timing:
        print(_wrap("(Only the long-run horizon is available: the training sources don't record "
                    "WHEN cases cleared.)"))
    head = f"  {'':>9}  {'model':>6}"
    if sim and sim["n"]:
        head += f"   {'similar real cases':>18}"
    print(head)
    for h in engine.horizons:
        line = f"  {h:>4} days  {_pct(pay['p_cleared'][str(h)])}"
        if sim and sim["n"]:
            o = obs.get(int(h))
            if o is None and h == engine.long_run:
                o = sim.get("observed_long_run")
            line += f"   {_pct(o):>18}"
        print(line)
    if sim and sim["n"]:
        src = ", ".join(f"{k} {v:,}" for k, v in sim["sources"].items())
        yrs = f"{sim['years'][0]}–{sim['years'][1]}" if sim["years"] else ""
        print(_wrap(f"Similar real cases: {sim['n']:,} reported thefts ({src}; {yrs}) matching "
                    + ("; ".join(sim["criteria"]) or "nothing specific (all cases)") + "."))
        if sim["relaxed"]:
            print(_wrap("Too few exact matches, so these were not required to match: "
                        + ", ".join(r.replace("_", " ") for r in sim["relaxed"]) + "."))
        if sim["median_days_to_clear"] is not None:
            print(_wrap(f"Among those that were cleared, the median time to clearance was "
                        f"{sim['median_days_to_clear']:.0f} days."))
    print(f"\nWhat moves the estimate at {pay['factor_horizon_days']} days (each answer vs. not knowing it):")
    for f in pay["top_factors"]:
        print(f"  {f['delta_pp']:+6.1f} pts  {f['label']}: {f['value']}")
    if not pay["top_factors"]:
        print("  (no answers given: this is the average over all real cases)")
    print(_wrap(f"Unanswered questions are averaged over {pay['averaged_over_real_cases']} real "
                "cases that match the answers given.", ""))
    for w in pay["warnings"]:
        print(_wrap("! " + w, ""))
    if plot:
        sub = ", ".join(_case_phrase(k, v) for k, v in fields.items())
        case_plot(engine, wi.population(engine, fields), sim, plot, subtitle=sub or None)
        print(f"\nchart -> {plot}")
    _print_caveats(engine)


def cmd_predict(args):
    engine = _load_engine(args.model)
    fields = _case_fields(args)
    pay = _prediction_payload(engine, fields, args.horizon, args.top)
    if args.json:
        print(json.dumps(pay, indent=2, default=str))
        return
    _render_prediction(engine, fields, pay, args.plot)


QUESTIONS = {
    "item_category": "What was stolen?",
    "location_type": "Where was it taken from?",
    "loss_value": "Roughly what was it worth, in dollars?",
    "days_to_report": "How many days after the theft was it reported? (0 = same day)",
    "victim_type": "Who was the victim?",
    "jurisdiction": "Which police agency? (type part of the name)",
    "has_footage": "Was there video footage?",
    "has_witness": "Were there witnesses?",
    "has_plate_read": "Was a licence plate captured?",
    "insider": "Is an employee or employer suspected?",
}


def _ask_one(engine, f, inp, out):
    ref = engine.reference_cases
    q = QUESTIONS[f]
    if f in ("item_category", "location_type", "victim_type"):
        levels = [lv for lv in engine.builder.levels_[f] if lv not in ("unknown",)]
        counts = ref[f].astype(str).value_counts() if ref is not None else pd.Series(dtype=int)
        levels.sort(key=lambda lv: (lv == "other", -counts.get(lv, 0)))
        print(f"\n{q}", file=out)
        for i, lv in enumerate(levels, 1):
            print(f"  {i:>2}) {lv.replace('_', ' '):<26} {counts.get(lv, 0):>9,} real cases", file=out)
        a = inp("  choose a number (Enter = don't know): ").strip()
        if a.isdigit() and 1 <= int(a) <= len(levels):
            return levels[int(a) - 1]
        return None
    if f == "jurisdiction":
        levels = [lv for lv in engine.builder.levels_[f] if lv != "other"]
        a = inp(f"\n{q} Enter = skip: ").strip().lower()
        if not a:
            return None
        hits = [lv for lv in levels if a in lv]
        if not hits:
            print("  no modelled agency matches; using the pooled average", file=out)
            return None
        for i, lv in enumerate(hits[:12], 1):
            print(f"  {i:>2}) {lv}", file=out)
        b = inp("  choose a number: ").strip()
        return hits[int(b) - 1] if b.isdigit() and 1 <= int(b) <= len(hits[:12]) else None
    if f in ("loss_value", "days_to_report"):
        a = inp(f"\n{q} (Enter = don't know): ").strip()
        v = _parse_value(a) if a else None
        return v if isinstance(v, float) else None
    a = inp(f"\n{q} [y/n, Enter = don't know]: ").strip().lower()
    return 1 if a.startswith("y") else 0 if a.startswith("n") else None


def cmd_ask(args, inp=input, out=None):
    out = out or sys.stdout
    engine = _load_engine(args.model)
    fields_ok = engine.askable_fields()
    skipped = [QUESTIONS[f] for f in QUESTIONS if f not in fields_ok]
    print("Answer a few questions about a reported theft. Every question is about something "
          "the real data records, so the answer is based on actual cases.", file=out)
    if skipped:
        print(_wrap("Not asked, because the training data doesn't record it: "
                    + "; ".join(s.split("?")[0].lower() for s in skipped) + "."), file=out)
    fields = {}
    for f in fields_ok:
        v = _ask_one(engine, f, inp, out)
        if v is not None:
            fields[f] = v
    print("", file=out)
    pay = _prediction_payload(engine, fields, None, args.top)
    _render_prediction(engine, fields, pay, args.plot)


def _parse_scenarios(items) -> dict:
    out = {}
    for i, s in enumerate(items or []):
        if ":" in s.split("=")[0]:
            name, body = s.split(":", 1)
        else:
            name, body = f"scenario {i + 1}", s
        out[name.strip()] = _parse_assignments([p.strip() for p in body.split(",")])
    return out


def cmd_whatif(args):
    engine = _load_engine(args.model)
    base = _case_fields(args)
    try:
        if args.sweep:
            field, vals = args.sweep.split("=", 1)
            values = [_parse_value(v) for v in vals.split(",")]
            tbl = wi.sweep(engine, base, field, values)
        else:
            scen = _parse_scenarios(args.scenario)
            if not scen:
                raise SystemExit("give --scenario 'name: key=value,...' (repeatable) or --sweep key=v1,v2,...")
            for ov in scen.values():
                wi.normalise_fields(ov)
            tbl = wi.compare(engine, base, scen)
    except wi.CaseInputError as e:
        raise SystemExit(str(e))
    if args.json:
        print(tbl.to_json(orient="records", indent=2))
        return
    fmt = {c: (lambda x: _pct(x).strip()) for c in tbl.columns if c.startswith("by_")}
    print("P(case cleared within N days) under each hypothetical\n")
    print(tbl.to_string(index=False, formatters=fmt))
    for w in wi.check_case(engine, wi.make_case(base, engine), base):
        print(_wrap("! " + w, ""))
    _print_caveats(engine)


def cmd_wait(args):
    engine = _load_engine(args.model)
    fields = _case_fields(args)
    try:
        res = wi.wait_analysis(engine, fields, window=args.window, give_up_below=args.threshold)
    except ValueError as e:
        raise SystemExit(str(e))
    if args.json:
        res = {**res, "table": res["table"].to_dict(orient="records")}
        print(json.dumps(res, indent=2, default=str))
        return
    L, W = int(res["long_run"]), res["window"]
    print(f"If the case is still open after N days, chance it is cleared...\n")
    print(f"  {'days open':>9}   {'in next ' + str(W) + 'd':>11}   {'by day ' + str(L):>10}")
    for r in res["table"].itertuples(index=False):
        print(f"  {r[0]:>9}   {_pct(r[1]):>11}   {_pct(r[2]):>10}")
    d = res["diminishing_returns_day"]
    print()
    if d is not None:
        print(_wrap(f"After about {d} days with no clearance, the remaining chance of clearance "
                    f"within {L} days falls below {res['threshold']:.0%}. That can be a sensible "
                    f"point to move on with the insurance claim or replace the item.", ""))
    else:
        print(_wrap(f"The remaining chance stays above {res['threshold']:.0%} through day {L}.", ""))
    print(_wrap("The model only sees clearance within the follow-up window. It cannot speak to "
                "property recovered without an arrest, or to anything after the longest horizon.", ""))
    for w in res["warnings"]:
        print(_wrap("! " + w, ""))
    _print_caveats(engine)


def cmd_info(args):
    e = _load_engine(args.model)
    print(json.dumps(e.meta, indent=2, default=str))
    if e.cox is not None:
        hr = pd.Series(e.cox.hazard_ratios(), index=e.builder.columns_)
        print("\nCox hazard ratios per unit (largest effects; >1 = clears sooner):")
        order = np.argsort(-np.abs(np.log(hr.values)))[: args.top]
        for i in order:
            print(f"  {hr.index[i]:<40} {hr.values[i]:6.2f}")
    _print_caveats(e)


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="solvability",
                                description="Looks back at reported property thefts and estimates P(case cleared) over time.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write synthetic demo data (canonical schema)")
    s.add_argument("--n", type=int, default=20000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--as-of", default="2025-12-31")
    s.add_argument("--out", default="data/synth.csv")
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("fetch", help="download real theft records (FBI NIBRS or city portals)")
    s.add_argument("portal", choices=sorted(PORTALS) + ["nibrs"])
    s.add_argument("--state", nargs="*", help="nibrs: state abbreviations, e.g. CO VA")
    s.add_argument("--years", nargs="*", type=int, help="nibrs: data years, e.g. 2023 2024")
    s.add_argument("--since", help="portals: YYYY-MM-DD")
    s.add_argument("--until")
    s.add_argument("--limit", type=int, default=100000)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_fetch)

    data_help = "SOURCE=PATH, repeatable. Sources: nibrs (folder), chicago, la, canonical (CSV)"

    s = sub.add_parser("train", help="fit, calibrate and evaluate")
    s.add_argument("--data", action="append", required=True, help=data_help)
    s.add_argument("--as-of", help="date follow-up ends (default: latest date in the data)")
    s.add_argument("--split", choices=["random", "temporal"], default="random")
    s.add_argument("--horizons", help="comma list of days, default 1,7,30,90,365")
    s.add_argument("--headline", choices=["auto", "blend", "gbm", "cox"], default="auto")
    s.add_argument("--no-strata", action="store_true", help="don't stratify Cox baseline by jurisdiction")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="artifacts/model.pkl")
    s.add_argument("--plot", default="artifacts/calibration.png")
    s.add_argument("--metrics-json")
    s.add_argument("--allow-news-training", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="evaluate a saved model on other official data")
    s.add_argument("--model", required=True)
    s.add_argument("--data", action="append", required=True, help=data_help)
    s.add_argument("--as-of")
    s.add_argument("--plot")
    s.set_defaults(fn=cmd_evaluate)

    def case_args(sp):
        sp.add_argument("--model", required=True)
        sp.add_argument("--case", help="JSON file or JSON string of case fields")
        sp.add_argument("--set", nargs="*", help="field=value ... (e.g. item=bicycle value=900 footage=yes)")
        sp.add_argument("--json", action="store_true")

    s = sub.add_parser("predict", help="probability + top factors for one case (real or hypothetical)")
    case_args(s)
    s.add_argument("--horizon", type=int, help="horizon for factor attribution (default 90)")
    s.add_argument("--top", type=int, default=5)
    s.set_defaults(fn=cmd_predict)

    s.add_argument("--plot", help="save a chart of the case (PNG) for slides")

    s = sub.add_parser("ask", help="interactive: answer questions about a theft, get the odds")
    s.add_argument("--model", required=True)
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--plot", help="save a chart of the case (PNG) for slides")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("whatif", help="compare hypothetical variants of a case")
    case_args(s)
    s.add_argument("--scenario", action="append", help="'name: field=value,field=value' (repeatable)")
    s.add_argument("--sweep", help="field=v1,v2,v3")
    s.set_defaults(fn=cmd_whatif)

    s = sub.add_parser("wait", help="if a case is still open after N days, what are the odds now?")
    case_args(s)
    s.add_argument("--window", type=int, default=30)
    s.add_argument("--threshold", type=float, default=0.02)
    s.set_defaults(fn=cmd_wait)

    s = sub.add_parser("info", help="model card")
    s.add_argument("--model", required=True)
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(fn=cmd_info)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
