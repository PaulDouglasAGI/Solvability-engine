import numpy as np
import pandas as pd
import pytest

from solvability.cleaning import NewsSourceError, assert_no_news, clean, derive_pattern_count, horizon_labels
from solvability.data.loader import load_canonical, load_chicago, load_la, load_nibrs
from solvability.data.mappings import map_value
from solvability.data.synthetic import generate


# --------------------------------------------------------------------- mappings
@pytest.mark.parametrize("text,kind,want", [
    ("PARKING LOT / GARAGE (NON RESIDENTIAL)", "location", "street_parking"),
    ("RESIDENCE - GARAGE", "location", "residence"),
    ("GAS STATION", "location", "retail_commercial"),
    ("COMMERCIAL / BUSINESS OFFICE", "location", "retail_commercial"),
    ("CTA TRAIN", "location", "transit"),
    ("THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)", "item", "vehicle_parts_contents"),
    ("VEHICLE - STOLEN", "item", "vehicle"),
    ("BIKE - STOLEN", "item", "bicycle"),
    ("Household Goods", "item", "household"),
    ("Furniture", "item", "household"),
    ("Business", "victim", "business"),
])
def test_mappings(text, kind, want):
    assert map_value(text, kind) == want


# ------------------------------------------------------------------- labelling
def _rows(**cols):
    return pd.DataFrame(cols)


def test_horizon_labels_known_timing():
    df = _rows(cleared=[1, 1, 0, 0], duration_days=[5, 40, 100, 10], followup_days=[200, 200, 100, 10],
               timing_known=[True] * 4)
    y = horizon_labels(df, 30, 365)
    assert y.tolist()[:3] == [1.0, 0.0, 0.0]
    assert np.isnan(y.iloc[3])            # censored at day 10: unknown at day 30


def test_horizon_labels_unknown_timing_only_long_run():
    df = _rows(cleared=[1, 0, 1], duration_days=[np.nan, 500, np.nan], followup_days=[500, 500, 100],
               timing_known=[False] * 3)
    assert horizon_labels(df, 30, 365).isna().all()
    y = horizon_labels(df, 365, 365)
    assert y.tolist()[:2] == [1.0, 0.0]
    assert np.isnan(y.iloc[2])            # watched for only 100 days


def test_news_guard():
    df = pd.DataFrame({"source": ["nibrs", "local news scrape"]})
    with pytest.raises(NewsSourceError):
        assert_no_news(df)
    assert_no_news(pd.DataFrame({"source": ["nibrs", "chicago"]}))


def test_pattern_count_uses_only_prior_cases():
    df = pd.DataFrame({
        "area": ["a"] * 4 + ["b"],
        "item_category": ["bicycle"] * 5,
        "incident_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-05", "2024-02-01", "2024-01-05"]),
    })
    pc = derive_pattern_count(df, window=14)
    assert pc.tolist() == [0, 1, 1, 0, 0]


def test_clean_handles_bad_values():
    df = generate(200, seed=1)
    df.loc[0, "loss_value"] = -5
    df.loc[1, "days_to_report"] = -3
    df.loc[2, "item_category"] = "spaceship"
    df.loc[3, "incident_date"] = None
    df = pd.concat([df, df.iloc[[10]]])
    rep = {}
    out = clean(df, report=rep)
    assert rep["dropped_no_date"] == 1 and rep["dropped_duplicates"] == 1
    assert out["loss_value"].isna().sum() >= 1
    assert (out["item_category"] == "other").any()
    assert out["days_to_report"].min() >= 0


# --------------------------------------------------------------------- loaders
def _write_nibrs(folder):
    t = {
        "NIBRS_incident": pd.DataFrame({
            "AGENCY_ID": [1, 1, 1, 2], "INCIDENT_ID": [10, 11, 12, 13],
            "INCIDENT_DATE": ["2022-01-01", "2022-03-01", "2022-06-01", "2022-07-01"],
            "REPORT_DATE_FLAG": ["", "R", "", ""],
            "CLEARED_EXCEPT_ID": [6, 6, 2, 6], "CLEARED_EXCEPT_DATE": [None, None, "2022-06-20", None]}),
        "NIBRS_OFFENSE": pd.DataFrame({"OFFENSE_ID": [1, 2, 3, 4, 5], "INCIDENT_ID": [10, 11, 12, 13, 13],
                                       "OFFENSE_TYPE_ID": [100, 200, 100, 300, 100],
                                       "LOCATION_ID": [20, 13, 8, 13, 13]}),
        "NIBRS_OFFENSE_TYPE": pd.DataFrame({"OFFENSE_TYPE_ID": [100, 200, 300],
                                            "OFFENSE_CODE": ["23H", "240", "13A"]}),
        "NIBRS_LOCATION_TYPE": pd.DataFrame({"LOCATION_ID": [8, 13, 20],
                                             "LOCATION_NAME": ["Department/Discount Store",
                                                               "Highway/Road/Alley/Street/Sidewalk", "Residence/Home"]}),
        "NIBRS_PROPERTY": pd.DataFrame({"PROPERTY_ID": [1, 2, 3], "INCIDENT_ID": [10, 12, 12],
                                        "PROP_LOSS_ID": [7, 7, 5]}),
        "NIBRS_PROPERTY_DESC": pd.DataFrame({"PROPERTY_ID": [1, 1, 2, 3], "PROP_DESC_ID": [4, 26, 19, 17],
                                             "PROPERTY_VALUE": [400, 900, 120, 5000]}),
        "NIBRS_PROP_DESC_TYPE": pd.DataFrame({"PROP_DESC_ID": [4, 17, 19, 26],
                                              "PROP_DESC_NAME": ["Bicycles", "Jewelry/Precious Metals/Gems",
                                                                 "Merchandise", "Radios/TVs/VCRs/DVD Players"]}),
        "NIBRS_VICTIM": pd.DataFrame({"VICTIM_ID": [1, 2, 3, 4], "INCIDENT_ID": [10, 11, 12, 13],
                                      "VICTIM_TYPE_ID": [4, 4, 1, 4]}),
        "NIBRS_VICTIM_TYPE": pd.DataFrame({"VICTIM_TYPE_ID": [1, 4], "VICTIM_TYPE_NAME": ["Business", "Individual"]}),
        "NIBRS_ARRESTEE": pd.DataFrame({"ARRESTEE_ID": [1, 2], "INCIDENT_ID": [10, 10],
                                        "ARREST_DATE": ["2022-01-09", "2022-01-03"]}),
        "NIBRS_OFFENDER": pd.DataFrame({"OFFENDER_ID": [1, 2, 3], "INCIDENT_ID": [10, 10, 12]}),
        "NIBRS_VICTIM_OFFENDER_REL": pd.DataFrame({"VICTIM_ID": [3], "OFFENDER_ID": [3], "RELATIONSHIP_ID": [9]}),
        "NIBRS_RELATIONSHIP": pd.DataFrame({"RELATIONSHIP_ID": [9], "RELATIONSHIP_CODE": ["EE"]}),
        "agencies": pd.DataFrame({"AGENCY_ID": [1, 2], "ORI": ["XX0010000", "XX0020000"]}),
    }
    for name, df in t.items():
        df.to_csv(folder / f"{name}.csv", index=False)


def test_nibrs_loader(tmp_path):
    _write_nibrs(tmp_path)
    df = load_nibrs(tmp_path, as_of="2022-12-31").set_index("case_id")
    assert set(df.index) == {"nibrs:10", "nibrs:11", "nibrs:12", "nibrs:13"}
    a = df.loc["nibrs:10"]
    assert a.cleared == 1 and a.clearance_type == "arrest" and a.duration_days == 2    # first arrest
    assert a.loss_value == 1300 and a.item_category == "electronics"                   # top-value item
    assert a.location_type == "residence" and np.isnan(a.offender_count) and a.timing_known
    b = df.loc["nibrs:11"]
    assert b.cleared == 0 and b.item_category == "vehicle" and b.duration_days == b.followup_days
    c = df.loc["nibrs:12"]
    assert c.clearance_type == "exceptional" and c.duration_days == 19
    assert c.loss_value == 120                              # recovered (loss 5) item excluded
    assert c.victim_type == "business" and c.insider == 1.0
    assert df.loc["nibrs:13"].jurisdiction == "nibrs:XX0020000"


def test_chicago_loader(tmp_path):
    p = tmp_path / "c.csv"
    pd.DataFrame({
        "id": [1, 2, 3, 4], "date": ["2023-01-01T10:00:00", "2023-01-02T10:00:00", "2023-01-03T00:00:00", "2023-01-04"],
        "primary_type": ["THEFT", "MOTOR VEHICLE THEFT", "BATTERY", "THEFT"],
        "description": ["RETAIL THEFT", "AUTOMOBILE", "SIMPLE", "ATTEMPT THEFT"],
        "location_description": ["DEPARTMENT STORE", "STREET", "STREET", "STREET"],
        "arrest": [True, False, True, False], "beat": [111, 112, 113, 114],
        "updated_on": ["2023-02-01"] * 4,
    }).to_csv(p, index=False)
    df = load_chicago(p, as_of="2024-01-01")
    assert len(df) == 2                                    # battery and attempts dropped
    assert df.cleared.tolist() == [1, 0]
    assert df.item_category.tolist() == ["retail_merchandise", "vehicle"]
    assert df.victim_type.iloc[0] == "business"
    assert not df.timing_known.any() and df.duration_days.isna().iloc[0]


def test_la_loader(tmp_path):
    p = tmp_path / "la.csv"
    pd.DataFrame({
        "dr_no": [1, 2, 3], "date_rptd": ["2022-01-05", "2022-01-02", "2022-01-02"],
        "date_occ": ["2022-01-01", "2022-01-02", "2022-01-02"],
        "crm_cd_desc": ["BIKE - STOLEN", "SHOPLIFTING - PETTY THEFT ($950 & UNDER)", "THEFT OF IDENTITY"],
        "premis_desc": ["SIDEWALK", "DEPARTMENT STORE", "SINGLE FAMILY DWELLING"],
        "status": ["IC", "AA", "IC"], "vict_sex": ["M", "X", "F"], "vict_age": [30, 0, 40],
        "rpt_dist_no": [101, 102, 103],
    }).to_csv(p, index=False)
    df = load_la(p, as_of="2023-01-01")
    assert len(df) == 2
    assert df.days_to_report.tolist() == [4, 0]
    assert df.cleared.tolist() == [0, 1]
    assert df.victim_type.tolist() == ["individual", "business"]


def test_canonical_roundtrip(tmp_path):
    p = tmp_path / "s.csv"
    generate(100, seed=2).to_csv(p, index=False)
    df = load_canonical(p)
    assert df["timing_known"].all() and (df["source"] == "synthetic").all()
