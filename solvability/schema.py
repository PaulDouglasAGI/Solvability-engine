"""Canonical case schema, category vocabularies, and the caveats every output carries.

Every loader converts its source into this one schema. Downstream code never sees
source-specific column names.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Canonical columns
# --------------------------------------------------------------------------- #
# Identity / bookkeeping
ID_COLS = ["case_id", "source", "jurisdiction", "area"]

# Facts about the reported case (model inputs, before featurisation)
NUMERIC_FIELDS = ["loss_value", "days_to_report", "pattern_count", "offender_count"]
CATEGORICAL_FIELDS = ["item_category", "location_type", "victim_type", "jurisdiction"]
# Tri-state flags: 1 = yes, 0 = no, NaN = not recorded
FLAG_FIELDS = ["has_footage", "has_witness", "has_plate_read", "insider"]
INPUT_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS + FLAG_FIELDS

# Outcome columns
#   incident_date  : when the theft happened (or was reported, if occurrence unknown)
#   report_date    : when it was reported
#   cleared        : 1 if cleared (arrest or exceptional) by the as-of date
#   clearance_type : "arrest" | "exceptional" | ""
#   duration_days  : days from incident to clearance (if cleared and timing known)
#                    or to the as-of date (if not cleared): the survival time
#   followup_days  : days from incident to the as-of date (how long we watched)
#   timing_known   : whether duration_days is a real clearance date (True) or the
#                    source only tells us *whether* it cleared, not *when* (False)
OUTCOME_COLS = [
    "incident_date", "report_date", "cleared", "clearance_type",
    "duration_days", "followup_days", "timing_known",
]

CANONICAL_COLS = ID_COLS + [c for c in INPUT_FIELDS if c not in ID_COLS] + OUTCOME_COLS

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #
ITEM_CATEGORIES = [
    "vehicle", "vehicle_parts_contents", "bicycle", "electronics", "cash_cards",
    "jewelry_precious", "tools_equipment", "retail_merchandise", "personal_items",
    "household", "other", "unknown",
]
LOCATION_TYPES = [
    "residence", "street_parking", "retail_commercial", "vehicle", "transit",
    "school_govt", "hotel_lodging", "other", "unknown",
]
VICTIM_TYPES = ["individual", "business", "government_institution", "unknown"]

VOCAB = {
    "item_category": ITEM_CATEGORIES,
    "location_type": LOCATION_TYPES,
    "victim_type": VICTIM_TYPES,
}

DEFAULT_HORIZONS = (1, 7, 30, 90, 365)

# --------------------------------------------------------------------------- #
# Caveats, shown on every prediction and evaluation report
# --------------------------------------------------------------------------- #
STANDING_CAVEATS = [
    "Only REPORTED thefts are in the data. Many thefts are never reported, and the ones "
    "that are tend to be the more solvable ones (higher value, insured, with a suspect). "
    "These numbers describe reported cases, not all thefts.",
    "'Cleared' means different things in different places. Departments differ on when "
    "they record an arrest or an exceptional clearance, how fast they update records, and "
    "whether an arrest for one incident clears related ones. Compare jurisdictions "
    "cautiously.",
    "Clearance is not conviction, and not getting your property back. An arrest can clear "
    "a case even if no one is convicted.",
    "These are rates for groups of similar cases. They are not a forecast for any single "
    "case and say nothing about the people involved.",
    "Never validate this model on news stories. Solved cases get far more coverage than "
    "unsolved ones, so news-sourced cases make any model look better than it is.",
]

SOURCE_CAVEATS = {
    "nibrs": "NIBRS: arrest and exceptional-clearance dates are recorded. Follow-up ends "
             "at the file snapshot, so later clearances are right-censored. Many agencies "
             "flag the date as the REPORT date when the occurrence date is unknown.",
    "chicago": "Chicago portal: records only whether an arrest was made, not when, and "
               "has no exceptional clearances or loss values (only $500 bands). Used for "
               "the long-run horizon only.",
    "la": "LAPD portal: case status (arrest/other) is recorded without a date, so it is "
          "used for the long-run horizon only. Victim type is inferred from blank "
          "demographics. Check the portal notes for gaps in coverage after the records "
          "system changed.",
    "canonical": "User-supplied records in the canonical schema. Their quality depends "
                 "on how they were collected.",
    "synthetic": "SYNTHETIC DEMO DATA: made up by `solvability synth`. Numbers from a "
                 "model trained on it are NOT real estimates.",
}

# Fields the public portals do not carry. If they are never observed during training,
# the model can say nothing about them and what-if toggles are reported as no-ops.
RARELY_PUBLIC_FIELDS = ["has_footage", "has_witness", "has_plate_read"]

NEWS_PATTERN = r"(?i)\b(?:news|media|press|article|headline|newspaper|tv|broadcast|blog)\b"
