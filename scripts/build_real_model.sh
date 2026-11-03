#!/usr/bin/env bash
# Download real public data and build the presentation model (~15 min, ~3 GB disk).
# Illinois NIBRS covers Chicago and all of DuPage County, with clearance dates.
set -euo pipefail
STATES="${STATES:-IL CO VA MI WA VT}"
YEARS="${YEARS:-2023 2024}"
solvability fetch nibrs --state $STATES --years $YEARS --out data/raw/nibrs
solvability fetch la --since 2022-01-01 --until 2023-01-01 --limit 60000 --out data/raw/la_2022.csv
solvability train --data nibrs=data/raw/nibrs --data la=data/raw/la_2022.csv \
  --out artifacts/real.pkl --plot docs/calibration_real.png --metrics-json docs/metrics_real.json
echo "Model ready: solvability ask --model artifacts/real.pkl --plot case.png"
