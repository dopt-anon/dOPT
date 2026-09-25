#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dopt-matplotlib}"

PYTHON="${PYTHON:-python}"
MODULE=vehicle_platoon_learning.compare_stability_layers
OUTPUT=vehicle_platoon_learning/results
mkdir -p "$OUTPUT"

COMMON=(
  --dt 0.1 --n-train 16 --relative-observation-noise 0.01
  --horizon 30 --true-switch-time 14 --initial-switch-time 20.3
  --ridge 1e-5 --init-perturbation 1e-4
  --soft-epochs 50 --hard-epochs 200
  --learning-rate 0.001 --hard-learning-rate 0.003
  --switch-learning-rate 0.05 --initial-temperature 3.0 --final-temperature 0.25
  --normalized-margin-weight 0.0015 --seed 12 --p-floor 0.001
  --scs-eps 1e-7
)

run_case() {
  local name=$1
  shift
  "$PYTHON" -m "$MODULE" "$@" "${COMMON[@]}" --output "$OUTPUT/$name.json"
}

run_case 3vehicle_dOPT_ffo_original_patch \
  --benchmark 3vehicle \
  --methods dOPT ffolayer_lifted ffolayer_nonlifted --solver MOSEK
run_case 5vehicle_dOPT_ffo_original_patch \
  --benchmark 5vehicle \
  --xml vehicle_platoon_learning/data/5_vehicle_official/5_vehicle_platoon.xml \
  --methods dOPT ffolayer_lifted ffolayer_nonlifted --solver MOSEK
run_case 10vehicle_dOPT_ffo_original_patch \
  --benchmark 10vehicle \
  --xml vehicle_platoon_learning/data/10_vehicle_official/10_vehicle_platoon.xml \
  --methods dOPT ffolayer_lifted ffolayer_nonlifted --solver MOSEK
