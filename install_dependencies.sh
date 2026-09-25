#!/usr/bin/env bash
set -euo pipefail

# Install the pinned Python environment used for the paper experiments.
# Run this script after activating any clean Python 3.13 environment:
#   bash install_dependencies.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${python_bin}" >&2
  exit 1
fi

if ! "$python_bin" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 13))'; then
  echo "This repository requires Python 3.13; found $("$python_bin" --version 2>&1)." >&2
  exit 1
fi

"$python_bin" -m pip install \
  "pip==25.3" \
  "setuptools==80.9.0" \
  "wheel==0.45.1"

"$python_bin" -m pip install \
  "numpy==2.3.5" \
  "scipy==1.16.3" \
  "matplotlib==3.10.7" \
  "torch==2.9.1" \
  "cvxpy==1.9.2" \
  "cvxpylayers==1.2.0" \
  "diffcp==1.1.9" \
  "scs==3.2.11" \
  "qpsolvers==4.8.2" \
  "threadpoolctl==3.6.0" \
  "mosek==11.0.29"

# FFOLayer's parametric-cone path uses cvxtorch. Pin the exact upstream commit
# used to validate this repository; FFOLayer itself is vendored under
# third_party/ and imported directly from there.
"$python_bin" -m pip install --no-deps \
  "git+https://github.com/cvxpy/cvxtorch.git@bae2d6494695a19cf1d2ee275d9058de3311a272"

cd "$repo_root"
"$python_bin" -c '
import cvxpy
import cvxpylayers
import cvxtorch
import diffcp
import matplotlib
import mosek
import numpy
import qpsolvers
import scipy
import scs
import threadpoolctl
import torch
from src.dSDP import dSDPLayer
from src.dSOCP import dSOCPLayer
from third_party.FFOLayer.src.ffolayer import FFOLayer
print("Environment import check passed.")
'

cat <<EOF
Dependency installation complete.

MOSEK is required for the paper benchmarks. The Python package is installed,
but you must provide a valid MOSEK license before running those experiments:
  https://www.mosek.com/products/academic-licenses/

Run all repository commands from:
  ${repo_root}
EOF
