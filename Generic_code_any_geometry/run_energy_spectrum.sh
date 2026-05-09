#!/bin/bash

# Local run script for tilted_manybody_spectrum_general_v2.py.
# This version uses nonnegative integer momentum representatives only.
#
# Run with:
#   bash run_tilted_manybody_spectrum_v2.sh

set -euo pipefail

# Physical/model parameters.
THETA=2.0
NUP=4 #8
VD=0.0
NK=8
NEV=5

# Tilted finite cluster vectors in the original a1,a2 integer basis.
# Number of orbitals/sites is abs(L1X*L2Y - L1Y*L2X).
#
# 24-site geometry:
#   L1 = 6*a1
#   L2 = -2*a1 + 4*a2
L1X=4 #6
L1Y=0 #0
L2X=0 #-2
L2Y=3 #4
#
# 27-site geometry for later:
#   L1 = 3*(2*a1 - a2) = 6*a1 - 3*a2
#   L2 = 3*(2*a2 - a1) = -3*a1 + 6*a2
# To use it, comment the four lines above and uncomment these:
# L1X=6
# L1Y=-3
# L2X=-3
# L2Y=6

# Number of CPU workers used by joblib inside Python.
N_JOBS=12

OUTPUT_PREFIX="ED_tilted_v2_L1_${L1X}_${L1Y}_L2_${L2X}_${L2Y}_nup_${NUP}_theta_${THETA}_nk_${NK}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

python3 tilted_manybody_spectrum_general_v2.py \
  --theta "$THETA" \
  --nup "$NUP" \
  --vd "$VD" \
  --l1 "$L1X" "$L1Y" \
  --l2 "$L2X" "$L2Y" \
  --nk "$NK" \
  --nev "$NEV" \
  --n-jobs "$N_JOBS" \
  --output-prefix "$OUTPUT_PREFIX"
