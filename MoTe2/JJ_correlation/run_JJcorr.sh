#!/bin/bash
set -euo pipefail

# Physical/system parameters.
THETA=2.0
N1=3
N2=3
NUP=3
VD=0.0

# Flux grid: total number of flux points is NTHETA*NTHETA.
NTHETA=14

# Ktot-sector indices containing the degenerate ground states.
# Examples:
#   same Ktot sector:        GS_KTOT_IND=(0)
#   different Ktot sectors:  GS_KTOT_IND=(0 3 6)
GS_KTOT_IND=(0)

# Number of parallel flux workers.
N_JOBS=12

# Avoid nested BLAS/OpenMP oversubscription inside each flux worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-JJcorr-minimal"
mkdir -p "${MPLCONFIGDIR}"

OUTPUT="JJcorr_theta${THETA}_N${N1}x${N2}_nup${NUP}_vd${VD}_ntheta${NTHETA}.npz"

python3 JJ_corr_with_flux_insertion.py \
  --theta "${THETA}" \
  --n1 "${N1}" \
  --n2 "${N2}" \
  --nup "${NUP}" \
  --vd "${VD}" \
  --ntheta "${NTHETA}" \
  --gs-ktot-ind "${GS_KTOT_IND[@]}" \
  --n-jobs "${N_JOBS}" \
  --output "${OUTPUT}"

echo "Saved ${OUTPUT}"
echo "Saved ${OUTPUT%.npz}_spectral_phi2.png"
echo "Saved ${OUTPUT%.npz}_energy_heatmap.png"
