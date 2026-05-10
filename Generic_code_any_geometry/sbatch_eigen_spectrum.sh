#!/bin/bash
#SBATCH -p cpuq
#SBATCH --job-name=mbSpectrum
#SBATCH --output=mb_spectrum_%j.out
#SBATCH --error=mb_spectrum_%j.err
#SBATCH --time=200:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=700G

# Submit from the folder containing this script and manybody_energy_spectrum.py:
#   sbatch submit_manybody_spectrum.sh
#
# For a local test without Slurm, run:
#   bash submit_manybody_spectrum.sh

set -euo pipefail

# Physical/model parameters.
THETA=2.0
NUP=6
VD=0.0
NK=8
NEV=5

# Tilted finite cluster vectors in the original a1,a2 integer basis.
# Number of orbitals/sites is abs(L1X*L2Y - L1Y*L2X).
# Current 24-site geometry: L1=(6,0), L2=(-2,4), so Norb=24.
# Current 27-site geometry: L1=(6,-3), L2=(-3,6), so Norb=24.
L1X=6
L1Y=0
L2X=0
L2Y=3

# Parallelism inside Python/joblib. Match this to --cpus-per-task above.
N_JOBS=${SLURM_CPUS_PER_TASK:-18}

# Run in the folder where you submitted the job. This keeps the .log, .npz,
# .png, and Slurm .out/.err files in the same folder where you ran sbatch.
RUN_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$RUN_DIR"

# Keep Matplotlib cache local/writable on clusters where $HOME/.config is read-only.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# Avoid nested BLAS/OpenMP oversubscription.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Output files will be ${OUTPUT_PREFIX}.png and ${OUTPUT_PREFIX}.npz.
OUTPUT_PREFIX="ED_nup_${NUP}_theta_${THETA}_L1_${L1X}_${L1Y}_L2_${L2X}_${L2Y}"
LOG="${RUN_DIR}/${OUTPUT_PREFIX}.log"

if [[ -e "$LOG" && ! -w "$LOG" ]]; then
    echo "Existing log file is not writable: $LOG" >&2
    exit 1
fi

touch "$LOG"

# -----------------------------
# Run
# -----------------------------
echo "Run directory: $RUN_DIR" > "$LOG"
echo "Output prefix: $OUTPUT_PREFIX" >> "$LOG"
echo "Python workers: $N_JOBS" >> "$LOG"

python3 manybody_energy_spectrum.py \
    --theta "$THETA" \
    --nup "$NUP" \
    --vd "$VD" \
    --l1 "$L1X" "$L1Y" \
    --l2 "$L2X" "$L2Y" \
    --nk "$NK" \
    --nev "$NEV" \
    --n-jobs "$N_JOBS" \
    --output-prefix "$OUTPUT_PREFIX" \
    >> "$LOG" 2>&1
