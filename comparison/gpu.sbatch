#!/bin/bash
#SBATCH --partition=zjhu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4090:1
#SBATCH --time=2-00:00:00

set -euo pipefail
workdir=$1
python=$2
shift 2
cd "$workdir"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
exec "$python" -B -u "$@"
