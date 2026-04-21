#!/bin/bash
#SBATCH --job-name=ploidy_inf
#SBATCH --output=ploidy_inf_%j.out
#SBATCH --error=ploidy_inf_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# ============================================================
# Example SLURM job script for ploidy inference.
#
# Before submitting, edit the three paths below:
#   REPO_DIR  — path to your cloned ploidy-classifier repository
#   BAM_DIR   — directory containing your .bam files
#   REF_DIR   — directory containing L*.fasta probe references
# ============================================================

REPO_DIR="/path/to/ploidy-classifier"
BAM_DIR="/path/to/bam_files"
REF_DIR="/path/to/reference_fastas"
CONDA_ENV="ploidy-env"

set -euo pipefail

module load conda 2>/dev/null || true
conda activate "${CONDA_ENV}"

export MAFFT_MODE="accurate"
export MAFFT_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_DIR}"
bash run_inference.sh "${BAM_DIR}" "${REF_DIR}"
