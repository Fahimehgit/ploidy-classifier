#!/bin/bash
#SBATCH --job-name=ploidy_inf
#SBATCH --output=ploidy_inf_%j.out
#SBATCH --error=ploidy_inf_%j.err
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# ============================================================
# SLURM job for ploidy inference
#
# Before submitting, edit the paths below:
#   1. REPO_DIR  — where you cloned ploidy-classifier-release
#   2. BAM_DIR   — directory with your .bam files
#   3. REF_DIR   — directory with L*.fasta reference sequences
#   4. CONDA_ENV — name of your conda environment
# ============================================================

REPO_DIR="/blue/YOUR_GROUP/YOUR_USER/ploidy-classifier-release"
BAM_DIR="/path/to/your/bam_files"
REF_DIR="/orange/juannanzhou/ploidy/query_fasta_files"
CONDA_ENV="ploidy-env"

# --- Setup ---
set -euo pipefail

module load conda 2>/dev/null || true
conda activate "${CONDA_ENV}"

export MAFFT_MODE="accurate"
export MAFFT_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_DIR}"
bash run_inference.sh "${BAM_DIR}" "${REF_DIR}"
