#!/bin/bash
# ============================================================
# run_inference.sh — Run ploidy inference on all BAM files in a directory
#
# Usage:
#   bash run_inference.sh /path/to/bam_directory /path/to/reference_fastas
#
# Or for a single BAM:
#   BAM_SINGLE=/path/to/sample.bam bash run_inference.sh /ignored /path/to/reference_fastas
# ============================================================
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <BAM_DIR> <REFERENCE_DIR>"
  echo ""
  echo "  BAM_DIR       Directory containing .bam files"
  echo "  REFERENCE_DIR Directory containing L*.fasta reference sequences"
  echo ""
  echo "Optional environment variables:"
  echo "  BAM_SINGLE    Path to a single BAM (overrides BAM_DIR)"
  echo "  MAFFT_MODE    accurate|auto|fast (default: accurate)"
  echo "  MAFFT_THREADS Number of MAFFT threads (default: 8)"
  exit 1
fi

BAM_DIR="$1"
REFERENCE_DIR="$2"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_ROOT="${REPO_DIR}/output"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAFFT_MODE="${MAFFT_MODE:-accurate}"
MAFFT_THREADS="${MAFFT_THREADS:-8}"

if ! command -v mafft &>/dev/null; then
  echo "ERROR: mafft not found. Make sure your conda environment is activated."
  exit 1
fi

mkdir -p "${OUT_ROOT}"

if [ -n "${BAM_SINGLE:-}" ]; then
  BAM_LIST=("${BAM_SINGLE}")
else
  BAM_LIST=("${BAM_DIR}"/*.bam)
fi

echo "============================================"
echo "Ploidy Classifier — Batch Inference"
echo "============================================"
echo "BAMs:        ${#BAM_LIST[@]} file(s)"
echo "Reference:   ${REFERENCE_DIR}"
echo "Output:      ${OUT_ROOT}"
echo "MAFFT mode:  ${MAFFT_MODE}"
echo "============================================"
echo ""

for bam in "${BAM_LIST[@]}"; do
  if [ ! -f "${bam}" ]; then
    continue
  fi

  base="$(basename "${bam}" .bam)"
  work_dir="${OUT_ROOT}/${base}"

  echo "--------------------------------------------"
  echo "Processing: ${base}"
  echo "--------------------------------------------"

  # --- Step 1: CNN top-10 inference ---
  "${PYTHON_BIN}" "${REPO_DIR}/src/run_cnn_top10_inference.py" \
    --bam "${bam}" \
    --work-dir "${work_dir}/cnn" \
    --cnn-dir "${REPO_DIR}/models/cnn_weights" \
    --reference-dir "${REFERENCE_DIR}" \
    --mafft-mode "${MAFFT_MODE}" \
    --mafft-threads "${MAFFT_THREADS}" \
    --output-dir "${work_dir}" \
    --skip-if-exists

  # --- Step 2: Ensemble pair inference (run each pair) ---
  for PAIR in "34 42" "42 209" "42 270" "55 254" "57 229"; do
    P1=$(echo $PAIR | cut -d' ' -f1)
    P2=$(echo $PAIR | cut -d' ' -f2)
    "${PYTHON_BIN}" "${REPO_DIR}/src/run_ensemble_inference.py" \
      --bam "${bam}" \
      --work-dir "${work_dir}/ensemble_${P1}_${P2}" \
      --cnn-dir "${REPO_DIR}/models/cnn_weights" \
      --reference-dir "${REFERENCE_DIR}" \
      --mafft-mode "${MAFFT_MODE}" \
      --mafft-threads "${MAFFT_THREADS}" \
      --probe1 "${P1}" \
      --probe2 "${P2}" \
      --output-csv "${work_dir}/ensemble_${P1}_${P2}_predictions.csv" \
      --skip-if-exists
  done

  echo "  -> Results in ${work_dir}/"
  echo ""
done

echo "============================================"
echo "All BAMs processed. Results in: ${OUT_ROOT}"
echo "============================================"
