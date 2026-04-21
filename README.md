# Ploidy Classifier

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.9-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Predict **diploid vs. polyploid** from targeted sequence capture BAM files using an ensemble of CNN and machine learning models.

> **Machine learning prediction of plant polyploidy from target capture data**
>
> Fahimeh Rahimi, Jessie A. Pelosi, J. Gordon Burleigh, Juannan Zhou

---

## Overview

The pipeline processes BAM files from targeted sequence capture experiments and classifies each sample as diploid or polyploid. It uses two complementary model types:

- **CNN models**: 10 independently trained convolutional neural networks, each operating on a single probe locus. Predictions are averaged across probes.
- **Ensemble models**: Pairs of CNN embeddings are combined via a trained classifier (MLP or SVM) for higher accuracy.

The best results are obtained by averaging predictions from both model types (the **hybrid** approach).

---

## Repository Structure

```
ploidy-classifier/
├── README.md
├── environment.yml              # Conda environment definition
├── requirements.txt             # Pip dependencies (alternative)
├── run_inference.sh             # Batch inference runner
├── src/
│   ├── run_cnn_top10_inference.py
│   └── run_ensemble_inference.py
├── models/
│   ├── README.md                # Model weight inventory
│   ├── cnn_weights/             # 16 CNN checkpoints (.pt)
│   ├── reducer_weights/         # 8 MLP reducer weights (.pt)
│   └── ensemble_models/         # 5 ensemble classifiers (.joblib)
├── data/
│   ├── final_ranked_results.npz
│   ├── probe_index_to_id_manifest.csv
│   └── example_truth.csv
└── slurm/
    └── run_inference.slurm      # Example SLURM job script
```

---

## Installation

### Option A: Conda (recommended)

```bash
git clone https://github.com/Fahimehgit/ploidy-classifier.git
cd ploidy-classifier
conda env create -f environment.yml
conda activate ploidy-env
```

### Option B: Pip

```bash
git clone https://github.com/Fahimehgit/ploidy-classifier.git
cd ploidy-classifier
pip install -r requirements.txt
```

Note: if using pip, you must install [MAFFT](https://mafft.cbrc.jp/alignment/software/) and [samtools](http://www.htslib.org/) separately.

### Dependencies

- Python 3.9+
- PyTorch >= 1.9
- scikit-learn >= 1.0
- pandas, numpy, joblib
- pysam
- MAFFT (multiple sequence alignment)
- samtools

All trained model weights are included in this repository (~164 MB).

---

## Input Requirements

1. **BAM files** — targeted sequence capture alignments (one per sample), sorted and indexed.
2. **Reference FASTA files** — probe reference sequences, one file per locus, named `L*.fasta`.
3. *(Optional)* **Truth labels** — a CSV with columns `RapidID2` and `Ploidy.binary` (0 = diploid, 1 = polyploid) for accuracy evaluation. See `data/example_truth.csv` for the format.

---

## Usage

### Batch inference (all BAMs in a directory)

```bash
conda activate ploidy-env
bash run_inference.sh /path/to/bam_files /path/to/reference_fastas
```

Output is written to `output/<sample_name>/`.

### Single-sample inference

```bash
# CNN-only inference
python src/run_cnn_top10_inference.py \
  --bam /path/to/sample.bam \
  --work-dir output/sample_work \
  --reference-dir /path/to/reference_fastas

# Ensemble inference (run for each probe pair)
python src/run_ensemble_inference.py \
  --bam /path/to/sample.bam \
  --work-dir output/sample_work \
  --reference-dir /path/to/reference_fastas \
  --probe1 34 --probe2 42 \
  --output-csv output/ensemble_34_42_predictions.csv
```

### SLURM cluster submission

Edit paths in `slurm/run_inference.slurm`, then:

```bash
sbatch slurm/run_inference.slurm
```

---

## Output Format

### CNN output (`cnn_top10_summary.csv`)

| Column | Description |
|--------|-------------|
| `species_id` | Sample identifier (derived from BAM filename) |
| `probe` | Model index of the probe used |
| `probe_id` | Probe locus ID (e.g., L238) |
| `prob_polyploid` | Predicted probability of polyploidy (0–1) |
| `pred_label` | Binary prediction at threshold 0.5 |

To obtain a final CNN prediction, average `prob_polyploid` across all probes for each sample.

### Ensemble output (`ensemble_<P1>_<P2>_predictions.csv`)

| Column | Description |
|--------|-------------|
| `species_id` | Sample identifier |
| `prob_polyploid` | Predicted probability of polyploidy (0–1) |
| `pred_label` | Binary prediction at threshold 0.5 |

### Hybrid prediction

For the most accurate classification, average the predicted probabilities from all CNN probes and all ensemble pairs, then threshold at 0.5.

---

## Models

### CNN top-10

The 10 highest-ranked individual probe CNNs (by test AUC on the original training/test split). Each CNN takes a MAFFT-aligned read matrix for a single probe locus as input and outputs a polyploidy probability.

| Rank | Probe Index | Test AUC |
|------|-------------|----------|
| 1 | 128 | 0.879 |
| 2 | 57 | 0.853 |
| 3 | 33 | 0.825 |
| 4 | 229 | 0.820 |
| 5 | 9 | 0.806 |
| 6 | 347 | 0.805 |
| 7 | 164 | 0.795 |
| 8 | 182 | 0.785 |
| 9 | 382 | 0.777 |
| 10 | 147 | 0.776 |

### Ensemble pairs

Each ensemble model takes 500-dimensional embeddings from two probe-specific CNN+reducer pipelines and classifies them jointly.

| Probe pair | Algorithm | Test AUC | Test accuracy |
|------------|-----------|----------|---------------|
| 34, 42 | MLP | 0.989 | 33/34 (97.1%) |
| 42, 270 | SVM (RBF) | 0.982 | 33/34 (97.1%) |
| 57, 229 | SVM (RBF) | 0.982 | 33/34 (97.1%) |
| 55, 254 | SVM (RBF) | 0.960 | 33/34 (97.1%) |
| 42, 209 | SVM (RBF) | 0.956 | 33/34 (97.1%) |

See `models/README.md` for a complete inventory of all weight files.

---

## Pipeline Architecture

```
BAM file
    │
    ▼
Extract reads per probe locus (samtools + pysam)
    │
    ▼
Multiple sequence alignment against reference (MAFFT)
    │
    ▼
Tokenize alignment (A=1, C=2, G=3, T=4, gap=0)
    │
    ▼
CNN model → polyploidy probability per probe
    │
    ├──→ CNN-only: average probabilities across top-10 probes
    │
    └──→ Ensemble: extract CNN features → MLP reducer → 500-d embedding
                                                            │
                                                            ▼
                                                Classifier (MLP/SVM) → final prediction
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `mafft: command not found` | Activate the conda environment or install MAFFT |
| `No CNN found for probe X` | Verify model weights are present in `models/cnn_weights/` |
| `pysam` import error | `conda install -c bioconda pysam` or `pip install pysam` |
| `CUDA out of memory` | Pass `--device cpu` to run on CPU |
| SLURM job fails | Check the `.err` log file for path or memory issues |

---

## Citation

If you use this software, please cite:

> Fahimeh Rahimi, Jessie A. Pelosi, J. Gordon Burleigh, Juannan Zhou. *Machine learning prediction of plant polyploidy from target capture data.* (2025)

---

## License

[MIT License](LICENSE)

---

## Contact

For questions or issues, please open a [GitHub issue](https://github.com/Fahimehgit/ploidy-classifier/issues) or contact Fahimeh Rahimi (fahimeh.rahimi@ufl.edu).
