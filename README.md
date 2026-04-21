# Ploidy Classifier — Inference Pipeline

Predict **diploid vs. polyploid** from targeted sequence capture BAM files using trained CNN and ensemble models.

---

## Repository Structure

```
ploidy-classifier-release/
├── README.md                 ← you are here
├── environment.yml           ← conda environment definition
├── requirements.txt          ← pip dependencies (alternative to conda)
├── run_inference.sh          ← one-command batch runner
├── src/
│   ├── run_cnn_top10_inference.py      ← CNN-only inference (10 best probes)
│   └── run_ensemble_inference.py       ← ensemble pair inference (any probe pair)
├── models/
│   ├── README.md             ← model weight details
│   ├── cnn_weights/          ← 16 CNN checkpoints (.pt)
│   ├── reducer_weights/      ← 8 reducer MLP weights (.pt)
│   └── ensemble_models/      ← 5 ensemble classifiers (.joblib)
├── data/
│   ├── final_ranked_results.npz        ← probe ranking (determines top-10)
│   ├── probe_index_to_id_manifest.csv  ← model index ↔ probe ID mapping
│   └── example_truth.csv               ← example truth labels (Lygodium)
├── slurm/
│   └── run_inference.slurm   ← example SLURM job script
└── example_output/           ← example output will appear here
```

---

## Quick Start (step by step)

### Step 0: Clone this repo

```bash
# On HiPerGator
cd /blue/YOUR_GROUP/YOUR_USER/
git clone https://github.com/Fahimehgit/ploidy-classifier.git
cd ploidy-classifier
```

### Step 1: Create the conda environment

```bash
module load conda
conda env create -f environment.yml
conda activate ploidy-env
```

This installs Python 3.9, PyTorch, scikit-learn, MAFFT, samtools, and pysam.

### Step 2: Model weights (included)

All model weights are included in this repository (~164 MB total). No extra download needed. They live in `models/`:
- 16 CNN checkpoints in `models/cnn_weights/`
- 8 reducer MLP weights in `models/reducer_weights/`
- 5 ensemble classifier files in `models/ensemble_models/`

### Step 3: Prepare your data

You need:
1. **BAM files** — your targeted sequence capture alignments (one per sample)
2. **Reference FASTAs** — the L*.fasta probe reference sequences (already on HiPerGator at `/orange/juannanzhou/ploidy/query_fasta_files/`)

Optionally, a **truth file** (CSV) with columns `RapidID2` and `Ploidy.binary` for accuracy evaluation. See `data/example_truth.csv` for the format.

### Step 4: Run inference

**Option A: Run on all BAMs at once**

```bash
conda activate ploidy-env
bash run_inference.sh /path/to/your/bam_files /orange/juannanzhou/ploidy/query_fasta_files
```

Results go to `output/<sample_name>/`.

**Option B: Run on a single BAM**

```bash
# CNN-only (simpler, no reducer/ensemble weights needed)
python src/run_cnn_top10_inference.py \
  --bam /path/to/sample.bam \
  --work-dir output/sample_work \
  --reference-dir /orange/juannanzhou/ploidy/query_fasta_files

# Ensemble (higher accuracy, needs all model weights)
# Run for each probe pair (34_42, 42_209, 42_270, 55_254, 57_229):
python src/run_ensemble_inference.py \
  --bam /path/to/sample.bam \
  --work-dir output/sample_work \
  --reference-dir /orange/juannanzhou/ploidy/query_fasta_files \
  --probe1 34 --probe2 42 \
  --output-csv output/ensemble_34_42_predictions.csv
```

**Option C: Submit as a SLURM job**

Edit `slurm/run_inference.slurm` with your paths, then:

```bash
sbatch slurm/run_inference.slurm
```

---

## Understanding the Output

### CNN-only output (`cnn_top10_summary.csv`)

| Column | Meaning |
|--------|---------|
| `probe` | Probe ID used for this prediction |
| `prob` | Predicted probability of being polyploid (0–1) |
| `pred` | Binary prediction (0 = diploid, 1 = polyploid) at threshold 0.5 |
| `true_label` | Ground-truth label (if truth file provided) |

The CNN script runs the **top 10 best-ranked probes** independently on each sample. Each probe gives its own probability. To get a final answer, take the **mean probability** across probes and threshold at 0.5.

### Ensemble output (`ensemble_predictions.csv`)

| Column | Meaning |
|--------|---------|
| `species_id` | Sample identifier |
| `prob_polyploid` | Predicted probability of being polyploid (0–1) |
| `pred_label` | Binary prediction (0 = diploid, 1 = polyploid) |

The ensemble combines embeddings from two probes through a trained classifier (MLP or SVM) for more accurate predictions.

---

## Which Models to Run

The pipeline uses **two types of models** that complement each other:

| Model type | Script | What it does |
|------------|--------|-------------|
| **CNN top-10** | `run_cnn_top10_inference.py` | Runs the 10 best individual probe CNNs. Each probe gives its own probability. Average them for a final answer. |
| **Ensemble pairs** | `run_ensemble_inference.py` | Combines CNN embeddings from two probes via a trained classifier (MLP/SVM). More accurate but needs reducer + ensemble weights. |

### Ensemble probe pairs

The batch runner (`run_inference.sh`) runs these 5 ensemble pairs by default:

| Pair | Probe 1 | Probe 2 | Algorithm |
|------|---------|---------|-----------|
| 34_42 | 34 | 42 | MLP |
| 42_209 | 42 | 209 | SVM (RBF) |
| 42_270 | 42 | 270 | SVM (RBF) |
| 55_254 | 55 | 254 | SVM (RBF) |
| 57_229 | 57 | 229 | SVM (RBF) |

To get the best prediction, average the probabilities from all models (CNN + ensemble pairs) — this is the **hybrid** approach, which achieved the best overall accuracy (64.6%) on the Lygodium validation set.

---

## How It Works

```
BAM file
    │
    ▼
[BAM → FASTA] ──→ extract reads per probe
    │
    ▼
[MAFFT alignment] ──→ align reads against reference
    │
    ▼
[Tokenization] ──→ encode nucleotides as integers (A=1, C=2, G=3, T=4, gap=0)
    │
    ▼
[CNN model] ──→ predict polyploidy probability per probe
    │
    ├──→ CNN-only: average probabilities across top-10 probes
    │
    └──→ Ensemble: CNN features → MLP reducer → 500-d embedding
                                                    │
                                                    ▼
                                        [Classifier (MLP/SVM)] ──→ final prediction
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mafft not found` | Activate your conda env: `conda activate ploidy-env` |
| `No CNN found for probe X` | Make sure CNN weights are in `models/cnn_weights/` |
| `pysam` import error | Install it: `conda install -c bioconda pysam` |
| `CUDA out of memory` | Add `--device cpu` to use CPU instead |
| SLURM job fails | Check the `.err` file; often a path or memory issue |

---

## Reference

This pipeline accompanies the paper:

> **Machine learning prediction of plant polyploidy from target capture data**
> Fahimeh Rahimi, Jessie A. Pelosi, J. Gordon Burleigh, Juannan Zhou

For questions, contact Fahimeh Rahimi (fahimeh.rahimi@ufl.edu).
