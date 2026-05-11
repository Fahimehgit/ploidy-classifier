# Model Weights

All model weights are included in this repository (~164 MB total).

## Contents

### 1. CNN weights (`cnn_weights/`)

16 trained CNN checkpoints: the top-10 ranked probes plus additional probes needed by the ensemble pairs.

| File | Probe | Used by |
|------|-------|---------|
| `cnn_L11.pt` | L11 | CNN top-10 |
| `cnn_L138.pt` | L138 | CNN top-10 |
| `cnn_L139.pt` | L139 | Ensemble (L139+L146) |
| `cnn_L146.pt` | L146 | Ensemble (L139+L146), (L146+L320), (L146+L389) |
| `cnn_L158.pt` | L158 | Ensemble (L158+L373) |
| `cnn_L16.pt` | L16 | CNN top-10, Ensemble (L16+L342) |
| `cnn_L238.pt` | L238 | CNN top-10 |
| `cnn_L257.pt` | L257 | CNN top-10 |
| `cnn_L278.pt` | L278 | CNN top-10 |
| `cnn_L295.pt` | L295 | CNN top-10 |
| `cnn_L320.pt` | L320 | Ensemble (L146+L320) |
| `cnn_L342.pt` | L342 | CNN top-10, Ensemble (L16+L342) |
| `cnn_L373.pt` | L373 | Ensemble (L158+L373) |
| `cnn_L389.pt` | L389 | Ensemble (L146+L389) |
| `cnn_L63.pt` | L63 | CNN top-10 |
| `cnn_L98.pt` | L98 | CNN top-10 |

### 2. Reducer weights (`reducer_weights/`)

8 MLP reducer checkpoints that map CNN features to 500-dimensional embeddings.
Only needed for **ensemble** inference.

| File | Probe |
|------|-------|
| `reducer_L139.pt` | L139 |
| `reducer_L146.pt` | L146 |
| `reducer_L158.pt` | L158 |
| `reducer_L16.pt` | L16 |
| `reducer_L320.pt` | L320 |
| `reducer_L342.pt` | L342 |
| `reducer_L373.pt` | L373 |
| `reducer_L389.pt` | L389 |

### 3. Ensemble models (`ensemble_models/`)

5 trained classifiers (`.joblib`) for the best ensemble probe pairs.

| File | Pair | Algorithm |
|------|------|-----------|
| `ensemble_L139_L146_mlp.joblib` | L139 + L146 | MLP |
| `ensemble_L146_L320_svm_rbf.joblib` | L146 + L320 | SVM (RBF) |
| `ensemble_L146_L389_svm_rbf.joblib` | L146 + L389 | SVM (RBF) |
| `ensemble_L158_L373_svm_rbf.joblib` | L158 + L373 | SVM (RBF) |
| `ensemble_L16_L342_svm_rbf.joblib` | L16 + L342 | SVM (RBF) |
