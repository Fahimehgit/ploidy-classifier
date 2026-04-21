# Model Weights

All model weights are included in this repository (~164 MB total).

## Contents

### 1. CNN weights (`cnn_weights/`)

16 trained CNN checkpoints: the top-10 ranked probes plus additional probes needed by the ensemble pairs.

| File | Probe Index | Used by |
|------|-------------|---------|
| `best_model_probe_3_9.pt` | 9 | CNN top-10 |
| `best_model_probe_3_33.pt` | 33 | CNN top-10 |
| `best_model_probe_3_34.pt` | 34 | Ensemble (34,42) |
| `best_model_probe_3_42.pt` | 42 | Ensemble (34,42), (42,209), (42,270) |
| `best_model_probe_3_55.pt` | 55 | Ensemble (55,254) |
| `best_model_probe_3_57.pt` | 57 | CNN top-10, Ensemble (57,229) |
| `best_model_probe_3_128.pt` | 128 | CNN top-10 |
| `best_model_probe_3_147.pt` | 147 | CNN top-10 |
| `best_model_probe_3_164.pt` | 164 | CNN top-10 |
| `best_model_probe_3_182.pt` | 182 | CNN top-10 |
| `best_model_probe_3_209.pt` | 209 | Ensemble (42,209) |
| `best_model_probe_3_229.pt` | 229 | CNN top-10, Ensemble (57,229) |
| `best_model_probe_3_254.pt` | 254 | Ensemble (55,254) |
| `best_model_probe_3_270.pt` | 270 | Ensemble (42,270) |
| `best_model_probe_3_347.pt` | 347 | CNN top-10 |
| `best_model_probe_3_382.pt` | 382 | CNN top-10 |

### 2. Reducer weights (`reducer_weights/`)

8 MLP reducer checkpoints that map CNN features to 500-dimensional embeddings.
Only needed for **ensemble** inference.

| File | Probe Index |
|------|-------------|
| `reducer_probe_34.pt` | 34 |
| `reducer_probe_42.pt` | 42 |
| `reducer_probe_55.pt` | 55 |
| `reducer_probe_57.pt` | 57 |
| `reducer_probe_209.pt` | 209 |
| `reducer_probe_229.pt` | 229 |
| `reducer_probe_254.pt` | 254 |
| `reducer_probe_270.pt` | 270 |

### 3. Ensemble models (`ensemble_models/`)

5 trained classifiers (`.joblib`) for the best ensemble probe pairs.

| File | Pair | Algorithm |
|------|------|-----------|
| `ensemble_34_42_mlp.joblib` | 34 + 42 | MLP |
| `ensemble_42_209_svm_rbf.joblib` | 42 + 209 | SVM (RBF) |
| `ensemble_42_270_svm_rbf.joblib` | 42 + 270 | SVM (RBF) |
| `ensemble_55_254_svm_rbf.joblib` | 55 + 254 | SVM (RBF) |
| `ensemble_57_229_svm_rbf.joblib` | 57 + 229 | SVM (RBF) |

