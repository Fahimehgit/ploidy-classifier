#!/usr/bin/env python3
"""
Ensemble inference pipeline for any probe pair.

Steps:
1) (Optional) BAM -> FASTA conversion
2) (Optional) Gap removal (internal + external) on aligned FASTA
3) (Optional) Tokenize to Probe15 JSON format
4) Load CNNs + reducers for the specified probe pair
5) Generate embeddings and run ensemble model inference

Use --probe1 and --probe2 to specify which probe pair to run.
Default is probe pair [42, 198], but any trained pair can be used.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# -------------------------
# Model definitions
# -------------------------

class DynamicLeNet(nn.Module):
    def __init__(
        self,
        input_channels,
        input_size,
        num_layers,
        num_filters,
        kernel_sizes,
        dropout_rate,
        activation_fn,
        pool_type,
        fc_layer_sizes,
        num_classes=1,
    ):
        super().__init__()
        layers = []
        in_channels = input_channels
        current_size = input_size

        for i in range(num_layers):
            out_channels = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=kernel_sizes[i] // 2,
                )
            )
            if activation_fn == "relu":
                layers.append(nn.ReLU())
            elif activation_fn == "leaky_relu":
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ELU())
            layers.append(nn.MaxPool2d(2) if pool_type == "max" else nn.AvgPool2d(2))
            current_size = (current_size - kernel_sizes[i] + 2 * (kernel_sizes[i] // 2)) // 2
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((current_size, current_size))
        self.dropout = nn.Dropout(dropout_rate)

        fc_layers = []
        in_features = out_channels * current_size * current_size
        for size in fc_layer_sizes:
            fc_layers.append(nn.Linear(in_features, size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            in_features = size
        fc_layers.append(nn.Linear(in_features, num_classes))

        self.fc_layers = nn.Sequential(*fc_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return self.sigmoid(x)


class ReducerMLP(nn.Module):
    """
    Maps CNN feature vector -> 500-d embedding.
    """

    def __init__(self, in_dim: int, emb_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        z = self.embed(x)
        logit = self.cls(z).squeeze(-1)
        return z, logit


def cnn_features(cnn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = cnn.conv_layers(x)
    x = cnn.adaptive_pool(x)
    x = x.view(x.size(0), -1)
    layers = list(cnn.fc_layers.children())
    if len(layers) >= 1:
        x = nn.Sequential(*layers[:-1])(x)
    return x


def resolve_cnn_path(cnn_dir: str, probe_idx: int) -> str:
    c1 = os.path.join(cnn_dir, f"best_model_probe_3_{probe_idx}.pt")
    c2 = os.path.join(cnn_dir, f"best_model_probe_{probe_idx}.pt")
    if os.path.exists(c1):
        return c1
    if os.path.exists(c2):
        return c2
    raise FileNotFoundError(f"No CNN found for probe {probe_idx} (checked {c1} and {c2})")


# -------------------------
# Data loading
# -------------------------

def load_probe_json(json_path: Path, sort_species_keys: bool) -> Tuple[List[np.ndarray], List[str]]:
    with json_path.open("r") as f:
        data = json.load(f)
    keys = list(data.keys())
    if sort_species_keys:
        keys = sorted(keys)

    seqs = []
    ids = []
    for sp in keys:
        info = data[sp]
        ref = np.array(info["reference_tokens"])
        aux = np.array(info["auxiliary_tokens"])
        full = np.vstack([ref.reshape(1, -1), aux]).reshape(1, 351, 154)
        seqs.append(full)
        ids.append(sp)
    return seqs, ids


def resolve_json_for_probe(json_dir: Path, probe_idx: int, sort_json_files: bool) -> Path:
    candidates = [
        json_dir / f"probe_{probe_idx}.fasta.json",
        json_dir / f"probe_{probe_idx}.json",
        json_dir / f"L{probe_idx}.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    files = sorted(files) if sort_json_files else files
    if probe_idx < len(files):
        return json_dir / files[probe_idx]
    raise FileNotFoundError(f"Could not resolve JSON for probe {probe_idx} in {json_dir}")


def load_probe_id_mapping(json_reference_dir: Path) -> Dict[int, int]:
    """
    Backward-compatible fallback for model index -> probe ID using lexicographic
    ordering of JSON filenames (historical training convention).
    """
    files = sorted([f for f in os.listdir(json_reference_dir) if f.endswith(".json")])
    mapping: Dict[int, int] = {}
    for idx, name in enumerate(files):
        stem = Path(name).stem
        if stem.startswith("L"):
            try:
                mapping[idx] = int(stem.replace("L", ""))
            except ValueError:
                continue
    return mapping


def load_probe_id_mapping_from_csv(probe_id_map_csv: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    with probe_id_map_csv.open("r", newline="") as fh:
        rdr = csv.DictReader(fh)
        cols = set(rdr.fieldnames or [])
        if not {"model_index", "probe_id"}.issubset(cols):
            raise ValueError(
                f"Invalid probe-id mapping CSV {probe_id_map_csv}: required columns model_index, probe_id"
            )
        for row in rdr:
            try:
                model_idx = int(str(row.get("model_index", "")).strip())
                probe_id = int(str(row.get("probe_id", "")).strip())
            except Exception:
                continue
            mapping[model_idx] = probe_id
    if not mapping:
        raise ValueError(f"Probe-id mapping CSV {probe_id_map_csv} contained no valid rows.")
    return mapping


class InferenceDataset(Dataset):
    def __init__(self, seqs: List[np.ndarray], ids: List[str]):
        self.seqs = seqs
        self.ids = ids

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.tensor(self.seqs[i], dtype=torch.float32), self.ids[i]


def embed_probe(
    json_dir: Path,
    json_probe_id: int,
    model_idx: int,
    cnn_dir: str,
    reducer_dir: str,
    sort_json_files: bool,
    sort_species_keys: bool,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    json_path = resolve_json_for_probe(json_dir, json_probe_id, sort_json_files)

    seqs, ids = load_probe_json(json_path, sort_species_keys)
    ds = InferenceDataset(seqs, ids)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    cnn_path = resolve_cnn_path(cnn_dir, model_idx)
    cnn = torch.load(cnn_path, map_location=device, weights_only=False).to(device).eval()

    reducer_path = os.path.join(reducer_dir, f"reducer_probe_{model_idx}.pt")
    reducer_ckpt = torch.load(reducer_path, map_location="cpu")
    reducer = ReducerMLP(
        in_dim=int(reducer_ckpt["in_dim"]),
        emb_dim=int(reducer_ckpt["emb_dim"]),
        hidden=int(reducer_ckpt["hidden"]),
        dropout=float(reducer_ckpt["dropout"]),
    ).to(device)
    reducer.load_state_dict(reducer_ckpt["state_dict"])
    reducer.eval()

    all_embs = []
    all_ids = []
    with torch.no_grad():
        for x, bid in dl:
            x = x.to(device)
            feat = cnn_features(cnn, x)
            z, _ = reducer(feat)
            all_embs.append(z.cpu().numpy())
            all_ids.extend(list(bid))

    embs = np.vstack(all_embs).astype(np.float32)
    ids_arr = np.array(all_ids, dtype=object)
    return embs, ids_arr


def align_by_ids(emb1: np.ndarray, ids1: np.ndarray, emb2: np.ndarray, ids2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m2 = {str(i): idx for idx, i in enumerate(ids2)}
    keep = []
    for idx1, i in enumerate(ids1):
        j = m2.get(str(i), None)
        if j is not None:
            keep.append((idx1, j, str(i)))
    if not keep:
        raise ValueError("No overlapping ids between probes for inference.")
    idx1 = np.array([a for a, _, _ in keep], dtype=int)
    idx2 = np.array([b for _, b, _ in keep], dtype=int)
    ids = np.array([c for _, _, c in keep], dtype=object)
    X = np.concatenate([emb1[idx1], emb2[idx2]], axis=1)
    return X, ids


# -------------------------
# Data prep helpers
# -------------------------

def run_cmd(cmd: str, desc: str) -> None:
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed with exit code {result.returncode}")


def load_embedding_split(embeddings_dir: Path, probe_idx: int, split: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    pdir = embeddings_dir / f"probe_{probe_idx}"
    candidates = [
        pdir / f"{split}.npz",
        pdir / f"reduced_embeddings_{split}.npz",
    ]
    npz_path = None
    for c in candidates:
        if c.exists():
            npz_path = c
            break
    if npz_path is None:
        raise FileNotFoundError(f"Missing embeddings file for probe {probe_idx}, split={split}")
    d = np.load(npz_path, allow_pickle=True)
    if "X" in d.files:
        X = d["X"]
        y = d["y"]
        ids = d["ids"] if "ids" in d.files else None
    else:
        X = d["embeddings"]
        y = d["labels"]
        ids = d["ids"] if "ids" in d.files else None
    return X.astype(np.float32, copy=False), y.astype(int, copy=False), ids


def align_train_by_ids(X1, y1, ids1, X2, y2, ids2) -> Tuple[np.ndarray, np.ndarray]:
    if ids1 is None or ids2 is None:
        if len(y1) != len(y2):
            raise ValueError("Row mismatch and missing ids for training alignment.")
        return np.concatenate([X1, X2], axis=1), y1
    m2 = {str(i): idx for idx, i in enumerate(ids2)}
    keep = []
    for idx1, i in enumerate(ids1):
        j = m2.get(str(i), None)
        if j is not None:
            keep.append((idx1, j))
    if not keep:
        raise ValueError("No overlapping ids between probes for training.")
    idx1 = np.array([a for a, _ in keep], dtype=int)
    idx2 = np.array([b for _, b in keep], dtype=int)
    X = np.concatenate([X1[idx1], X2[idx2]], axis=1)
    y = y1[idx1]
    if not np.array_equal(y, y2[idx2]):
        raise ValueError("Label mismatch after id alignment in training.")
    return X, y


def bam_to_fasta(bam_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{bam_path.stem}.txt"
    # Keep the same logic as complete_pipeline.py to avoid behavioral differences
    try:
        import pysam  # type: ignore
    except Exception as e:
        if shutil.which("samtools") is None:
            raise RuntimeError(
                "pysam is required for BAM -> FASTA conversion. Activate the conda env with pysam installed."
            ) from e
        # Fallback to samtools if pysam fails (e.g., libcrypto mismatch)
        with out_file.open("w") as output_file:
            proc = subprocess.Popen(
                ["samtools", "view", "-h", str(bam_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if not line or line.startswith("@"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 10:
                    continue
                read_name = parts[0]
                reference_name = parts[2]
                sequence = parts[9]
                if reference_name == "*" or not sequence:
                    continue
                fasta_id = ">" + reference_name + "-" + read_name
                output_file.write(fasta_id + "\n" + sequence + "\n")
            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"samtools view failed: {err}")
        return

    truncated_error: Optional[Exception] = None
    with open(out_file, "w") as output_file, pysam.AlignmentFile(str(bam_path), "rb") as bam_file:
        while True:
            try:
                read = next(bam_file)
            except StopIteration:
                break
            except OSError as e:
                truncated_error = e
                break
            read_name = read.query_name
            reference_id = read.reference_id
            if reference_id is None or reference_id < 0:
                continue
            reference_name = bam_file.get_reference_name(reference_id)
            if reference_name is None:
                continue
            sequence = read.query_sequence
            if sequence:
                fasta_id = ">" + reference_name + "-" + read_name
                output_file.write(fasta_id + "\n" + sequence + "\n")
    if truncated_error is not None:
        print(f"WARN: truncated BAM tail detected; using partial reads from {bam_path}: {truncated_error}")


def parse_fasta_by_probe(fasta_path: Path, probe_prefix: str, probes: List[int]) -> Dict[int, List[Tuple[str, str]]]:
    keep = {int(p) for p in probes}
    out: Dict[int, List[Tuple[str, str]]] = {int(p): [] for p in keep}
    header = None
    seq_lines: List[str] = []
    with fasta_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    ref = header[1:].split("-")[0]
                    if ref.startswith(probe_prefix):
                        try:
                            idx = int(ref.replace(probe_prefix, ""))
                        except ValueError:
                            idx = None
                        if idx in keep:
                            out[idx].append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            ref = header[1:].split("-")[0]
            if ref.startswith(probe_prefix):
                try:
                    idx = int(ref.replace(probe_prefix, ""))
                except ValueError:
                    idx = None
                if idx in keep:
                    out[idx].append((header, "".join(seq_lines)))
    return out


def parse_fasta_records(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = None
    seq_lines: List[str] = []
    with path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, "".join(seq_lines)))
    return records


def dna_tokenizer(seq: str, max_length: int = 154) -> List[int]:
    token_mapping = {"A": 1, "C": 2, "G": 3, "T": 4, "-": 0}
    tokens = [token_mapping.get(nuc.upper(), 0) for nuc in seq]
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    elif len(tokens) < max_length:
        tokens += [0] * (max_length - len(tokens))
    return tokens


def select_aux_by_gap(aux_seqs: List[str], max_auxiliary: int) -> List[str]:
    if len(aux_seqs) <= max_auxiliary:
        return aux_seqs
    gap_counts = [seq.count("-") for seq in aux_seqs]
    idxs = np.argsort(gap_counts)[:max_auxiliary]
    return [aux_seqs[i] for i in idxs]


def find_fasta_for_probe(input_dir: Path, probe_prefix: str, probe_idx: int) -> Path:
    candidates = [
        f"{probe_prefix}{probe_idx}_aligned.fasta",
        f"{probe_prefix}{probe_idx}.fasta",
    ]
    for name in candidates:
        p = input_dir / name
        if p.exists():
            return p
    # fallback: search recursively
    matches = list(input_dir.rglob(f"{probe_prefix}{probe_idx}*.fasta"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No FASTA found for probe {probe_idx} in {input_dir}")


def write_probe_json(
    fasta_path: Path,
    output_dir: Path,
    probe_idx: int,
    species_id: str,
    max_length: int = 154,
    max_auxiliary: int = 350,
) -> Path:
    records = parse_fasta_records(fasta_path)
    if not records:
        raise RuntimeError(f"No sequences in {fasta_path}")
    ref_seq = records[0][1]
    aux_seqs = [s for _, s in records[1:]]
    aux_seqs = select_aux_by_gap(aux_seqs, max_auxiliary)

    data = {
        species_id: {
            "label": 0,
            "reference_tokens": dna_tokenizer(ref_seq, max_length),
            "auxiliary_tokens": [dna_tokenizer(s, max_length) for s in aux_seqs],
        }
    }
    aux_tokens = data[species_id]["auxiliary_tokens"]
    if len(aux_tokens) < max_auxiliary:
        pad = [0] * max_length
        aux_tokens.extend([pad] * (max_auxiliary - len(aux_tokens)))
    else:
        aux_tokens[:] = aux_tokens[:max_auxiliary]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"L{probe_idx}.json"
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
    return out_path


def write_probe_fastas(
    reads_by_probe: Dict[int, List[Tuple[str, str]]],
    out_dir: Path,
    probe_prefix: str,
    max_reads_per_probe: Optional[int] = None,
    sampling_seed: int = 13,
) -> List[Path]:
    def maybe_cap_records(records: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if not max_reads_per_probe or max_reads_per_probe <= 0 or len(records) <= max_reads_per_probe:
            return records
        rng = random.Random(sampling_seed)
        idxs = sorted(rng.sample(range(len(records)), max_reads_per_probe))
        return [records[i] for i in idxs]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for probe_idx, records in reads_by_probe.items():
        if not records:
            continue
        records = maybe_cap_records(records)
        out_path = out_dir / f"{probe_prefix}{probe_idx}.fasta"
        with out_path.open("w") as f:
            for h, s in records:
                f.write(f"{h}\n{s}\n")
        out_paths.append(out_path)
    return out_paths


def resolve_mafft_bin() -> str:
    env_bin = os.environ.get("MAFFT_BIN")
    if env_bin:
        return env_bin
    mafft_bin = shutil.which("mafft")
    if mafft_bin:
        return mafft_bin
    raise FileNotFoundError("mafft not found in PATH. Install mafft or set MAFFT_BIN.")


def run_mafft_align(
    reference_dir: Path,
    probe_fasta: Path,
    out_dir: Path,
    probe_prefix: str,
    mafft_mode: str = "accurate",
    mafft_threads: int = 1,
) -> Path:
    ref_name = probe_fasta.stem
    ref_path = reference_dir / f"{ref_name}.fasta"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference fasta not found: {ref_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / f"{ref_name}_combined.fasta"
    aligned = out_dir / f"{ref_name}_aligned.fasta"
    with combined.open("w") as f:
        f.write(ref_path.read_text())
        f.write(probe_fasta.read_text())
    log_path = out_dir / f"{ref_name}_mafft.log"
    mafft_bin = resolve_mafft_bin()
    if mafft_mode == "accurate":
        mafft_args = [
            "--maxiterate",
            "1000",
            "--localpair",
            "--adjustdirection",
            "--op",
            "2.0",
            "--ep",
            "0.2",
            "--leavegappyregion",
        ]
    elif mafft_mode == "auto":
        mafft_args = ["--auto", "--adjustdirection", "--leavegappyregion"]
    elif mafft_mode == "fast":
        mafft_args = [
            "--retree",
            "1",
            "--maxiterate",
            "0",
            "--6merpair",
            "--adjustdirection",
            "--leavegappyregion",
        ]
    else:
        raise ValueError(f"Unsupported --mafft-mode: {mafft_mode}")
    with aligned.open("w") as out_f, log_path.open("w") as err_f:
        result = subprocess.run(
            [
                mafft_bin,
                "--thread",
                str(max(1, int(mafft_threads))),
                *mafft_args,
                str(combined),
            ],
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"MAFFT alignment for {ref_name} failed (see {log_path})")
    return aligned


def write_unaligned_with_reference(reference_dir: Path, probe_fasta: Path, out_dir: Path) -> Path:
    ref_name = probe_fasta.stem
    ref_path = reference_dir / f"{ref_name}.fasta"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference fasta not found: {ref_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ref_name}_aligned.fasta"
    with out_path.open("w") as f:
        # Keep output shape compatible with downstream code: reference first, then reads.
        f.write(ref_path.read_text())
        f.write(probe_fasta.read_text())
    return out_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Run ensemble model inference (probe pair) on a BAM file to predict ploidy."
    )
    ap.add_argument("--bam", required=True, help="Input BAM file")
    ap.add_argument("--work-dir", required=True, help="Working directory for intermediate and output files")
    ap.add_argument("--aligned-dir", default=None, help="Aligned FASTA dir (skip alignment if provided)")
    ap.add_argument(
        "--skip-mafft",
        action="store_true",
        help="Skip MAFFT and use reference+reads concatenation (faster, may reduce accuracy).",
    )
    ap.add_argument("--reference-dir", required=True, help="Directory with L*.fasta reference files")
    ap.add_argument("--probe-prefix", default="L", help="Probe prefix in BAM reference names (default: L)")
    ap.add_argument("--num-cores", type=int, default=8, help="Cores for gap removal/tokenization")
    ap.add_argument("--mafft-threads", type=int, default=8, help="Threads to pass to MAFFT")
    ap.add_argument(
        "--mafft-mode",
        choices=["accurate", "auto", "fast"],
        default="accurate",
        help="MAFFT strategy: accurate=slowest/best, auto=balanced, fast=fastest.",
    )
    ap.add_argument(
        "--max-reads-per-probe",
        type=int,
        default=0,
        help="If >0, randomly cap reads per probe before MAFFT to speed up alignment.",
    )
    ap.add_argument("--sampling-seed", type=int, default=13, help="Random seed for read capping.")
    ap.add_argument("--skip-data-prep", action="store_true", help="Skip BAM->JSON, use --json-dir instead")
    ap.add_argument("--json-dir", default=None, help="JSON dir if skipping data prep")
    ap.add_argument("--cnn-dir", default=str(repo_root / "models" / "cnn_weights"),
                     help="Directory with best_model_probe_3_*.pt CNN checkpoints")
    ap.add_argument("--reducers-dir", default=str(repo_root / "models" / "reducer_weights"),
                     help="Directory with reducer_probe_*.pt reducer checkpoints")
    ap.add_argument(
        "--ensemble-model",
        default=None,
        help="Path to ensemble .joblib file. If not provided, auto-detects from --probe1/--probe2 in models/ensemble_models/.",
    )
    ap.add_argument(
        "--embeddings-dir",
        default=str(repo_root / "models" / "embeddings_500_ids"),
        help="Embeddings dir for fallback retrain (only needed if ensemble-model is missing)",
    )
    ap.add_argument("--probe1", type=int, default=42, help="Model index for probe1")
    ap.add_argument("--probe2", type=int, default=198, help="Model index for probe2")
    ap.add_argument(
        "--json-reference-dir",
        default=None,
        help="Reference JSON dir (for legacy probe-id mapping fallback).",
    )
    ap.add_argument(
        "--probe-id-map-csv",
        default=str(repo_root / "data" / "probe_index_to_id_manifest.csv"),
        help="Mapping CSV with columns model_index,probe_id.",
    )
    ap.add_argument(
        "--sort-json-files",
        action="store_true",
        help="Deprecated (JSON sorting is now default).",
    )
    ap.add_argument(
        "--no-sort-json-files",
        action="store_true",
        help="Disable JSON filename sorting (not recommended).",
    )
    ap.add_argument("--sort-species-keys", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-csv", default="ensemble_42_198_predictions.csv")
    ap.add_argument("--skip-if-exists", action="store_true", help="Skip if output CSV already exists")
    args = ap.parse_args()

    if args.ensemble_model is None:
        ens_dir = repo_root / "models" / "ensemble_models"
        p1, p2 = min(args.probe1, args.probe2), max(args.probe1, args.probe2)
        candidates = list(ens_dir.glob(f"ensemble_{p1}_{p2}_*.joblib")) + \
                     list(ens_dir.glob(f"ensemble_{args.probe1}_{args.probe2}_*.joblib"))
        if candidates:
            args.ensemble_model = str(candidates[0])
            print(f"Auto-detected ensemble model: {args.ensemble_model}")
        else:
            args.ensemble_model = str(ens_dir / f"ensemble_{args.probe1}_{args.probe2}_gradient_boosting.joblib")

    probe_map_csv = Path(args.probe_id_map_csv) if args.probe_id_map_csv else None
    if probe_map_csv is not None and probe_map_csv.exists():
        probe_id_map = load_probe_id_mapping_from_csv(probe_map_csv)
    elif args.json_reference_dir is not None:
        probe_id_map = load_probe_id_mapping(Path(args.json_reference_dir))
    else:
        raise ValueError(
            "No probe-id mapping available. Provide --probe-id-map-csv or --json-reference-dir."
        )
    probe1_id = probe_id_map.get(args.probe1)
    probe2_id = probe_id_map.get(args.probe2)
    if probe1_id is None or probe2_id is None:
        raise ValueError(
            f"Could not map model indices to probe IDs. "
            f"probe1={args.probe1}->{probe1_id}, probe2={args.probe2}->{probe2_id}. "
            f"Check --json-reference-dir."
        )

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_data_prep:
        if not args.json_dir:
            raise ValueError("--json-dir is required when --skip-data-prep is set.")
        json_dir = Path(args.json_dir)
    else:
        bam_path = Path(args.bam)
        fasta_dir = work_dir / "fasta"
        processed_dir = work_dir / "processed"
        json_dir = work_dir / "json"

        out_csv = Path(args.output_csv)
        if args.skip_if_exists and out_csv.exists() and out_csv.stat().st_size > 0:
            print(f"Output already exists, skipping: {out_csv}")
            return

        try:
            bam_to_fasta(bam_path, fasta_dir)
        except RuntimeError as e:
            status_path = work_dir / "status.json"
            status = {
                "status": "bam_read_error",
                "error": str(e),
                "bam": str(bam_path),
            }
            status_path.write_text(json.dumps(status, indent=2))
            out_csv = Path(args.output_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out_csv.write_text("species_id,prob_polyploid,pred_label\n")
            print(f"{e}. Wrote {status_path} and empty CSV.")
            return

        # Build aligned FASTA files if not provided
        if args.aligned_dir:
            aligned_dir = Path(args.aligned_dir)
        else:
            reference_dir = Path(args.reference_dir)
            reads_fasta = fasta_dir / f"{bam_path.stem}.txt"
            reads_by_probe = parse_fasta_by_probe(reads_fasta, args.probe_prefix, [probe1_id, probe2_id])
            missing = [p for p in [probe1_id, probe2_id] if not reads_by_probe.get(p)]
            if missing:
                status_path = work_dir / "status.json"
                status = {
                    "status": "missing_probe_reads",
                    "missing_probes": missing,
                    "bam": str(bam_path),
                }
                status_path.write_text(json.dumps(status, indent=2))
                out_csv = Path(args.output_csv)
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                out_csv.write_text("species_id,prob_polyploid,pred_label\n")
                print(f"No reads for probes {missing}. Wrote {status_path} and empty CSV.")
                return

            probe_fasta_dir = work_dir / "probe_fastas"
            probe_fastas = write_probe_fastas(
                reads_by_probe,
                probe_fasta_dir,
                args.probe_prefix,
                max_reads_per_probe=(args.max_reads_per_probe if args.max_reads_per_probe > 0 else None),
                sampling_seed=args.sampling_seed,
            )
            if not probe_fastas:
                raise RuntimeError("No probe FASTA files created from BAM.")
            aligned_dir = work_dir / "aligned"
            for pf in probe_fastas:
                if args.skip_mafft:
                    write_unaligned_with_reference(reference_dir, pf, aligned_dir)
                else:
                    run_mafft_align(
                        reference_dir,
                        pf,
                        aligned_dir,
                        args.probe_prefix,
                        mafft_mode=args.mafft_mode,
                        mafft_threads=args.mafft_threads,
                    )

        # Gap removal (internal + external)
        internal_out = processed_dir / "internal_gaps_removed"
        external_out = processed_dir / "external_gaps_removed"
        run_cmd(
            f"{sys.executable} /blue/juannanzhou/fahimeh.rahimi/ensemble_ploidy_classifier/data_preparation/remove_gaps.py "
            f"--input_dir {aligned_dir} --output_dir {processed_dir} --mode internal --num_cores {args.num_cores}",
            "Internal gap removal",
        )
        run_cmd(
            f"{sys.executable} /blue/juannanzhou/fahimeh.rahimi/ensemble_ploidy_classifier/data_preparation/remove_gaps.py "
            f"--input_dir {internal_out} --output_dir {processed_dir} --mode external --num_cores {args.num_cores}",
            "External gap removal",
        )

        # Tokenize only the two probes into probe15-like JSON (one species = BAM sample)
        sample_id = bam_path.stem.replace(".Target", "")
        token_source = external_out
        try:
            _ = list(token_source.rglob("*.fasta"))
            if not _:
                raise FileNotFoundError
        except Exception:
            token_source = internal_out

        for p in [probe1_id, probe2_id]:
            fasta_path = find_fasta_for_probe(token_source, args.probe_prefix, p)
            write_probe_json(
                fasta_path=fasta_path,
                output_dir=json_dir,
                probe_idx=p,
                species_id=sample_id,
                max_length=154,
                max_auxiliary=350,
            )

    sort_json_files = not args.no_sort_json_files

    # Inference
    emb1, ids1 = embed_probe(
        json_dir=json_dir,
        json_probe_id=probe1_id,
        model_idx=args.probe1,
        cnn_dir=args.cnn_dir,
        reducer_dir=args.reducers_dir,
        sort_json_files=sort_json_files,
        sort_species_keys=args.sort_species_keys,
        device=device,
        batch_size=args.batch_size,
    )
    emb2, ids2 = embed_probe(
        json_dir=json_dir,
        json_probe_id=probe2_id,
        model_idx=args.probe2,
        cnn_dir=args.cnn_dir,
        reducer_dir=args.reducers_dir,
        sort_json_files=sort_json_files,
        sort_species_keys=args.sort_species_keys,
        device=device,
        batch_size=args.batch_size,
    )

    X, ids = align_by_ids(emb1, ids1, emb2, ids2)
    try:
        bundle = joblib.load(args.ensemble_model)
        model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    except Exception as e:
        # Fallback: retrain GradientBoosting on stored embeddings for probes 42/198
        embeddings_dir = Path(args.embeddings_dir)
        X1_tr, y1_tr, ids1_tr = load_embedding_split(embeddings_dir, args.probe1, "train")
        X2_tr, y2_tr, ids2_tr = load_embedding_split(embeddings_dir, args.probe2, "train")
        X1_val, y1_val, ids1_val = load_embedding_split(embeddings_dir, args.probe1, "val")
        X2_val, y2_val, ids2_val = load_embedding_split(embeddings_dir, args.probe2, "val")
        X1_tv, y1_tv, ids1_tv = load_embedding_split(embeddings_dir, args.probe1, "true_val")
        X2_tv, y2_tv, ids2_tv = load_embedding_split(embeddings_dir, args.probe2, "true_val")

        X_tr, y_tr = align_train_by_ids(X1_tr, y1_tr, ids1_tr, X2_tr, y2_tr, ids2_tr)
        X_val, y_val = align_train_by_ids(X1_val, y1_val, ids1_val, X2_val, y2_val, ids2_val)
        X_tv, y_tv = align_train_by_ids(X1_tv, y1_tv, ids1_tv, X2_tv, y2_tv, ids2_tv)

        X_fit = np.vstack([X_tr, X_val, X_tv])
        y_fit = np.concatenate([y_tr, y_val, y_tv])

        model = GradientBoostingClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=1.0,
        )
        model.fit(X_fit, y_fit)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("species_id,prob_polyploid,pred_label\n")
        for sid, p, y in zip(ids, proba, pred):
            f.write(f"{sid},{p:.6f},{int(y)}\n")

    print(f"Saved predictions to {out_csv}")


if __name__ == "__main__":
    main()

