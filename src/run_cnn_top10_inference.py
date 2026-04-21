#!/usr/bin/env python3
"""
Run top-N CNN models on BAM files (no ensemble).
Uses the same data preparation steps as run_ensemble_inference_42_198.py.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd


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


def resolve_cnn_path(cnn_dir: str, probe_idx: int) -> str:
    c1 = os.path.join(cnn_dir, f"best_model_probe_3_{probe_idx}.pt")
    c2 = os.path.join(cnn_dir, f"best_model_probe_{probe_idx}.pt")
    if os.path.exists(c1):
        return c1
    if os.path.exists(c2):
        return c2
    raise FileNotFoundError(f"No CNN found for probe {probe_idx} (checked {c1} and {c2})")


# -------------------------
# Data prep helpers
# -------------------------

def resolve_mafft_bin() -> str:
    env_bin = os.environ.get("MAFFT_BIN")
    if env_bin:
        return env_bin
    mafft_bin = shutil.which("mafft")
    if mafft_bin:
        return mafft_bin
    raise FileNotFoundError("mafft not found in PATH. Install mafft or set MAFFT_BIN.")


def bam_to_fasta(bam_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{bam_path.stem}.txt"
    try:
        import pysam  # type: ignore
    except Exception as e:
        if shutil.which("samtools") is None:
            raise RuntimeError(
                "pysam is required for BAM -> FASTA conversion. Activate the conda env with pysam installed."
            ) from e
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
        return out_file

    truncated_error: Optional[Exception] = None
    with out_file.open("w") as output_file, pysam.AlignmentFile(str(bam_path), "rb") as bam_file:
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
    return out_file


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


def run_mafft_align(
    reference_dir: Path,
    probe_fasta: Path,
    out_dir: Path,
    mafft_mode: str = "accurate",
    mafft_threads: int = 1,
) -> Optional[Path]:
    ref_name = probe_fasta.stem
    ref_path = reference_dir / f"{ref_name}.fasta"
    if not ref_path.exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / f"{ref_name}_combined.fasta"
    aligned = out_dir / f"{ref_name}_aligned.fasta"
    with combined.open("w") as f:
        f.write(ref_path.read_text())
        f.write(probe_fasta.read_text())

    mafft_bin = resolve_mafft_bin()
    log_path = out_dir / f"{ref_name}_mafft.log"
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


def write_unaligned_with_reference(reference_dir: Path, probe_fasta: Path, out_dir: Path) -> Optional[Path]:
    ref_name = probe_fasta.stem
    ref_path = reference_dir / f"{ref_name}.fasta"
    if not ref_path.exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ref_name}_aligned.fasta"
    with out_path.open("w") as f:
        # Keep output shape compatible with downstream code: reference first, then reads.
        f.write(ref_path.read_text())
        f.write(probe_fasta.read_text())
    return out_path


def run_cmd(cmd: str, desc: str) -> None:
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed with exit code {result.returncode}")


def dna_tokenizer(seq: str, max_length: int = 154) -> List[int]:
    token_mapping = {"A": 1, "C": 2, "G": 3, "T": 4, "-": 0}
    tokens = [token_mapping.get(nuc.upper(), 0) for nuc in seq]
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    elif len(tokens) < max_length:
        tokens += [0] * (max_length - len(tokens))
    return tokens


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


def select_aux_by_gap(aux_seqs: List[str], max_auxiliary: int) -> List[str]:
    if len(aux_seqs) <= max_auxiliary:
        return aux_seqs
    gap_counts = [seq.count("-") for seq in aux_seqs]
    idxs = np.argsort(gap_counts)[:max_auxiliary]
    return [aux_seqs[i] for i in idxs]


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


# -------------------------
# CNN inference
# -------------------------

class InferenceDataset(Dataset):
    def __init__(self, seqs: List[np.ndarray]):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.tensor(self.seqs[i], dtype=torch.float32)


def load_probe_json(json_path: Path) -> List[np.ndarray]:
    with json_path.open("r") as f:
        data = json.load(f)
    seqs = []
    for info in data.values():
        ref = np.array(info["reference_tokens"])
        aux = np.array(info["auxiliary_tokens"])
        full = np.vstack([ref.reshape(1, -1), aux]).reshape(1, 351, 154)
        seqs.append(full)
    return seqs


def run_cnn_predict(json_path: Path, cnn_path: str, device: torch.device) -> float:
    seqs = load_probe_json(json_path)
    ds = InferenceDataset(seqs)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    cnn = torch.load(cnn_path, map_location=device, weights_only=False).to(device).eval()
    probs = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            x = x.view(x.size(0), 1, 351, -1)
            p = cnn(x).cpu().numpy().reshape(-1)
            probs.extend(list(p))
    # For single-sample BAM, take mean
    return float(np.mean(probs)) if probs else 0.0


def get_top_probes(npz_path: Path, top_n: int) -> List[int]:
    data = np.load(npz_path)
    probes = list(data["probe_indices"])
    return [int(p) for p in probes[:top_n]]


def load_probe_id_mapping(json_reference_dir: Path, probe_id_map_csv: Optional[Path] = None) -> List[int]:
    # Prefer an explicit frozen mapping manifest to avoid any ordering ambiguity.
    if probe_id_map_csv is not None and probe_id_map_csv.exists():
        df = pd.read_csv(probe_id_map_csv)
        if not {"model_index", "probe_id"}.issubset(df.columns):
            raise ValueError(
                f"Invalid probe-id mapping CSV {probe_id_map_csv}: required columns model_index, probe_id"
            )
        out: Dict[int, int] = {}
        for _, row in df.iterrows():
            mi = pd.to_numeric(row.get("model_index"), errors="coerce")
            pid = pd.to_numeric(row.get("probe_id"), errors="coerce")
            if pd.isna(mi) or pd.isna(pid):
                continue
            out[int(mi)] = int(pid)
        if not out:
            raise ValueError(f"Probe-id mapping CSV {probe_id_map_csv} contained no valid rows.")
        max_idx = max(out.keys())
        return [out.get(i, -1) for i in range(max_idx + 1)]

    # Backward-compatible fallback: legacy lexicographic ordering used by historical artifacts.
    files = sorted([f for f in os.listdir(json_reference_dir) if f.endswith(".json")])
    probe_ids: List[int] = []
    for name in files:
        stem = Path(name).stem
        if stem.startswith("L"):
            try:
                probe_ids.append(int(stem.replace("L", "")))
            except ValueError:
                probe_ids.append(-1)
        else:
            probe_ids.append(-1)
    return probe_ids


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Run top-N CNN probe models on a BAM file to predict ploidy."
    )
    ap.add_argument("--bam", required=True, help="Input BAM file")
    ap.add_argument("--work-dir", required=True, help="Working directory for intermediate and output files")
    ap.add_argument("--cnn-dir", default=str(repo_root / "models" / "cnn_weights"),
                     help="Directory with best_model_probe_3_*.pt CNN checkpoints")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--ranked-npz", default=str(repo_root / "data" / "final_ranked_results.npz"))
    ap.add_argument("--probe-prefix", default="L")
    ap.add_argument("--reference-dir", required=True, help="Directory with L*.fasta reference files")
    ap.add_argument("--json-reference-dir", default=None,
                     help="Reference JSON dir (for legacy probe-id mapping fallback)")
    ap.add_argument(
        "--probe-id-map-csv",
        default=str(repo_root / "data" / "probe_index_to_id_manifest.csv"),
        help="Mapping CSV with columns model_index,probe_id.",
    )
    ap.add_argument("--num-cores", type=int, default=8)
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
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--truth-file",
        default=None,
        help="Optional truth labels file (.csv or .xlsx) for accuracy evaluation.",
    )
    ap.add_argument("--skip-if-exists", action="store_true")
    ap.add_argument(
        "--skip-mafft",
        action="store_true",
        help="Skip MAFFT and use reference+reads concatenation (faster, may reduce accuracy).",
    )
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.output_dir) if args.output_dir else work_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if summary exists
    summary_path = out_dir / "cnn_top10_summary.csv"
    if args.skip_if_exists and summary_path.exists() and summary_path.stat().st_size > 0:
        print(f"Output already exists, skipping: {summary_path}")
        return

    bam_path = Path(args.bam)
    top_probes = get_top_probes(Path(args.ranked_npz), args.top_n)
    fasta_dir = work_dir / "fasta"
    processed_dir = work_dir / "processed"
    json_dir = work_dir / "json"

    try:
        bam_to_fasta(bam_path, fasta_dir)
    except RuntimeError as e:
        sample_id = bam_path.stem.replace(".Target", "")
        status = {"status": "bam_read_error", "error": str(e), "bam": str(bam_path)}
        (out_dir / "status.json").write_text(json.dumps(status, indent=2))
        summary_rows = [
            {
                "species_id": sample_id,
                "probe": p,
                "probe_id": None,
                "prob_polyploid": None,
                "pred_label": None,
                "status": "bam_read_error",
                "true_label": None,
                "correct": None,
            }
            for p in top_probes
        ]
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"{e}. Wrote {(out_dir / 'status.json')} and {summary_path}.")
        return

    probe_map_csv = Path(args.probe_id_map_csv) if args.probe_id_map_csv else None
    mapping = load_probe_id_mapping(Path(args.json_reference_dir), probe_map_csv)
    probe_id_by_index = {idx: probe_id for idx, probe_id in enumerate(mapping) if probe_id >= 0}
    # map model indices -> probe ids for reference/alignment
    probe_ids = []
    for idx in top_probes:
        pid = probe_id_by_index.get(idx)
        if pid is not None:
            probe_ids.append(pid)
    reads_fasta = fasta_dir / f"{bam_path.stem}.txt"
    reads_by_probe = parse_fasta_by_probe(reads_fasta, args.probe_prefix, probe_ids)
    missing = []
    for idx in top_probes:
        probe_id = probe_id_by_index.get(idx)
        if probe_id is None or not reads_by_probe.get(probe_id):
            missing.append(idx)

    if len(missing) == len(top_probes):
        status = {"status": "missing_all_probe_reads", "missing_probes": missing, "bam": str(bam_path)}
        (out_dir / "status.json").write_text(json.dumps(status, indent=2))
        print("No reads for any top probes. Skipping.")
        return

    probe_fasta_dir = work_dir / "probe_fastas"
    probe_fastas = write_probe_fastas(
        reads_by_probe,
        probe_fasta_dir,
        args.probe_prefix,
        max_reads_per_probe=(args.max_reads_per_probe if args.max_reads_per_probe > 0 else None),
        sampling_seed=args.sampling_seed,
    )
    aligned_dir = work_dir / "aligned"
    missing_reference = set()
    for pf in probe_fastas:
        if args.skip_mafft:
            aligned = write_unaligned_with_reference(Path(args.reference_dir), pf, aligned_dir)
        else:
            aligned = run_mafft_align(
                Path(args.reference_dir),
                pf,
                aligned_dir,
                mafft_mode=args.mafft_mode,
                mafft_threads=args.mafft_threads,
            )
        if aligned is None:
            try:
                idx = int(pf.stem.replace(args.probe_prefix, ""))
                missing_reference.add(idx)
            except ValueError:
                pass

    # Gap removal
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

    sample_id = bam_path.stem.replace(".Target", "")
    predictions = []
    for p in top_probes:
        probe_id = probe_id_by_index.get(p)
        if probe_id is None:
            predictions.append(
                {"probe": p, "probe_id": None, "prob_polyploid": None, "pred_label": None, "status": "missing_probe_id"}
            )
            continue
        if p in missing:
            predictions.append(
                {"probe": p, "probe_id": probe_id, "prob_polyploid": None, "pred_label": None, "status": "missing_probe_reads"}
            )
            continue
        if probe_id in missing_reference:
            predictions.append(
                {"probe": p, "probe_id": probe_id, "prob_polyploid": None, "pred_label": None, "status": "missing_reference_fasta"}
            )
            continue
        # find aligned fasta for probe
        fasta_matches = list(external_out.rglob(f"{args.probe_prefix}{probe_id}*.fasta"))
        if not fasta_matches:
            predictions.append(
                {"probe": p, "probe_id": probe_id, "prob_polyploid": None, "pred_label": None, "status": "missing_fasta"}
            )
            continue
        fasta_path = fasta_matches[0]
        json_path = write_probe_json(
            fasta_path=fasta_path,
            output_dir=json_dir,
            probe_idx=probe_id,
            species_id=sample_id,
            max_length=154,
            max_auxiliary=350,
        )
        cnn_path = resolve_cnn_path(args.cnn_dir, p)
        prob = run_cnn_predict(json_path, cnn_path, device)
        pred = int(prob >= 0.5)
        predictions.append(
            {"probe": p, "probe_id": probe_id, "prob_polyploid": prob, "pred_label": pred, "status": "success"}
        )

        # per-model CSV
        out_csv = out_dir / f"cnn_probe_{p}_predictions.csv"
        out_csv.write_text("species_id,prob_polyploid,pred_label\n" + f"{sample_id},{prob:.6f},{pred}\n")

    # compare to truth
    truth_path = Path(args.truth_file)
    if truth_path.suffix.lower() == ".csv":
        truth_df = pd.read_csv(truth_path)
    else:
        truth_df = pd.read_excel(truth_path)

    if {"RapidID2", "Ploidy.binary"}.issubset(truth_df.columns):
        truth_df = truth_df[["RapidID2", "Ploidy.binary"]].rename(
            columns={"RapidID2": "species_id", "Ploidy.binary": "true_label"}
        )
    elif {"species_id", "true_label"}.issubset(truth_df.columns):
        truth_df = truth_df[["species_id", "true_label"]].copy()
    else:
        raise ValueError(
            f"Unsupported truth file columns in {truth_path}. "
            "Expected either [RapidID2, Ploidy.binary] or [species_id, true_label]."
        )
    truth_df["species_id"] = truth_df["species_id"].astype(str).str.strip()
    truth_label = None
    match = truth_df[truth_df["species_id"] == sample_id]
    if not match.empty:
        truth_label = int(match.iloc[0]["true_label"])

    summary_rows = []
    for row in predictions:
        row_out = {
            "species_id": sample_id,
            "probe": row["probe"],
            "probe_id": row.get("probe_id"),
            "prob_polyploid": row["prob_polyploid"],
            "pred_label": row["pred_label"],
            "status": row["status"],
            "true_label": truth_label,
            "correct": None,
        }
        if truth_label is not None and row["pred_label"] is not None:
            row_out["correct"] = int(row["pred_label"] == truth_label)
        summary_rows.append(row_out)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

