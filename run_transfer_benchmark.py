"""
Complete Transfer Learning Benchmark: Motor Execution → Motor Imagery
6 models x 3 channel configs x 3 conditions x 5 splits = 270 runs

Saves results to results/transfer_results.csv
Run with: python run_transfer_benchmark.py
"""

import random
import time
import warnings
from copy import deepcopy
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from braindecode.models import CTNet, EEGInceptionMI, EEGNeX, EEGSimpleConv, EEGSym, MSVTNet
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# ── Config ──────────────────────────────────────────────────────────────
SEED = 7
DATA_ROOT = Path("/NAS/aniruddham/mne/data")
RESULTS_DIR = Path("results")
MOVEMENT_RUNS = [3, 7, 11]
IMAGERY_RUNS = [4, 8, 12]
RESAMPLE_SFREQ = 160.0
FILTER_BAND = (8.0, 30.0)
EPOCH_WINDOW = (0.0, 4.0)
CROP_WINDOW = (0.5, 2.5)

CHANNEL_OPTIONS = {
    "all_64": None,
    "sensorimotor_17": [
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CPz", "CP4",
    ],
    "sensorimotor_9": [
        "FC3", "FCz", "FC4",
        "C3", "Cz", "C4",
        "CP3", "CPz", "CP4",
    ],
}

MODEL_NAMES = [
    "EEGSimpleConv",
    "EEGInceptionMI",
    "EEGSym",
    "MSVTNet",
    "EEGNeX",
    "CTNet",
]

N_SPLITS = 5
MAX_SUBJECTS = None
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 12
FINETUNE_EPOCHS = 8
FINETUNE_LR = 5e-4
BASELINE_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.2
VAL_SIZE = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONTAGE = make_standard_montage("standard_1005")

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# ── Helpers ─────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_available_subjects(data_root: Path = DATA_ROOT) -> list[int]:
    return sorted(
        int(p.name[1:])
        for p in data_root.glob("S[0-9][0-9][0-9]")
        if p.is_dir()
    )


def local_run_paths(subject: int, runs: list[int], data_root: Path = DATA_ROOT) -> list[Path]:
    s = f"S{subject:03d}"
    return [data_root / s / f"{s}R{r:02d}.edf" for r in runs]


def load_subject_raw(subject: int, runs: list[int]) -> mne.io.BaseRaw:
    paths = local_run_paths(subject, runs)
    raw = concatenate_raws(
        [read_raw_edf(p, preload=True, verbose="ERROR") for p in paths]
    )
    eegbci.standardize(raw)
    raw.pick("eeg")
    raw.set_montage(MONTAGE, match_case=False, on_missing="warn")
    raw.annotations.rename({"T1": "left_hand", "T2": "right_hand"})
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.resample(RESAMPLE_SFREQ, verbose="ERROR")
    raw.filter(*FILTER_BAND, fir_design="firwin", skip_by_annotation="edge", verbose="ERROR")
    return raw


def epoch_subject(raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray, Epochs]:
    events, event_map = mne.events_from_annotations(raw, verbose="ERROR")
    epochs = Epochs(
        raw, events,
        event_id={"left_hand": event_map["left_hand"], "right_hand": event_map["right_hand"]},
        tmin=EPOCH_WINDOW[0], tmax=EPOCH_WINDOW[1],
        proj=False,
        picks=pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"),
        baseline=None, preload=True, reject_by_annotation=True, verbose="ERROR",
    ).crop(*CROP_WINDOW)
    if len(epochs.times) % 2 != 0:
        epochs.crop(tmin=epochs.tmin, tmax=epochs.times[-2])
    X = epochs.get_data(copy=False).astype(np.float32)
    y = (epochs.events[:, -1] == event_map["right_hand"]).astype(np.int64)
    return X, y, epochs


def select_channels(X, channel_names, chs_info, wanted):
    if wanted is None:
        return X, channel_names, chs_info
    name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
    picks = [name_to_idx[ch] for ch in wanted if ch in name_to_idx]
    return (
        X[:, picks, :],
        [channel_names[idx] for idx in picks],
        [deepcopy(chs_info[idx]) for idx in picks],
    )


def build_dataset(runs: list[int], label: str) -> dict:
    subjects = infer_available_subjects()
    if MAX_SUBJECTS is not None:
        subjects = subjects[:MAX_SUBJECTS]

    X_parts, y_parts, groups = [], [], []
    channel_names = None
    chs_info = None

    for i, subject in enumerate(subjects):
        raw = load_subject_raw(subject, runs)
        X, y, epochs = epoch_subject(raw)
        mean = X.mean(axis=2, keepdims=True)
        std = X.std(axis=2, keepdims=True) + 1e-6
        X = (X - mean) / std

        if channel_names is None:
            channel_names = epochs.ch_names
            chs_info = deepcopy(epochs.info["chs"])

        X_parts.append(X)
        y_parts.append(y)
        groups.append(np.full(len(y), subject))

        if (i + 1) % 20 == 0:
            print(f"  [{label}] Loaded {i + 1}/{len(subjects)} subjects")

    print(f"  [{label}] Done: {len(subjects)} subjects")
    return {
        "X": np.concatenate(X_parts, axis=0),
        "y": np.concatenate(y_parts, axis=0),
        "groups": np.concatenate(groups, axis=0),
        "channel_names": list(channel_names),
        "chs_info": chs_info,
        "sfreq": RESAMPLE_SFREQ,
    }


def make_group_split(groups: np.ndarray, seed: int) -> dict:
    index = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_val_idx, test_idx = next(outer.split(index, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=seed + 100)
    inner_train_rel, inner_val_rel = next(
        inner.split(np.arange(len(train_val_idx)), groups=groups[train_val_idx])
    )
    return {
        "train": train_val_idx[inner_train_rel],
        "val": train_val_idx[inner_val_rel],
        "test": test_idx,
    }


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False,
    )


def unpack_logits(logits):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    while logits.ndim > 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if logits.ndim > 2:
        logits = logits.mean(dim=tuple(range(2, logits.ndim)))
    return logits


@torch.no_grad()
def predict_torch(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = []
    for batch_X, _ in loader:
        batch_X = batch_X.to(DEVICE)
        logits = unpack_logits(model(batch_X))
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def build_model(model_name: str, n_chans: int, n_times: int, sfreq: float, chs_info: list[dict]) -> nn.Module:
    common = dict(n_chans=n_chans, n_times=n_times, n_outputs=2, sfreq=sfreq, chs_info=chs_info)
    if model_name == "EEGSimpleConv":
        return EEGSimpleConv(**common).to(DEVICE)
    if model_name == "EEGInceptionMI":
        return EEGInceptionMI(**common).to(DEVICE)
    if model_name == "EEGSym":
        return EEGSym(**common, scales_time=(525, 275, 125)).to(DEVICE)
    if model_name == "MSVTNet":
        return MSVTNet(**common).to(DEVICE)
    if model_name == "EEGNeX":
        return EEGNeX(**common).to(DEVICE)
    if model_name == "CTNet":
        return CTNet(**common).to(DEVICE)
    raise KeyError(model_name)


def train_phase(model, train_loader, val_loader, y_val, max_epochs, lr=LEARNING_RATE):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val_bal = -np.inf
    best_epoch = 0
    patience_counter = 0
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = unpack_logits(model(batch_X))
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        val_pred = predict_torch(model, val_loader)
        val_bal = balanced_accuracy_score(y_val, val_pred)
        if val_bal > best_val_bal + 1e-4:
            best_val_bal = val_bal
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start
    return model, best_epoch, elapsed


# ── Main ────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "transfer_results.csv"

    print(f"Device: {DEVICE}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Channels: {list(CHANNEL_OPTIONS.keys())}")
    print(f"Splits: {N_SPLITS}")
    total = len(CHANNEL_OPTIONS) * N_SPLITS * len(MODEL_NAMES) * 3
    print(f"Total runs: {total}")
    print()

    # Load data
    print("Loading datasets...")
    set_seed()
    imagery_ds = build_dataset(IMAGERY_RUNS, "imagery")
    movement_ds = build_dataset(MOVEMENT_RUNS, "movement")
    print(f"Imagery:  {len(imagery_ds['y'])} epochs")
    print(f"Movement: {len(movement_ds['y'])} epochs")
    print()

    all_rows = []
    run_count = 0

    for subset_name, subset_channels in CHANNEL_OPTIONS.items():
        Xi, chi_names, chi_info = select_channels(
            imagery_ds["X"], imagery_ds["channel_names"], imagery_ds["chs_info"], subset_channels
        )
        Xm, _, _ = select_channels(
            movement_ds["X"], movement_ds["channel_names"], movement_ds["chs_info"], subset_channels
        )
        n_chans, n_times, sfreq = Xi.shape[1], Xi.shape[2], imagery_ds["sfreq"]

        for split_idx in range(N_SPLITS):
            seed = SEED + split_idx
            i_split = make_group_split(imagery_ds["groups"], seed)
            m_split = make_group_split(movement_ds["groups"], seed)

            Xi_train, yi_train = Xi[i_split["train"]], imagery_ds["y"][i_split["train"]]
            Xi_val, yi_val = Xi[i_split["val"]], imagery_ds["y"][i_split["val"]]
            Xi_test, yi_test = Xi[i_split["test"]], imagery_ds["y"][i_split["test"]]
            Xm_train, ym_train = Xm[m_split["train"]], movement_ds["y"][m_split["train"]]
            Xm_val, ym_val = Xm[m_split["val"]], movement_ds["y"][m_split["val"]]

            i_train_loader = make_loader(Xi_train, yi_train, shuffle=True)
            i_val_loader = make_loader(Xi_val, yi_val, shuffle=False)
            i_test_loader = make_loader(Xi_test, yi_test, shuffle=False)
            m_train_loader = make_loader(Xm_train, ym_train, shuffle=True)
            m_val_loader = make_loader(Xm_val, ym_val, shuffle=False)

            for model_name in MODEL_NAMES:
                base = dict(model=model_name, channel_subset=subset_name,
                            n_channels=n_chans, split=split_idx)

                # Condition 1: Imagery only
                set_seed(seed)
                model = build_model(model_name, n_chans, n_times, sfreq, chi_info)
                model, best_ep, elapsed = train_phase(
                    model, i_train_loader, i_val_loader, yi_val, BASELINE_EPOCHS
                )
                test_pred = predict_torch(model, i_test_loader)
                all_rows.append({
                    **base, "condition": "imagery_only",
                    "test_balanced_accuracy": balanced_accuracy_score(yi_test, test_pred),
                    "best_epoch": best_ep, "train_time_s": elapsed,
                })
                run_count += 1

                # Condition 2: Zero-shot
                set_seed(seed)
                model = build_model(model_name, n_chans, n_times, sfreq, chi_info)
                model, best_ep, elapsed = train_phase(
                    model, m_train_loader, m_val_loader, ym_val, PRETRAIN_EPOCHS
                )
                test_pred = predict_torch(model, i_test_loader)
                all_rows.append({
                    **base, "condition": "zero_shot",
                    "test_balanced_accuracy": balanced_accuracy_score(yi_test, test_pred),
                    "best_epoch": best_ep, "train_time_s": elapsed,
                })
                run_count += 1

                # Condition 3: Pretrain + finetune
                set_seed(seed)
                model = build_model(model_name, n_chans, n_times, sfreq, chi_info)
                model, _, pre_time = train_phase(
                    model, m_train_loader, m_val_loader, ym_val, PRETRAIN_EPOCHS
                )
                model, ft_ep, ft_time = train_phase(
                    model, i_train_loader, i_val_loader, yi_val, FINETUNE_EPOCHS, lr=FINETUNE_LR
                )
                test_pred = predict_torch(model, i_test_loader)
                all_rows.append({
                    **base, "condition": "pretrain_finetune",
                    "test_balanced_accuracy": balanced_accuracy_score(yi_test, test_pred),
                    "best_epoch": ft_ep, "train_time_s": pre_time + ft_time,
                })
                run_count += 1

                print(f"[{run_count:3d}/{total}] {model_name:18s} | {subset_name:17s} | split {split_idx}")

                # Save incrementally so nothing is lost
                pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(csv_path, index=False)
    print(f"\nDone. {len(results_df)} results saved to {csv_path}")

    # Print quick summary
    summary = (
        results_df.groupby(["model", "channel_subset", "condition"], as_index=False)
        .agg(mean_bal_acc=("test_balanced_accuracy", "mean"),
             std_bal_acc=("test_balanced_accuracy", "std"))
    )
    pivot = summary.pivot_table(
        index=["model", "channel_subset"], columns="condition", values="mean_bal_acc"
    )
    print("\n" + "=" * 80)
    print("SUMMARY (mean balanced accuracy across 5 splits)")
    print("=" * 80)
    print(pivot.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
