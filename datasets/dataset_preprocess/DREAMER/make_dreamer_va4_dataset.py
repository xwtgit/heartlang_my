import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from QRSTokenizer import QRSTokenizer


LABEL_NAMES = {
    0: "LVHA",
    1: "HVHA",
    2: "LVLA",
    3: "HVLA",
}


def get_args():
    parser = argparse.ArgumentParser(
        "Prepare DREAMER ECG for HeartLang VA quadrant linear probing."
    )
    parser.add_argument("--dreamer_mat", required=True, help="Path to DREAMER.mat.")
    parser.add_argument(
        "--output_dir",
        default="datasets/ecg_datasets/DREAMER_QRS/va4",
        help="Output directory for HeartLang QRS files.",
    )
    parser.add_argument(
        "--raw_output_dir",
        default="datasets/ecg_datasets/DREAMER/va4",
        help="Optional output directory for raw windowed ECG before QRS tokenization.",
    )
    parser.add_argument("--original_fs", default=256, type=int)
    parser.add_argument("--target_fs", default=100, type=int)
    parser.add_argument("--window_sec", default=60.0, type=float)
    parser.add_argument("--step_sec", default=None, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--train_subjects", default=16, type=int)
    parser.add_argument("--val_subjects", default=3, type=int)
    parser.add_argument("--test_subjects", default=4, type=int)
    parser.add_argument("--used_channels", nargs="+", default=[0, 1], type=int)
    parser.add_argument("--max_len", default=256, type=int)
    parser.add_argument("--token_len", default=96, type=int)
    parser.add_argument("--no_zscore", action="store_true")
    parser.add_argument("--plot_qrs", action="store_true")
    return parser.parse_args()


def get_field(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        return obj[name]

    lower_name = name.lower()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() == lower_name:
                return value
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        for key in obj.dtype.names:
            if key.lower() == lower_name:
                return obj[key]
    if hasattr(obj, "_fieldnames"):
        for key in obj._fieldnames:
            if key.lower() == lower_name:
                return getattr(obj, key)
    raise KeyError(f"Cannot find field '{name}' in DREAMER structure.")


def to_list(value):
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return [arr.item()]
    return [item for item in arr.reshape(-1)]


def to_1d_float(value):
    return np.asarray(value, dtype=np.float32).reshape(-1)


def to_channels_first(ecg):
    arr = np.asarray(ecg, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D ECG trial, got shape {arr.shape}.")
    if arr.shape[0] == 2:
        return arr
    if arr.shape[1] == 2:
        return arr.T
    if arr.shape[0] < arr.shape[1]:
        return arr
    return arr.T


def zscore_channels(ecg):
    mean = ecg.mean(axis=1, keepdims=True)
    std = ecg.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return (ecg - mean) / std


def va_to_quadrant(valence, arousal):
    if valence < 3 and arousal >= 3:
        return 0
    if valence >= 3 and arousal >= 3:
        return 1
    if valence < 3 and arousal < 3:
        return 2
    return 3


def load_dreamer_subjects(dreamer_mat):
    try:
        mat = loadmat(dreamer_mat, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as err:
        raise RuntimeError(
            "This script expects the public DREAMER.mat layout readable by "
            "scipy.io.loadmat. If your file is MATLAB v7.3/HDF5, save it as a "
            "non-v7.3 .mat file first."
        ) from err
    dreamer = mat["DREAMER"]
    return to_list(get_field(dreamer, "Data"))


def split_subjects(num_subjects, args):
    expected = args.train_subjects + args.val_subjects + args.test_subjects
    if expected != num_subjects:
        raise ValueError(
            "Subject split counts must sum to the number of DREAMER subjects: "
            f"{expected} != {num_subjects}"
        )

    subject_ids = np.arange(1, num_subjects + 1)
    rng = np.random.default_rng(args.seed)
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)

    train_end = args.train_subjects
    val_end = train_end + args.val_subjects
    split = {
        "train": sorted(shuffled[:train_end].tolist()),
        "val": sorted(shuffled[train_end:val_end].tolist()),
        "test": sorted(shuffled[val_end:].tolist()),
    }
    return split


def split_for_subject(subject_id, split):
    for stage, subject_ids in split.items():
        if subject_id in subject_ids:
            return stage
    raise ValueError(f"Subject {subject_id} is not assigned to any split.")


def iter_windows(ecg, window_len, step_len):
    if ecg.shape[1] < window_len:
        return
    for start in range(0, ecg.shape[1] - window_len + 1, step_len):
        yield start, ecg[:, start : start + window_len]


def prepare_windows(subjects, split, args):
    window_len = int(round(args.window_sec * args.target_fs))
    step_sec = args.step_sec if args.step_sec is not None else args.window_sec
    step_len = int(round(step_sec * args.target_fs))
    if window_len <= 0 or step_len <= 0:
        raise ValueError("window_sec and step_sec must produce positive lengths.")

    stages = {
        stage: {"data": [], "labels": [], "metadata": []}
        for stage in ["train", "val", "test"]
    }

    for subject_index, subject in enumerate(subjects, start=1):
        stage = split_for_subject(subject_index, split)
        ecg = get_field(subject, "ECG")
        stimuli = to_list(get_field(ecg, "stimuli"))
        valence_scores = to_1d_float(get_field(subject, "ScoreValence"))
        arousal_scores = to_1d_float(get_field(subject, "ScoreArousal"))

        if len(stimuli) != len(valence_scores) or len(stimuli) != len(arousal_scores):
            raise ValueError(
                f"Subject {subject_index} has mismatched trial/label counts: "
                f"{len(stimuli)}, {len(valence_scores)}, {len(arousal_scores)}"
            )

        for trial_index, trial_ecg in enumerate(stimuli, start=1):
            ecg_cf = to_channels_first(trial_ecg)
            ecg_100hz = resample_poly(
                ecg_cf, args.target_fs, args.original_fs, axis=1
            ).astype(np.float32)
            if not args.no_zscore:
                ecg_100hz = zscore_channels(ecg_100hz).astype(np.float32)

            valence = float(valence_scores[trial_index - 1])
            arousal = float(arousal_scores[trial_index - 1])
            label = va_to_quadrant(valence, arousal)

            for start, window in iter_windows(ecg_100hz, window_len, step_len):
                stages[stage]["data"].append(window)
                stages[stage]["labels"].append(label)
                stages[stage]["metadata"].append(
                    {
                        "subject": subject_index,
                        "trial": trial_index,
                        "start_sample_100hz": int(start),
                        "end_sample_100hz": int(start + window_len),
                        "valence": valence,
                        "arousal": arousal,
                        "label": label,
                        "label_name": LABEL_NAMES[label],
                    }
                )

    return stages


def save_raw_windows(stages, raw_output_dir):
    raw_output_dir = Path(raw_output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    for stage, payload in stages.items():
        data = np.asarray(payload["data"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        np.save(raw_output_dir / f"{stage}_data.npy", data)
        np.save(raw_output_dir / f"{stage}_labels.npy", labels)
        with open(raw_output_dir / f"{stage}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(payload["metadata"], f, indent=2)
        print(f"{stage}: raw data {data.shape}, labels {labels.shape}")
        print_label_distribution(labels, prefix=f"{stage} raw")


def tokenize_and_save(stages, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stage, payload in stages.items():
        data = np.asarray(payload["data"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        if len(data) == 0:
            raise ValueError(f"No ECG windows were generated for split '{stage}'.")

        tokenizer = QRSTokenizer(
            fs=args.target_fs,
            max_len=args.max_len,
            token_len=args.token_len,
            save_path=str(output_dir),
            stage=stage,
            used_channels=args.used_channels,
        )
        tokenizer(data, plot=args.plot_qrs)
        np.save(output_dir / f"{stage}_labels.npy", labels)
        with open(output_dir / f"{stage}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(payload["metadata"], f, indent=2)

        qrs_data = np.load(output_dir / f"{stage}_data.npy", mmap_mode="r")
        in_chans = np.load(output_dir / f"{stage}_data_in_chans.npy", mmap_mode="r")
        in_times = np.load(output_dir / f"{stage}_data_in_times.npy", mmap_mode="r")
        print(
            f"{stage}: QRS data {qrs_data.shape}, labels {labels.shape}, "
            f"in_chans {in_chans.shape}, in_times {in_times.shape}"
        )
        print_label_distribution(labels, prefix=f"{stage} QRS")


def print_label_distribution(labels, prefix):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=4)
    readable = ", ".join(
        f"{class_id}:{LABEL_NAMES[class_id]}={count}"
        for class_id, count in enumerate(counts)
    )
    print(f"{prefix} label distribution: {readable}")


def save_split_metadata(split, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "split": split,
        "label_names": LABEL_NAMES,
        "valence_arousal_rule": {
            "0": "LVHA: V < 3, A >= 3",
            "1": "HVHA: V >= 3, A >= 3",
            "2": "LVLA: V < 3, A < 3",
            "3": "HVLA: V >= 3, A < 3",
        },
        "original_fs": args.original_fs,
        "target_fs": args.target_fs,
        "window_sec": args.window_sec,
        "step_sec": args.step_sec if args.step_sec is not None else args.window_sec,
        "used_channels": args.used_channels,
        "zscore": not args.no_zscore,
        "seed": args.seed,
    }
    with open(output_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    args = get_args()
    if len(args.used_channels) != 2:
        print(
            "Warning: this script was designed for DREAMER's two ECG channels; "
            f"received used_channels={args.used_channels}."
        )

    subjects = load_dreamer_subjects(args.dreamer_mat)
    split = split_subjects(len(subjects), args)
    print(f"Loaded {len(subjects)} DREAMER subjects.")
    print(f"Subject split: {split}")

    stages = prepare_windows(subjects, split, args)
    save_raw_windows(stages, args.raw_output_dir)
    tokenize_and_save(stages, args)
    save_split_metadata(split, args)
    print(f"Done. HeartLang QRS dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
