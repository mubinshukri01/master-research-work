"""
split_generator.py

Generates ONE canonical Train / Validate / Calibrate / Test split (60/20/10/10)
for the Manglish sentiment dataset, and saves:
  1. A cleaned, canonical copy of the dataset  -> canonical_dataset.csv
  2. The four index arrays                     -> splits.npz

Every other script (Train_GRU.py, Train_CNN_BERT.py, the ensemble evaluation
script) must load canonical_dataset.csv + splits.npz instead of reading the
raw Excel file and performing its own split. This is what eliminates the
leakage between pretraining data and held-out evaluation data.

Usage:
    python split_generator.py \
        --data "original data/TweeterManglishDS - filtered.xlsx" \
        --text_col "comment/tweet" \
        --label_col "majority_sent" \
        --out splits.npz
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean(data_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    if data_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    n_before = len(df)

    # Drop rows with missing text or missing label. These can't be used for
    # training/evaluation, and if left in (or dropped inconsistently across
    # scripts) they'd break index alignment between splits.npz and whatever
    # each script independently loads.
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)

    n_after = len(df)
    if n_after < n_before:
        print(f"Dropped {n_before - n_after} row(s) with missing text/label "
              f"({n_before} -> {n_after})")

    return df


def generate_canonical_split(
    n_samples: int,
    labels: np.ndarray,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    cal_frac: float = 0.10,
    test_frac: float = 0.10,
    random_state: int = 42,
):
    assert abs((train_frac + val_frac + cal_frac + test_frac) - 1.0) < 1e-9, \
        "Fractions must sum to 1.0"

    all_idx = np.arange(n_samples)

    # Step 1: Train (60%) vs. the rest (40%)
    train_idx, rest_idx, _, rest_y = train_test_split(
        all_idx, labels,
        train_size=train_frac,
        stratify=labels,
        random_state=random_state,
    )

    # Step 2: split the remaining 40% into Val / Cal / Test
    val_share_of_rest = val_frac / (val_frac + cal_frac + test_frac)
    val_idx, remainder_idx, _, remainder_y = train_test_split(
        rest_idx, rest_y,
        train_size=val_share_of_rest,
        stratify=rest_y,
        random_state=random_state,
    )

    # Step 3: split remainder into Cal / Test (even split, since cal_frac == test_frac)
    cal_share_of_remainder = cal_frac / (cal_frac + test_frac)
    cal_idx, test_idx = train_test_split(
        remainder_idx,
        train_size=cal_share_of_remainder,
        stratify=remainder_y,
        random_state=random_state,
    )

    return {"train": train_idx, "val": val_idx, "cal": cal_idx, "test": test_idx}


def verify_no_overlap(splits: dict):
    """Hard assertion: every pairwise intersection between partitions must be empty."""
    keys = list(splits.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            overlap = np.intersect1d(splits[a], splits[b])
            assert overlap.size == 0, (
                f"LEAKAGE DETECTED between '{a}' and '{b}': "
                f"{overlap.size} overlapping indices: {overlap[:10]}..."
            )
    total = sum(len(v) for v in splits.values())
    print(f"Overlap check passed. Total indices covered: {total}")


def report_class_balance(df: pd.DataFrame, label_col: str, splits: dict):
    print("\nClass distribution per partition (proportions):")
    for name, idx in splits.items():
        counts = df.loc[idx, label_col].value_counts(normalize=True).round(3)
        print(f"  {name:5s} (n={len(idx):5d}): {counts.to_dict()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="original data/TweeterManglishDS - filtered.xlsx",
                         help="Path to full dataset (.xlsx or .csv)")
    parser.add_argument("--text_col", default="comment/tweet", help="Column name for text")
    parser.add_argument("--label_col", default="majority_sent", help="Column name for sentiment label")
    parser.add_argument("--out", default="splits.npz", help="Output path for saved index arrays")
    parser.add_argument("--clean_out", default="canonical_dataset.csv",
                         help="Output path for the cleaned canonical dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_and_clean(args.data, args.text_col, args.label_col)
    labels = df[args.label_col].values

    splits = generate_canonical_split(
        n_samples=len(df),
        labels=labels,
        random_state=args.seed,
    )

    verify_no_overlap(splits)
    report_class_balance(df, args.label_col, splits)

    # Save the cleaned dataset so every downstream script reads the SAME
    # rows in the SAME order. The saved indices are only valid against this file.
    df.to_csv(args.clean_out, index=False)

    np.savez(args.out, **splits)
    summary = {k: len(v) for k, v in splits.items()}
    with open(args.out.replace(".npz", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved canonical dataset to {args.clean_out}")
    print(f"Saved canonical split to {args.out}")
    print(f"Partition sizes: {summary}")


if __name__ == "__main__":
    main()