# ===============================================================
# run_baselines_canonical.py
#
# Re-runs the eight single-model baselines on the CANONICAL split so
# they are directly comparable to every ensemble number in
# RESULTS_CONSOLIDATED.xlsx.
#
#     {Keras word tokenizer, BERT tokenizer} x {CNN, GRU, LSTM, RNN}
#
# What differs from the originals in ../NLP and ../BERT:
#
#   * partition comes from redo/splits.npz, not a local
#     train_test_split, so these numbers sit on the same test set as
#     the ensembles (984 rows, support 420/257/307)
#   * epochs are SELECTED on val by macro-F1 rather than fixed at 5
#     (NLP) or 3 (BERT), removing that confound
#   * both families right-pad, and the recurrent readout gathers the
#     true final timestep instead of h_n[-1]
#   * padding_idx=0, and pad positions are masked before the CNN pool
#   * every model exports to disk - none of the originals' BERT-side
#     export blocks were left commented out
#
# Run from the experiment root:
#
#   uv run --with pandas --with openpyxl --with torch \
#          --with transformers --with tensorflow \
#          python "Baseline experiment/canonical rerun/run_baselines_canonical.py"
#
# Useful flags:
#   --smoke                1 epoch, 200 train rows - proves the wiring
#   --compare-weighting    one model weighted vs unweighted, then exit
#   --models keras_gru bert_cnn      run a subset
# ===============================================================

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_baseline import (  # noqa: E402
    OUT_DIR,
    build_bert_inputs,
    build_keras_inputs,
    evaluate,
    load_canonical,
    make_model,
    set_seed,
    train_and_select,
)

HEADS = ["cnn", "gru", "lstm", "rnn"]
FAMILIES = ["keras", "bert"]

PRETTY = {
    "keras": "Keras Word Tokenizer",
    # Deliberately NOT "BERT". These baselines use the BERT tokenizer
    # over a randomly-initialised embedding - there is no BERT encoder
    # in them. The originals' own headers say so; only the figures
    # overclaimed.
    "bert": "BERT Tokenizer",
}


def run_one(family, head, ids, lengths, vocab, y, idx, label_names,
            max_epochs, weighted):
    name = f"{PRETTY[family]} + {head.upper()}"
    print(f"\n=== {name} ===")

    # Re-seed before EVERY model so initialisation does not depend on
    # construction order. The old sweep script seeded once before
    # building all five models, which made each model's init a function
    # of how many were built before it.
    set_seed(42)
    model = make_model(head, vocab, len(label_names))

    t0 = time.time()
    best_state, best_epoch, history = train_and_select(
        model, ids, lengths, y, idx, max_epochs=max_epochs, weighted=weighted
    )
    model.load_state_dict(best_state)

    # Test is read here and only here.
    row, per_class, cm, preds = evaluate(
        model, ids, lengths, y, idx, "test", label_names
    )

    row.update(
        {
            "model": name,
            "family": family,
            "head": head,
            "best_epoch": best_epoch,
            "epochs_searched": max_epochs,
            "val_macro_f1_at_best": max(h["val_macro_f1"] for h in history),
            "class_weighted": weighted,
            "vocab_size": int(vocab),
            "train_seconds": round(time.time() - t0, 1),
        }
    )

    print(
        f"  -> best epoch {best_epoch} | "
        f"test acc {row['accuracy']:.4f} macro-F1 {row['f1_macro']:.4f}"
    )

    slug = f"{family}_{head}"
    with pd.ExcelWriter(os.path.join(OUT_DIR, f"{slug}.xlsx"), engine="openpyxl") as w:
        pd.DataFrame([row]).to_excel(w, sheet_name="metrics", index=False)
        pd.DataFrame(per_class).to_excel(w, sheet_name="per_class", index=False)
        pd.DataFrame(history).to_excel(w, sheet_name="history", index=False)
        pd.DataFrame(
            cm, index=[f"true_{c}" for c in label_names],
            columns=[f"pred_{c}" for c in label_names],
        ).to_excel(w, sheet_name="confusion")
        # Predicted vs human distribution on test - the figure the
        # originals drew, but now backed by a file.
        dist = pd.DataFrame(
            {
                "class": label_names,
                "human_count": np.bincount(
                    np.asarray(y)[idx["test"]], minlength=len(label_names)
                ),
                "model_count": np.bincount(preds, minlength=len(label_names)),
            }
        )
        dist["human_pct"] = dist["human_count"] / dist["human_count"].sum()
        dist["model_pct"] = dist["model_count"] / dist["model_count"].sum()
        dist.to_excel(w, sheet_name="distribution", index=False)

    return row, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compare-weighting", action="store_true")
    ap.add_argument("--unweighted", action="store_true")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset like: keras_gru bert_cnn")
    args = ap.parse_args()

    texts, y, label_names, idx = load_canonical()
    # common_v2 returns numpy str_ objects; coerce so they render as
    # plain strings in the exported sheet headers and index labels.
    label_names = [str(c) for c in label_names]
    print(f"canonical split: " + " ".join(f"{k}={len(v)}" for k, v in idx.items()))
    print(f"labels: {label_names}")

    if args.smoke:
        args.epochs = 1
        idx = {k: v[:200] if k == "train" else v[:100] for k, v in idx.items()}
        # Smoke output goes to a SEPARATE directory. This folder already
        # has one bad experience with fake results sitting at real-looking
        # paths (_smoketest_sandbox_DO_NOT_USE), and consolidate_results.py
        # reads results/Baseline_Canonical_Results.xlsx blind. Keeping the
        # two apart is what stops a 100-row smoke run being cited.
        globals()["OUT_DIR"] = os.path.join(
            os.path.dirname(OUT_DIR), "results_SMOKE_DO_NOT_CITE"
        )
        os.makedirs(OUT_DIR, exist_ok=True)
        print("SMOKE MODE - results meaningless, wiring only")
        print(f"           writing to {OUT_DIR}")

    # Build each family's inputs once; both families cover ALL rows and
    # are indexed by the canonical index arrays, so no re-tokenisation
    # per model and no chance of a split/tokenizer mismatch.
    print("\ntokenising...")
    inputs = {}
    wanted = set(args.models) if args.models else None
    for fam in FAMILIES:
        if wanted and not any(m.startswith(fam + "_") for m in wanted):
            continue
        if fam == "keras":
            ids, lengths, vocab, _ = build_keras_inputs(texts, idx["train"])
        else:
            ids, lengths, vocab, _ = build_bert_inputs(texts)
        inputs[fam] = (ids, lengths, vocab)
        print(f"  {PRETTY[fam]}: vocab={vocab} "
              f"len median={int(np.median(lengths))} max={int(lengths.max())} "
              f"empty={(lengths == 0).sum()}")

    if args.compare_weighting:
        # Settles the one open decision in the plan: does weighting the
        # loss actually help macro-F1 enough to justify departing from
        # the original experiment?
        #
        # Writes to its own directory. This mode trains the same model
        # twice under different loss settings, so pointing it at
        # results/ would overwrite a committed result with the losing
        # half of a comparison.
        globals()["OUT_DIR"] = os.path.join(
            os.path.dirname(OUT_DIR), "results_weighting_comparison"
        )
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"comparison output -> {OUT_DIR}")
        fam, head = "keras", "gru"
        ids, lengths, vocab = inputs[fam]
        out = {}
        for weighted in (False, True):
            row, _ = run_one(fam, head, ids, lengths, vocab, y, idx,
                             label_names, args.epochs, weighted)
            out["weighted" if weighted else "unweighted"] = row
        print("\n=== class-weighting comparison (%s + %s) ===" % (fam, head))
        for k, r in out.items():
            print(f"  {k:>11}: acc {r['accuracy']:.4f}  macro-F1 {r['f1_macro']:.4f}")
        d = out["weighted"]["f1_macro"] - out["unweighted"]["f1_macro"]
        print(f"  delta macro-F1 = {d:+.4f}")
        return

    weighted = not args.unweighted
    rows, per_class_all = [], []
    for fam in FAMILIES:
        if fam not in inputs:
            continue
        ids, lengths, vocab = inputs[fam]
        for head in HEADS:
            if wanted and f"{fam}_{head}" not in wanted:
                continue
            row, pc = run_one(fam, head, ids, lengths, vocab, y, idx,
                              label_names, args.epochs, weighted)
            rows.append(row)
            for r in pc:
                r["model"] = row["model"]
            per_class_all.extend(pc)

    if not rows:
        print("nothing ran")
        return

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    lead = ["model", "family", "head", "accuracy", "f1_macro", "f1_weighted",
            "best_epoch", "class_weighted"]
    df = df[lead + [c for c in df.columns if c not in lead]]

    combined = os.path.join(OUT_DIR, "Baseline_Canonical_Results.xlsx")
    with pd.ExcelWriter(combined, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="summary", index=False)
        pd.DataFrame(per_class_all).to_excel(w, sheet_name="per_class", index=False)

    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump(
            {
                "split": {k: int(len(v)) for k, v in idx.items()},
                "labels": label_names,
                "class_weighted": weighted,
                "epochs_searched": args.epochs,
                "models": len(rows),
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 64)
    print(df[["model", "accuracy", "f1_macro", "best_epoch"]].to_string(index=False))
    print("=" * 64)
    print(f"\nwrote {combined}")


if __name__ == "__main__":
    main()
