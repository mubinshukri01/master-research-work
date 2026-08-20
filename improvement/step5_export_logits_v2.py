# ===============================================================
# step5_export_logits_v2.py
#
# Re-runs step 1 against the tuned branches from step 4 and writes
# logits_dump_v2.npz in the IDENTICAL schema, so step 2 and step 3
# work on the new branches with no code changes.
#
# Usage:
#   python improvement/step5_export_logits_v2.py
#   python improvement/step5_export_logits_v2.py --install
#
# --install additionally copies the result over logits_dump.npz
# (backing up the original first), which is what lets you then run:
#   python improvement/step2_diagnose_and_fit.py
#   python improvement/step3_final_fusion.py
# ===============================================================

import os
import sys
import json
import shutil
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v2 import (  # noqa: E402
    ROOT, WORK_DIR, DEVICE, load_canonical, macro_f1, load_pickle,
    GRUBranch, CNNTransformerBranch,
)
from sklearn.metrics import accuracy_score  # noqa: E402

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = os.path.join(ROOT, "improvement")
BATCH = 32


def load_params(tag):
    p = os.path.join(WORK_DIR, tag, f"{tag}_best_params.json")
    if not os.path.exists(p):
        sys.exit(f"Missing {p}\nRun step 4 for the '{tag}' branch first.")
    with open(p) as f:
        d = json.load(f)
    if d.get("smoke"):
        sys.exit(f"'{tag}' params came from a --smoke run. Re-run step 4 properly.")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true",
                    help="overwrite logits_dump.npz so steps 2-3 use the new branches")
    args = ap.parse_args()

    texts, y, label_names, idx = load_canonical()
    K = len(label_names)
    print("Classes:", label_names, " device:", DEVICE)

    gru_meta = load_params("gru")
    xlm_meta = load_params("xlmr")
    gp, xp = gru_meta["params"], xlm_meta["params"]
    print(f"GRU   val macro-F1 at tuning time: {gru_meta['best_val_macro_f1']:.4f}")
    print(f"XLM-R val macro-F1 at tuning time: {xlm_meta['best_val_macro_f1']:.4f}")

    # ---- GRU branch ----
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    keras_tok = load_pickle(os.path.join(WORK_DIR, "gru", "keras_tokenizer.pkl"))

    gru = GRUBranch(vocab=gp["vocab"], num_classes=K, embed_dim=gp["embed_dim"],
                    hidden=gp["hidden"], layers=gp["layers"],
                    bidirectional=gp["bidirectional"], dropout=gp["dropout"]).to(DEVICE)
    gru.load_state_dict(torch.load(os.path.join(WORK_DIR, "gru", "gru_best.pt"),
                                   map_location=DEVICE))
    gru.eval()

    def gru_tensor(tl):
        return torch.tensor(
            pad_sequences(keras_tok.texts_to_sequences(tl), maxlen=gp["max_len"]),
            dtype=torch.long)

    # ---- XLM-R branch ----
    from transformers import AutoTokenizer
    xtok = AutoTokenizer.from_pretrained(xp["encoder"])
    cnn = CNNTransformerBranch(
        encoder_name=xp["encoder"], num_classes=K, n_filters=xp["n_filters"],
        kernel_sizes=tuple(xp["kernel_sizes"]), dropout=xp["dropout"],
        freeze_embeddings=xp["freeze_embeddings"],
        unfrozen_layers=xp["unfrozen_layers"]).to(DEVICE)
    cnn.load_state_dict(torch.load(os.path.join(WORK_DIR, "xlmr", "xlmr_best.pt"),
                                   map_location=DEVICE))
    cnn.eval()

    # ---- ordered inference, same contract as step 1 ----
    @torch.no_grad()
    def run(part):
        ii = idx[part]
        tl = [texts[i] for i in ii]
        yy = y[ii]
        n = len(tl)
        xk = gru_tensor(tl)
        gl, cl = [], []
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            gl.append(gru(xk[s:e].to(DEVICE)).float().cpu())
            enc = xtok(tl[s:e], return_tensors="pt", padding="max_length",
                       truncation=True, max_length=xp["max_len"])
            cl.append(cnn(enc["input_ids"].to(DEVICE),
                          enc["attention_mask"].to(DEVICE)).float().cpu())
            if (s // BATCH) % 10 == 0:
                print(f"  {part}: {e}/{n}", end="\r")
        print(f"  {part}: {n}/{n} done        ")
        g = torch.cat(gl).numpy().astype(np.float64)
        c = torch.cat(cl).numpy().astype(np.float64)
        assert g.shape == (n, K) and c.shape == (n, K)
        return g, c, yy

    out = {}
    for part in ["val", "cal", "test"]:
        g, c, yy = run(part)
        out[f"gru_logits_{part}"] = g
        out[f"cnn_logits_{part}"] = c
        out[f"y_{part}"] = yy
        out[f"idx_{part}"] = idx[part]

    # ---- sanity: val performance must track what step 4 reported ----
    print("\n===== SANITY CHECK (validation partition) =====")
    ok = True
    for nm, key, ref in [("GRU", "gru", gru_meta["best_val_macro_f1"]),
                         ("XLM-R", "cnn", xlm_meta["best_val_macro_f1"])]:
        f1 = macro_f1(out["y_val"], out[f"{key}_logits_val"].argmax(1))
        drift = abs(f1 - ref)
        print(f"  {nm:6s} val macro-F1 now {f1:.4f}  vs step4 {ref:.4f}  drift {drift:.4f}")
        if drift > 0.02:
            ok = False
    print("  PASS" if ok else "  WARN - large drift. Check tokenizer/params alignment.")

    print("\n===== NEW BRANCH PERFORMANCE (test partition) =====")
    for nm, key in [("GRU", "gru"), ("XLM-R CNN", "cnn")]:
        pr = out[f"{key}_logits_test"].argmax(1)
        print(f"  {nm:10s} accuracy {accuracy_score(out['y_test'], pr):.4f}   "
              f"macro-F1 {macro_f1(out['y_test'], pr):.4f}")
    print("  (previous branches: GRU 0.5459, CNN-BERT 0.6053 macro-F1)")

    dst = os.path.join(OUT_DIR, "logits_dump_v2.npz")
    np.savez_compressed(dst, label_names=np.array(label_names), **out)
    print(f"\nSaved: {dst}")

    if args.install:
        orig = os.path.join(OUT_DIR, "logits_dump.npz")
        bak = os.path.join(OUT_DIR, "logits_dump_v1_backup.npz")
        if os.path.exists(orig) and not os.path.exists(bak):
            shutil.copy2(orig, bak)
            print(f"Backed up original to: {bak}")
        shutil.copy2(dst, orig)
        print("Installed as logits_dump.npz. Now run:")
        print("  python improvement/step2_diagnose_and_fit.py")
        print("  python improvement/step3_final_fusion.py")
        print("\nNote: step2's BASE_ALPHA/expected-value comments refer to the")
        print("OLD branches. The 2x2 attribution section is only meaningful")
        print("for the v1 dump; everything else applies to v2 unchanged.")
    else:
        print("Re-run with --install to point steps 2-3 at these branches.")


if __name__ == "__main__":
    main()
