"""Build RESULTS_CONSOLIDATED.xlsx from the scattered result workbooks.

Re-runnable: every number is read from a source file, nothing is typed in by
hand. The 'Source file' column on each row says where it came from.
"""
import os
import json
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "RESULTS_CONSOLIDATED.xlsx")

BASE1 = "base essemble model experiment #1 (BERT + CNN & NLP + GRU)"
OPT1 = "optimised essemble model experiment #1 fix"

rows = []


def add(stage, model, branches, split, safe, n, acc, mf1, wf1, src, note="", ckpt=""):
    rows.append({
        "Stage": stage,
        "Model": model,
        "Branches": branches,
        "Branch checkpoints": ckpt,
        "Split scheme": split,
        "Leakage-safe": safe,
        "Test n": n,
        "Accuracy": acc,
        "Macro-F1": mf1,
        "Weighted-F1": wf1,
        "Source file": src,
        "Note": note,
    })


CKPT_V2 = "V2 (epoch Version 2: gru_best.pt / cnn_bert_best.pt)"
CKPT_V3 = "V3 tuned (tuned GRU|CNN BERT model/V3)"
CKPT_LEGACY = "V1 era (epoch_10: gru_keras10.pt / cnn_bert10.pt)"


def rd(rel, sheet):
    return pd.read_excel(os.path.join(ROOT, rel), sheet_name=sheet)


# ---------------------------------------------------------------
# 0. Single-model baselines, CANONICAL split (the reference floor)
#
# Produced by "Baseline experiment/canonical rerun/run_baselines_canonical.py".
# These sit on the same 984-row test set as every ensemble row below, so they
# are the floor those numbers should be read against. The pre-canonical 80/20
# versions are kept further down for provenance only.
# ---------------------------------------------------------------
BASE_CANON = "Baseline experiment/canonical rerun/results/Baseline_Canonical_Results.xlsx"
if os.path.exists(os.path.join(ROOT, BASE_CANON)):
    b = rd(BASE_CANON, "summary")
    for _, r in b.iterrows():
        add("0. Baseline (canonical)", r["model"], r["model"],
            "60/20/10/10 canonical", "Yes", 984,
            r["accuracy"], r["f1_macro"], r["f1_weighted"], BASE_CANON,
            f"epoch {int(r['best_epoch'])} selected on val"
            + ("; class-weighted loss" if r.get("class_weighted") else ""),
            "none (single model)")
else:
    print(f"NOTE: {BASE_CANON} not found - canonical baselines omitted. "
          "Run run_baselines_canonical.py to populate them.")

# ---------------------------------------------------------------
# 1. Single-model baselines (own 80/20 split, pre-canonical)
# ---------------------------------------------------------------
src = "Mubin Master Finalize experiment.xlsx"
b = rd(src, "Performance Baseline")
b.columns = [str(c).strip() for c in b.columns]
for _, r in b.iterrows():
    name = str(r["ensemble model"]).strip()
    if not name or name == "nan":
        continue
    add("1. Baseline (single model, superseded)", name, name, "80/20 own split",
        "No - own split", 1967, r["Accuracy"], r["F1-Macro"], None, src,
        "Own train_test_split; test set not comparable to canonical 984-row test "
        "set. Superseded by stage 0. Note the four 'BERT + X' rows are BERT "
        "TOKENIZER + X - there is no BERT encoder in the baseline scripts.")

# ---------------------------------------------------------------
# 2. Legacy ensemble (pre-canonical, LEAKY)
# ---------------------------------------------------------------
b = rd(src, "Performance ensemble")
b.columns = [str(c).strip() for c in b.columns]
for _, r in b.iterrows():
    name = " ".join(str(r["ensemble voting / model"]).split())
    if not name or name == "nan":
        continue
    add("2. Base ensemble V1 (legacy)", name, "NLP+GRU / BERT+CNN or LSTM", "80/20 own split",
        "No - own split", 1967, r["Accuracy"], r["F1-Macro"], None, src)

for ep in [5, 6, 7, 8, 9, 10]:
    rel = f"{OPT1}/ablation study/ablation_study_3way_vs_4way_split_epoch{ep}.xlsx"
    m = rd(rel, "Metrics").set_index("Metric")["Value"]
    add("3. Optimised V1 (legacy, LEAKY)", f"3-way split, epoch {ep}",
        "NLP+GRU + BERT+CNN", "60/30/10 own split", "NO - LEAKAGE",
        984, m["Accuracy"], m["Macro F1"], None, rel,
        "Branches trained on 85% split; ensemble re-split independently -> test rows seen in training",
        CKPT_LEGACY)

b = rd(src, "ablation study performance")
b.columns = [str(c).strip() for c in b.columns]
# The sheet stacks two tables and uses merged cells for the split label, so the
# strategy only appears on each group's first row and the delta table below
# reuses the same columns with no Epoch.
b["Split strategy"] = b["Split strategy"].ffill()
b = b[b["Epoch"].notna() & b["Accuracy"].notna() & b["Macro F1"].notna()]
for _, r in b.iterrows():
    add("3. Optimised V1 (legacy, LEAKY)",
        f"{r['Split strategy']}, epoch {int(r['Epoch'])}",
        "NLP+GRU + BERT+CNN", str(r["Split strategy"]), "NO - LEAKAGE",
        984, r["Accuracy"], r["Macro F1"], None, src,
        "Inflated by leakage - superseded by canonical-split reruns", CKPT_LEGACY)

# ---------------------------------------------------------------
# 4. Canonical-split era (redo/splits.npz, 60/20/10/10)
# ---------------------------------------------------------------
rel = f"{BASE1}/Result/Version 2 results/Base_Ensemble_Results.xlsx"
m = rd(rel, "Overall Metrics").iloc[0]
add("4. Base ensemble V2 (canonical)", "Base ensemble (alpha=0.6, no temperature)",
    "GRU+Keras + CNN+BERT", "60/20/10/10 canonical", "Yes", 984,
    m["Accuracy"], m["F1 (Macro)"], m["F1 (Weighted)"], rel, "", CKPT_V2)

rel = f"{OPT1}/result 4 way V2/Optimized_Ensemble_Results.xlsx"
m = rd(rel, "Overall Metrics").iloc[0]
add("5. Optimised V2 (canonical)", "Alpha grid search + temperature",
    "GRU+Keras + CNN+BERT", "60/20/10/10 canonical", "Yes", 984,
    m["Accuracy"], m["F1 (Macro)"], m["F1 (Weighted)"], rel,
    f"alpha={m['Best Alpha']}, T_cnn={round(m['CNN Temperature'],3)}, T_gru={round(m['GRU Temperature'],3)}",
    CKPT_V2)

# ---------------------------------------------------------------
# 6. Fusion analysis on V2 branches (improvement/, steps 1-3)
# ---------------------------------------------------------------
rel = "improvement/Chapter4_Final_Tables.xlsx"
t1 = rd(rel, "T1 Model Comparison")
for _, r in t1.iterrows():
    name = str(r["Model"]).strip()
    if not name or name == "nan":
        continue
    add("6. Fusion analysis (V2 branches)", name, "GRU+Keras + CNN+BERT",
        "60/20/10/10 canonical", "Yes", 984,
        r["Test accuracy"], r["Test macro-F1"], r["Test weighted-F1"], rel,
        f"CV macro-F1={r['CV macro-F1']}" if pd.notna(r.get("CV macro-F1")) else "not CV-selected",
        CKPT_V2)

# ---------------------------------------------------------------
# 7. Tuned branches V3 (random search) + tuned fusion
# ---------------------------------------------------------------
rel = f"{BASE1}/Result/Version 3 tuned results/Tuned_Ensemble_Results.xlsx"
bt = rd(rel, "Base vs Tuned")
for _, r in bt.iterrows():
    add("7. Tuned V3 (canonical)", str(r["Model"]).strip(), "GRU+Keras + CNN+BERT",
        "60/20/10/10 canonical", "Yes", 984,
        r["Accuracy"], r["F1 (Macro)"], r["F1 (Weighted)"], rel, "", CKPT_V3)

df = pd.DataFrame(rows)
for c in ["Accuracy", "Macro-F1", "Weighted-F1"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").round(4)

# ---------------------------------------------------------------
# Headline: leakage-safe rows only, best first
# ---------------------------------------------------------------
safe = (df[df["Leakage-safe"] == "Yes"]
        .sort_values("Macro-F1", ascending=False)
        .reset_index(drop=True))

# Per-class recall for the models that report it
percls = []
rel = f"{BASE1}/Result/Version 3 tuned results/Tuned_Ensemble_Results.xlsx"
p = rd(rel, "Per-Class PRF")
for _, r in p.iterrows():
    percls.append({"Model": "Tuned ensemble V3", "Sentiment": r["Sentiment"],
                   "Precision": r["Precision"], "Recall": r["Recall"],
                   "F1": r["F1"], "Support": r["Support"], "Source file": rel})
rel = f"{BASE1}/Result/Version 2 results/Base_Ensemble_Results.xlsx"
p = rd(rel, "Per-Class PRF")
for _, r in p.iterrows():
    percls.append({"Model": "Base ensemble V2", "Sentiment": r["Sentiment"],
                   "Precision": r["Precision"], "Recall": r["Recall"],
                   "F1": r["F1"], "Support": r["Support"], "Source file": rel})
rel = "improvement/Chapter4_Final_Tables.xlsx"
p = rd(rel, "T6 Headline Report")
for _, r in p.iterrows():
    lab = str(r.iloc[0])
    if lab in ("negative", "neutral", "positive"):
        percls.append({"Model": "S1 stacker (improvement headline)", "Sentiment": lab,
                       "Precision": r["precision"], "Recall": r["recall"],
                       "F1": r["f1-score"], "Support": r["support"], "Source file": rel})
# Best canonical baseline, so the per-class sheet shows the floor's
# neutral recall alongside the ensembles' - that trade-off is the
# recurring theme in Chapter 4.
if os.path.exists(os.path.join(ROOT, BASE_CANON)):
    s = rd(BASE_CANON, "summary").sort_values("f1_macro", ascending=False)
    best_baseline = s.iloc[0]["model"]
    p = rd(BASE_CANON, "per_class")
    for _, r in p[p["model"] == best_baseline].iterrows():
        percls.append({"Model": f"{best_baseline} (best canonical baseline)",
                       "Sentiment": r["class"], "Precision": r["precision"],
                       "Recall": r["recall"], "F1": r["f1"],
                       "Support": r["support"], "Source file": BASE_CANON})

percls = pd.DataFrame(percls)

# Provenance
with open(os.path.join(ROOT, "redo", "splits_summary.json")) as f:
    sizes = json.load(f)
prov = pd.DataFrame([
    {"Item": "Canonical dataset", "Value": "redo/canonical_dataset.csv (9,835 rows)"},
    {"Item": "Canonical splits", "Value": "redo/splits.npz — " + ", ".join(f"{k}={v}" for k, v in sizes.items())},
    {"Item": "Split seed", "Value": "42, stratified, generated by redo/split_generator.py"},
    {"Item": "Label set", "Value": "negative / neutral / positive (majority_sent)"},
    {"Item": "Text column", "Value": "comment/tweet"},
    {"Item": "Best leakage-safe model", "Value": f"{safe.iloc[0]['Model']} — macro-F1 {safe.iloc[0]['Macro-F1']}"},
    {"Item": "Generated by", "Value": "consolidate_results.py at the experiment root - re-run to refresh"},
])

with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    prov.to_excel(w, sheet_name="00 Provenance", index=False)
    safe.to_excel(w, sheet_name="01 Headline (leakage-safe)", index=False)
    df.to_excel(w, sheet_name="02 All results", index=False)
    percls.to_excel(w, sheet_name="03 Per-class", index=False)
    df[df["Leakage-safe"] != "Yes"].to_excel(w, sheet_name="04 Legacy (do not cite)", index=False)

    for name in w.sheets:
        ws = w.sheets[name]
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 62)
        ws.freeze_panes = "A2"

print("Wrote", OUT)
print("\nHeadline (leakage-safe), top 8:")
print(safe[["Stage", "Model", "Accuracy", "Macro-F1"]].head(8).to_string(index=False))
print(f"\nTotal rows: {len(df)}  |  leakage-safe: {len(safe)}  |  legacy: {len(df) - len(safe)}")
