# Stage-by-stage correctness audit — Stage 1: Baseline experiment

## Context

The thesis compares 8 single-model baselines, then a two-branch ensemble, then an
optimised ensemble. A prior audit established that the ensemble results split into
a leaky pre-canonical era (~0.86 accuracy) and an honest canonical-split era
(0.63–0.67), and that `redo/splits.npz` is the authoritative partition.

This plan covers **Stage 1 only** — the Baseline experiment. Stages 2 (base
ensemble) and 3 (optimised ensemble) get their own plans after Stage 1 closes, per
the agreed one-stage-at-a-time cadence.

Audit scope agreed: methodology soundness, implementation bugs, and cross-script
consistency.

**Intent confirmed by the user:**
- The four `BERT + X` scripts are *meant* to be tokenizer-only comparisons
  (WordPiece vs Keras word tokenisation, model held fixed). The code is therefore
  correct in intent — the **figure and table labels are wrong**, not the models.
- The baseline stage must both motivate the ensemble **and** provide a comparable
  reference floor for Chapter 4. A comparable floor requires the canonical split.
- The four BERT rows must be re-derived so every number traces to a file on disk.

## What the audit found

Verified directly against the source, not inferred:

| # | Finding | Evidence |
|---|---|---|
| 1 | **Padding direction is opposite between families and breaks the recurrent readout.** Keras `pad_sequences` defaults to left-pad; HF `padding="max_length"` right-pads. All recurrent models read `h_n[-1]`, so BERT+GRU/LSTM/RNN read the state after up to 99 `[PAD]` steps. | `NLP/GRU.py:50` vs `BERT/BERT + GRU.py:71`; readout `:87` / `:127` |
| 2 | **`padding_idx` is never set**, so PAD is a trainable non-zero vector that actively overwrites recurrent state and feeds the CNN max-pool. | `nn.Embedding(vocab_size, 128)` at `:116` in all 4 BERT scripts; same in NLP |
| 3 | **`attention_mask` is discarded** — only `["input_ids"]` is kept, so nothing downstream *can* mask. | `BERT + CNN.py:65-83` |
| 4 | **No validation set in any of the 8.** One `train_test_split`, and that same test set drives metrics, figures and exports. The 5-vs-3 epoch choice was made with only test feedback. | `NLP/CNN.py:37-42`, `BERT + CNN.py:55-60` |
| 5 | **`baseline NLP+RNN.py` is a CNN clone at 3 epochs that overwrites the CNN result.** Header `# KERAS WORD TOKENIZER + CNN`; writes the same path as `CNN.py:262`. Timestamps indicate the reported CNN row came from this file, not `CNN.py`. | `NLP/baseline NLP+RNN.py:1`, `:96`, `:264` |
| 6 | **Epoch budget differs by family** — 5 (NLP) vs 3 (BERT), a confound in every NLP-vs-BERT comparison. | `NLP/CNN.py:94` vs `BERT + CNN.py:135` |
| 7 | **No class weights** on a 42.7/26.1/31.2 split, while macro-F1 is the headline metric. | `CrossEntropyLoss()` with no `weight=`, all 8 |
| 8 | **The 4 BERT rows have no on-disk provenance** — export blocks commented out; numbers exist only as literals in chart scripts. | `BERT + CNN.py:295-314` and siblings |
| 9 | **Labels overclaim.** Figures say `BERT + CNN`; the code is BERT *tokenizer* + CNN. | `spider chart generator all.py:12-15` |
| 10 | `sentiment distribution Keras word token.py` feeds Keras word IDs to mBERT at lr 1e-3 with all output commented out — dead code end to end. | `:118`, `:125`, `:148-153` |

**Not bugs** (checked and cleared): the Keras `Tokenizer` is fit *after* the split
in every NLP script, so no tokenizer leakage. `LabelEncoder` on the same column
gives the same `negative/neutral/positive` → 0/1/2 order everywhere. `LSTM.py`
really does build an `nn.LSTM`.

**Environment correction:** the dataset and `Phase 3/` output folder exist under
`Master/initial experiment/`, not `Final experiment/`. The scripts are relocated
copies that were originally run from that working directory — they are not broken.

## Execution moves to the desktop PC

This laptop is a **Snapdragon X Plus (ARM64)**, no CUDA, and TensorFlow has no
Windows-ARM64 build. The user will instead run this stage on the **desktop PC** —
the CUDA machine the original experiments were run on.

Consequences, all simplifying:
- **The TensorFlow port is unnecessary.** Keep the original
  `from tensorflow.keras.preprocessing...` imports; the PC already has the working
  environment that produced the current results.
- No `keras-preprocessing` substitution, no ARM64 wheel risk.
- Compute is trivial either way: all 8 models are ~2.6–4M params (embedding + one
  layer). 5,901 train rows / batch 32 = 185 steps/epoch → **well under 30 minutes
  for all 8** on the GPU.

### Step -1 — Hand off to the PC (do this first)

Project files already sync via OneDrive (confirmed reparse point), so all scripts,
`README.md` and `consolidate_results.py` are already on the PC. Only the Claude Code
conversation state does not travel — it lives outside OneDrive in `~/.claude\`.

Write two files **into the OneDrive project root** so they sync automatically:
1. `CLAUDE.md` — auto-loaded by Claude Code from the working directory on the PC.
   Records: the canonical-split rule, the leakage-era boundary, the folder hazards,
   and the Stage 1 audit findings table above.
2. `STAGE1_BASELINE_PLAN.md` — this plan, so the PC session executes the same spec.

Optionally copy `~/.claude\projects\<project-key>\memory\` (4 files, ~7 KB) to the
same path on the PC for literal memory continuity. The transcript itself
(1.4 MB `.jsonl`) is not worth copying — `CLAUDE.md` carries the durable content.

## Plan (executed on the PC)

### Step 0 — Environment check (gate)

Confirm the PC's existing environment still runs: `torch` sees CUDA, `tensorflow`
and `transformers` import, and `redo/splits.npz` loads. One model, 1 epoch, tiny
subset.

### Step 1 — Shared module

Create `Baseline experiment/canonical rerun/common_baseline.py`.

Reuse the existing canonical-split loader rather than reimplementing it:

```python
sys.path.insert(0, os.path.join(ROOT, "improvement"))
from common_v2 import load_canonical, class_weights, macro_f1, set_seed
```

`improvement/common_v2.py` already provides `load_canonical()` (loads
`redo/canonical_dataset.csv` + `redo/splits.npz`, asserts zero pairwise overlap),
`class_weights()` (inverse-frequency, mean-normalised), `macro_f1()` and
`set_seed()`. The project already uses the `sys.path.insert` idiom in
`improvement/step5_export_logits_v2.py`.

`common_baseline.py` adds only what is baseline-specific:
- `build_keras_inputs(texts, train_idx)` — fit the Keras `Tokenizer` on **train rows
  only**, `num_words=20000`, `maxlen=100`, **right-pad** (`padding='post'`,
  `truncating='post'`), returning ids + lengths.
- `build_bert_inputs(texts)` — `BertTokenizer`, `max_length=100`, right-pad,
  returning ids **and `attention_mask`**.
- The 4 model classes, each taking `padding_idx=0` and a mask/lengths argument.

### Step 2 — Correct the models

| Bug | Fix |
|---|---|
| 1, 2 | Right-pad both families; `nn.Embedding(..., padding_idx=0)`; recurrent models gather the hidden state at the **true final index** rather than `h_n[-1]` |
| 3 | Keep `attention_mask`; zero pad positions before the CNN max-pool, mirroring the idiom already in `common_v2.py:198` (`h = h * mask.unsqueeze(-1)`) |
| 4, 6 | Train on canonical `train`, select the best epoch on canonical `val` by macro-F1, report once on canonical `test`. This removes the 5-vs-3 confound — epochs become a selected quantity, not a fixed guess |
| 7 | `CrossEntropyLoss(weight=class_weights(...))`, matching the ensemble branches' `class_weight: balanced` so the floor is comparable |
| 5 | Do **not** port `baseline NLP+RNN.py`; it is a duplicate. Add a note to the original |
| 8 | Export every model's metrics to disk — no commented-out blocks |

### Step 3 — Run all 8

`{Keras, BERT-tokenizer} × {CNN, GRU, LSTM, RNN}` on the canonical split, seed 42.
Write one workbook per model plus a combined
`Baseline experiment/canonical rerun/Baseline_Canonical_Results.xlsx`.

### Step 4 — Wire into the existing consolidation

Extend `consolidate_results.py` to read the new workbooks as a
`0. Baseline (canonical)` stage marked `Leakage-safe = Yes`, so they land in sheet
`01 Headline (leakage-safe)` alongside the ensemble numbers. This is the
comparable reference floor.

Keep the old 80/20 rows in `04 Legacy (do not cite)` for provenance.

### Step 5 — Fix the labelling

Correct `BERT + X` → `BERT Tokenizer + X` in `spider chart generator all.py:12-15`,
`performance comparison spider chart BERT.py:49`, `README.md` and the consolidated
workbook. Also remove the stale commented arrays at
`NLP/performance comparison.py:14-15`, which provably disagree with the exported
results (micro-F1 must equal accuracy; they don't).

### Open decision to settle during execution

**Class weighting** (Step 2, bug 7) changes the experiment rather than merely fixing
it. Weighted loss makes the baseline a fairer floor under macro-F1 and matches the
ensemble branches. Unweighted preserves continuity with the current thesis text. I
will run **both** for one model and show the gap before applying either across all 8.

## Verification

1. `load_canonical()`'s overlap assertion passes (already enforced in `common_v2.py:58-61`).
2. Split sizes match `redo/splits_summary.json` — train 5901 / val 1967 / cal 983 / test 984.
3. Test-set per-class support is 420 / 257 / 307, matching every other canonical result.
4. Micro-F1 == accuracy for every model — an arithmetic identity in single-label
   multiclass, and a check the old `performance comparison.py` numbers fail.
5. Re-run `consolidate_results.py`; confirm 8 new rows appear in
   `01 Headline (leakage-safe)` and the legacy rows are untouched.
6. Spot-check one model end to end: confirm the selected epoch came from `val`, and
   that `test` is read exactly once.

## Explicitly not doing

- Not modifying the original baseline scripts — they are the provenance record for
  the numbers currently in the thesis. Corrected versions live in a new folder.
- Not renaming the misspelled `essemble` folders — scripts hardcode those paths.
- Not touching Stage 2 or Stage 3 yet.

## After this stage

Stage 2 (base ensemble: `experiment V1/V2/V3`, branch trainers) and Stage 3
(optimised ensemble: 3-way/4-way splits, alpha grid, temperature scaling) each get
their own audit and plan, briefed by you the same way.
