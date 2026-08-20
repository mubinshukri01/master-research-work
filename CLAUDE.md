# CLAUDE.md — Final Experiment (DRSE)

Manglish (Malay-English code-mixed) tweet sentiment → **negative / neutral /
positive**. Two-branch ensemble: CNN head over BERT (`bert-base-cased`) fused with
a GRU over Keras word-tokenised sequences, `alpha * CNN + (1 - alpha) * GRU`.

`README.md` in this folder is the full reference — folder guide, script→output map,
results tables. Read it before answering questions about results. This file records
only the rules and the current state of work.

---

## Hard rules

1. **Never call `train_test_split`.** The partition comes from `redo/splits.npz`
   and nowhere else — train 5901 / val 1967 / cal 983 / test 984, seed 42,
   stratified. Load it via `load_canonical()` in `improvement/common_v2.py`, which
   asserts zero pairwise overlap.
2. **The test set is touched once**, at the very end, for reporting only. Epoch
   selection, alpha, and temperature are all fitted on `val` (or `cal`).
3. **Seed 42 everywhere.**
4. **Headline metric is macro-F1**, not accuracy — the classes are imbalanced
   (test support 420 / 257 / 307).
5. Columns are `comment/tweet` (text) and `majority_sent` (label). `LabelEncoder`
   gives `negative/neutral/positive` → 0/1/2.
6. Run scripts **from this folder** — most resolve paths against the working
   directory. (`common_v2.py` is the exception: it resolves from `__file__`.)

## Two eras — do not mix numbers across them

- **Pre-fix (leaky), ~0.86 accuracy.** Branch checkpoints and ensemble scripts each
  called their own `train_test_split`, so ~85% of "test" rows had been seen in
  branch training. These are artefacts, not results.
- **Post-fix (canonical), ~0.63–0.67 accuracy.** Everything loading `redo/splits.npz`.

`Mubin Master Finalize experiment.xlsx` still holds pre-fix numbers in its
*Performance optimized ensembled* and *ablation study performance* sheets. The
`04 Legacy (do not cite)` sheet of `RESULTS_CONSOLIDATED.xlsx` lists every affected row.

Established conclusion: the fusion ceiling on the V2 branches is ~0.654 macro-F1
while the base was already 0.622, so **branch quality, not fusion weights, is the
binding constraint.**

## Two folders that look real and are not

- `_smoketest_sandbox_DO_NOT_USE/` — fakes TensorFlow and the BERT weights, and its
  `redo/canonical_dataset.csv` is 420 synthetic rows. It mirrors the real directory
  layout exactly, so its checkpoints sit at real-looking paths. Never copy from it.
- `improvement/v2/` — steps 4–5 were only ever run with `--smoke` (1 trial, 1 epoch).
  Both `*_best_params.json` carry `"smoke": true`. Steps 1–3 are real; 4–5 are not.

## Machines

- **Laptop** (this profile, `cloud`): Snapdragon X Plus, **ARM64, no CUDA**.
  Verified working: `torch` 2.13.0+cpu, `transformers`, and **TensorFlow 2.21.0**
  — but TF needs **Python 3.12**, since it ships no `cp314` wheels and uv otherwise
  defaults to 3.14. Pin it with `-p 3.12`. CPU-only, so the small baseline models
  are fine here (~2 min each) but nothing involving a BERT *encoder* is.
- **Desktop PC**: the CUDA machine the experiments were run on. Anything that
  fine-tunes an encoder (branch training, V3 tuning, `improvement/step1`) runs
  there. The project folder syncs via OneDrive, so scripts authored on the laptop
  are immediately runnable on the PC.

Working invocation on the laptop:

```bash
uv run --no-project -p 3.12 --with torch --with transformers --with tensorflow --with pandas --with scikit-learn --with openpyxl python "Baseline experiment/canonical rerun/run_baselines_canonical.py"
```

Old virtualenv `…/initial experiment/archieve/tf_env` is broken (points at the
retired `MUBIN` profile). Use `uv run --with …` instead.

---

## Current work: Stage 1 of a three-stage correctness audit

Auditing all experiment code stage by stage — Baseline → base ensemble → optimised
ensemble — for methodology soundness, implementation bugs, and cross-script
consistency. **One stage at a time, with the user briefing intent before each.**

Stage 1 spec: `STAGE1_BASELINE_PLAN.md`.

### Confirmed intent for Stage 1

- The four `BERT + X` baselines are **meant** to be tokenizer-only comparisons
  (WordPiece vs Keras word tokenisation, model held fixed at a from-scratch
  embedding). The code is correct in intent — **the figure/table labels are wrong**,
  not the models. Label them `BERT Tokenizer + X`.
- The baseline stage must motivate the ensemble **and** provide a reference floor
  comparable to the ensemble numbers → hence the canonical-split re-run.

### Stage 1 audit findings (verified against source)

| # | Finding | Evidence |
|---|---|---|
| 1 | Padding direction is opposite between families and breaks the recurrent readout. Keras `pad_sequences` defaults to left-pad; HF `padding="max_length"` right-pads. All recurrent models read `h_n[-1]`, so BERT+GRU/LSTM/RNN read state after up to 99 `[PAD]` steps. | `NLP/GRU.py:50` vs `BERT/BERT + GRU.py:71`; readout `:87` / `:127` |
| 2 | `padding_idx` is never set, so PAD is a trainable non-zero vector. | `nn.Embedding(vocab_size, 128)` at `:116`, all 4 BERT scripts |
| 3 | `attention_mask` is discarded — only `["input_ids"]` kept. | `BERT + CNN.py:65-83` |
| 4 | No validation set in any of the 8; the test set drives metrics, figures and exports. | `NLP/CNN.py:37-42`, `BERT + CNN.py:55-60` |
| 5 | `baseline NLP+RNN.py` is a CNN clone at 3 epochs that overwrites the CNN result. | `:1`, `:96`, `:264` (same path as `CNN.py:262`) |
| 6 | Epoch budget differs by family — 5 (NLP) vs 3 (BERT). | `NLP/CNN.py:94` vs `BERT + CNN.py:135` |
| 7 | No class weights, on a 42.7/26.1/31.2 split, with macro-F1 as headline. | all 8 |
| 8 | The 4 BERT rows have no on-disk provenance — exports commented out, numbers only as literals in chart scripts. | `BERT + CNN.py:295-314` |
| 9 | Figures overclaim `BERT + CNN` where the code is BERT *tokenizer* + CNN. | `spider chart generator all.py:12-15` |
| 10 | `sentiment distribution Keras word token.py` feeds Keras word IDs to mBERT at lr 1e-3, all output commented out — dead code. | `:118`, `:125`, `:148-153` |

**Checked and cleared (not bugs):** the Keras `Tokenizer` is fit *after* the split
in every NLP script — no tokenizer leakage. `LabelEncoder` order is consistent
across all 8. `LSTM.py` really does build an `nn.LSTM`.

**Path note:** the baseline scripts' dataset and `Phase 3/` output folder live under
`Master/initial experiment/`, not here. They are relocated copies, not broken.

### Stage 1 status: COMPLETE

- [x] Audit complete, findings verified
- [x] `Baseline experiment/canonical rerun/common_baseline.py`
- [x] `Baseline experiment/canonical rerun/run_baselines_canonical.py`
- [x] All 8 run on the canonical split (laptop CPU, class-weighted, 10 epochs searched)
- [x] `consolidate_results.py` gained a `0. Baseline (canonical)` stage
- [x] `BERT + X` → `BERT Tokenizer + X` labelling fixed

**The reference floor** (984-row canonical test set, epoch selected on val):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Keras Word Tokenizer + LSTM | 0.5996 | **0.5896** |
| BERT Tokenizer + GRU | 0.5823 | 0.5737 |
| Keras Word Tokenizer + GRU | 0.5843 | 0.5721 |
| BERT Tokenizer + CNN | 0.5854 | 0.5702 |
| Keras Word Tokenizer + CNN | 0.5925 | 0.5696 |
| BERT Tokenizer + LSTM | 0.5711 | 0.5619 |
| Keras Word Tokenizer + RNN | 0.5254 | 0.5167 |
| BERT Tokenizer + RNN | 0.4421 | 0.4385 |

Best ensemble (tuned V3) is 0.6664 → **+0.0768 macro-F1 over the best single model.**

**Class weighting — decided, and the reason matters.** Weighted vs unweighted on
Keras+GRU: macro-F1 0.5721 vs 0.5688 (+0.0033), accuracy 0.5843 vs 0.5864
(−0.0021). That is within single-seed noise. The 8 numbers above use weighted loss
for **consistency with the ensemble branches** (`class_weight: balanced`), not
because it measurably helps. Do not claim an improvement from it.

**Two findings to carry into Stage 2:**
1. The corrected Keras+GRU baseline (0.5721) beats the thesis's V2 GRU *branch*
   (0.5459) at the same architecture. The delta is the methodology fix — weighted
   loss plus epoch selection on val, neither of which the branches had. Supports
   the existing "branch quality is the binding constraint" conclusion and implies
   recoverable headroom.
2. Neutral is the weak class at baseline too (recall 0.4942 vs 0.6095 negative /
   0.6743 positive), so the neutral collapse is **inherited, not introduced by
   fusion.**

Originals in `Baseline experiment/` are **not** to be modified — they are the
provenance record for the numbers currently in the thesis. Corrected versions live
in `canonical rerun/`.

### Next: Stage 2 (base ensemble)

`experiment V1/V2/V3` plus the branch trainers. Not started. Ask the user what each
version was meant to establish before auditing, same as Stage 1.
