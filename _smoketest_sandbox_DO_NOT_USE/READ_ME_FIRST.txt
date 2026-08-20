SMOKE-TEST SANDBOX - NOT RESULTS
================================

This folder was named "outputs/" until 2026-08-20. Nothing in it is a result.

_smoketest_harness.py exists only to prove the three V3 scripts execute end to
end without a GPU, without TensorFlow, and without downloading BERT weights.
It does this by faking them:

  * tensorflow.keras.preprocessing  -> pure-Python stand-ins
  * BertTokenizerFast / BertModel   -> a tiny randomly-initialised BERT
  * search budgets                  -> shrunk so a run takes seconds

WHY THIS FOLDER IS A HAZARD
---------------------------
It mirrors the real directory layout exactly. These paths look real and are not:

  base essemble model experiment #1 (...)/optimized essemble model experiment/
      tuned GRU keras model/V3/gru_best.pt          <- FAKE (random weights)
      tuned GRU keras model/V3/gru_best_config.json <- FAKE
      tuned GRU keras model/V3/GRU_Tuning_Trials.xlsx <- FAKE

  redo/canonical_dataset.csv   <- 420 synthetic rows, 2 columns
                                  (the real one has 9,835 rows, 12 columns)
  redo/splits.npz              <- splits over those 420 synthetic rows

DO NOT copy anything from this folder into the main tree.
DO NOT cite any number produced from it.

The real equivalents live at the same paths from the experiment root, without
the "_smoketest_sandbox_DO_NOT_USE/" prefix.

The three real V3 scripts that used to sit here were moved into the main tree
on 2026-08-20:

  Tune CNN + BERT Tokenizer V3.py
      -> base essemble model experiment #1 (...)/optimized essemble model
         experiment/tuned CNN BERT model/
  Tune GRU + Keras Word Tokenizer V3.py
      -> base essemble model experiment #1 (...)/optimized essemble model
         experiment/tuned GRU keras model/
  experiment V3 (tuned ensemble model).py
      -> base essemble model experiment #1 (...)/

They resolve their paths relative to the working directory, so they still run
unchanged as long as you run them from the experiment root.

This folder is safe to delete once you no longer need the harness.
