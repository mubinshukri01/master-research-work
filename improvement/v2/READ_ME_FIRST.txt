UNFINISHED SMOKE RUN - NOT RESULTS
==================================

Everything in this folder came from a --smoke invocation of
improvement/step4_tune_branches.py, i.e. 1 trial and 1 epoch per branch.
The script itself prints on exit:

    "SMOKE TEST ONLY - results meaningless. Re-run without --smoke."

Both parameter files record this:

    gru/gru_best_params.json    "smoke": true, n_trials: 1, epochs_per_trial: 1
    xlmr/xlmr_best_params.json  "smoke": true, n_trials: 1, epochs_per_trial: 1

xlmr/xlmr_best.pt is 1.1 GB because it is a genuine xlm-roberta-base
architecture - but it was trained for a single epoch on a single trial. It is a
placeholder, not a tuned model. Its reported val macro-F1 of 0.566 is below the
existing V3 CNN-BERT branch (0.615).

Do not cite these numbers. Do not use these checkpoints in a fusion run.

WHAT IS ACTUALLY FINISHED IN improvement/
-----------------------------------------
Steps 1-3 are complete and real, and produced the Chapter 4 tables:

    step1_export_logits.py     -> logits_dump.npz (+ manifest)
    step2_diagnose_and_fit.py  -> Improved_Ensemble_Analysis.xlsx, figures/
    step3_final_fusion.py      -> Chapter4_Final_Tables.xlsx, step3_summary.json

Steps 4-5 (branch retuning with XLM-R, and re-export of logits against the
retuned branches) were written but never run for real:

    step4_tune_branches.py     -> this folder, smoke only
    step5_export_logits_v2.py  -> never produced logits_dump_v2.npz

TO FINISH THE WORK
------------------
Run step 4 without --smoke (needs a GPU and real time), then step 5, then
re-run steps 2 and 3 against the new logits:

    python improvement/step4_tune_branches.py --branch gru
    python improvement/step4_tune_branches.py --branch xlmr
    python improvement/step5_export_logits_v2.py --install
    python improvement/step2_diagnose_and_fit.py
    python improvement/step3_final_fusion.py

--install backs up the original logits_dump.npz before overwriting it.

Worth knowing before spending the GPU hours: the ceiling analysis in
Chapter4_Final_Tables.xlsx (sheet "T4 Ceiling Analysis") shows that no
reweighting of the current V2 branches can exceed ~0.654 macro-F1, and 24.3% of
test cases are missed by both branches. Retuning the branches - which is what
step 4 does - is the right lever. Fusion tuning alone is close to exhausted.
