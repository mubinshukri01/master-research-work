# ===============================================================
# step2_diagnose_and_fit.py
#
# CPU-only. Consumes logits_dump.npz from step 1.
#
# Sections:
#   A. Branch diagnostics  - oracle, disagreement, Q-statistic, ECE
#   B. Temperature scaling - fit on cal, report calibration gain
#   C. 2x2 attribution     - alpha {0.6,0.7} x temperature {off,on}
#   D. Alpha curve         - 5-fold CV with error bars, 1-SE rule
#   E. Fusion model search - scalar alpha -> +class bias -> per-class
#                            weights, selected by CV, not by test
#   F. Significance        - McNemar + bootstrap CI vs base
#   G. Export              - Excel tables + figures for Chapter 4
#
# Run from the "Final experiment" folder:
#     python improvement/step2_diagnose_and_fit.py
# ===============================================================

import os
import json
import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "improvement")
DUMP = os.path.join(OUT_DIR, "logits_dump.npz")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BASE_ALPHA = 0.6      # what the base ensemble actually uses
OLD_OPT_ALPHA = 0.7   # what the current optimized model selected
N_BOOT = 10000
N_FOLDS = 5

d = np.load(DUMP, allow_pickle=True)
label_names = [str(s) for s in d["label_names"]]
K = len(label_names)
print("Classes:", label_names)

L = {p: {"gru": d[f"gru_logits_{p}"], "cnn": d[f"cnn_logits_{p}"], "y": d[f"y_{p}"]}
     for p in ["val", "cal", "test"]}
for p in L:
    print(f"  {p}: n={len(L[p]['y'])}")

NEUTRAL = label_names.index("neutral") if "neutral" in label_names else None

# Self-test: the fast macro-F1 below must agree with sklearn exactly.
_yt = RNG.integers(0, K, 500)
_pt = RNG.integers(0, K, 500)


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------
def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1(y, pred):
    """Vectorised macro-F1 via a bincount confusion matrix.

    Identical output to sklearn's f1_score(average='macro', zero_division=0)
    but ~20x faster, which matters because the bias grid search below
    evaluates it tens of thousands of times. Verified against sklearn in
    the self-test at the bottom of this file.
    """
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(np.float64)
    denom = 2.0 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1e-12), 0.0).mean())


assert abs(macro_f1(_yt, _pt) -
           f1_score(_yt, _pt, average="macro", zero_division=0)) < 1e-12, \
    "fast macro_f1 disagrees with sklearn"
print("Self-test passed: fast macro-F1 matches sklearn.")


def ece(probs, y, n_bins=15):
    """Expected calibration error over max-probability bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        total += (m.sum() / len(y)) * abs(correct[m].mean() - conf[m].mean())
    return total


def nll(probs, y):
    eps = 1e-12
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + eps)))


def fit_temperature(logits, y):
    """Single-parameter temperature minimising NLL. Guo et al. (2017)."""
    def obj(logT):
        return nll(softmax(logits, T=np.exp(logT)), y)
    r = minimize_scalar(obj, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(r.x))


# ===============================================================
# A. BRANCH DIAGNOSTICS
# ===============================================================
print("\n" + "=" * 66)
print("A. BRANCH DIAGNOSTICS (test partition, uncalibrated)")
print("=" * 66)

y_te = L["test"]["y"]
p_gru_raw = softmax(L["test"]["gru"])
p_cnn_raw = softmax(L["test"]["cnn"])
pred_gru = p_gru_raw.argmax(1)
pred_cnn = p_cnn_raw.argmax(1)

cg = pred_gru == y_te
cc = pred_cnn == y_te

oracle = (cg | cc).mean()
both = (cg & cc).mean()
disagree = (pred_gru != pred_cnn).mean()

n11 = (cg & cc).sum()
n00 = (~cg & ~cc).sum()
n10 = (cg & ~cc).sum()
n01 = (~cg & cc).sum()
q_stat = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10 + 1e-12)

diag_rows = [
    {"Metric": "GRU accuracy", "Value": cg.mean()},
    {"Metric": "CNN-BERT accuracy", "Value": cc.mean()},
    {"Metric": "GRU macro-F1", "Value": macro_f1(y_te, pred_gru)},
    {"Metric": "CNN-BERT macro-F1", "Value": macro_f1(y_te, pred_cnn)},
    {"Metric": "Oracle accuracy (either branch correct)", "Value": oracle},
    {"Metric": "Both branches correct", "Value": both},
    {"Metric": "Branch disagreement rate", "Value": disagree},
    {"Metric": "Yule's Q (1 = identical errors)", "Value": q_stat},
    {"Metric": "Headroom: oracle - best branch", "Value": oracle - max(cg.mean(), cc.mean())},
]
for r in diag_rows:
    print(f"  {r['Metric']:<45s} {r['Value']:.4f}")

print("\n  Reading: oracle - best branch is the absolute ceiling for ANY")
print("  fusion rule over these two frozen branches. If it is small, the")
print("  branches fail on the same examples and no weighting can help.")


# ===============================================================
# B. TEMPERATURE SCALING
# ===============================================================
print("\n" + "=" * 66)
print("B. TEMPERATURE SCALING (fitted on calibration, n=%d)" % len(L["cal"]["y"]))
print("=" * 66)

T_gru = fit_temperature(L["cal"]["gru"], L["cal"]["y"])
T_cnn = fit_temperature(L["cal"]["cnn"], L["cal"]["y"])
print(f"  T_gru = {T_gru:.4f}   T_cnn = {T_cnn:.4f}")
print("  (T > 1 means the branch was overconfident)")

calib_rows = []
for nm, lg, T in [("GRU", L["test"]["gru"], T_gru), ("CNN-BERT", L["test"]["cnn"], T_cnn)]:
    p0, p1 = softmax(lg), softmax(lg, T=T)
    calib_rows.append({
        "Branch": nm,
        "Temperature": T,
        "ECE before": ece(p0, y_te), "ECE after": ece(p1, y_te),
        "NLL before": nll(p0, y_te), "NLL after": nll(p1, y_te),
        "Accuracy before": accuracy_score(y_te, p0.argmax(1)),
        "Accuracy after": accuracy_score(y_te, p1.argmax(1)),
    })
calib_df = pd.DataFrame(calib_rows)
print("\n" + calib_df.to_string(index=False))
print("\n  Temperature scaling is monotone per branch, so single-branch")
print("  accuracy cannot change. Any ECE/NLL reduction here is a real,")
print("  reportable result even when accuracy is flat.")


# probability builders
def P(part, T_g=1.0, T_c=1.0):
    return softmax(L[part]["gru"], T=T_g), softmax(L[part]["cnn"], T=T_c)


def fuse(p_gru, p_cnn, w_cnn, bias=None):
    """w_cnn: scalar or per-class array. bias: additive log-space class offset."""
    w = np.asarray(w_cnn, dtype=np.float64)
    out = w * p_cnn + (1.0 - w) * p_gru
    if bias is not None:
        out = out * np.exp(np.asarray(bias, dtype=np.float64))
    return out


# ===============================================================
# C. 2x2 ATTRIBUTION
# ===============================================================
print("\n" + "=" * 66)
print("C. ATTRIBUTION: what actually caused the regression?")
print("=" * 66)
print("  Base and optimized differ in TWO ways at once (alpha 0.6->0.7")
print("  AND temperature off->on). This isolates each.\n")

attr = []
for a in [BASE_ALPHA, OLD_OPT_ALPHA]:
    for tname, tg, tc in [("off", 1.0, 1.0), ("on", T_gru, T_cnn)]:
        pg, pc = P("test", tg, tc)
        pr = fuse(pg, pc, a).argmax(1)
        attr.append({
            "alpha (CNN)": a, "temperature": tname,
            "accuracy": accuracy_score(y_te, pr),
            "macro-F1": macro_f1(y_te, pr),
            "neutral recall": precision_recall_fscore_support(
                y_te, pr, labels=[NEUTRAL], zero_division=0)[1][0] if NEUTRAL is not None else np.nan,
            "config": f"a={a}, T={tname}",
            "_pred": pr,
        })
attr_df = pd.DataFrame(attr).drop(columns=["_pred"])
print(attr_df.to_string(index=False))

preds_store = {r["config"]: r["_pred"] for r in attr}
base_pred = preds_store[f"a={BASE_ALPHA}, T=off"]
old_opt_pred = preds_store[f"a={OLD_OPT_ALPHA}, T=on"]


# ===============================================================
# D. ALPHA CURVE WITH CROSS-VALIDATED ERROR BARS
# ===============================================================
print("\n" + "=" * 66)
print("D. ALPHA CURVE - is the peak real or is it noise?")
print("=" * 66)

alphas = np.linspace(0, 1, 21)
pg_cal, pc_cal = P("cal", T_gru, T_cnn)
y_cal = L["cal"]["y"]

full_curve = np.array([macro_f1(y_cal, fuse(pg_cal, pc_cal, a).argmax(1)) for a in alphas])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(len(y_cal)), y_cal))
fold_curves = np.array([
    [macro_f1(y_cal[te], fuse(pg_cal[te], pc_cal[te], a).argmax(1)) for a in alphas]
    for _, te in folds
])
cv_mean = fold_curves.mean(0)
cv_se = fold_curves.std(0, ddof=1) / np.sqrt(N_FOLDS)

best_i = int(np.argmax(cv_mean))
thresh = cv_mean[best_i] - cv_se[best_i]
within = np.where(cv_mean >= thresh)[0]
alpha_1se = float(alphas[within[np.argmin(np.abs(alphas[within] - 0.5))]])

curve_df = pd.DataFrame({
    "alpha": alphas,
    "macro_f1_full_cal": full_curve,
    "cv_mean": cv_mean,
    "cv_se": cv_se,
    "within_1se_of_best": [i in within for i in range(len(alphas))],
})
print(curve_df.round(4).to_string(index=False))
print(f"\n  Full-cal argmax          : alpha = {alphas[int(np.argmax(full_curve))]:.2f}")
print(f"  CV argmax                : alpha = {alphas[best_i]:.2f}  "
      f"(CV macro-F1 {cv_mean[best_i]:.4f} +/- {cv_se[best_i]:.4f})")
print(f"  Alphas within 1 SE       : {alphas[within].round(2).tolist()}")
print(f"  1-SE selected alpha      : {alpha_1se:.2f}")
print(f"\n  Total curve spread: {100*(full_curve.max()-full_curve.min()):.2f} pp")
print(f"  Mean CV standard error  : {100*cv_se.mean():.2f} pp")
if (full_curve.max() - full_curve.min()) < 2 * cv_se.mean():
    print("  -> The entire curve fits inside the noise band. Alpha is not")
    print("     an informative hyperparameter for these two branches.")


# ===============================================================
# E. FUSION MODEL SEARCH
#
# Three nested model classes of increasing capacity. Selection is by
# 5-fold CV on the calibration partition with a 1-SE rule, NOT by
# test performance. Test is touched exactly once, at the end.
# ===============================================================
print("\n" + "=" * 66)
print("E. FUSION MODEL SEARCH (selection on calibration CV only)")
print("=" * 66)

BIAS_GRID = np.arange(-0.8, 1.21, 0.05)


def fit_M1(pg, pc, yy):
    """Scalar alpha."""
    sc = [macro_f1(yy, fuse(pg, pc, a).argmax(1)) for a in alphas]
    return {"w": float(alphas[int(np.argmax(sc))]), "b": np.zeros(K)}


def fit_M2(pg, pc, yy):
    """Scalar alpha + class-bias vector.

    Class 0's offset is pinned at 0 because argmax is invariant to a
    constant shift across all classes - only K-1 offsets are identifiable.
    """
    best, par = -1.0, None
    logb = np.exp(BIAS_GRID)
    for a in alphas:
        base = fuse(pg, pc, a)
        for i1, e1 in enumerate(logb):
            scaled = base.copy()
            scaled[:, 1] *= e1
            col2 = base[:, 2]
            for i2, e2 in enumerate(logb):
                scaled[:, 2] = col2 * e2
                s = macro_f1(yy, scaled.argmax(1))
                if s > best:
                    best = s
                    b = np.zeros(K)
                    b[1], b[2] = BIAS_GRID[i1], BIAS_GRID[i2]
                    par = {"w": float(a), "b": b}
    return par


def fit_M3(pg, pc, yy, init=None):
    """Per-class fusion weights + class bias, coordinate ascent from M2."""
    cur = init or fit_M2(pg, pc, yy)
    w = np.full(K, cur["w"], dtype=np.float64)
    b = cur["b"].copy()
    best = macro_f1(yy, fuse(pg, pc, w, b).argmax(1))
    for _ in range(4):
        improved = False
        for c in range(K):
            for cand in alphas:
                w2 = w.copy(); w2[c] = cand
                s = macro_f1(yy, fuse(pg, pc, w2, b).argmax(1))
                if s > best + 1e-9:
                    best, w, improved = s, w2, True
        for c in range(1, K):
            for cand in BIAS_GRID:
                b2 = b.copy(); b2[c] = cand
                s = macro_f1(yy, fuse(pg, pc, w, b2).argmax(1))
                if s > best + 1e-9:
                    best, b, improved = s, b2, True
        if not improved:
            break
    return {"w": w, "b": b}


MODELS = {
    "M0 base (alpha=0.6 fixed, no temp)": None,
    "M1 scalar alpha": fit_M1,
    "M2 alpha + class bias": fit_M2,
    "M3 per-class weights + bias": fit_M3,
}

print(f"  Running {N_FOLDS}-fold CV on the calibration partition...")
cv_results = {}
for name, fitter in MODELS.items():
    if fitter is None:
        # M0 fits nothing, so it is scored directly on each held-out fold
        # using uncalibrated probabilities at the fixed base alpha.
        scores = [
            macro_f1(y_cal[te],
                     fuse(softmax(L["cal"]["gru"][te]),
                          softmax(L["cal"]["cnn"][te]),
                          BASE_ALPHA).argmax(1))
            for _, te in folds
        ]
    else:
        scores = []
        for tr, te in folds:
            par = fitter(pg_cal[tr], pc_cal[tr], y_cal[tr])
            scores.append(macro_f1(y_cal[te],
                                   fuse(pg_cal[te], pc_cal[te], par["w"], par["b"]).argmax(1)))
    scores = np.array(scores)
    cv_results[name] = {"mean": scores.mean(), "se": scores.std(ddof=1) / np.sqrt(N_FOLDS)}
    print(f"    {name:<38s} CV macro-F1 {scores.mean():.4f} +/- {scores.std(ddof=1)/np.sqrt(N_FOLDS):.4f}")

cv_df = pd.DataFrame([{"Model": k, "CV macro-F1": v["mean"], "CV SE": v["se"]}
                      for k, v in cv_results.items()])

best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
cut = cv_results[best_name]["mean"] - cv_results[best_name]["se"]
eligible = [k for k in MODELS if cv_results[k]["mean"] >= cut]
selected = eligible[0]  # dict order = increasing complexity
print(f"\n  Best CV model      : {best_name}")
print(f"  1-SE threshold     : {cut:.4f}")
print(f"  Within 1 SE        : {eligible}")
print(f"  SELECTED (simplest within 1 SE): {selected}")

# refit selected + all variants on FULL calibration, then score test once
pg_te, pc_te = P("test", T_gru, T_cnn)
final_rows, final_preds = [], {}

final_rows.append({"Model": "M0 base (alpha=0.6, no temp)", "params": "alpha=0.60",
                   "pred": base_pred})
final_rows.append({"Model": "Old optimized (alpha=0.7, temp)", "params":
                   f"alpha=0.70, T_gru={T_gru:.3f}, T_cnn={T_cnn:.3f}",
                   "pred": old_opt_pred})
final_rows.append({"Model": "1-SE alpha (temp)", "params": f"alpha={alpha_1se:.2f}",
                   "pred": fuse(pg_te, pc_te, alpha_1se).argmax(1)})

fitted_params = {}
for name, fitter in MODELS.items():
    if fitter is None:
        continue
    par = fitter(pg_cal, pc_cal, y_cal)
    fitted_params[name] = {"w": np.atleast_1d(par["w"]).tolist(), "b": par["b"].tolist()}
    final_rows.append({
        "Model": name,
        "params": f"w={np.round(np.atleast_1d(par['w']),3).tolist()}, b={np.round(par['b'],3).tolist()}",
        "pred": fuse(pg_te, pc_te, par["w"], par["b"]).argmax(1),
    })


# ===============================================================
# F. SIGNIFICANCE
# ===============================================================
def mcnemar(y, p1, p2):
    c1, c2 = p1 == y, p2 == y
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    if b + c == 0:
        return b, c, 1.0
    p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, p


# One shared set of bootstrap resamples, drawn once and reused for every
# model. This makes the intervals a PAIRED bootstrap: each replicate scores
# the candidate and the base on the identical resampled rows, so the CI
# reflects the difference between models rather than the sum of their
# independent sampling noise.
BOOT_IDX = RNG.integers(0, len(y_te), size=(N_BOOT, len(y_te)), dtype=np.int32)


def boot_ci(y, p_ref, p_new):
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = BOOT_IDX[i]
        ys = y[s]
        d[i] = macro_f1(ys, p_new[s]) - macro_f1(ys, p_ref[s])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


print("\n" + "=" * 66)
print("F. TEST-SET RESULTS (each model scored exactly once)")
print("=" * 66)

summary = []
for r in final_rows:
    pr = r["pred"]
    final_preds[r["Model"]] = pr
    pm, rm, fm, _ = precision_recall_fscore_support(y_te, pr, average="macro", zero_division=0)
    pc_, rc_, fc_, _ = precision_recall_fscore_support(
        y_te, pr, labels=range(K), zero_division=0)
    row = {
        "Model": r["Model"], "Params": r["params"],
        "Accuracy": accuracy_score(y_te, pr),
        "Macro-P": pm, "Macro-R": rm, "Macro-F1": fm,
        "Weighted-F1": f1_score(y_te, pr, average="weighted", zero_division=0),
    }
    for i, cn in enumerate(label_names):
        row[f"{cn} R"] = rc_[i]
        row[f"{cn} F1"] = fc_[i]
    if r["Model"] != "M0 base (alpha=0.6, no temp)":
        b, c, p = mcnemar(y_te, base_pred, pr)
        lo, hi = boot_ci(y_te, base_pred, pr)
        row.update({"d Macro-F1 vs base": fm - macro_f1(y_te, base_pred),
                    "95% CI low": lo, "95% CI high": hi,
                    "McNemar b": b, "McNemar c": c, "McNemar p": p})
    summary.append(row)

sum_df = pd.DataFrame(summary)
show = ["Model", "Accuracy", "Macro-F1"] + [f"{c} R" for c in label_names] + \
       ["d Macro-F1 vs base", "95% CI low", "95% CI high", "McNemar p"]
print(sum_df[[c for c in show if c in sum_df.columns]].round(4).to_string(index=False))

best_final = sum_df.loc[sum_df["Macro-F1"].idxmax(), "Model"]
print(f"\n  Highest test macro-F1: {best_final}")
print(f"  CV-SELECTED model    : {selected}")
print("\n  Report the CV-selected model as your headline. If a different")
print("  model tops the test set, that gap is selection noise - quoting it")
print("  as your result would repeat the error this analysis exists to fix.")


# ===============================================================
# G. EXPORT
# ===============================================================
XL = os.path.join(OUT_DIR, "Improved_Ensemble_Analysis.xlsx")
with pd.ExcelWriter(XL, engine="openpyxl") as w:
    sum_df.to_excel(w, sheet_name="Test Results", index=False)
    pd.DataFrame(diag_rows).to_excel(w, sheet_name="Branch Diagnostics", index=False)
    calib_df.to_excel(w, sheet_name="Calibration", index=False)
    attr_df.to_excel(w, sheet_name="2x2 Attribution", index=False)
    curve_df.to_excel(w, sheet_name="Alpha Curve CV", index=False)
    cv_df.to_excel(w, sheet_name="Model Selection CV", index=False)
    for nm, pr in final_preds.items():
        safe = nm[:28].replace("/", "-").replace("=", "").replace(",", "")
        pd.DataFrame(classification_report(
            y_te, pr, target_names=label_names, output_dict=True, zero_division=0
        )).T.reset_index().to_excel(w, sheet_name=f"CR {safe}"[:31], index=False)

# figure 1: alpha curve with CV band
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.fill_between(alphas, cv_mean - cv_se, cv_mean + cv_se, alpha=0.22,
                color="#888780", label="CV mean +/- 1 SE")
ax.plot(alphas, cv_mean, "-", color="#2a78d6", lw=2, label="CV mean macro-F1")
ax.plot(alphas, full_curve, "--", color="#eb6834", lw=1.4, label="Full-cal macro-F1")
ax.axvline(BASE_ALPHA, color="#1baf7a", ls=":", lw=1.6, label=f"base alpha={BASE_ALPHA}")
ax.axvline(OLD_OPT_ALPHA, color="#e34948", ls=":", lw=1.6, label=f"old selected={OLD_OPT_ALPHA}")
ax.axvline(alpha_1se, color="#4a3aa7", ls="-.", lw=1.6, label=f"1-SE alpha={alpha_1se:.2f}")
ax.set_xlabel("alpha (weight on CNN-BERT)")
ax.set_ylabel("Calibration macro-F1")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "alpha_curve_with_ci.png"), dpi=300)
plt.close(fig)

# figure 2: per-class recall comparison
fig, ax = plt.subplots(figsize=(7.2, 4.4))
names = list(final_preds.keys())
x = np.arange(K)
wdt = 0.8 / len(names)
for i, nm in enumerate(names):
    rc_ = precision_recall_fscore_support(y_te, final_preds[nm], labels=range(K),
                                          zero_division=0)[1]
    ax.bar(x + i * wdt, rc_, wdt, label=nm[:30])
ax.set_xticks(x + 0.4 - wdt / 2)
ax.set_xticklabels(label_names)
ax.set_ylabel("Recall")
ax.legend(fontsize=7)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "per_class_recall.png"), dpi=300)
plt.close(fig)

# figure 3: confusion matrix of the CV-selected model
sel_pred = final_preds.get(selected, base_pred)
cm = confusion_matrix(y_te, sel_pred, labels=range(K))
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(cm, cmap="Blues")
for i in range(K):
    for j in range(K):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(K)); ax.set_xticklabels(label_names)
ax.set_yticks(range(K)); ax.set_yticklabels(label_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "confusion_selected.png"), dpi=300)
plt.close(fig)

with open(os.path.join(OUT_DIR, "step2_selection.json"), "w") as f:
    json.dump({
        "T_gru": T_gru, "T_cnn": T_cnn,
        "alpha_1se": alpha_1se,
        "cv_results": {k: {"mean": float(v["mean"]), "se": float(v["se"])}
                       for k, v in cv_results.items()},
        "cv_selected_model": selected,
        "fitted_params": fitted_params,
        "oracle_accuracy": float(oracle),
        "disagreement_rate": float(disagree),
        "q_statistic": float(q_stat),
    }, f, indent=2)

print(f"\nSaved: {XL}")
print(f"Saved figures to: {FIG_DIR}")
