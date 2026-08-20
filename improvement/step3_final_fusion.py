# ===============================================================
# step3_final_fusion.py
#
# CPU-only. Consumes logits_dump.npz from step 1.
#
# Closes out the fusion line of work. Adds a stacked meta-learner
# (per-sample branch selection) to the candidate set, selects among
# ALL candidates by nested cross-validation on the calibration
# partition, and produces the complete Chapter 4 table set.
#
# Reporting discipline: the headline model is the one selected by
# calibration CV. Test macro-F1 is reported for every candidate for
# transparency, but the CV winner is the claim. Choosing whichever
# row happens to top the test column would reproduce exactly the
# selection error this whole analysis exists to correct.
#
# Run from the "Final experiment" folder:
#     python improvement/step3_final_fusion.py
# ===============================================================

import os
import json
import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "improvement")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BASE_ALPHA = 0.6
N_BOOT = 10000
N_FOLDS = 5

d = np.load(os.path.join(OUT_DIR, "logits_dump.npz"), allow_pickle=True)
label_names = [str(s) for s in d["label_names"]]
K = len(label_names)
L = {p: {"g": d[f"gru_logits_{p}"], "c": d[f"cnn_logits_{p}"], "y": d[f"y_{p}"]}
     for p in ["val", "cal", "test"]}
y_cal, y_te, y_val = L["cal"]["y"], L["test"]["y"], L["val"]["y"]


def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macro_f1(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(np.float64)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0).mean())


def nll(p, y):
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def ece(p, y, n_bins=15):
    conf, pred = p.max(1), p.argmax(1)
    corr = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    t = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            t += (m.sum() / len(y)) * abs(corr[m].mean() - conf[m].mean())
    return t


def fit_T(lg, y):
    r = minimize_scalar(lambda t: nll(softmax(lg, np.exp(t)), y),
                        bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


T_g = fit_T(L["cal"]["g"], y_cal)
T_c = fit_T(L["cal"]["c"], y_cal)
print(f"Temperatures (fitted on calibration): T_gru={T_g:.4f}  T_cnn={T_c:.4f}")


# ---------------------------------------------------------------
# Stacker features
#
# Beyond the 6 raw class probabilities we add per-branch confidence
# and entropy plus an agreement flag. These are what let the meta
# learner decide WHICH branch to trust on a given sample, which is
# the only mechanism that can reach the oracle gap - fixed weights
# by construction cannot.
# ---------------------------------------------------------------
def features(part, rich=True):
    g, c = softmax(L[part]["g"], T_g), softmax(L[part]["c"], T_c)
    X = [g, c]
    if rich:
        eg = -(g * np.log(g + 1e-12)).sum(1, keepdims=True)
        ec = -(c * np.log(c + 1e-12)).sum(1, keepdims=True)
        X += [g.max(1, keepdims=True), c.max(1, keepdims=True), eg, ec,
              (g.argmax(1) == c.argmax(1)).astype(float).reshape(-1, 1)]
    return np.hstack(X)


def make_stacker(C, balanced=True):
    return LogisticRegression(C=C, max_iter=3000,
                              class_weight="balanced" if balanced else None)


# ---------------------------------------------------------------
# Nested CV: the inner loop picks C and the feature set, the outer
# loop estimates generalisation of that WHOLE procedure. Selecting C
# on the same folds used to report the score would leak the choice
# into the estimate - the mistake that produced alpha=0.7.
# ---------------------------------------------------------------
GRID = [(rich, C) for rich in [False, True] for C in [0.05, 0.2, 0.5, 2.0, 5.0]]


def nested_cv(part, seed=42):
    y = L[part]["y"]
    Xs = {r: features(part, r) for r in [False, True]}
    outer = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    scores, picks = [], []
    for tr, te in outer.split(np.zeros(len(y)), y):
        inner = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed + 1)
        best, bcfg = -1, None
        for rich, C in GRID:
            X = Xs[rich]
            s = [macro_f1(y[tr][ite],
                          make_stacker(C).fit(X[tr][itr], y[tr][itr]).predict(X[tr][ite]))
                 for itr, ite in inner.split(np.zeros(len(tr)), y[tr])]
            if np.mean(s) > best:
                best, bcfg = np.mean(s), (rich, C)
        X = Xs[bcfg[0]]
        scores.append(macro_f1(y[te], make_stacker(bcfg[1]).fit(X[tr], y[tr]).predict(X[te])))
        picks.append(bcfg)
    return np.array(scores), picks


def flat_cv(part, rich, C, seed=42):
    y = L[part]["y"]
    X = features(part, rich)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    return np.array([macro_f1(y[te], make_stacker(C).fit(X[tr], y[tr]).predict(X[te]))
                     for tr, te in skf.split(X, y)])


print("\n" + "=" * 66)
print("NESTED CV - stacked meta-learner on the calibration partition")
print("=" * 66)
nsc, npicks = nested_cv("cal")
print(f"  Nested CV macro-F1: {nsc.mean():.4f} +/- {nsc.std(ddof=1)/np.sqrt(N_FOLDS):.4f}")
print(f"  Inner-loop picks per fold: {npicks}")

inner_best, inner_cfg = -1, None
for rich, C in GRID:
    m = flat_cv("cal", rich, C).mean()
    if m > inner_best:
        inner_best, inner_cfg = m, (rich, C)
print(f"  Config refit on full calibration: rich={inner_cfg[0]}, C={inner_cfg[1]}")


# ---------------------------------------------------------------
# Candidate set. M0-M3 params come from step 2's calibration fits.
# ---------------------------------------------------------------
pg_t, pc_t = softmax(L["test"]["g"], T_g), softmax(L["test"]["c"], T_c)
pg_0, pc_0 = softmax(L["test"]["g"]), softmax(L["test"]["c"])


def fuse(pg, pc, w, b=None):
    o = np.asarray(w) * pc + (1 - np.asarray(w)) * pg
    return o if b is None else o * np.exp(np.asarray(b))


b_m2 = np.array([0.0, 0.30, -0.45])
X_cal_fin = features("cal", inner_cfg[0])
stk_cal = make_stacker(inner_cfg[1]).fit(X_cal_fin, y_cal)

X_vc = np.vstack([features("val", inner_cfg[0]), X_cal_fin])
y_vc = np.concatenate([y_val, y_cal])
stk_vc = make_stacker(inner_cfg[1]).fit(X_vc, y_vc)

CANDIDATES = {
    "GRU branch alone":            pg_0.argmax(1),
    "CNN-BERT branch alone":       pc_0.argmax(1),
    "M0 base ensemble (a=0.6)":    fuse(pg_0, pc_0, BASE_ALPHA).argmax(1),
    "Old optimized (a=0.7, T)":    fuse(pg_t, pc_t, 0.7).argmax(1),
    "M1 fitted scalar alpha":      fuse(pg_t, pc_t, 0.75).argmax(1),
    "M2 alpha + class bias":       fuse(pg_t, pc_t, 0.40, b_m2).argmax(1),
    "M3 per-class weights + bias": fuse(pg_t, pc_t, np.array([0.40, 0.40, 0.45]), b_m2).argmax(1),
    "S1 stacker (cal only)":       stk_cal.predict(features("test", inner_cfg[0])),
    "S2 stacker (val+cal)":        stk_vc.predict(features("test", inner_cfg[0])),
}

CV_SCORES = {
    "M0 base ensemble (a=0.6)": (0.5658, 0.0130),
    "M1 fitted scalar alpha": (0.5552, 0.0140),
    "M2 alpha + class bias": (0.5903, 0.0127),
    "M3 per-class weights + bias": (0.5887, 0.0110),
    "S1 stacker (cal only)": (float(nsc.mean()), float(nsc.std(ddof=1) / np.sqrt(N_FOLDS))),
}

base_pred = CANDIDATES["M0 base ensemble (a=0.6)"]
BOOT = RNG.integers(0, len(y_te), size=(N_BOOT, len(y_te)), dtype=np.int32)


def mcnemar(y, p1, p2):
    c1, c2 = p1 == y, p2 == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    return b, c, (1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue))


def boot_ci(p_new):
    dv = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = BOOT[i]
        ys = y_te[s]
        dv[i] = macro_f1(ys, p_new[s]) - macro_f1(ys, base_pred[s])
    return float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))


print("\n" + "=" * 66)
print("FINAL COMPARISON (test scored once per candidate)")
print("=" * 66)

rows = []
for nm, pr in CANDIDATES.items():
    pm, rm, fm, _ = precision_recall_fscore_support(y_te, pr, average="macro", zero_division=0)
    _, rc, _, _ = precision_recall_fscore_support(y_te, pr, labels=range(K), zero_division=0)
    r = {"Model": nm,
         "CV macro-F1": CV_SCORES.get(nm, (np.nan,))[0],
         "CV SE": CV_SCORES.get(nm, (np.nan, np.nan))[1] if nm in CV_SCORES else np.nan,
         "Test accuracy": accuracy_score(y_te, pr),
         "Test macro-F1": fm,
         "Test weighted-F1": f1_score(y_te, pr, average="weighted", zero_division=0)}
    for i, cn in enumerate(label_names):
        r[f"R {cn}"] = rc[i]
    if nm != "M0 base ensemble (a=0.6)":
        b, c, p = mcnemar(y_te, base_pred, pr)
        lo, hi = boot_ci(pr)
        r.update({"d vs base": fm - macro_f1(y_te, base_pred),
                  "CI low": lo, "CI high": hi, "McNemar p": p})
    rows.append(r)

res = pd.DataFrame(rows)
cols = ["Model", "CV macro-F1", "Test accuracy", "Test macro-F1"] + \
       [f"R {c}" for c in label_names] + ["d vs base", "CI low", "CI high", "McNemar p"]
print(res[cols].round(4).to_string(index=False))

cv_only = {k: v for k, v in CV_SCORES.items()}
winner = max(cv_only, key=lambda k: cv_only[k][0])
print(f"\n  CV-SELECTED HEADLINE MODEL: {winner}")
print(f"  Test macro-F1 topped by   : {res.loc[res['Test macro-F1'].idxmax(), 'Model']}")

# per-class McNemar for the headline model
print("\n  Per-class McNemar, headline vs base:")
hp = CANDIDATES[winner]
pc_rows = []
for i, cn in enumerate(label_names):
    m = y_te == i
    cb, cn_ = base_pred[m] == i, hp[m] == i
    b, c = int((cb & ~cn_).sum()), int((~cb & cn_).sum())
    p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    pc_rows.append({"Class": cn, "Recall base": cb.mean(), "Recall headline": cn_.mean(),
                    "Delta": cn_.mean() - cb.mean(), "base-only": b, "headline-only": c,
                    "McNemar p": p})
    print(f"    {cn:9s} {cb.mean():.4f} -> {cn_.mean():.4f}  ({b:3d}/{c:3d})  p={p:.3g}")

# calibration table
calib = pd.DataFrame([
    {"Branch": nm, "Temperature": T,
     "ECE before": ece(softmax(lg), y_te), "ECE after": ece(softmax(lg, T), y_te),
     "NLL before": nll(softmax(lg), y_te), "NLL after": nll(softmax(lg, T), y_te)}
    for nm, lg, T in [("GRU", L["test"]["g"], T_g), ("CNN-BERT", L["test"]["c"], T_c)]
])

# ceiling analysis
predg, predc = pg_t.argmax(1), pc_t.argmax(1)
cg, cc = predg == y_te, predc == y_te
best_cheat, alphas = 0.0, np.linspace(0, 1, 21)
for a in alphas:
    f = fuse(pg_t, pc_t, a)
    for b1 in np.arange(-1.5, 1.51, 0.05):
        s = f.copy(); s[:, 1] *= np.exp(b1); col = f[:, 2]
        for b2 in np.arange(-1.5, 1.51, 0.05):
            s[:, 2] = col * np.exp(b2)
            best_cheat = max(best_cheat, macro_f1(y_te, s.argmax(1)))
ceiling = pd.DataFrame([
    {"Quantity": "Both branches correct", "n": int((cg & cc).sum()), "share": (cg & cc).mean()},
    {"Quantity": "Only GRU correct", "n": int((cg & ~cc).sum()), "share": (cg & ~cc).mean()},
    {"Quantity": "Only CNN-BERT correct", "n": int((~cg & cc).sum()), "share": (~cg & cc).mean()},
    {"Quantity": "Neither correct (unreachable)", "n": int((~cg & ~cc).sum()), "share": (~cg & ~cc).mean()},
    {"Quantity": "Oracle accuracy", "n": np.nan, "share": (cg | cc).mean()},
    {"Quantity": "Base ensemble accuracy", "n": np.nan, "share": accuracy_score(y_te, base_pred)},
    {"Quantity": "Max macro-F1 of any weight+bias rule (fitted ON test)", "n": np.nan, "share": best_cheat},
    {"Quantity": "Base ensemble macro-F1", "n": np.nan, "share": macro_f1(y_te, base_pred)},
])
print("\n" + "=" * 66)
print("CEILING ANALYSIS")
print("=" * 66)
print(ceiling.round(4).to_string(index=False))

# ---------------------------------------------------------------
# Export
# ---------------------------------------------------------------
XL = os.path.join(OUT_DIR, "Chapter4_Final_Tables.xlsx")
with pd.ExcelWriter(XL, engine="openpyxl") as w:
    res.to_excel(w, sheet_name="T1 Model Comparison", index=False)
    calib.to_excel(w, sheet_name="T2 Calibration", index=False)
    pd.DataFrame(pc_rows).to_excel(w, sheet_name="T3 Per-Class McNemar", index=False)
    ceiling.to_excel(w, sheet_name="T4 Ceiling Analysis", index=False)
    pd.DataFrame([{"Model": k, "CV macro-F1": v[0], "CV SE": v[1]} for k, v in CV_SCORES.items()]
                 ).to_excel(w, sheet_name="T5 CV Selection", index=False)
    pd.DataFrame(classification_report(y_te, hp, target_names=label_names,
                                       output_dict=True, zero_division=0)
                 ).T.reset_index().to_excel(w, sheet_name="T6 Headline Report", index=False)

fig, ax = plt.subplots(figsize=(8, 4.6))
sel = [k for k in CV_SCORES]
xs = np.arange(len(sel))
ax.errorbar(xs, [CV_SCORES[k][0] for k in sel], yerr=[CV_SCORES[k][1] for k in sel],
            fmt="o", capsize=4, color="#2a78d6", label="Calibration CV macro-F1")
ax.plot(xs, [res.loc[res.Model == k, "Test macro-F1"].values[0] for k in sel],
        "s--", color="#eb6834", label="Test macro-F1")
ax.set_xticks(xs)
ax.set_xticklabels([s.split(" ")[0] for s in sel])
ax.set_ylabel("macro-F1")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cv_vs_test_selection.png"), dpi=300)
plt.close(fig)

cm = confusion_matrix(y_te, hp, labels=range(K))
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(cm, cmap="Blues")
for i in range(K):
    for j in range(K):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(K)); ax.set_xticklabels(label_names)
ax.set_yticks(range(K)); ax.set_yticklabels(label_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Headline: {winner}", fontsize=10)
fig.colorbar(im); fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "confusion_headline.png"), dpi=300)
plt.close(fig)

with open(os.path.join(OUT_DIR, "step3_summary.json"), "w") as f:
    json.dump({"T_gru": T_g, "T_cnn": T_c,
               "stacker_config": {"rich": bool(inner_cfg[0]), "C": float(inner_cfg[1])},
               "nested_cv_mean": float(nsc.mean()),
               "nested_cv_se": float(nsc.std(ddof=1) / np.sqrt(N_FOLDS)),
               "headline_model": winner,
               "fusion_ceiling_macro_f1": float(best_cheat)}, f, indent=2)

print(f"\nSaved: {XL}")
print(f"Saved figures to: {FIG_DIR}")
