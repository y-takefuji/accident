import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from xgboost import XGBClassifier
import shap

RANDOM_STATE = 42
TOP_K_FULL = 8
TOP_K_REDUCED = 7
N_SPLITS = 5
SHAP_SAMPLE = 100

def sig3(x):
    """Format a number to 3 significant digits."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    if x == 0:
        return "0"
    return f"{x:.{max(0, 2 - int(np.floor(np.log10(abs(x)))))}f}"

# ------------------------------------------------------------
# 1. Load data and show shape
# ------------------------------------------------------------
file_path = "OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv"
df = pd.read_csv(file_path)
print("Original shape:", df.shape)

# ------------------------------------------------------------
# 2. Binary target filter
# ------------------------------------------------------------
target_col = "Degree of Injury"
df = df[df[target_col].isin(["Nonfatal", "Fatal"])].copy()

# ------------------------------------------------------------
# 3. Drop first three variables
# ------------------------------------------------------------
df = df.drop(columns=df.columns[:3])

# ------------------------------------------------------------
# 4. Encode strings to numbers using dtype
# ------------------------------------------------------------
for col in df.columns:
    if col == target_col:
        df[col] = df[col].map({"Nonfatal": 0, "Fatal": 1})
    elif df[col].dtype == object or str(df[col].dtype) == "string":
        df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

# ------------------------------------------------------------
# 5. Drop instances with more than 20% empty cells
# ------------------------------------------------------------
df = df[df.isna().mean(axis=1) <= 0.20].copy()

# ------------------------------------------------------------
# 6. Shape and target distribution after cleaning
# ------------------------------------------------------------
print("Shape after cleaning:", df.shape)
print("Target distribution:")
print(df[target_col].value_counts())
print(df[target_col].value_counts(normalize=True).apply(sig3))

X_full = df.drop(columns=[target_col]).fillna(-1)
y_full = df[target_col].astype(int).values
feature_names_full = list(X_full.columns)

# ------------------------------------------------------------
# 7. Feature selection functions (default settings, fit on given data only)
# ------------------------------------------------------------
def score_rf(X, y, names):
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=names)

def score_rf_shap(X, y, names, rng):
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X, y)
    idx = rng.choice(len(X), size=min(SHAP_SAMPLE, len(X)), replace=False)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X.iloc[idx])
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    return pd.Series(np.abs(sv).mean(axis=0), index=names)

def score_xgb(X, y, names):
    model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=names)

def score_xgb_shap(X, y, names, rng):
    model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
    model.fit(X, y)
    idx = rng.choice(len(X), size=min(SHAP_SAMPLE, len(X)), replace=False)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X.iloc[idx])
    return pd.Series(np.abs(sv).mean(axis=0), index=names)

def score_gmm(X, y, names):
    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
    gmm.fit(X)
    means = gmm.means_
    return pd.Series(np.abs(means[0] - means[1]), index=names)

def score_hvgs(X, y, names):
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    dispersion = var / (mean.abs() + 1e-12)
    return pd.Series(dispersion.values, index=names)

def score_spearman(X, y, names):
    scores = []
    for j in range(X.shape[1]):
        rho, _ = spearmanr(X.iloc[:, j], y)
        scores.append(abs(rho) if not np.isnan(rho) else 0.0)
    return pd.Series(scores, index=names)

def score_dea(X, y, names):
    t_vals = []
    for j in range(X.shape[1]):
        a = X.iloc[:, j][y == 0]
        b = X.iloc[:, j][y == 1]
        t, _ = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        t_vals.append(abs(t) if not np.isnan(t) else 0.0)
    return pd.Series(t_vals, index=names)

# Method name -> (selector key, predictive model used in cross-validation)
METHODS = {
    "RF":       ("rf",       "rf"),
    "RF-SHAP":  ("rf_shap",  "rf"),
    "XGB":      ("xgb",      "xgb"),
    "XGB-SHAP": ("xgb_shap", "xgb"),
    "GMM":      ("gmm",      "rf"),
    "HVGS":     ("hvgs",     "rf"),
    "Spearman": ("spearman", "rf"),
    "DEA":      ("dea",      "rf"),
}

def run_selector(key, X, y, names, rng):
    if key == "rf":
        return score_rf(X, y, names)
    if key == "rf_shap":
        return score_rf_shap(X, y, names, rng)
    if key == "xgb":
        return score_xgb(X, y, names)
    if key == "xgb_shap":
        return score_xgb_shap(X, y, names, rng)
    if key == "gmm":
        return score_gmm(X, y, names)
    if key == "hvgs":
        return score_hvgs(X, y, names)
    if key == "spearman":
        return score_spearman(X, y, names)
    if key == "dea":
        return score_dea(X, y, names)

def build_predictor(model_key):
    """Predictive model used in cross-validation for each method."""
    if model_key == "xgb":
        return XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
    return RandomForestClassifier(random_state=RANDOM_STATE)

def top_k(scores, k):
    return list(scores.sort_values(ascending=False).head(k).index)

def ci95(values):
    values = np.asarray(values)
    n = len(values)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t_crit * se, mean + t_crit * se

def exact_position_rate(top8_list, top7_list):
    """
    For i = 1..7: compare top-8 r(i+1) with top-7 r(i).
    Rate = matches / 7.
    """
    matches = sum(1 for i in range(7) if top8_list[i + 1] == top7_list[i])
    return matches / 7

# ------------------------------------------------------------
# 8. Cross-validation pipeline with progress reporting
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)
results = []
total_methods = len(METHODS)

for m_idx, (name, (sel_key, model_key)) in enumerate(METHODS.items(), start=1):
    print(f"\n[{m_idx}/{total_methods}] Algorithm: {name} (CV model: {model_key.upper()})")
    acc_top8, acc_top7 = [], []
    fold_pairs = []   # (top8_list, top7_list) from the SAME fold

    for f_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full), start=1):
        print(f"  Fold {f_idx}/{N_SPLITS}: selecting features...", end=" ", flush=True)

        X_tr, X_te = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_tr, y_te = y_full[train_idx], y_full[test_idx]
        predictor = build_predictor(model_key)

        # Step 1: full-feature ranking on training fold -> top 8
        scores_full = run_selector(sel_key, X_tr, y_tr, feature_names_full, rng)
        top8 = top_k(scores_full, TOP_K_FULL)

        m8 = clone(predictor).fit(X_tr[top8], y_tr)
        acc8 = accuracy_score(y_te, m8.predict(X_te[top8]))
        acc_top8.append(acc8)

        # Step 2: remove the rank-1 feature from the full ranking, re-rank the rest -> top 7
        highest = scores_full.sort_values(ascending=False).index[0]
        reduced_names = [f for f in feature_names_full if f != highest]
        X_tr_red = X_tr[reduced_names]
        X_te_red = X_te[reduced_names]

        scores_red = run_selector(sel_key, X_tr_red, y_tr, reduced_names, rng)
        top7 = top_k(scores_red, TOP_K_REDUCED)

        m7 = clone(predictor).fit(X_tr_red[top7], y_tr)
        acc7 = accuracy_score(y_te, m7.predict(X_te_red[top7]))
        acc_top7.append(acc7)

        # Store the pair from the same fold
        fold_pairs.append((tuple(top8), tuple(top7)))

        print(f"done | acc top8={sig3(acc8)}, top7={sig3(acc7)}")

    mean8, lo8, hi8 = ci95(acc_top8)
    mean7, lo7, hi7 = ci95(acc_top7)

    # Most frequent (top8, top7) pair across folds -> reported lists come from one fold
    most_pair = max(set(fold_pairs), key=fold_pairs.count)
    most8, most7 = list(most_pair[0]), list(most_pair[1])

    pos_rate = exact_position_rate(most8, most7)

    print(f"  Finished {name}: top8 mean={sig3(mean8)} [{sig3(lo8)}, {sig3(hi8)}]"
          f" | top7 mean={sig3(mean7)} [{sig3(lo7)}, {sig3(hi7)}]"
          f" | exact position rate={sig3(pos_rate)}")

    results.append({
        "Method": name,
        "Top 8 (95% CI mean [lower, upper])": f"{sig3(mean8)} [{sig3(lo8)}, {sig3(hi8)}]",
        "Top 8 features (r1..r8)": ", ".join(most8),
        "Top 7 features (r1..r7)": ", ".join(most7),
        "Exact position rate": sig3(pos_rate),
    })

# ------------------------------------------------------------
# 9. Summary table and save to result.csv
# ------------------------------------------------------------
summary = pd.DataFrame(results)
summary.to_csv("result.csv", index=False, encoding="utf-8-sig")
print("\n=== SUMMARY TABLE ===")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 300)
print(summary.to_string(index=False))
print("\nSaved to result.csv")