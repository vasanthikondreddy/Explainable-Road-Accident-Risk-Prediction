"""Research experiments for the paper. Run:  python experiments.py
Writes research_results.json (used by the app's "Research" tab) and prints tables.

1. Hotspot method comparison  - temporal validation (detect on Jan-Sep, test on Oct-Dec)
2. Model comparison           - same temporal split, severity (Serious/Fatal vs Slight)
3. Ablation study             - drop one feature group at a time
4. Feature importance         - permutation importance of the best model
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEV_W = {"Fatal": 10, "Serious": 3, "Slight": 1}
EARTH_KM = 6371.0088
CELL = 0.01
SEED = 42

d = pd.read_csv("accidents_clean.csv", parse_dates=["Date"])
d["w"] = d["Accident_Severity_label"].map(SEV_W)
train = d[d["Date"].dt.month <= 9].copy()
test = d[d["Date"].dt.month >= 10].copy()
print(f"records: {len(d):,} | train (Jan-Sep): {len(train):,} | test (Oct-Dec): {len(test):,}")


# ------------------------------------------------------------------ 1. hotspots
def cell_ids(df):
    return (np.floor(df["latitude"] / CELL).astype(int).astype(str) + "_" +
            np.floor(df["longitude"] / CELL).astype(int).astype(str))


def score_grid_count(tr):
    return tr.groupby(cell_ids(tr)).size().astype(float)


def score_grid_severity(tr):
    return tr.groupby(cell_ids(tr))["w"].sum().astype(float)


def score_kde(tr):
    """Severity grid smoothed over the 3x3 neighbourhood (a simple KDE)."""
    gy = np.floor(tr["latitude"] / CELL).astype(int)
    gx = np.floor(tr["longitude"] / CELL).astype(int)
    base = tr.assign(gy=gy, gx=gx).groupby(["gy", "gx"])["w"].sum()
    acc = {}
    for (y, x), v in base.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                k = f"{y + dy}_{x + dx}"
                acc[k] = acc.get(k, 0.0) + v * (1.0 if (dy == 0 and dx == 0) else 0.5)
    return pd.Series(acc)


def score_dbscan(tr):
    xy = np.radians(tr[["latitude", "longitude"]].values)
    lab = DBSCAN(eps=1.0 / EARTH_KM, min_samples=10, metric="haversine",
                 algorithm="ball_tree").fit_predict(xy)
    t = tr.assign(c=lab)
    t = t[t["c"] >= 0]
    cw = t.groupby("c")["w"].sum()
    t["cw"] = t["c"].map(cw)
    return t.groupby(cell_ids(t))["cw"].max().astype(float)


def evaluate_hotspots(score, k, rng):
    """Take top-k cells by score; report share of test accidents / severity captured."""
    top = score.sort_values(ascending=False).head(k).index
    tc = cell_ids(test)
    hit = tc.isin(top)
    return {"capture_pct": float(hit.mean() * 100),
            "severity_capture_pct": float(test.loc[hit, "w"].sum() / test["w"].sum() * 100)}


rng = np.random.default_rng(SEED)
methods = {"Grid (count)": score_grid_count, "Grid (severity-weighted)": score_grid_severity,
           "KDE (smoothed severity)": score_kde, "DBSCAN clusters": score_dbscan}
hot = []
for k in (100, 300, 500):
    all_cells = cell_ids(train).unique()
    rand = pd.Series(rng.random(len(all_cells)), index=all_cells)
    row = {"top_k_cells": k, "Random baseline": evaluate_hotspots(rand, k, rng)}
    for name, fn in methods.items():
        row[name] = evaluate_hotspots(fn(train), k, rng)
    hot.append(row)
    print(f"\nHotspot capture of Oct-Dec accidents, top {k} cells (% accidents / % severity-weighted)")
    for name, v in row.items():
        if name != "top_k_cells":
            print(f"  {name:28s} {v['capture_pct']:6.1f} / {v['severity_capture_pct']:6.1f}")

# ------------------------------------------------------------------ 2. models
CAT = ["Light_Conditions_label", "Weather_Conditions_label", "Road_Surface_Conditions_label",
       "Road_Type_label", "Junction_Detail_label", "1st_Road_Class_label", "Urban_or_Rural_Area_label"]
NUM = ["hour", "Day_of_Week", "Speed_limit", "is_weekend"]
GROUPS = {"Weather & surface": ["Weather_Conditions_label", "Road_Surface_Conditions_label"],
          "Time (hour, day, weekend)": ["hour", "Day_of_Week", "is_weekend", "Light_Conditions_label"],
          "Road type & junction": ["Road_Type_label", "Junction_Detail_label", "1st_Road_Class_label"],
          "Speed limit": ["Speed_limit"], "Urban/rural": ["Urban_or_Rural_Area_label"]}

ytr = train["Accident_Severity"].isin([1, 2]).astype(int)
yte = test["Accident_Severity"].isin([1, 2]).astype(int)


def make_model(kind, cat, num):
    enc = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                             ("n", StandardScaler(), num)])
    if kind == "Majority baseline":
        m = DummyClassifier(strategy="prior")
    elif kind == "Logistic regression":
        m = LogisticRegression(max_iter=3000, class_weight="balanced")
    elif kind == "Random forest":
        m = RandomForestClassifier(n_estimators=150, min_samples_leaf=10, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=SEED)
    else:
        m = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=150,
                                           class_weight="balanced", random_state=SEED)
    return Pipeline([("pre", enc), ("m", m)])


def metrics(model, cat, num):
    model.fit(train[cat + num], ytr)
    p = model.predict_proba(test[cat + num])[:, 1]
    pr, rc, th = precision_recall_curve(yte, p)
    f1s = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    b = int(np.nanargmax(f1s[:-1])) if len(th) else 0
    return {"roc_auc": float(roc_auc_score(yte, p)), "pr_auc": float(average_precision_score(yte, p)),
            "best_f1": float(f1s[b]), "precision_at_best": float(pr[b]), "recall_at_best": float(rc[b])}, p


print(f"\nModel comparison (train Jan-Sep, test Oct-Dec; base rate {yte.mean():.3f})")
models, fitted = {}, {}
for kind in ["Majority baseline", "Logistic regression", "Random forest", "Gradient boosting"]:
    mdl = make_model(kind, CAT, NUM)
    models[kind], _ = metrics(mdl, CAT, NUM)
    fitted[kind] = mdl
    v = models[kind]
    print(f"  {kind:22s} ROC-AUC {v['roc_auc']:.3f}  PR-AUC {v['pr_auc']:.3f}  F1 {v['best_f1']:.3f}")

# ------------------------------------------------------------------ 3. ablation
best_kind = max([k for k in models if k != "Majority baseline"], key=lambda k: models[k]["roc_auc"])
full_auc = models[best_kind]["roc_auc"]
print(f"\nAblation with {best_kind} (full model ROC-AUC {full_auc:.3f})")
abl = []
for g, cols in GROUPS.items():
    cat = [c for c in CAT if c not in cols]
    num = [c for c in NUM if c not in cols]
    v, _ = metrics(make_model(best_kind, cat, num), cat, num)
    abl.append({"removed": g, "roc_auc": v["roc_auc"], "delta": v["roc_auc"] - full_auc})
    print(f"  without {g:28s} ROC-AUC {v['roc_auc']:.3f} ({v['roc_auc'] - full_auc:+.3f})")

# ------------------------------------------------------------------ 4. importance
sub = test.sample(min(6000, len(test)), random_state=SEED)
ysub = sub["Accident_Severity"].isin([1, 2]).astype(int)
pi = permutation_importance(fitted[best_kind], sub[CAT + NUM], ysub, scoring="roc_auc",
                            n_repeats=5, random_state=SEED, n_jobs=-1)
imp = (pd.Series(pi.importances_mean, index=CAT + NUM).sort_values(ascending=False))
print("\nPermutation importance (drop in ROC-AUC)")
print(imp.round(4).to_string())

json.dump({"dataset": {"records": len(d), "train": len(train), "test": len(test),
                       "base_rate_serious_fatal": float(yte.mean()),
                       "period": "2018 (train Jan-Sep, test Oct-Dec)"},
           "hotspots": hot, "models": models, "best_model": best_kind, "ablation": abl,
           "importance": {k: float(v) for k, v in imp.items()}},
          open("research_results.json", "w"), indent=2)
print("\nSaved research_results.json")
