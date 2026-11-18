"""Explainable AI layer: SHAP explanations for the Random Forest severity model.

Why group columns?  The model one-hot encodes categories (49 columns), but a human wants
"Light conditions" as ONE factor.  SHAP values are additive, so summing the one-hot columns of
the same feature gives that feature's exact total contribution.
"""
import json
import os

import numpy as np
import pandas as pd
import shap

CACHE = "shap_global.csv"

PRETTY = {
    "Light_Conditions_label": "Light conditions",
    "Weather_Conditions_label": "Weather",
    "Road_Surface_Conditions_label": "Road surface",
    "Road_Type_label": "Road type",
    "Junction_Detail_label": "Junction type",
    "1st_Road_Class_label": "Road class",
    "Urban_or_Rural_Area_label": "Urban / rural",
    "hour": "Hour of day",
    "Day_of_Week": "Day of week",
    "Speed_limit": "Speed limit",
    "is_weekend": "Weekend",
}
DAYS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}


# ---------- helpers ----------
def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _positive_class(sv):
    """Different SHAP versions return a list, a 3-D array, or a 2-D array. Return (n, features)."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def _base_value(explainer):
    ev = np.atleast_1d(explainer.expected_value)
    return float(ev[1] if len(ev) > 1 else ev[0])


def _group_index(model, feats):
    """For each transformed column, which original feature does it belong to?"""
    names = model.named_steps["pre"].get_feature_names_out()
    originals = sorted(feats["cat"] + feats["num"], key=len, reverse=True)  # longest first
    owner = []
    for n in names:
        n = n.split("__", 1)[1]
        owner.append(next(o for o in originals if n == o or n.startswith(o + "_")))
    return np.array(owner)


def _grouped(sv, owner, order):
    return np.column_stack([sv[:, owner == f].sum(axis=1) for f in order])


def pretty_value(feat, v):
    if feat == "Day_of_Week":
        return DAYS.get(int(v), str(v))
    if feat == "hour":
        return f"{int(v):02d}:00"
    if feat == "is_weekend":
        return "Yes" if int(v) else "No"
    if feat == "Speed_limit":
        return f"{int(v)}"
    return str(v)


def get_explainer(model):
    return shap.TreeExplainer(model.named_steps["m"])


# ---------- local: why THIS trip? ----------
def local_explanation(model, feats, row, explainer=None):
    """row = one-row DataFrame with feats['cat'] + feats['num'].
    Returns base rate, predicted probability and each factor's signed contribution."""
    explainer = explainer or get_explainer(model)
    order = feats["cat"] + feats["num"]
    X = row[order]
    Xt = _dense(model.named_steps["pre"].transform(X))
    sv = _positive_class(explainer.shap_values(Xt))
    owner = _group_index(model, feats)
    g = _grouped(sv, owner, order)[0]
    df = pd.DataFrame({
        "feature": [PRETTY[f] for f in order],
        "value": [pretty_value(f, X.iloc[0][f]) for f in order],
        "shap": g,
    })
    df = df.reindex(df["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    base = _base_value(explainer)
    return {"base": base, "prob": base + float(g.sum()),
            "model_prob": float(model.predict_proba(X)[0, 1]), "table": df}


def narrate(exp, top=3):
    """Plain-English sentences from a local explanation."""
    t = exp["table"]
    out = []
    for _, r in t.head(top).iterrows():
        if abs(r["shap"]) < 0.002:
            continue
        verb = "raises" if r["shap"] > 0 else "lowers"
        out.append(f"{r['feature']} = {r['value']} {verb} the serious/fatal probability "
                   f"by {abs(r['shap']) * 100:.1f} percentage points.")
    return out or ["No single factor stands out for this trip."]


# ---------- global: what does the model rely on overall? ----------
def build_global_cache(model, data, feats, n=500, path=CACHE, seed=42):
    """One-off (about 1 minute for n=500): SHAP values for a random sample, saved to CSV."""
    order = feats["cat"] + feats["num"]
    sample = data[order].sample(min(n, len(data)), random_state=seed).reset_index(drop=True)
    Xt = _dense(model.named_steps["pre"].transform(sample))
    sv = _positive_class(get_explainer(model).shap_values(Xt))
    g = _grouped(sv, _group_index(model, feats), order)
    out = pd.concat([sample.add_prefix("val_"),
                     pd.DataFrame(g, columns=[f"shap_{f}" for f in order])], axis=1)
    out.to_csv(path, index=False)
    return out


def load_global(model=None, data=None, feats=None, path=CACHE):
    if os.path.exists(path):
        return pd.read_csv(path)
    if model is None:
        raise FileNotFoundError(f"{path} missing. Run: python build_shap_cache.py")
    return build_global_cache(model, data, feats, path=path)


def global_importance(g, feats):
    """Mean |SHAP| per factor (overall importance), as a sorted Series."""
    order = feats["cat"] + feats["num"]
    s = pd.Series({PRETTY[f]: g[f"shap_{f}"].abs().mean() for f in order})
    return s.sort_values(ascending=False)


def value_effects(g, feats, min_count=15, top=8):
    """Which specific values push risk up / down on average? (e.g. 'Dark - unlit' -> +0.04)"""
    rows = []
    for f in feats["cat"] + ["is_weekend"]:
        for v, grp in g.groupby(f"val_{f}"):
            if len(grp) >= min_count:
                rows.append({"factor": PRETTY[f], "value": pretty_value(f, v),
                             "avg_effect": grp[f"shap_{f}"].mean(), "n": len(grp)})
    df = pd.DataFrame(rows)
    return (df.sort_values("avg_effect", ascending=False).head(top),
            df.sort_values("avg_effect").head(top))


# =====================================================================
# Extra explainability logic (HTML lives in xai_ui.py)
# =====================================================================
def batch_local(model, feats, rows, explainer=None):
    """Local SHAP explanations for many scenarios in ONE call. Same output shape as
    local_explanation(), one dict per row."""
    explainer = explainer or get_explainer(model)
    order = feats["cat"] + feats["num"]
    X = rows[order].reset_index(drop=True)
    Xt = _dense(model.named_steps["pre"].transform(X))
    g = _grouped(_positive_class(explainer.shap_values(Xt)), _group_index(model, feats), order)
    probs = model.predict_proba(X)[:, 1]
    base = _base_value(explainer)
    out = []
    for i in range(len(X)):
        df = pd.DataFrame({"feature": [PRETTY[f] for f in order],
                           "value": [pretty_value(f, X.iloc[i][f]) for f in order],
                           "shap": g[i]})
        df = df.reindex(df["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        out.append({"base": base, "prob": base + float(g[i].sum()),
                    "model_prob": float(probs[i]), "table": df})
    return out


def headline(exp, top=2):
    """One sentence: what pushes this scenario up and what eases it."""
    t = exp["table"]
    up, dn = t[t["shap"] >= 0.005].head(top), t[t["shap"] <= -0.005].head(1)
    parts = []
    if len(up):
        parts.append("pushed up by " + " and ".join(
            f"{r['feature'].lower()} ({r['value']}, {r['shap'] * 100:+.1f} pp)" for _, r in up.iterrows()))
    if len(dn):
        r = dn.iloc[0]
        parts.append(f"eased by {r['feature'].lower()} ({r['value']}, {r['shap'] * 100:+.1f} pp)")
    return ("Model risk is " + "; ".join(parts) + ".") if parts else "No single factor stands out."


def compare(exp_a, exp_b, top=3):
    """Why does scenario B differ from A? Per-factor SHAP difference (B minus A), largest first."""
    a = exp_a["table"].set_index("feature")
    b = exp_b["table"].set_index("feature")
    d = pd.DataFrame({"a_value": a["value"], "b_value": b["value"], "delta": b["shap"] - a["shap"]})
    d = d.reindex(d["delta"].abs().sort_values(ascending=False).index)
    return exp_b["prob"] - exp_a["prob"], d.head(top).reset_index()


WHAT_IF = {
    "Leave at midday (12:00, daylight)": {"hour": 12, "Light_Conditions_label": "Daylight"},
    "Leave at night (23:00, street-lit)": {"hour": 23, "Light_Conditions_label": "Dark - lit"},
    "Rain and wet road": {"Weather_Conditions_label": "Raining", "Road_Surface_Conditions_label": "Wet/damp"},
    "Fine weather and dry road": {"Weather_Conditions_label": "Fine", "Road_Surface_Conditions_label": "Dry"},
    "Travel on a weekday (Wed)": {"Day_of_Week": 4, "is_weekend": 0},
    "Travel on a weekend (Sat)": {"Day_of_Week": 7, "is_weekend": 1},
}


def what_if(model, feats, row):
    """Counterfactuals: how does the model probability change if one thing about the trip changes?"""
    cols = feats["cat"] + feats["num"]
    now = row[cols].iloc[0].to_dict()
    # skip scenarios that would not change anything (e.g. "fine weather" when it is already fine)
    todo = {k: ch for k, ch in WHAT_IF.items() if any(now.get(c) != v for c, v in ch.items())}
    rows = [now] + [{**now, **ch} for ch in todo.values()]
    p = model.predict_proba(pd.DataFrame(rows)[cols])[:, 1]
    return pd.DataFrame({"Scenario": list(todo), "Serious/fatal probability": p[1:],
                         "Change (pp)": (p[1:] - p[0]) * 100}), float(p[0])


def score_breakdown(route):
    """The route score is a plain formula, so show it exactly: (10*fatal + 3*serious + 1*slight) / km."""
    f, s = route["fatal"], route["serious"]
    l = max(route["n"] - f - s, 0)
    bs = route.get("blackspots")
    pen = 0.5 * float(bs["weight"].sum()) if bs is not None and len(bs) else 0.0
    return {"fatal": f, "serious": s, "slight": l, "km": route["km"], "penalty": pen}


def accidents_near(data, lat, lon, r_km):
    dy = (data["latitude"] - lat) * 111.0
    dx = (data["longitude"] - lon) * 111.0 * np.cos(np.radians(lat))
    return data[(dx ** 2 + dy ** 2) <= r_km ** 2]


def typical_row(sub, feats):
    """Most common value of every model input among the accidents in `sub`."""
    row = {c: (sub[c].dropna().mode().iloc[0] if sub[c].notna().any() else sub[c].iloc[0])
           for c in feats["cat"] + feats["num"]}
    row["is_weekend"] = int(row["Day_of_Week"] in (1, 7))
    return pd.DataFrame([row])[feats["cat"] + feats["num"]]


def blackspot_explanations(model, feats, data, spots, explainer=None):
    """SHAP for the 'typical accident' at each blackspot -> {blackspot id: explanation}."""
    rows, ids = [], []
    for _, p in spots.iterrows():
        sub = accidents_near(data, p["lat"], p["lon"], p["radius_km"])
        if len(sub) >= 5:
            rows.append(typical_row(sub, feats))
            ids.append(int(p["id"]))
    if not rows:
        return {}
    return dict(zip(ids, batch_local(model, feats, pd.concat(rows, ignore_index=True), explainer)))
