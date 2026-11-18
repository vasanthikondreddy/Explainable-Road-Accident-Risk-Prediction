"""Run once after train.py:  python build_shap_cache.py   (about 1 minute)"""
import risk, explain_shap
data, model, feats = risk.load_assets()
out = explain_shap.build_global_cache(model, data, feats, n=500)
print("saved shap_global.csv with", len(out), "rows")
