"""Route-level experiment for the paper (needs internet for OSRM). Run:  python route_experiment.py
For each origin-destination pair compares the FASTEST route with the LOWEST-RISK route and
saves routes_results.csv. Edit PAIRS to add your own routes (aim for 20+)."""
import numpy as np, pandas as pd, risk
from scipy.stats import wilcoxon

PAIRS = [("Hyderabad", "Bengaluru"), ("Pune", "Mumbai"), ("Delhi", "Jaipur"), ("Chennai", "Bengaluru"),
         ("Delhi", "Agra"), ("Mumbai", "Ahmedabad"), ("Kolkata", "Bhubaneswar"), ("Lucknow", "Kanpur"),
         ("Jaipur", "Ahmedabad"), ("Nagpur", "Hyderabad"), ("Indore", "Bhopal"), ("Chandigarh", "Delhi")]

data, model, feats = risk.load_assets()
bsp = risk.detect_blackspots(data, min_accidents=5, top_share=0.35)
rows = []
for a, b in PAIRS:
    try:
        s, e = risk.geocode(a), risk.geocode(b)
        routes = risk.get_routes((s[0], s[1]), (e[0], e[1]))
        res = risk.analyze(data, model, feats, routes, 12, 4, "Fine", 5, bsp)
        t = risk.tradeoff(res)
        f, sf = res[t["fastest"]], res[t["safest"]]
        rows.append({"origin": a, "destination": b, "routes_found": len(res),
                     "fastest_km": f["km"], "fastest_min": f["minutes"], "fastest_risk": f["risk_total"],
                     "fastest_blackspots": f["n_blackspots"],
                     "safest_km": sf["km"], "safest_min": sf["minutes"], "safest_risk": sf["risk_total"],
                     "safest_blackspots": sf["n_blackspots"],
                     "risk_reduction_pct": t["risk_reduction_pct"], "extra_minutes": t["extra_minutes"]})
        print(f"{a} -> {b}: {len(res)} routes, risk -{t['risk_reduction_pct']:.1f}%, +{t['extra_minutes']:.0f} min")
    except Exception as ex:
        print(f"{a} -> {b}: skipped ({ex})")
df = pd.DataFrame(rows)
df.to_csv("routes_results.csv", index=False)
if len(df):
    print("\nMean risk reduction: %.1f%% | mean extra time: %.1f min" %
          (df.risk_reduction_pct.mean(), df.extra_minutes.mean()))
    multi = df[df.routes_found > 1]
    if len(multi) >= 6 and (multi.fastest_risk - multi.safest_risk).abs().sum() > 0:
        print("Wilcoxon signed-rank (fastest vs safest risk): p = %.4f" %
              wilcoxon(multi.fastest_risk, multi.safest_risk).pvalue)
