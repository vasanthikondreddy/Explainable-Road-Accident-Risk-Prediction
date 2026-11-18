"""Analytics used by the dashboard: risk-along-route profile, day x hour heat-map, explainable ranking
and the rule-based trip briefing. Pure calculations, no Streamlit code."""
import altair as alt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import risk

DAY_ORDER = [(2, "Mon"), (3, "Tue"), (4, "Wed"), (5, "Thu"), (6, "Fri"), (7, "Sat"), (1, "Sun")]
LEVEL_COL = {"Critical": "#E5484D", "High": "#F5A524", "Moderate": "#E5C100"}


# ---------- risk profile along the route ----------
def _cum_km(pts):
    seg = risk._hav_km(pts[:-1, 1], pts[:-1, 0], pts[1:, 1], pts[1:, 0])
    return np.concatenate([[0.0], np.cumsum(seg)])


def _position_km(pts, lat, lon):
    """Distance from the start (km) of the route vertex nearest to each (lat, lon)."""
    tree = BallTree(np.radians(pts[:, ::-1]), metric="haversine")
    _, ix = tree.query(np.radians(np.column_stack([lat, lon])), k=1)
    return _cum_km(pts)[ix[:, 0]]


def risk_profile(r, step=10):
    """Severity-weighted accidents per km for every `step` km of the route, plus blackspot positions."""
    cum = _cum_km(r["pts"])
    edges = np.arange(0, cum[-1] + step, step)
    cor = r["corridor"]
    if len(cor):
        pos = _position_km(r["pts"], cor["latitude"].values, cor["longitude"].values)
        w = cor["Accident_Severity_label"].map(risk.SEV_W).values
        h, _ = np.histogram(pos, bins=edges, weights=w)
    else:
        h = np.zeros(len(edges) - 1)
    prof = pd.DataFrame({"km": edges[:-1] + step / 2, "risk": h / step})
    bs = r["blackspots"]
    spots = pd.DataFrame(columns=["km", "level", "accidents"])
    if bs is not None and len(bs):
        spots = pd.DataFrame({"km": _position_km(r["pts"], bs["lat"].values, bs["lon"].values),
                              "level": bs["level"].values, "accidents": bs["n"].values})
    return prof, spots


def profile_chart(res, colors, rec, height=300):
    frames, spot_frames = [], []
    for i, r in enumerate(res, 1):
        prof, spots = risk_profile(r)
        prof["Route"] = f"Route {i}"
        frames.append(prof)
        if i - 1 == rec and len(spots):
            peak = prof["risk"].max()
            spots = spots.assign(Route=f"Route {i}", y=peak * 1.08)
            spot_frames.append(spots)
    df = pd.concat(frames)
    names = [f"Route {i}" for i in range(1, len(res) + 1)]
    cols = [colors[r["label"]] for r in res]
    area = alt.Chart(df).mark_area(opacity=.18, interpolate="monotone").encode(
        x=alt.X("km:Q", title="Distance from start (km)"),
        y=alt.Y("risk:Q", title="Severity-weighted accidents per km", stack=None),
        color=alt.Color("Route:N", scale=alt.Scale(domain=names, range=cols), legend=alt.Legend(orient="top")))
    line = alt.Chart(df).mark_line(interpolate="monotone", strokeWidth=2.5).encode(
        x="km:Q", y="risk:Q", color=alt.Color("Route:N", scale=alt.Scale(domain=names, range=cols), legend=None),
        tooltip=["Route", alt.Tooltip("km:Q", format=".0f"), alt.Tooltip("risk:Q", format=".2f")])
    chart = area + line
    if spot_frames:
        sp = pd.concat(spot_frames)
        pts = alt.Chart(sp).mark_point(shape="triangle-down", size=150, filled=True, stroke="white",
                                       strokeWidth=1).encode(
            x="km:Q", y="y:Q",
            color=alt.Color("level:N", scale=alt.Scale(domain=list(LEVEL_COL), range=list(LEVEL_COL.values())),
                            legend=alt.Legend(title="Blackspots (recommended route)", orient="top")),
            tooltip=["level", alt.Tooltip("km:Q", format=".0f", title="at km"), "accidents"])
        chart = chart + pts
    return chart.properties(height=height).configure_view(strokeWidth=0)


# ---------- when to leave: day x hour heat-map ----------
def day_hour_grid(model, feats, corridor, weather):
    """Relative model risk (1.0 = average slot) for every weekday and departure hour, or None."""
    rows = []
    for dow, name in DAY_ORDER:
        p = risk.hourly_profile(model, feats, corridor, dow, weather)
        if p is None:
            return None
        rows += [{"Day": name, "Hour": h, "p": float(v)} for h, v in enumerate(p)]
    df = pd.DataFrame(rows)
    df["Relative risk"] = df["p"] / df["p"].mean()
    return df


def heatmap_chart(df, sel_day, sel_hour):
    days = [n for _, n in DAY_ORDER]
    base = alt.Chart(df).encode(x=alt.X("Hour:O", title="Departure hour"),
                                y=alt.Y("Day:N", sort=days, title=None))
    rect = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color("Relative risk:Q", scale=alt.Scale(range=["#19C3B1", "#F5E6A8", "#E5484D"], domain=[
            df["Relative risk"].min(), 1.0, df["Relative risk"].max()]), legend=alt.Legend(orient="top")),
        tooltip=["Day", "Hour", alt.Tooltip("Relative risk:Q", format=".2f")])
    sel = base.mark_rect(fill=None, stroke="white", strokeWidth=2.5, cornerRadius=3).transform_filter(
        (alt.datum.Day == sel_day) & (alt.datum.Hour == sel_hour))
    return (rect + sel).properties(height=240).configure_view(strokeWidth=0)


# ---------- explainable ranking ----------
def rank_breakdown(res, w_safe):
    """Same maths as risk.rank_routes, but returning each part so the choice can be explained."""
    def norm(v):
        lo, hi = min(v), max(v)
        return [0.0 if hi - lo < 1e-9 else (x - lo) / (hi - lo) for x in v]
    nr = norm([x["risk_total"] for x in res])
    nt = norm([x["minutes"] for x in res])
    return [{"risk_part": w_safe * a, "time_part": (1 - w_safe) * b, "total": w_safe * a + (1 - w_safe) * b,
             "norm_risk": a, "norm_time": b} for a, b in zip(nr, nt)]


# ---------- rule-based trip briefing ----------
def briefing(res, rec, exps, hour, weather, dow, model, feats, w_safe):
    """A few plain-English sentences (HTML bold allowed). Deterministic: built from the numbers only."""
    r, to = res[rec], risk.tradeoff(res)
    fast = to["fastest"]
    out = [f"<b>Route {rec + 1}</b> is recommended: {r['km']:.0f} km, {r['minutes'] / 60:.1f} h, "
           f"with your safety priority set to {w_safe:.0%}."]
    if rec != fast and res[fast]["risk_total"] > 0:
        cut = (res[fast]["risk_total"] - r["risk_total"]) / res[fast]["risk_total"] * 100
        verb = "cuts" if cut > 0 else "raises"
        out.append(f"Versus the fastest route (Route {fast + 1}) it {verb} accident exposure by "
                   f"<b>{abs(cut):.0f}%</b> for <b>{r['minutes'] - res[fast]['minutes']:+.0f} min</b> of driving.")
    elif rec == fast:
        out.append("It is also the fastest option.")
    exp = exps.get(rec)
    if exp:
        top = exp["table"].iloc[0]
        wx = {"Fine": "fine weather", "Raining": "rain", "Fog/mist": "fog or mist"}.get(weather, weather.lower())
        out.append(f"The severity model rates a {hour:02d}:00 trip in {wx} at <b>{exp['prob']:.0%}</b> "
                   f"Serious/Fatal if an accident occurs; the biggest driver is <b>{top['feature'].lower()}</b> "
                   f"({top['value']}, {top['shap'] * 100:+.1f} pp).")
    bs = r["blackspots"]
    if bs is not None and len(bs):
        crit = int((bs["level"] == "Critical").sum())
        out.append(f"<b>{len(bs)}</b> accident blackspot(s) lie along it"
                   f"{f', {crit} critical' if crit else ''}; the worst has {int(bs['n'].max())} recorded accidents.")
    prof = risk.hourly_profile(model, feats, r["corridor"], dow, weather)
    if prof is not None and prof[hour] > 0:
        best = int(np.argmin(prof))
        gain = 1 - prof[best] / prof[hour]
        if gain > 0.02:
            out.append(f"Leaving at <b>{best:02d}:00</b> instead would lower the model's risk by about "
                       f"<b>{gain:.0%}</b>.")
    return out
