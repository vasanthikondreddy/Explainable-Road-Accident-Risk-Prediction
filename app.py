"""Road Risk Route Planner: explainable accident-risk routing (Streamlit)."""
import datetime as dt
import json
import os

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import analytics
import explain_shap as xai
import gmaps
import mapview
import risk
import ui
import xai_ui

st.set_page_config(page_title="Road Risk Route Planner", page_icon="🛣️", layout="wide")
ui.apply_theme(st)

COLORS = {"Lowest risk": "#19C3B1", "Medium risk": "#F5A524", "Highest risk": "#E5484D",
          "Similar": "#5B8DEF", "Only route": "#5B8DEF"}
EXAMPLES = {"Custom": None, "Hyderabad → Bengaluru": ("Hyderabad", "Bengaluru"),
            "Pune → Mumbai": ("Pune", "Mumbai"), "Delhi → Jaipur": ("Delhi", "Jaipur"),
            "Chennai → Bengaluru": ("Chennai", "Bengaluru")}


# ---------------- cached resources ----------------
@st.cache_resource
def assets():
    return risk.load_assets()


data, model, feats = assets()


@st.cache_data
def get_blackspots(min_acc=5, share=0.35):
    return risk.detect_blackspots(data, min_accidents=min_acc, top_share=share)


@st.cache_resource
def explainer():
    return xai.get_explainer(model)


@st.cache_data(show_spinner="Computing global SHAP values (one-off, ~1 min)...")
def global_shap():
    return xai.load_global(model, data, feats)


@st.cache_data(show_spinner="Explaining blackspots with SHAP...")
def blackspot_shap(spots):
    return xai.blackspot_explanations(model, feats, data, spots, explainer())


ui.show(st, ui.hero(len(data)))

# ---------------- sidebar ----------------
st.session_state.setdefault("src", "Hyderabad")
st.session_state.setdefault("dst", "Bengaluru")


def fill_example():
    ex = EXAMPLES[st.session_state["ex"]]
    if ex:
        st.session_state["src"], st.session_state["dst"] = ex


def use_live_weather():
    try:
        g = risk.geocode(st.session_state["src"])
        w = risk.live_weather(g[0], g[1])
        if w:
            st.session_state["weather"] = w[0]
            st.session_state["wnote"] = f"Live weather at {st.session_state['src']}: {w[0]} ({w[1]})"
        else:
            st.session_state["wnote"] = "Could not fetch live weather (check internet)."
    except Exception:
        st.session_state["wnote"] = "Could not fetch live weather."


with st.sidebar:
    st.markdown("### 🧭 Plan your trip")
    st.selectbox("Quick examples", list(EXAMPLES), key="ex", on_change=fill_example)
    src_txt = st.text_input("From", key="src")
    dst_txt = st.text_input("To", key="dst")
    c1, c2 = st.columns(2)
    d = c1.date_input("Date", dt.date.today())
    t = c2.time_input("Departure", dt.time(22, 0))
    weather = st.selectbox("Weather", ["Fine", "Raining", "Fog/mist"], key="weather")
    st.button("Use live weather at origin", on_click=use_live_weather, use_container_width=True)
    if st.session_state.get("wnote"):
        st.caption(st.session_state["wnote"])
    buffer_km = st.slider("Corridor width (km each side)", 1, 10, 5)
    st.markdown("### ⚖️ Optimisation")
    w_safe = st.slider("Priority: speed  ←→  safety", 0, 100, 60,
                       help="0 = fastest route, 100 = lowest accident risk") / 100
    st.markdown("### 🗺️ Map")
    map_style = st.radio("Map style", list(mapview.TILES), horizontal=True)
    direct_open = st.toggle("Tapping a route on the map opens Google Maps", True,
                            help="Off: click shows the SHAP explanation with an 'Open in Google Maps' button. "
                                 "On: clicking the route line opens Google Maps straight away.")
    animate = st.toggle("Animate recommended route", True)
    show_heat = st.toggle("Accident heatmap", False)
    go = st.button("Analyze routes", type="primary", use_container_width=True)

if go:
    try:
        with st.spinner("Finding routes and scoring risk..."):
            s = risk.geocode(src_txt)
            e = risk.geocode(dst_txt)
            routes = risk.get_routes((s[0], s[1]), (e[0], e[1]))
            bsp = get_blackspots()
            res = risk.analyze(data, model, feats, routes, t.hour, risk.stats_dow(d.weekday()), weather,
                               buffer_km, bsp)
            if any(r["n_blackspots"] == 0 for r in res):  # sparse area: detect more sensitively
                bsp = get_blackspots(3, 0.6)
                res = risk.analyze(data, model, feats, routes, t.hour, risk.stats_dow(d.weekday()), weather,
                                   buffer_km, bsp)
        st.session_state["out"] = (res, s, e, t, weather, bsp)
        st.session_state["ctx"] = {"dow": risk.stats_dow(d.weekday()), "date": d, "src": src_txt, "dst": dst_txt}
    except Exception as ex:
        st.error(f"Could not complete the search: {ex}")

if "out" not in st.session_state:
    ui.show(st, """<div class="brief"><div><h3>👋 START HERE</h3><ul>
    <li>Pick a quick example (or type two places) in the sidebar and press <b>Analyze routes</b>.</li>
    <li>You get up to 3 routes ranked by accident risk and drive time, each explained with SHAP.</li>
    <li>Click any route to see why, then open it in <b>Google Maps</b> to navigate.</li></ul></div><div></div></div>""")
    st.stop()

res, s, e, t_used, weather_used, bsp = st.session_state["out"]
ctx = st.session_state.get("ctx", {"dow": 4, "src": "", "dst": ""})
rec = risk.rank_routes(res, w_safe)
urls = [gmaps.directions_url(r["coords"]) for r in res]
exps = xai_ui.route_explanations(model, feats, res, explainer())
parts = [r["blackspots"] for r in res if r["blackspots"] is not None and len(r["blackspots"])]
bs_exp = blackspot_shap(pd.concat(parts).drop_duplicates("id").reset_index(drop=True)) if parts else {}
best = min(range(len(res)), key=lambda i: res[i]["score"])
fast = min(range(len(res)), key=lambda i: res[i]["minutes"])

# ---------------- KPI row + briefing ----------------
k = st.columns(4)
kp = [("Recommended", f"Route {rec + 1}", f"{res[rec]['km']:.0f} km · {res[rec]['minutes'] / 60:.1f} h · "
       f"{res[rec]['n_blackspots']} blackspots", "🏁", "#19C3B1"),
      ("Lowest risk", f"Route {best + 1}", f"score {res[best]['score']:.2f} · {res[best]['n_blackspots']} "
       "blackspots", "🛡️", "#5B8DEF"),
      ("Fastest", f"Route {fast + 1}", f"{res[fast]['minutes'] / 60:.1f} h · {res[fast]['km']:.0f} km", "⚡",
       "#F5A524"),
      ("Routes found", str(len(res)), f"departure {t_used.strftime('%H:%M')} · {weather_used}", "🧭", "#7C5CFF")]
for col, args in zip(k, kp):
    col.markdown(ui.kpi(*args), unsafe_allow_html=True)

lines = analytics.briefing(res, rec, exps, t_used.hour, weather_used, ctx["dow"], model, feats, w_safe)
ui.show(st, ui.briefing_card(lines, urls[rec], rec))
st.write("")

tab_map, tab_xai, tab_time, tab_alerts, tab_insights, tab_research = st.tabs(
    ["🗺️ Map & routes", "🧠 Explainable AI", "🕒 When to leave", "🚨 Safety alerts", "📈 Insights", "🔬 Research"])


def route_picker(label, key, index=None):
    pick = st.selectbox(label, [f"Route {i}" for i in range(1, len(res) + 1)],
                        index=rec if index is None else index, key=key)
    return int(pick.split()[1]) - 1


# ---------------- map & routes ----------------
with tab_map:
    st.markdown("#### 🛣️ Choose your route: tap a card to navigate in Google Maps")
    mx = max(r["score"] for r in res) or 1
    for c, (i, r) in zip(st.columns(len(res)), enumerate(res, 1)):
        c.markdown(ui.route_card(i, r, COLORS[r["label"]], i - 1 == rec, exps.get(i - 1), urls[i - 1],
                                 r["score"] / mx * 100), unsafe_allow_html=True)
    st.caption("Tapping a card opens Google Maps and starts navigation. Google Maps links pass the origin, destination and up to 8 waypoints along the analysed route, "
               "so it follows the same road. Google may adjust the path slightly or show the waypoints as stops.")
    st.write("")
    m = mapview.build_map(data, res, s, e, rec, exps, bs_exp, COLORS, urls, map_style, show_heat,
                          direct_open, animate)
    st_folium(m, height=560, use_container_width=True, returned_objects=[], key="map")
    ui.show(st, '<div class="legend">' + "".join(
        f'<span><i style="background:{COLORS[n]}"></i>{n}</span>'
        for n in ["Lowest risk", "Medium risk", "Highest risk"]) +
        '<span class="note">Pulsing dots = blackspots · labels show the SHAP top driver · '
        + ("click a route to open Google Maps." if direct_open else
           "click a route for its SHAP explanation and a Google Maps button.") + '</span></div>')
    st.write("")

    st.markdown("#### 📉 Risk profile along the route")
    st.altair_chart(analytics.profile_chart(res, COLORS, rec), use_container_width=True)
    st.caption("Severity-weighted accidents per km, in 10 km steps from the start. Triangles mark blackspots "
               "on the recommended route.")

    st.markdown("#### ⚖️ Speed vs safety trade-off")
    front = risk.pareto_front(res)
    tdf = pd.DataFrame({"Route": [f"Route {i + 1}" for i in range(len(res))],
                        "Drive time (min)": [r["minutes"] for r in res],
                        "Total risk": [r["risk_total"] for r in res],
                        "Pareto optimal": ["Yes" if i in front else "No" for i in range(len(res))]})
    st.scatter_chart(tdf, x="Drive time (min)", y="Total risk", color="Pareto optimal", size=120)
    to = risk.tradeoff(res)
    if to["fastest"] == to["safest"]:
        st.success(f"Route {to['fastest'] + 1} is both the fastest and the lowest-risk option.")
    else:
        st.info(f"Taking Route {to['safest'] + 1} (lowest risk) instead of Route {to['fastest'] + 1} (fastest) "
                f"cuts risk by {to['risk_reduction_pct']:.1f}% for {to['extra_minutes']:+.0f} min "
                f"({to['extra_km']:+.0f} km). Pareto-optimal routes: "
                + ", ".join(f"Route {i + 1}" for i in front) + ".")

    st.markdown("#### 📍 Blackspots along a route")
    bb = res[route_picker("Route", "bs_pick")]["blackspots"]
    if bb is None or len(bb) == 0:
        st.info("No accident blackspots were found within 3 km of this route in the dataset.")
    else:
        st.dataframe(bb.sort_values("weight", ascending=False)[["level", "n", "fatal", "serious", "lat", "lon"]]
                     .rename(columns={"n": "accidents"}).reset_index(drop=True), use_container_width=True)

# ---------------- explainable AI ----------------
with tab_xai:
    st.markdown("#### 1. Why was this route recommended?")
    parts_rank = analytics.rank_breakdown(res, w_safe)
    ui.show(st, ui.rank_panel(res, parts_rank, rec, w_safe, COLORS))

    st.markdown("#### 2. Why does the model rate this trip this way? (SHAP waterfall)")
    ix = route_picker("Route", "xai_route")
    rx = res[ix]
    if not rx["cond"]:
        st.info("Too few accidents near this route to build a trip scenario.")
    else:
        exp = xai.local_explanation(model, feats, rx["cond"]["row"], explainer())
        st.markdown(f"Starting from the model's baseline of **{exp['base']:.2f}**, each factor pushes the "
                    f"probability up (red) or down (teal) to reach **{exp['prob']:.2f}** for your trip.")
        st.altair_chart(xai_ui.waterfall(exp), use_container_width=True)
        for line in xai.narrate(exp):
            st.markdown(f"- {line}")
        st.caption("pp = percentage points. Contributions add up exactly to the model output (SHAP property).")

        st.markdown("#### 3. What if the trip changed?")
        xai_ui.what_if_panel(st, model, feats, rx["cond"])

    st.markdown("#### 4. Why do two routes differ?")
    xai_ui.compare_panel(st, exps)

    st.divider()
    st.markdown("#### 5. Global explanation: what does the model rely on overall?")
    g = global_shap()
    imp = xai.global_importance(g, feats)
    a, b = st.columns([3, 2])
    with a:
        st.bar_chart(imp, color="#19C3B1", horizontal=True)
        st.caption(f"Mean |SHAP| per factor over {len(g)} sampled accidents.")
    with b:
        up, dn = xai.value_effects(g, feats)
        st.markdown("**Values that push risk up**")
        st.dataframe(up[["factor", "value", "avg_effect"]].round(3).reset_index(drop=True),
                     use_container_width=True, hide_index=True)
        st.markdown("**Values that push risk down**")
        st.dataframe(dn[["factor", "value", "avg_effect"]].round(3).reset_index(drop=True),
                     use_container_width=True, hide_index=True)
    st.warning("SHAP explains what the **model** learned, not what causes accidents. Some patterns (for example "
               "rain lowering risk, or 'not at a junction' raising it) mostly reflect how this dataset was "
               "recorded, and the model is weak (ROC-AUC about 0.60).")

# ---------------- when to leave ----------------
with tab_time:
    rt_ = res[route_picker("Route", "tm_pick")]
    grid = analytics.day_hour_grid(model, feats, rt_["corridor"], weather_used)
    if grid is None:
        st.info("Not enough accidents near this route to estimate risk by hour.")
    else:
        names = dict(analytics.DAY_ORDER)
        sel_day, sel_hour = names[ctx["dow"]], t_used.hour
        st.markdown(f"**Relative model risk by day and departure hour** ({weather_used.lower()}). "
                    "1.0 = average slot; the white box is your departure.")
        st.altair_chart(analytics.heatmap_chart(grid, sel_day, sel_hour), use_container_width=True)
        day = grid[grid["Day"] == sel_day].set_index("Hour")["Relative risk"]
        order = day.sort_values()
        c1, c2, c3 = st.columns(3)
        c1.metric("Safest time to leave", f"{order.index[0]:02d}:00", f"{(order.iloc[0] - 1) * 100:+.0f}% vs average")
        c2.metric("Riskiest time to leave", f"{order.index[-1]:02d}:00",
                  f"{(order.iloc[-1] - 1) * 100:+.0f}% vs average")
        c3.metric("Your departure", f"{sel_hour:02d}:00", f"{(day[sel_hour] - 1) * 100:+.0f}% vs average")
        st.caption("Based on the severity model (weak, ROC-AUC about 0.60), so treat it as a rough guide.")
    st.markdown("**Recorded accidents by hour on this route**")
    st.bar_chart(rt_["corridor"]["hour"].value_counts().reindex(range(24), fill_value=0), color="#F5A524")

# ---------------- alerts ----------------
with tab_alerts:
    rr = res[route_picker("Route", "al_pick")]
    al, trip = risk.route_alerts(rr, t_used.hour, weather_used)
    st.markdown(f"**{len(al)} blackspot alert(s) on this {rr['km']:.0f} km route**")
    for msg in trip:
        st.info(msg)
    if not al:
        st.success("No blackspots on this route with the current thresholds.")
    for x in al:
        (st.error if x["level"] == "Critical" else st.warning if x["level"] == "High" else st.info)(x["text"])

# ---------------- insights ----------------
with tab_insights:
    ip = route_picker("Route", "in_pick", 0)
    cor = res[ip]["corridor"]
    if len(cor) == 0:
        st.info("No recorded accidents near this route.")
    else:
        st.download_button("Download corridor accidents (CSV)", cor.to_csv(index=False).encode(),
                           file_name=f"route_{ip + 1}_accidents.csv", mime="text/csv")
        a, b = st.columns(2)
        with a:
            st.markdown("**Accidents by hour of day**")
            st.bar_chart(cor["hour"].value_counts().reindex(range(24), fill_value=0), color="#19C3B1")
        with b:
            st.markdown("**Severity split**")
            st.bar_chart(cor["Accident_Severity_label"].value_counts().reindex(
                ["Fatal", "Serious", "Slight"], fill_value=0), color="#F5A524")
        a, b = st.columns(2)
        with a:
            st.markdown("**Light conditions**")
            st.bar_chart(cor["Light_Conditions_label"].value_counts(), color="#5B8DEF")
        with b:
            st.markdown("**Speed limit**")
            st.bar_chart(cor["Speed_limit"].value_counts().sort_index(), color="#5B8DEF")

# ---------------- research ----------------
with tab_research:
    st.markdown("### Evaluation results (for the paper)")
    if not os.path.exists("research_results.json"):
        st.info("Run `python experiments.py` in the project folder to generate the results.")
    else:
        R = json.load(open("research_results.json"))
        ds = R["dataset"]
        st.caption(f"{ds['records']:,} records | {ds['period']} | train {ds['train']:,}, test {ds['test']:,} | "
                   f"serious/fatal base rate {ds['base_rate_serious_fatal']:.1%}")
        st.markdown("**1. Hotspot detection methods** - share of future (Oct-Dec) accidents that fall in the "
                    "top-K hotspot cells found from past (Jan-Sep) data. Higher is better.")
        rows = [{"Top-K cells": h["top_k_cells"], "Method": name, "Accidents captured %": round(v["capture_pct"], 1)}
                for h in R["hotspots"] for name, v in h.items() if name != "top_k_cells"]
        hot = pd.DataFrame(rows).pivot(index="Method", columns="Top-K cells", values="Accidents captured %")
        st.dataframe(hot, use_container_width=True)
        st.markdown("**2. Severity model comparison** (Serious/Fatal vs Slight, temporal split)")
        md = pd.DataFrame(R["models"]).T.round(3)
        md.columns = ["ROC-AUC", "PR-AUC", "Best F1", "Precision", "Recall"]
        st.dataframe(md, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**3. Ablation** (model: {R['best_model']}) - change in ROC-AUC")
            st.bar_chart(pd.DataFrame(R["ablation"]).set_index("removed")["delta"], color="#E5484D")
        with c2:
            st.markdown("**4. Permutation importance** (drop in ROC-AUC)")
            st.bar_chart(pd.Series(R["importance"]).sort_values(), color="#19C3B1")
        e1, e2 = st.columns(2)
        e1.download_button("Export model comparison (CSV)", md.to_csv().encode(), "model_comparison.csv",
                           "text/csv", use_container_width=True)
        e2.download_button("Export hotspot comparison (CSV)", hot.to_csv().encode(), "hotspot_comparison.csv",
                           "text/csv", use_container_width=True)

    st.markdown("**5. Route-level results** (fastest vs lowest-risk route)")
    if os.path.exists("routes_results.csv"):
        rr_ = pd.read_csv("routes_results.csv")
        st.dataframe(rr_.round(2), use_container_width=True)
        st.metric("Mean risk reduction", f"{rr_['risk_reduction_pct'].mean():.1f}%",
                  f"{rr_['extra_minutes'].mean():+.1f} min extra on average")
    else:
        st.info("Run `python route_experiment.py` (needs internet) to generate route-level results.")

    st.markdown("**Honest limitations to state in the paper**")
    st.write("- The dataset covers one year (2018) and its coordinates appear synthetic, clustered on a few "
             "points, so results show the framework works, not real-world safety rates.\n"
             "- All severity models reach only about 0.59 ROC-AUC, so the limit is the available features, "
             "not the algorithm.\n"
             "- DBSCAN performed poorly because the data is extremely dense around a few points; grid-based "
             "methods generalised best.\n"
             "- SHAP explains the model's behaviour, not causes of accidents.")

with st.expander("About these scores"):
    st.write("The dataset's locations appear synthetic and cluster around a few points, and the severity model "
             "is weak (ROC-AUC about 0.60). Use scores to compare routes, not as real-world safety ratings. The "
             "tool summarises past accidents; it does not predict whether an accident will happen on a trip.")


# ---------------- downloadable report ----------------
def build_report():
    rows = [{"Route": i, "Label": r["label"], "Recommended": "Yes" if i - 1 == rec else "",
             "Distance km": round(r["km"], 1), "Drive time h": round(r["minutes"] / 60, 2),
             "Risk score": round(r["score"], 2), "Accidents nearby": r["n"], "Fatal": r["fatal"],
             "Serious": r["serious"], "Blackspots": r["n_blackspots"], "Google Maps": urls[i - 1]}
            for i, r in enumerate(res, 1)]
    df = pd.DataFrame(rows)
    html = ["<html><head><meta charset='utf-8'><title>Road Risk Trip Report</title>"
            "<style>body{font-family:Arial;margin:30px;color:#0F1B2D}table{border-collapse:collapse}"
            "td,th{border:1px solid #ccc;padding:6px 10px;font-size:13px}th{background:#0E7C86;color:#fff}"
            "</style></head><body><h1>Road Risk Trip Report</h1>",
            f"<p><b>From:</b> {ctx.get('src', '')}<br><b>To:</b> {ctx.get('dst', '')}<br>"
            f"<b>Departure:</b> {t_used.strftime('%H:%M')}, {weather_used}</p>",
            f"<p><b>Recommended:</b> Route {rec + 1}</p><h3>Briefing</h3><ul>"
            + "".join(f"<li>{x}</li>" for x in lines) + "</ul>",
            df.drop(columns=["Google Maps"]).to_html(index=False)]
    for i, r in enumerate(res, 1):
        al, trip = risk.route_alerts(r, t_used.hour, weather_used)
        html.append(f"<h3>Route {i} ({r['label']})</h3><p><a href='{urls[i - 1]}'>Open in Google Maps</a></p>")
        if exps.get(i - 1):
            html.append(f"<p><b>Model explanation:</b> {xai.headline(exps[i - 1])}</p>")
        html.append("<ul>" + "".join(f"<li>{x}</li>" for x in trip) + "".join(f"<li>{x['text']}</li>" for x in al)
                    + "</ul><ul>" + "".join(f"<li>{w}</li>" for w in r["why"]) + "</ul>")
    html.append("<p style='font-size:12px;color:#666'>Scores summarise past accidents in the dataset and do not "
                "predict whether an accident will happen.</p></body></html>")
    return df.to_csv(index=False).encode(), "\n".join(html).encode()


with st.sidebar:
    st.divider()
    st.markdown("### 📥 Download")
    _csv, _html = build_report()
    st.download_button("Trip report (HTML, print to PDF)", _html, "trip_report.html", "text/html",
                       use_container_width=True)
    st.download_button("Route summary (CSV)", _csv, "route_summary.csv", "text/csv", use_container_width=True)
