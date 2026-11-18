"""Explainable-AI display layer: SHAP bars, waterfall, popups, comparison and what-if panels.
All numbers come from explain_shap.py; this file only turns them into HTML / charts / widgets.
CSS colours use variables (with light defaults) so the same bars work on the dark dashboard and in the
light map popups."""
import altair as alt
import folium
import pandas as pd

import explain_shap as xai

UP, DOWN, MUTED = "#E5484D", "#19C3B1", "#8B98AD"

CSS = """
.shc{display:grid;grid-template-columns:104px 1fr 46px;align-items:center;gap:8px;margin:5px 0}
.shc .n{font-size:12px;color:var(--txt,#0F1B2D);line-height:1.15}.shc .n small{color:var(--mut,#5B6B7F)}
.shc .t{position:relative;height:12px;background:var(--trk,#EEF2F6);border-radius:6px}
.shc .t:after{content:'';position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--mut,#9AA7B5)}
.shc .t div{position:absolute;top:0;height:12px;border-radius:6px}
.shc .v{font-size:12px;font-weight:700;text-align:right}
"""
POP_CSS = ("<style>" + CSS + "body,div{font-family:Arial,sans-serif}.pt{font-size:15px;font-weight:700;"
           "margin-bottom:4px}.pm{font-size:12px;color:#33465C;margin:2px 0}.ph{font-size:12px;font-weight:700;"
           "margin:9px 0 2px}.pn{font-size:11px;color:#5B6B7F;margin-top:6px}.go{display:block;margin-top:10px;"
           "padding:8px 12px;border-radius:8px;background:linear-gradient(90deg,#19C3B1,#2D7FF9);color:#fff;"
           "text-align:center;font-weight:700;font-size:13px;text-decoration:none}</style>")


# ---------- building blocks ----------
def bars(exp, top=4):
    """Diverging SHAP bars: red raises the risk, teal lowers it."""
    t = exp["table"].head(top)
    mx = max(t["shap"].abs().max(), 1e-9)
    out = []
    for _, r in t.iterrows():
        w, c = abs(r["shap"]) / mx * 50, (UP if r["shap"] > 0 else DOWN)
        pos = f"left:50%;width:{w:.1f}%" if r["shap"] > 0 else f"left:{50 - w:.1f}%;width:{w:.1f}%"
        out.append(f'<div class="shc"><div class="n">{r["feature"]}<br><small>{r["value"]}</small></div>'
                   f'<div class="t"><div style="{pos};background:{c}"></div></div>'
                   f'<div class="v" style="color:{c}">{r["shap"] * 100:+.1f}</div></div>')
    return "".join(out)


def route_explanations(model, feats, res, explainer):
    """{route index: SHAP explanation} for every route that has a trip scenario (one batched call)."""
    idx = [i for i, r in enumerate(res) if r["cond"]]
    if not idx:
        return {}
    rows = pd.concat([res[i]["cond"]["row"] for i in idx], ignore_index=True)
    return dict(zip(idx, xai.batch_local(model, feats, rows, explainer)))


def waterfall(exp, top=8):
    """SHAP waterfall: starts at the model baseline, each factor moves the probability up or down."""
    t = exp["table"]
    rows, cur = [("Baseline", exp["base"] - .002, exp["base"] + .002, "base", 0.0)], exp["base"]
    for _, r in t.head(top).iterrows():
        rows.append((f"{r['feature']}: {r['value']}", cur, cur + r["shap"], "up" if r["shap"] > 0 else "down",
                     r["shap"] * 100))
        cur += r["shap"]
    rest = float(t["shap"].iloc[top:].sum())
    if abs(rest) > 1e-6:
        rows.append(("All other factors", cur, cur + rest, "up" if rest > 0 else "down", rest * 100))
    df = pd.DataFrame(rows, columns=["label", "start", "end", "kind", "pp"])
    df["order"] = range(len(df))
    bar = alt.Chart(df).mark_bar(cornerRadius=4, size=18).encode(
        y=alt.Y("label:N", sort=alt.EncodingSortField("order"), title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("start:Q", title="Probability of Serious/Fatal", scale=alt.Scale(zero=False),
                axis=alt.Axis(format="%")),
        x2="end:Q",
        color=alt.Color("kind:N", legend=None, scale=alt.Scale(domain=["up", "down", "base"],
                                                                 range=[UP, DOWN, MUTED])),
        tooltip=["label", alt.Tooltip("pp:Q", format="+.2f", title="effect (pp)")])
    rule = alt.Chart(pd.DataFrame({"x": [exp["prob"]], "t": [f"Trip: {exp['prob']:.1%}"]})).mark_rule(
        strokeDash=[5, 4], color="#E8EEF8").encode(x="x:Q")
    txt = alt.Chart(pd.DataFrame({"x": [exp["prob"]], "t": [f"Trip: {exp['prob']:.1%}"]})).mark_text(
        align="left", dx=5, dy=-4, color="#E8EEF8", fontWeight="bold").encode(x="x:Q", text="t:N",
                                                                               y=alt.value(8))
    return (bar + rule + txt).properties(height=max(150, 34 * len(df))).configure_view(strokeWidth=0)


# ---------- route cards ----------
def card_block(r, exp):
    """HTML appended to a route card: exact score formula + SHAP bars."""
    bd = xai.score_breakdown(r)
    pen = f' Blackspots add <b>{bd["penalty"]:.0f}</b> more to the ranking total.' if bd["penalty"] else ""
    html = (f'<div class="chip"><b>How the score is built:</b> (10 x {bd["fatal"]} fatal + 3 x {bd["serious"]} '
            f'serious + 1 x {bd["slight"]} slight) / {bd["km"]:.0f} km = <b>{r["score"]:.2f}</b>.{pen}</div>'
            if r["n"] else "")
    if exp:
        html += (f'<div class="xbox"><div class="why-h">Model view: what drives risk on your trip '
                 f'({exp["prob"]:.0%} serious/fatal)</div>{bars(exp)}'
                 f'<div class="xnote">{xai.headline(exp)}</div>'
                 f'<div class="xleg">Percentage points vs the {exp["base"]:.0%} baseline. '
                 f'Red raises risk, teal lowers it.</div></div>')
    return html


# ---------- map popups / labels ----------
def _go(url):
    return f'<a class="go" href="{url}" target="_blank" rel="noopener">Open this route in Google Maps &#8599;</a>'


def route_popup(i, r, exp, color, recommended, url):
    bd = xai.score_breakdown(r)
    h = (f'{POP_CSS}<div class="pt" style="color:{color}">Route {i}: {r["label"]}'
         f'{" (recommended)" if recommended else ""}</div>'
         f'<div class="pm">{r["km"]:.0f} km, {r["minutes"] / 60:.1f} h, risk score <b>{r["score"]:.2f}</b></div>'
         f'<div class="pm">{bd["fatal"]} fatal, {bd["serious"]} serious, {bd["slight"]} slight accidents '
         f'in the corridor, {r["n_blackspots"]} blackspot(s)</div>')
    if exp:
        h += (f'<div class="ph">Why the model rates your trip {exp["prob"]:.0%} serious/fatal</div>{bars(exp)}'
              f'<div class="pm">{xai.headline(exp)}</div>'
              f'<div class="pn">Percentage points. Red raises risk, teal lowers it.</div>')
    return folium.Popup(folium.Html(h + _go(url), script=True), max_width=360)


def blackspot_popup(p, exp, color):
    h = (f'{POP_CSS}<div class="pt" style="color:{color}">{p["level"]} blackspot</div>'
         f'<div class="pm"><b>{int(p["n"])}</b> accidents: {int(p["fatal"])} fatal, {int(p["serious"])} serious.</div>')
    if exp:
        h += (f'<div class="ph">Typical accident here: model rates it {exp["prob"]:.0%} serious/fatal</div>'
              f'{bars(exp)}<div class="pm">{xai.headline(exp)}</div>'
              f'<div class="pn">Built from the most common conditions recorded at this spot. '
              f'SHAP explains the model, not the cause of accidents.</div>')
    return folium.Popup(folium.Html(h, script=True), max_width=360)


def add_route_labels(group, res, exps, colors, rec):
    """Always-visible label on each route (no click needed): probability and its top driver."""
    for i, r in enumerate(res):
        exp = exps.get(i)
        if not exp:
            continue
        top = exp["table"].iloc[0]
        lon, lat = r["coords"][int(len(r["coords"]) * (0.25 + 0.2 * i))]
        star = " ★" if i == rec else ""
        html = (f'<div style="background:rgba(10,17,31,.92);border:2px solid {colors[r["label"]]};'
                f'border-radius:10px;padding:3px 9px;font:600 11px Arial;color:#E8EEF8;white-space:nowrap;'
                f'box-shadow:0 2px 10px rgba(0,0,0,.45);transform:translate(-50%,-50%)">'
                f'R{i + 1}{star}: {exp["prob"]:.0%} serious/fatal · top driver: {top["feature"]} '
                f'{top["shap"] * 100:+.1f} pp</div>')
        folium.Marker((lat, lon), icon=folium.DivIcon(html=html, icon_size=(0, 0))).add_to(group)


# ---------- extra panels ----------
def compare_panel(st, exps):
    """Why does one route's model risk differ from another's?"""
    if len(exps) < 2:
        return
    names = [f"Route {i + 1}" for i in exps]
    a, b = st.columns(2)
    ra = a.selectbox("Compare", names, index=0, key="cmp_a")
    rb = b.selectbox("with", names, index=min(1, len(names) - 1), key="cmp_b")
    if ra == rb:
        st.info("Pick two different routes.")
        return
    ia, ib = int(ra.split()[1]) - 1, int(rb.split()[1]) - 1
    diff, d = xai.compare(exps[ia], exps[ib])
    st.markdown(f"{rb} has a model risk **{abs(diff) * 100:.1f} pp {'higher' if diff > 0 else 'lower'}** than "
                f"{ra} ({exps[ib]['prob']:.0%} vs {exps[ia]['prob']:.0%}).")
    shown = False
    for _, x in d.iterrows():
        if abs(x["delta"]) >= 0.002:
            shown = True
            st.markdown(f"- **{x['feature']}**: {x['a_value']} → {x['b_value']} ({x['delta'] * 100:+.1f} pp)")
    if not shown:
        st.caption("These routes run on very similar roads, so the model sees almost no difference.")
    st.caption("Routes share the same time and weather, so differences come from each corridor's typical road "
               "(type, class, speed limit, junctions, urban/rural).")


def what_if_panel(st, model, feats, cond):
    """Counterfactuals: how would the model probability change if the trip changed?"""
    df, p0 = xai.what_if(model, feats, cond["row"])
    st.caption(f"Your trip: {p0:.1%} serious/fatal probability.")
    ch = alt.Chart(df).mark_bar(cornerRadius=4).encode(
        y=alt.Y("Scenario:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("Change (pp):Q", title="Change in probability (percentage points)"),
        color=alt.condition(alt.datum["Change (pp)"] > 0, alt.value(UP), alt.value(DOWN)),
        tooltip=["Scenario", alt.Tooltip("Serious/fatal probability:Q", format=".1%"),
                 alt.Tooltip("Change (pp):Q", format="+.2f")]).properties(height=220)
    st.altair_chart(ch.configure_view(strokeWidth=0), use_container_width=True)
    st.caption("Counter-intuitive results (for example rain lowering risk) reflect how this dataset was "
               "recorded, not real-world safety.")
