"""Design system for the dashboard: theme CSS and small HTML components (hero, KPI cards, route cards)."""
import re

import xai_ui

CSS = """
:root{--txt:#E8EEF8;--mut:#93A1B8;--trk:rgba(255,255,255,.09);--card:rgba(255,255,255,.045);
--line:rgba(255,255,255,.10);--teal:#19C3B1;--blue:#2D7FF9;--red:#E5484D;--amber:#F5A524}
#MainMenu,footer{visibility:hidden}
.stApp{background:radial-gradient(1200px 600px at 8% -10%,rgba(25,195,177,.14),transparent 60%),
radial-gradient(900px 500px at 100% 0%,rgba(45,127,249,.13),transparent 55%),#0A111F}
.block-container{padding-top:1.2rem;max-width:1320px}
section[data-testid=stSidebar]{background:linear-gradient(180deg,#0E1830,#0A111F);border-right:1px solid var(--line)}
.hero{position:relative;overflow:hidden;border-radius:22px;padding:30px 34px;margin-bottom:18px;
background:linear-gradient(120deg,rgba(25,195,177,.28),rgba(45,127,249,.30) 60%,rgba(124,92,255,.28));
border:1px solid var(--line)}
.hero:before{content:'';position:absolute;inset:0;opacity:.35;
background-image:linear-gradient(rgba(255,255,255,.07) 1px,transparent 1px),
linear-gradient(90deg,rgba(255,255,255,.07) 1px,transparent 1px);background-size:34px 34px}
.hero>*{position:relative}
.hero h1{margin:0;font-size:34px;letter-spacing:-.02em;color:#fff}
.hero h1 span{background:linear-gradient(90deg,#7CF3E4,#8DB8FF);-webkit-background-clip:text;color:transparent}
.hero p{margin:8px 0 14px;color:#CFE0F5;font-size:14px;max-width:860px}
.badge{display:inline-block;margin:0 8px 6px 0;padding:4px 12px;border-radius:999px;font-size:12px;
font-weight:600;color:#DDEBFA;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.16)}
.kpi{position:relative;background:var(--card);border:1px solid var(--line);border-radius:16px;
padding:16px 18px 14px;overflow:hidden;backdrop-filter:blur(8px);transition:transform .15s}
.kpi:hover{transform:translateY(-3px)}
.kpi:before{content:'';position:absolute;left:0;top:0;right:0;height:3px;background:var(--a)}
.kpi .ic{position:absolute;right:14px;top:10px;font-size:26px;opacity:.9}
.kpi .l{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.07em}
.kpi .v{font-size:28px;font-weight:800;color:#fff;margin-top:2px}
.kpi .s{font-size:12px;color:var(--mut)}
.brief{margin:16px 0 4px;border-radius:18px;padding:18px 22px;border:1px solid rgba(25,195,177,.35);
background:linear-gradient(135deg,rgba(25,195,177,.12),rgba(45,127,249,.10));display:grid;
grid-template-columns:1fr auto;gap:18px;align-items:center}
.brief h3{margin:0 0 6px;font-size:15px;color:#7CF3E4;letter-spacing:.03em}
.brief ul{margin:0;padding-left:18px}.brief li{font-size:13.5px;color:#DCE7F6;margin:3px 0}
.brief .ai{font-size:11px;color:var(--mut);margin-top:8px}
.gm{display:inline-block;padding:11px 18px;border-radius:12px;font-weight:700;font-size:14px;color:#fff!important;
text-decoration:none!important;background:linear-gradient(90deg,#19C3B1,#2D7FF9);
box-shadow:0 6px 18px rgba(45,127,249,.35);transition:transform .15s,box-shadow .15s;white-space:nowrap}
.gm:hover{transform:translateY(-2px);box-shadow:0 10px 24px rgba(45,127,249,.5)}
.gm.full{display:block;text-align:center;margin:12px 0 4px}
.rc{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--c);border-radius:18px;
padding:18px 20px;height:100%;backdrop-filter:blur(8px)}
.rc.best{box-shadow:0 0 0 1px var(--c),0 10px 34px color-mix(in srgb,var(--c) 28%,transparent)}
.rc-link,.rc-link:hover,.rc-link *{text-decoration:none!important;color:inherit}.rc-link{display:block;height:100%;cursor:pointer}.rc-link .rc{transition:transform .15s,box-shadow .15s}.rc-link:hover .rc{transform:translateY(-4px);box-shadow:0 14px 34px rgba(45,127,249,.35)}.rc-link .gm{color:#fff!important}
.rc-top{display:flex;align-items:center;gap:12px}
.rank{width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;
font-weight:800;color:#fff;background:var(--c);font-size:18px}
.rc-t{font-size:18px;font-weight:800;color:#fff;line-height:1.1}.rc-s{font-size:12px;color:var(--mut)}
.rc-top .sp{flex:1}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;color:#06231F;
background:#7CF3E4;margin-left:8px;vertical-align:middle}
.ring{text-align:center}.ring .cap{font-size:9px;color:var(--mut);margin-top:-2px;letter-spacing:.04em}
.stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:14px 0}
.stats .l{font-size:11px;color:var(--mut)}.stats .v{font-size:18px;font-weight:700;color:#fff}
.chip{background:rgba(255,255,255,.06);border-radius:10px;padding:9px 12px;font-size:12.5px;color:var(--txt);
margin:8px 0}
.why-h{font-size:12.5px;font-weight:700;color:#fff;margin:6px 0 4px}
.why{margin:0;padding-left:18px}.why li{font-size:12.5px;color:#C5D2E6;margin-bottom:3px}
.xbox{border-top:1px dashed var(--line);margin-top:12px;padding-top:10px}
.xnote{font-size:12px;color:#C5D2E6;margin-top:6px}.xleg{font-size:11px;color:var(--mut);margin-top:4px}
.legend span{display:inline-block;margin-right:18px;font-size:13px;color:#C5D2E6}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.note{font-size:12px;color:var(--mut)}
.stack{display:flex;height:16px;border-radius:8px;overflow:hidden;background:var(--trk);margin:4px 0 2px}
.stack div{height:16px}.rk{margin:12px 0}.rk .h{display:flex;justify-content:space-between;font-size:13px;
color:var(--txt)}
.sh{display:grid;grid-template-columns:150px 1fr 70px;align-items:center;gap:10px;margin:5px 0}
.sh .n{font-size:13px;color:var(--txt)}.sh .n small{color:var(--mut)}
.sh .t{position:relative;height:16px;background:var(--trk);border-radius:6px}
.sh .t div{position:absolute;top:0;height:16px;border-radius:6px}.sh .v{font-size:13px;font-weight:700;text-align:right}
.stTabs [data-baseweb=tab-list]{gap:6px;background:rgba(255,255,255,.04);padding:6px;border-radius:14px}
.stTabs [data-baseweb=tab]{border-radius:10px;padding:8px 16px;height:auto}
.stTabs [aria-selected=true]{background:linear-gradient(90deg,rgba(25,195,177,.35),rgba(45,127,249,.35))}
.stTabs [data-baseweb=tab-highlight],.stTabs [data-baseweb=tab-border]{display:none}
iframe[title="streamlit_folium.st_folium"]{border-radius:18px;border:1px solid var(--line)}
@media(max-width:800px){.brief{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}}
""" + xai_ui.CSS


def flat(html):
    """Remove newlines and indentation so Streamlit's markdown never mistakes HTML for a code block."""
    return re.sub(r"\n\s*", "", html)


def show(st, html):
    st.markdown(flat(html), unsafe_allow_html=True)


def apply_theme(st):
    st.markdown(f"<style>{flat(CSS)}</style>", unsafe_allow_html=True)


def hero(n_records):
    return f"""<div class="hero"><h1>🛣️ Road Risk <span>Route Planner</span></h1>
<p>Explainable accident-risk routing: every route is scored from {n_records:,} historical accidents, and a
SHAP-explained severity model tells you <b>why</b>. Click a route to open it in Google Maps.</p>
<span class="badge">📊 {n_records:,} accident records</span><span class="badge">🧠 Explainable AI (SHAP)</span>
<span class="badge">🗺️ Google Maps hand-off</span><span class="badge">⚖️ Speed vs safety optimiser</span></div>"""


def kpi(label, value, sub, icon, accent):
    return (f'<div class="kpi" style="--a:{accent}"><div class="ic">{icon}</div><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def ring(pct, color, size=66):
    c = 2 * 3.14159 * 26
    d = max(0.0, min(pct, 100)) / 100 * c
    return (f'<div class="ring"><svg width="{size}" height="{size}" viewBox="0 0 66 66">'
            f'<circle cx="33" cy="33" r="26" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="7"/>'
            f'<circle cx="33" cy="33" r="26" fill="none" stroke="{color}" stroke-width="7" stroke-linecap="round" '
            f'stroke-dasharray="{d:.1f} {c:.1f}" transform="rotate(-90 33 33)"/>'
            f'<text x="33" y="38" text-anchor="middle" fill="#fff" font-size="15" font-weight="800">{pct:.0f}</text>'
            f'</svg><div class="cap">RISK INDEX</div></div>')


def route_card(i, r, color, is_rec, exp, url, rel_pct):
    cond = ""
    if r["cond"]:
        ratio = r["cond"]["ratio"]
        word = "higher than" if ratio >= 1.1 else "lower than" if ratio <= .9 else "about the same as"
        cond = (f'<div class="chip">Your time and weather: <b>{word}</b> a midday, fine-weather trip on this '
                f'road type ({ratio:.2f}x). Weak model, rough indication only.</div>')
    why = "".join(f"<li>{x}</li>" for x in r["why"])
    rec = '<span class="pill">Recommended</span>' if is_rec else ""
    return flat(f"""<a class="rc-link" href="{url}" target="_blank" rel="noopener"><div class="rc{' best' if is_rec else ''}" style="--c:{color}">
<div class="rc-top"><div class="rank">{i}</div><div><div class="rc-t">Route {i}{rec}</div>
<div class="rc-s">{r['label']}</div></div><div class="sp"></div>{ring(rel_pct, color)}</div>
<div class="stats"><div><div class="l">Distance</div><div class="v">{r['km']:.0f} km</div></div>
<div><div class="l">Drive time</div><div class="v">{r['minutes'] / 60:.1f} h</div></div>
<div><div class="l">Risk score</div><div class="v">{r['score']:.2f}</div></div>
<div><div class="l">Accidents nearby</div><div class="v">{r['n']:,}</div></div>
<div><div class="l">Blackspots</div><div class="v">{r['n_blackspots']}</div></div>
<div><div class="l">Fatal / Serious</div><div class="v">{r['fatal']} / {r['serious']}</div></div></div>
<div class="gm full">📍 Tap to navigate Route {i} in Google Maps ↗</div>
{xai_ui.card_block(r, exp)}{cond}
<div class="why-h">Why this rating</div><ul class="why">{why}</ul></div></a>""")


def briefing_card(lines, url, rec):
    li = "".join(f"<li>{x}</li>" for x in lines)
    return flat(f"""<div class="brief"><div><h3>🧭 AI TRIP BRIEFING</h3><ul>{li}</ul>
<div class="ai">Rule-based summary generated from the route statistics and SHAP values (no external AI service).
</div></div><div><a class="gm" href="{url}" target="_blank" rel="noopener">Navigate Route {rec + 1} in
Google Maps ↗</a></div></div>""")


def rank_panel(res, parts, rec, w_safe, colors):
    """Stacked bars showing how each route's ranking cost is made of risk and time (lower total wins)."""
    out = [f'<div class="note">Ranking cost = {w_safe:.0%} × normalised risk + {1 - w_safe:.0%} × normalised '
           f'drive time. Lowest total wins.</div>']
    for i, (r, p) in enumerate(zip(res, parts)):
        wr, wt = p["risk_part"] * 100, p["time_part"] * 100
        star = " ✅ recommended" if i == rec else ""
        out.append(f'<div class="rk"><div class="h"><span><b>Route {i + 1}</b>{star}</span>'
                   f'<span>total {p["total"]:.2f}</span></div><div class="stack">'
                   f'<div style="width:{wr:.1f}%;background:#E5484D" title="risk part"></div>'
                   f'<div style="width:{wt:.1f}%;background:#2D7FF9" title="time part"></div></div>'
                   f'<div class="note">risk part {p["risk_part"]:.2f} (risk {r["risk_total"]:.0f}) · '
                   f'time part {p["time_part"]:.2f} ({r["minutes"]:.0f} min)</div></div>')
    return flat("".join(out))
