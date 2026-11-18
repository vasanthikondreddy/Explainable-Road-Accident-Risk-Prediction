"""Interactive map: layered Folium map with SHAP popups, pulsing blackspots and Google Maps hand-off."""
import folium
from branca.element import MacroElement, Template
from folium.plugins import AntPath, Fullscreen, HeatMap, MiniMap

import risk
import xai_ui

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
# "Streets" is the classic OpenStreetMap map (first = default). CARTO basemaps now need an API key, so the
# other styles use free Esri tiles.
TILES = {"Streets": ("OpenStreetMap", None),
         "Satellite": (ESRI.format("World_Imagery"), "Tiles &copy; Esri"),
         "Light gray": (ESRI.format("Canvas/World_Light_Gray_Base"), "Tiles &copy; Esri"),
         "Dark gray": (ESRI.format("Canvas/World_Dark_Gray_Base"), "Tiles &copy; Esri")}
BS_COL = {"Critical": "#E5484D", "High": "#F5A524", "Moderate": "#E5C100"}
MAP_CSS = ("<style>.bsp{width:18px;height:18px;border-radius:50%;background:var(--c);border:2px solid #fff;"
           "animation:pl 1.8s infinite}@keyframes pl{0%{box-shadow:0 0 0 0 var(--c)}70%{box-shadow:0 0 0 16px "
           "rgba(0,0,0,0)}100%{box-shadow:0 0 0 0 rgba(0,0,0,0)}}.pin{width:30px;height:30px;border-radius:50% 50% "
           "50% 0;transform:rotate(-45deg);display:flex;align-items:center;justify-content:center;border:2px solid "
           "#fff;box-shadow:0 2px 8px rgba(0,0,0,.5)}.pin b{transform:rotate(45deg);color:#fff;font:700 13px Arial}"
           ".leaflet-popup-content-wrapper{border-radius:14px}</style>")


class OpenOnClick(MacroElement):
    """Clicking the route line opens `url` in a new tab (used for the 'click a route to open Google Maps' mode)."""
    _template = Template("{% macro script(this, kwargs) %}{{ this.target.get_name() }}.on('click', function()"
                         "{ window.open({{ this.url|tojson }}, '_blank'); });{% endmacro %}")

    def __init__(self, target, url):
        super().__init__()
        self._name = "OpenOnClick"
        self.target, self.url = target, url


def _pin(letter, color):
    return folium.DivIcon(icon_size=(30, 30), icon_anchor=(15, 30),
                          html=f'<div class="pin" style="background:{color}"><b>{letter}</b></div>')


def build_map(data, res, s, e, rec, exps, bs_exp, colors, urls, style="Streets", show_heat=False,
              direct_open=False, animate=True):
    tiles, attr = TILES[style]
    m = folium.Map(location=[(s[0] + e[0]) / 2, (s[1] + e[1]) / 2], zoom_start=7, tiles=tiles, attr=attr,
                   control_scale=True)
    m.get_root().header.add_child(folium.Element(MAP_CSS))
    g_routes = folium.FeatureGroup("Routes").add_to(m)
    g_acc = folium.FeatureGroup("Recorded accidents", show=True).add_to(m)
    g_bs = folium.FeatureGroup("Blackspots").add_to(m)
    g_lab = folium.FeatureGroup("SHAP labels").add_to(m)

    for i, r in enumerate(res, 1):
        col, best = colors[r["label"]], i - 1 == rec
        ll = [(y, x) for x, y in r["coords"]]
        folium.PolyLine(ll, color=col, weight=16 if best else 11, opacity=.18).add_to(g_routes)  # glow
        poly = folium.PolyLine(ll, color=col, weight=7 if best else 4, opacity=1 if best else .75,
                               tooltip=f"Route {i}: {r['label']}{' (recommended)' if best else ''}. "
                                       + ("Click to open in Google Maps." if direct_open
                                          else "Click for the SHAP explanation + Google Maps link."))
        if not direct_open:
            poly.add_child(xai_ui.route_popup(i, r, exps.get(i - 1), col, best, urls[i - 1]))
        poly.add_to(g_routes)
        if direct_open:
            OpenOnClick(poly, urls[i - 1]).add_to(m)
        if best and animate:
            AntPath(ll, color="#FFFFFF", pulse_color=col, weight=3, delay=1200, dash_array=[8, 40],
                    opacity=.9).add_to(g_routes)
        for _, a in r["corridor"].sample(min(120, len(r["corridor"])), random_state=1).iterrows():
            folium.CircleMarker((a["latitude"], a["longitude"]), radius=3, color=col, fill=True,
                                fill_opacity=.7, weight=0).add_to(g_acc)

    if show_heat:
        hm = data.sample(min(8000, len(data)), random_state=1)
        HeatMap([[la, lo, w] for la, lo, w in zip(hm["latitude"], hm["longitude"],
                hm["Accident_Severity_label"].map(risk.SEV_W))], radius=14, blur=18, min_opacity=.3,
                name="Heatmap").add_to(m)

    seen = set()
    for r in res:
        b = r["blackspots"]
        for _, p in (b.iterrows() if b is not None else []):
            if p["id"] in seen:
                continue
            seen.add(p["id"])
            c = BS_COL[p["level"]]
            tip = (f"{p['level']} blackspot: {int(p['n'])} accidents, {int(p['fatal'])} fatal, "
                   f"{int(p['serious'])} serious. Click for why.")
            pop = xai_ui.blackspot_popup(p, bs_exp.get(int(p["id"])), c)
            folium.Circle((p["lat"], p["lon"]), radius=p["radius_km"] * 1000, color=c, fill=True,
                          fill_opacity=.12, weight=1).add_to(g_bs)
            folium.Marker((p["lat"], p["lon"]), tooltip=tip, popup=pop,
                          icon=folium.DivIcon(icon_size=(18, 18), icon_anchor=(9, 9),
                                              html=f'<div class="bsp" style="--c:{c}"></div>')).add_to(g_bs)

    xai_ui.add_route_labels(g_lab, res, exps, colors, rec)
    folium.Marker((s[0], s[1]), tooltip="Start", icon=_pin("A", "#19C3B1")).add_to(m)
    folium.Marker((e[0], e[1]), tooltip="Destination", icon=_pin("B", "#E5484D")).add_to(m)
    allp = [c for r in res for c in r["coords"][::20]]
    m.fit_bounds([[min(y for x, y in allp), min(x for x, y in allp)],
                  [max(y for x, y in allp), max(x for x, y in allp)]], max_zoom=12)
    Fullscreen(position="topright").add_to(m)
    MiniMap(toggle_display=True, minimized=True, position="bottomleft").add_to(m)
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    return m