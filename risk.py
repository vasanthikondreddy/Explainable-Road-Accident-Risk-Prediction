"""Route risk engine: corridor accident density + model-based conditions risk."""
import json
import numpy as np
import pandas as pd
import joblib
import requests
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN

EARTH_KM = 6371.0088
SEV_W = {"Fatal": 10, "Serious": 3, "Slight": 1}
HEADERS = {"User-Agent": "road-risk-major-project/1.0"}


# ---------- loading ----------
def load_assets(data_path="accidents_clean.csv", model_path="risk_model.pkl",
                feat_path="risk_features.json"):
    data = pd.read_csv(data_path)
    model = joblib.load(model_path)
    feats = json.load(open(feat_path))
    return data, model, feats


# ---------- geocoding & routing (need internet) ----------
def geocode(place):
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": place, "format": "json", "limit": 1, "countrycodes": "in"},
                     headers=HEADERS, timeout=20)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise ValueError(f"Could not find '{place}'. Try adding the city or state.")
    return float(js[0]["lat"]), float(js[0]["lon"]), js[0]["display_name"]


def _osrm(coords_str, alternatives=True):
    url = f"https://router.project-osrm.org/route/v1/driving/{coords_str}"
    r = requests.get(url, params={"alternatives": "true" if alternatives else "false",
                                  "overview": "full", "geometries": "geojson"},
                     headers=HEADERS, timeout=40)
    r.raise_for_status()
    js = r.json()
    if js.get("code") != "Ok":
        return []
    return [{"coords": rt["geometry"]["coordinates"], "km": rt["distance"] / 1000.0,
             "minutes": rt["duration"] / 60.0} for rt in js["routes"]]


def _same_route(a, b, tol_km=1.5):
    """True if two routes are essentially the same road (sampled points all close)."""
    pa, pb = densify(a["coords"], 5.0), densify(b["coords"], 5.0)
    tree = BallTree(np.radians(pb[:, ::-1]), metric="haversine")
    d, _ = tree.query(np.radians(pa[:, ::-1]), k=1)
    return float(np.mean(d[:, 0] * EARTH_KM <= tol_km)) > 0.9


def get_routes(src, dst, want=3):
    """src/dst = (lat, lon). Returns up to `want` distinct routes. If OSRM gives only one,
    force alternatives by routing through offset via-points either side of the straight line."""
    base = f"{src[1]},{src[0]};{dst[1]},{dst[0]}"
    routes = _osrm(base)
    if not routes:
        raise ValueError("No driving route found between these places.")
    if len(routes) >= want:
        return routes[:want]

    mid_lat, mid_lon = (src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2
    dlat, dlon = dst[0] - src[0], dst[1] - src[1]
    length = (dlat ** 2 + dlon ** 2) ** 0.5 or 1e-6
    # unit vector perpendicular to the straight line
    pl, po = -dlon / length, dlat / length
    for frac in (0.12, 0.25, 0.4):
        for sign in (1, -1):
            if len(routes) >= want:
                break
            vl, vo = mid_lat + sign * pl * length * frac, mid_lon + sign * po * length * frac
            try:
                cands = _osrm(f"{src[1]},{src[0]};{vo},{vl};{dst[1]},{dst[0]}", alternatives=False)
            except Exception:
                continue
            for c in cands:
                # reject silly detours (>60% longer than best) and duplicates
                if c["km"] <= min(r["km"] for r in routes) * 1.6 and \
                        not any(_same_route(c, r) for r in routes):
                    routes.append(c)
    return routes[:want]


# ---------- geometry ----------
def _hav_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def densify(coords, step_km=1.0):
    """Resample a [lon,lat] polyline to points roughly every step_km."""
    pts = np.asarray(coords, dtype=float)
    seg = _hav_km(pts[:-1, 1], pts[:-1, 0], pts[1:, 1], pts[1:, 0])
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total == 0:
        return pts
    grid = np.append(np.arange(0, total, step_km), total)
    return np.column_stack([np.interp(grid, cum, pts[:, 0]), np.interp(grid, cum, pts[:, 1])])


def corridor_accidents(data, route_pts, buffer_km=5.0):
    """Accidents within buffer_km of the route. route_pts = [[lon,lat],...]."""
    tree = BallTree(np.radians(route_pts[:, ::-1]), metric="haversine")
    acc = np.radians(data[["latitude", "longitude"]].values)
    dist, _ = tree.query(acc, k=1)
    return data[dist[:, 0] * EARTH_KM <= buffer_km]


# ---------- scoring ----------
def severity_score(corridor, km):
    w = corridor["Accident_Severity_label"].map(SEV_W).sum()
    return float(w) / max(km, 1.0)


def _mode(s, default):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else default


def _scenario_row(corridor, hour, dow, weather):
    """Model input for a trip on this corridor's typical road at the given conditions."""
    road = {c: _mode(corridor[c], "Unknown") for c in
            ["Road_Type_label", "Junction_Detail_label", "1st_Road_Class_label",
             "Urban_or_Rural_Area_label"]}
    speed = _mode(corridor["Speed_limit"], 30)
    light = "Daylight" if 7 <= hour <= 18 else "Dark - lit"
    surface = "Wet/damp" if "Rain" in weather else "Dry"
    row = {**road, "Light_Conditions_label": light, "Weather_Conditions_label": weather,
           "Road_Surface_Conditions_label": surface, "hour": hour, "Day_of_Week": dow,
           "Speed_limit": speed, "is_weekend": int(dow in (1, 7))}
    return row, road, speed


def conditions_risk(model, feats, corridor, hour, dow, weather):
    """Model probability (Serious/Fatal) for the traveller's conditions on the route's
    typical road, versus a reference trip (noon, weekday, fine, dry)."""
    if len(corridor) < 20:
        return None

    def prob(h, wd, w):
        row, _, _ = _scenario_row(corridor, h, wd, w)
        return float(model.predict_proba(pd.DataFrame([row])[feats["cat"] + feats["num"]])[0, 1])

    row, road, speed = _scenario_row(corridor, hour, dow, weather)
    p_now = prob(hour, dow, weather)
    p_ref = prob(12, 4, "Fine")
    return {"p_now": p_now, "p_ref": p_ref, "ratio": p_now / p_ref if p_ref else np.nan,
            "speed": int(speed), "road": road,
            "row": pd.DataFrame([row])[feats["cat"] + feats["num"]]}


def hotspots(corridor, top=5, cell_deg=0.02):
    """Group nearby accidents into ~2 km grid cells and rank cells by severity weight.
    These are the 'black spots' to highlight on the map."""
    if len(corridor) == 0:
        return pd.DataFrame()
    c = corridor.copy()
    c["w"] = c["Accident_Severity_label"].map(SEV_W)
    c["cell"] = list(zip((c["latitude"] / cell_deg).round().astype(int),
                         (c["longitude"] / cell_deg).round().astype(int)))
    g = c.groupby("cell").agg(lat=("latitude", "mean"), lon=("longitude", "mean"),
                              accidents=("w", "size"), weight=("w", "sum"),
                              fatal=("Accident_Severity", lambda s: int((s == 1).sum())),
                              peak_hour=("hour", lambda s: int(s.mode().iloc[0])),
                              junction=("Junction_Detail_label", lambda s: s.mode().iloc[0]),
                              road=("Road_Type_label", lambda s: s.mode().iloc[0]))
    return g.sort_values("weight", ascending=False).head(top).reset_index(drop=True)


def stats_dow(py_weekday):
    """Python Monday=0 -> STATS19 Sunday=1."""
    return (py_weekday + 1) % 7 + 1


# ---------- extra features ----------
def hourly_profile(model, feats, corridor, dow, weather):
    """Model probability of a Serious/Fatal accident for each hour 0-23 on this route's typical road."""
    if len(corridor) < 20:
        return None
    road = {c: _mode(corridor[c], "Unknown") for c in
            ["Road_Type_label", "Junction_Detail_label", "1st_Road_Class_label",
             "Urban_or_Rural_Area_label"]}
    speed = _mode(corridor["Speed_limit"], 30)
    surface = "Wet/damp" if "Rain" in weather else "Dry"
    rows = []
    for h in range(24):
        light = "Daylight" if 7 <= h <= 18 else "Dark - lit"
        rows.append({**road, "Light_Conditions_label": light, "Weather_Conditions_label": weather,
                     "Road_Surface_Conditions_label": surface, "hour": h, "Day_of_Week": dow,
                     "Speed_limit": speed, "is_weekend": int(dow in (1, 7))})
    X = pd.DataFrame(rows)[feats["cat"] + feats["num"]]
    return model.predict_proba(X)[:, 1]


def route_alerts(route, hour, weather):
    """Ordered list of alerts along a route: blackspots with distance from start + trip-level warnings."""
    out = []
    bsr = route.get("blackspots")
    pts = route["pts"]
    if bsr is not None and len(bsr):
        tree = BallTree(np.radians(pts[:, ::-1]), metric="haversine")
        _, idx = tree.query(np.radians(bsr[["lat", "lon"]].values), k=1)
        seg = _hav_km(pts[:-1, 1], pts[:-1, 0], pts[1:, 1], pts[1:, 0])
        cum = np.concatenate([[0], np.cumsum(seg)])
        for (_, p), i in zip(bsr.iterrows(), idx[:, 0]):
            km_at = float(cum[i])
            tip = {"Critical": "Drive slowly and avoid overtaking.",
                   "High": "Stay alert and keep a safe distance.",
                   "Moderate": "Take care."}[p["level"]]
            out.append({"km": km_at, "level": p["level"], "lat": p["lat"], "lon": p["lon"],
                        "text": f"{p['level']} blackspot at km {km_at:.0f}: {int(p['n'])} accidents "
                                f"({int(p['fatal'])} fatal, {int(p['serious'])} serious). {tip}"})
    out.sort(key=lambda a: a["km"])
    trip = []
    if hour >= 19 or hour <= 5:
        trip.append("Night driving: visibility is lower and fatigue risk is higher. Take a break every 2 hours.")
    if weather != "Fine":
        trip.append(f"Weather is {weather.lower()}: reduce speed and increase following distance.")
    if route["minutes"] > 240:
        trip.append(f"Long trip ({route['minutes'] / 60:.1f} h): plan a rest stop at least every 2 hours.")
    return out, trip


# ---------- multi-objective routing & live weather ----------
def pareto_front(results):
    """Indices of routes not dominated on (drive time, total risk): no other route is both
    faster and safer."""
    keep = []
    for i, a in enumerate(results):
        dominated = any(b["minutes"] <= a["minutes"] and b["risk_total"] <= a["risk_total"] and
                        (b["minutes"] < a["minutes"] or b["risk_total"] < a["risk_total"])
                        for j, b in enumerate(results) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def tradeoff(results):
    """Safest route vs fastest route: % risk reduction and extra minutes."""
    f = min(range(len(results)), key=lambda i: results[i]["minutes"])
    sf = min(range(len(results)), key=lambda i: results[i]["risk_total"])
    rf, rs = results[f], results[sf]
    red = (rf["risk_total"] - rs["risk_total"]) / rf["risk_total"] * 100 if rf["risk_total"] else 0.0
    return {"fastest": f, "safest": sf, "risk_reduction_pct": float(red),
            "extra_minutes": float(rs["minutes"] - rf["minutes"]),
            "extra_km": float(rs["km"] - rf["km"])}


def live_weather(lat, lon):
    """Current weather from Open-Meteo (free, no key). Returns (label, description) or None."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": lat, "longitude": lon, "current": "weather_code,temperature_2m"},
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        cur = r.json()["current"]
        code = int(cur["weather_code"])
        if code in (45, 48):
            lab = "Fog/mist"
        elif code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99):
            lab = "Raining"
        else:
            lab = "Fine"
        return lab, f"{cur['temperature_2m']:.0f} C, code {code}"
    except Exception:
        return None


# ---------- explanation ----------
def explain(corridor, data):
    n = len(corridor)
    if n < 20:
        return ["Too few recorded accidents near this route to draw conclusions."]
    out = []

    def pct(mask_c, mask_d):
        return mask_c.mean() * 100, mask_d.mean() * 100

    sf_c, sf_d = pct(corridor["Accident_Severity"] <= 2, data["Accident_Severity"] <= 2)
    out.append(f"{sf_c:.0f}% of nearby accidents were serious or fatal "
               f"({'above' if sf_c > sf_d * 1.1 else 'below' if sf_c < sf_d * 0.9 else 'close to'} "
               f"the dataset average of {sf_d:.0f}%).")

    dk_c, dk_d = pct(corridor["Light_Conditions_label"] != "Daylight",
                     data["Light_Conditions_label"] != "Daylight")
    if dk_c > dk_d * 1.1:
        out.append(f"{dk_c:.0f}% happened in darkness (average {dk_d:.0f}%): night driving adds risk here.")

    wt_c, wt_d = pct(corridor["Road_Surface_Conditions_label"] == "Wet/damp",
                     data["Road_Surface_Conditions_label"] == "Wet/damp")
    if wt_c > wt_d * 1.1:
        out.append(f"{wt_c:.0f}% happened on wet roads (average {wt_d:.0f}%).")

    jn_c, jn_d = pct(corridor["Junction_Detail_label"] != "Not at junction",
                     data["Junction_Detail_label"] != "Not at junction")
    if jn_c > jn_d * 1.1:
        out.append(f"{jn_c:.0f}% were at junctions (average {jn_d:.0f}%): watch intersections.")

    hi_c, hi_d = pct(corridor["Speed_limit"] >= 60, data["Speed_limit"] >= 60)
    if hi_c > hi_d * 1.1:
        out.append(f"{hi_c:.0f}% were on roads with a 60+ speed limit (average {hi_d:.0f}%).")

    peak = int(corridor["hour"].value_counts().idxmax())
    out.append(f"Most accidents here occur around {peak:02d}:00. "
               f"Most common road type: {_mode(corridor['Road_Type_label'], 'Unknown')}.")
    return out


def level_labels(scores):
    """Relative labels among alternatives."""
    if len(scores) == 1:
        return ["Only route"]
    lo, hi = min(scores), max(scores)
    labs = []
    for s in scores:
        if hi - lo < 1e-9 or (hi / max(lo, 1e-9)) < 1.15:
            labs.append("Similar")
        elif s == lo:
            labs.append("Lowest risk")
        elif s == hi:
            labs.append("Highest risk")
        else:
            labs.append("Medium risk")
    return labs


# ---------- blackspots ----------
def detect_blackspots(data, cell_deg=0.01, min_accidents=10, top_share=0.25):
    """Grid-based blackspots (cells ~1.1 km). A cell needs >= min_accidents records; of those,
    the top `top_share` by severity weight are blackspots. Grid works better than DBSCAN here
    because the data is extremely dense around a few points and DBSCAN merges it into
    city-sized blobs.
    Returns DataFrame: id, lat, lon, n, fatal, serious, weight, radius_km, level."""
    d = data.assign(_w=data["Accident_Severity_label"].map(SEV_W),
                    _gy=np.floor(data["latitude"] / cell_deg).astype(int),
                    _gx=np.floor(data["longitude"] / cell_deg).astype(int))
    g = d.groupby(["_gy", "_gx"]).agg(
        lat=("latitude", "mean"), lon=("longitude", "mean"), n=("_w", "size"),
        fatal=("Accident_Severity", lambda x: int((x == 1).sum())),
        serious=("Accident_Severity", lambda x: int((x == 2).sum())),
        weight=("_w", "sum")).reset_index(drop=True)
    g = g[g["n"] >= min_accidents]
    if g.empty:
        return pd.DataFrame(columns=["id", "lat", "lon", "n", "fatal", "serious",
                                     "weight", "radius_km", "level"])
    g = g[g["weight"] >= g["weight"].quantile(1 - top_share)].copy()
    g = g.sort_values("weight", ascending=False).reset_index(drop=True)
    g["id"] = g.index
    g["radius_km"] = cell_deg * 111.0 / 2 + 0.3
    q_hi, q_mid = g["weight"].quantile([0.8, 0.4])
    g["level"] = np.where(g["weight"] >= q_hi, "Critical",
                          np.where(g["weight"] >= q_mid, "High", "Moderate"))
    return g[["id", "lat", "lon", "n", "fatal", "serious", "weight", "radius_km", "level"]]


def blackspots_on_route(blackspots, route_pts, buffer_km=5.0):
    """Blackspots whose centre lies within buffer_km of the route."""
    if blackspots is None or len(blackspots) == 0:
        return blackspots.iloc[0:0] if blackspots is not None else pd.DataFrame()
    tree = BallTree(np.radians(route_pts[:, ::-1]), metric="haversine")
    d, _ = tree.query(np.radians(blackspots[["lat", "lon"]].values), k=1)
    return blackspots[d[:, 0] * EARTH_KM <= buffer_km].copy()


def rank_routes(results, w_risk=0.5):
    """Optimise: combined = w_risk * norm(risk) + (1-w_risk) * norm(time). Lower is better."""
    def norm(v):
        lo, hi = min(v), max(v)
        return [0.0 if hi - lo < 1e-9 else (x - lo) / (hi - lo) for x in v]
    r = norm([x["risk_total"] for x in results])
    t = norm([x["minutes"] for x in results])
    for x, a, b in zip(results, r, t):
        x["combined"] = w_risk * a + (1 - w_risk) * b
    return min(range(len(results)), key=lambda i: results[i]["combined"])


def analyze(data, model, feats, routes, hour, dow, weather, buffer_km=5.0, blackspots=None):
    results = []
    for rt in routes:
        pts = densify(rt["coords"], 1.0)
        cor = corridor_accidents(data, pts, buffer_km)
        bsr = blackspots_on_route(blackspots, pts, min(buffer_km, 3.0)) if blackspots is not None else None
        results.append({
            **rt, "pts": pts, "corridor": cor,
            "blackspots": bsr,
            "n_blackspots": 0 if bsr is None else len(bsr),
            # total risk = severity-weighted accidents in corridor + extra penalty per blackspot weight
            "risk_total": severity_score(cor, rt["km"]) * rt["km"] + (0 if bsr is None else float(bsr["weight"].sum()) * 0.5),
            "n": len(cor),
            "fatal": int((cor["Accident_Severity"] == 1).sum()),
            "serious": int((cor["Accident_Severity"] == 2).sum()),
            "score": severity_score(cor, rt["km"]),
            "cond": conditions_risk(model, feats, cor, hour, dow, weather),
            "why": explain(cor, data),
            "spots": hotspots(cor),
        })
    for r, lab in zip(results, level_labels([r["score"] for r in results])):
        r["label"] = lab
    return results