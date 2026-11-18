"""Google Maps deep links. The URL carries origin, destination and evenly spaced waypoints taken from
the analysed route, so Google Maps follows (approximately) the same road the risk analysis used."""
from urllib.parse import quote

import numpy as np

import risk


def waypoints(coords, n=8):
    """n points spread evenly along the route (not including the two ends). coords = [[lon, lat], ...]."""
    pts = risk.densify(coords, 1.0)
    if len(pts) < n + 2:
        return []
    idx = np.linspace(0, len(pts) - 1, n + 2)[1:-1].astype(int)
    return [(float(pts[i][1]), float(pts[i][0])) for i in idx]


def directions_url(coords, n_way=8, mode="driving"):
    """Link that opens Google Maps (web or app) with this exact trip. Max 9 waypoints are allowed."""
    o, d = coords[0], coords[-1]
    url = (f"https://www.google.com/maps/dir/?api=1&origin={o[1]:.6f},{o[0]:.6f}"
           f"&destination={d[1]:.6f},{d[0]:.6f}&travelmode={mode}&dir_action=navigate")
    wp = waypoints(coords, min(n_way, 9))
    if wp:
        url += "&waypoints=" + quote("|".join(f"{a:.6f},{b:.6f}" for a, b in wp), safe=",")
    return url
