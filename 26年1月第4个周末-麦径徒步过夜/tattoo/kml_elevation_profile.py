#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract distance-elevation profile from a KML and export a PURE LINE SVG.
No matplotlib needed.

Output: <basename>_elev_line.svg and <basename>_profile.csv

Usage:
  python kml_to_elevation_line_svg.py "麦理浩径2段＋破边洲.kml" --width-mm 180 --height-mm 25
  python kml_to_elevation_line_svg.py "xxx.kml" --width-mm 220 --height-mm 28 --smooth 7 --samples 1500
"""

import argparse
import csv
import math
import os
from xml.etree import ElementTree as ET


KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def parse_kml_points(kml_path: str):
    """
    Return list of (lon, lat, ele_m or None) in file order.
    Supports:
      - LineString/coordinates: "lon,lat,alt"
      - gx:Track/gx:coord: "lon lat alt"
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": KML_NS, "gx": GX_NS}

    pts = []

    for node in root.findall(".//kml:LineString/kml:coordinates", ns):
        if not node.text:
            continue
        text = node.text.strip()
        for token in text.replace("\n", " ").split():
            parts = token.split(",")
            if len(parts) < 2:
                continue
            lon = float(parts[0])
            lat = float(parts[1])
            ele = float(parts[2]) if len(parts) >= 3 and parts[2] != "" else None
            pts.append((lon, lat, ele))

    for node in root.findall(".//gx:Track/gx:coord", ns):
        if not node.text:
            continue
        parts = node.text.strip().split()
        if len(parts) < 2:
            continue
        lon = float(parts[0])
        lat = float(parts[1])
        ele = float(parts[2]) if len(parts) >= 3 else None
        pts.append((lon, lat, ele))

    return pts


def fill_elevations(elev):
    e = list(elev)

    # forward fill
    last = None
    for i in range(len(e)):
        if e[i] is None:
            e[i] = last
        else:
            last = e[i]

    # back fill
    last = None
    for i in range(len(e) - 1, -1, -1):
        if e[i] is None:
            e[i] = last
        else:
            last = e[i]

    # if all missing
    if all(v is None for v in e):
        return [0.0] * len(e)

    return [0.0 if v is None else float(v) for v in e]


def compute_distance_km(points, gap_threshold_m=None):
    lons, lats, _ = zip(*points)
    dist_m = [0.0]
    cum = 0.0
    for i in range(1, len(points)):
        d = haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i])
        if gap_threshold_m is not None and d > gap_threshold_m:
            # don't connect across big jumps
            dist_m.append(cum)
            continue
        cum += d
        dist_m.append(cum)
    return [d / 1000.0 for d in dist_m]


def moving_average(y, window):
    if window <= 1:
        return y
    w = int(window)
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = []
    for i in range(len(y)):
        a = max(0, i - half)
        b = min(len(y), i + half + 1)
        out.append(sum(y[a:b]) / (b - a))
    return out


def resample_by_distance(dist_km, elev_m, n_samples):
    """
    Resample to fixed count along distance axis (linear interpolation).
    """
    if n_samples is None or n_samples <= 0 or n_samples >= len(dist_km):
        return dist_km, elev_m

    total = dist_km[-1]
    if total <= 0:
        return dist_km, elev_m

    xs = [i * total / (n_samples - 1) for i in range(n_samples)]

    j = 0
    out_e = []
    for x in xs:
        while j < len(dist_km) - 2 and dist_km[j + 1] < x:
            j += 1
        x0, x1 = dist_km[j], dist_km[j + 1]
        y0, y1 = elev_m[j], elev_m[j + 1]
        if x1 == x0:
            out_e.append(y0)
        else:
            t = (x - x0) / (x1 - x0)
            out_e.append(y0 + t * (y1 - y0))
    return xs, out_e


def write_csv(path, dist_km, elev_m):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distance_km", "elevation_m"])
        for d, e in zip(dist_km, elev_m):
            w.writerow([f"{d:.6f}", f"{e:.2f}"])


def to_svg_polyline(dist_km, elev_m, width_mm, height_mm, padding_mm, invert_y=True, stroke_mm=0.35):
    """
    Map (dist, elev) into SVG coordinates.
    Pure polyline, no axes, no text.
    """
    W = float(width_mm)
    H = float(height_mm)
    P = float(padding_mm)

    # Prevent division by zero
    x_min, x_max = dist_km[0], dist_km[-1]
    if x_max <= x_min:
        x_max = x_min + 1e-9

    y_min, y_max = min(elev_m), max(elev_m)
    if y_max <= y_min:
        y_max = y_min + 1e-9

    inner_w = max(1e-9, W - 2 * P)
    inner_h = max(1e-9, H - 2 * P)

    pts = []
    for x, y in zip(dist_km, elev_m):
        sx = P + (x - x_min) / (x_max - x_min) * inner_w
        sy = P + (y - y_min) / (y_max - y_min) * inner_h
        if invert_y:
            sy = H - sy  # SVG y goes down
        pts.append(f"{sx:.3f},{sy:.3f}")

    poly = " ".join(pts)

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">
  <polyline fill="none" stroke="black" stroke-width="{stroke_mm}mm" stroke-linejoin="round" stroke-linecap="round"
            points="{poly}" />
</svg>
"""
    return svg


def main():
    ap = argparse.ArgumentParser(description="KML -> pure elevation line SVG (distance vs elevation). No matplotlib.")
    ap.add_argument("kml", help="Input .kml file path")
    ap.add_argument("--width-mm", type=float, default=180.0, help="SVG width in mm (longer => more 'stretched')")
    ap.add_argument("--height-mm", type=float, default=25.0, help="SVG height in mm (smaller => flatter)")
    ap.add_argument("--padding-mm", type=float, default=2.0, help="Padding in mm")
    ap.add_argument("--stroke-mm", type=float, default=0.35, help="Line stroke width in mm")
    ap.add_argument("--smooth", type=int, default=1, help="Moving-average window (odd preferred). 1=off")
    ap.add_argument("--samples", type=int, default=1500, help="Resample points to this count (0 disables)")
    ap.add_argument("--gap-threshold-m", type=float, default=None,
                    help="If set, do not connect segments if jump > threshold meters.")
    args = ap.parse_args()

    kml_path = args.kml
    if not os.path.isfile(kml_path):
        raise SystemExit(f"File not found: {kml_path}")

    pts = parse_kml_points(kml_path)
    if len(pts) < 2:
        raise SystemExit("No usable track points found in KML.")

    dist_km = compute_distance_km(pts, gap_threshold_m=args.gap_threshold_m)
    elev = fill_elevations([p[2] for p in pts])

    # Optional resample (makes line cleaner + file lighter)
    if args.samples and args.samples > 0:
        dist_km, elev = resample_by_distance(dist_km, elev, args.samples)

    # Optional smooth (reduces GPS elevation noise)
    elev = moving_average(elev, args.smooth)

    base = os.path.splitext(os.path.basename(kml_path))[0]
    out_csv = f"{base}_profile.csv"
    out_svg = f"{base}_elev_line.svg"

    write_csv(out_csv, dist_km, elev)

    svg = to_svg_polyline(
        dist_km=dist_km,
        elev_m=elev,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
        padding_mm=args.padding_mm,
        stroke_mm=args.stroke_mm,
    )
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)

    total_km = dist_km[-1]
    print(f"Points used: {len(dist_km)}")
    print(f"Total distance: {total_km:.3f} km")
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
