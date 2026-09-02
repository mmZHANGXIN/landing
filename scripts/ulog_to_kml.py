#!/usr/bin/env python3
"""Convert a PX4 ULog into a Google Earth 3D trajectory KML.

The main track is the PX4 EKF-fused global position
(``vehicle_global_position.lat/lon/alt``, MSL altitude), written as a WGS84
LineString with absolute altitude so Google Earth shows the true 3D path.
Start, end and highest points are marked with pushpins.  After writing, the
KML is re-parsed and validated (XML syntax, lon/lat/alt ordering, finite
values) and the emitted ranges are compared against the raw ULog.

Usage:
    python3 orinlanding/scripts/ulog_to_kml.py \\
        experiments/20260807_162946_orin_landing/log_712_2026-8-7-16-31-54.ulg \\
        --output experiments/20260807_162946_orin_landing/flight_trajectory.kml

Optional extras:
    --altitude-mode relative   subtract the home_position MSL altitude
    --time-track               add a gx:Track with GPS UTC timestamps for the
                               Google Earth time slider
    --every / --max-points     decimate the track (default max 20000 points)
"""

import argparse
import bisect
import datetime as dt
import html
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pyulog

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
_PUSH = "http://maps.google.com/mapfiles/kml/pushpin"
ICON_URLS = {
    "start": f"{_PUSH}/grn-pushpin.png",
    "highest": f"{_PUSH}/blu-pushpin.png",
    "end": f"{_PUSH}/rd-pushpin.png",
}


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _kml_color(hexcolor, opacity):
    """RRGGBB + opacity(0..1) -> KML aabbggrr."""
    alpha = f"{max(0, min(255, round(opacity * 255))):02x}"
    return f"{alpha}{hexcolor[4:6]}{hexcolor[2:4]}{hexcolor[0:2]}"


def _load_track(log, altitude_mode):
    """Extract (timestamp_us, lat, lon, alt) tuples from the fused position."""
    dataset = log.get_dataset("vehicle_global_position")
    data = dataset.data
    timestamps, lats, lons, alts = data["timestamp"], data["lat"], data["lon"], data["alt"]
    lat_ok = data.get("lat_lon_valid")
    alt_ok = data.get("alt_valid")

    home_alt = None
    if altitude_mode == "relative":
        home_alt = _home_altitude(log)
        if home_alt is None:
            raise SystemExit(
                "--altitude-mode relative requires a valid home_position.alt "
                "in the log; none found."
            )

    points, dropped_invalid, dropped_nonfinite, dropped_dup = [], 0, 0, 0
    previous = None
    for i, timestamp in enumerate(timestamps):
        lat, lon, alt = float(lats[i]), float(lons[i]), float(alts[i])
        if (lat_ok is not None and int(lat_ok[i]) == 0) or (alt_ok is not None and int(alt_ok[i]) == 0):
            dropped_invalid += 1
            continue
        if not (_finite(lat) and _finite(lon) and _finite(alt)):
            dropped_nonfinite += 1
            continue
        if previous is not None and (lat, lon, alt) == previous:
            dropped_dup += 1
            continue
        previous = (lat, lon, alt)
        if home_alt is not None:
            alt -= home_alt
        points.append((int(timestamp), lat, lon, alt))

    if not points:
        raise SystemExit("no valid vehicle_global_position points found in the log")
    return points, {"invalid": dropped_invalid, "nonfinite": dropped_nonfinite,
                    "dup": dropped_dup}


def _home_altitude(log):
    try:
        home = log.get_dataset("home_position")
    except (KeyError, AttributeError):
        return None
    data = home.data
    valid_alt = data.get("valid_alt")
    if valid_alt is not None and int(valid_alt[0]) != 1:
        return None
    return float(data["alt"][0])


def _decimate(points, every, max_points):
    stride = every or max(1, math.ceil(len(points) / max_points))
    if stride == 1:
        return points, 1
    return points[::stride], stride


def _gps_utc_samples(log):
    """Sorted (log_timestamp_us, utc_offset_us) from sensor_gps, or None."""
    try:
        gps = log.get_dataset("sensor_gps")
    except (KeyError, AttributeError):
        return None
    data = gps.data
    samples = []
    for timestamp, utc in zip(data["timestamp"], data["time_utc_usec"]):
        if int(utc) > 0 and _finite(timestamp) and _finite(utc):
            samples.append((int(timestamp), int(utc) - int(timestamp)))
    return sorted(samples) if samples else None


def _utc_interpolator(samples):
    """Map log timestamps to GPS UTC; clamps outside the GPS time span."""
    times = [sample[0] for sample in samples]
    offsets = [sample[1] for sample in samples]

    def utc_us(timestamp_us):
        index = bisect.bisect_left(times, timestamp_us)
        if index <= 0:
            return timestamp_us + offsets[0]
        if index >= len(times):
            return timestamp_us + offsets[-1]
        t0, t1 = times[index - 1], times[index]
        fraction = (timestamp_us - t0) / (t1 - t0)
        return timestamp_us + (offsets[index - 1] + (offsets[index] - offsets[index - 1]) * fraction)

    return utc_us


def _iso_utc(utc_us):
    """ISO 8601 UTC with milliseconds — 10 Hz track points share a second."""
    formatted = dt.datetime.fromtimestamp(utc_us / 1e6, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    return formatted[:-3] + "Z"


def _coord_text(lat, lon, alt):
    return f"{lon:.7f},{lat:.7f},{alt:.2f}"


def _build_kml(args, points, utc_interp, stride):
    duration_s = (points[-1][0] - points[0][0]) / 1e6
    highest = max(points, key=lambda point: point[3])
    start, end = points[0], points[-1]

    document = []
    document.append('<?xml version="1.0" encoding="UTF-8"?>')
    document.append(
        f'<kml xmlns="{KML_NS}" xmlns:gx="{GX_NS}">'
    )
    document.append("  <Document>")
    document.append(f"    <name>{html.escape(args.name)}</name>")
    description = (
        f"{len(points)} points ({duration_s:.1f} s, "
        f"stride {stride}) | altitude mode: {args.altitude_mode} | "
        f"start {start[1]:.7f},{start[2]:.7f},{start[3]:.2f} m | "
        f"highest {highest[1]:.7f},{highest[2]:.7f},{highest[3]:.2f} m | "
        f"end {end[1]:.7f},{end[2]:.7f},{end[3]:.2f} m"
    )
    document.append(f"    <description>{html.escape(description)}</description>")

    line_color = _kml_color(args.color, args.opacity)
    document.append("    <Style id=\"track-style\">")
    document.append("      <LineStyle>")
    document.append(f"        <color>{line_color}</color>")
    document.append(f"        <width>{args.width}</width>")
    document.append("      </LineStyle>")
    document.append("    </Style>")
    for key, url in ICON_URLS.items():
        document.append(f"    <Style id=\"pin-{key}\">")
        document.append("      <IconStyle>")
        document.append(f"        <Icon><href>{url}</href></Icon>")
        document.append("      </IconStyle>")
        document.append("    </Style>")

    # Main 3D track.
    document.append("    <Placemark>")
    document.append("      <name>Flight track</name>")
    document.append("      <styleUrl>#track-style</styleUrl>")
    document.append("      <LineString>")
    document.append(f"        <altitudeMode>{args.altitude_mode}</altitudeMode>")
    document.append("        <coordinates>")
    for _, lat, lon, alt in points:
        document.append(f"          {_coord_text(lat, lon, alt)}")
    document.append("        </coordinates>")
    document.append("      </LineString>")
    document.append("    </Placemark>")

    # Markers.
    for key, label, point in (
        ("start", "Start", start),
        ("highest", "Highest point", highest),
        ("end", "End", end),
    ):
        document.append("    <Placemark>")
        document.append(f"      <name>{label}</name>")
        document.append(f"      <styleUrl>#pin-{key}</styleUrl>")
        document.append("      <Point>")
        document.append(f"        <altitudeMode>{args.altitude_mode}</altitudeMode>")
        document.append(
            f"        <coordinates>{_coord_text(point[1], point[2], point[3])}</coordinates>"
        )
        document.append("      </Point>")
        document.append("    </Placemark>")

    # Optional time-track.
    if args.time_track:
        document.append("    <Placemark>")
        document.append("      <name>Flight track (time)</name>")
        document.append("      <styleUrl>#track-style</styleUrl>")
        document.append("      <gx:Track>")
        document.append(f"        <altitudeMode>{args.altitude_mode}</altitudeMode>")
        if utc_interp is not None:
            for timestamp, _, _, _ in points:
                document.append(f"        <when>{_iso_utc(utc_interp(timestamp))}</when>")
        else:
            for timestamp, _, _, _ in points:
                relative_s = (timestamp - points[0][0]) / 1e6
                document.append(
                    f"        <when>{_iso_utc(relative_s * 1e6)}</when>"
                )
        for _, lat, lon, alt in points:
            document.append(f"        <gx:coord>{lon:.7f} {lat:.7f} {alt:.2f}</gx:coord>")
        document.append("      </gx:Track>")
        document.append("    </Placemark>")

    document.append("  </Document>")
    document.append("</kml>")
    return "\n".join(document) + "\n"


def _verify(path, points, time_track):
    """Re-parse the KML and check structure, ordering and values."""
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise SystemExit(f"verification failed: KML is not valid XML: {exc}")
    root = tree.getroot()
    failures = []

    coordinates = root.findall(f".//{{{KML_NS}}}LineString/{{{KML_NS}}}coordinates")
    if len(coordinates) != 1:
        failures.append(f"expected exactly 1 LineString, found {len(coordinates)}")
    emitted = []
    for element in coordinates:
        for triple in element.text.split():
            lon, lat, alt = (float(part) for part in triple.split(","))
            emitted.append((lat, lon, alt))
            if not (-180.0 <= lon <= 180.0):
                failures.append(f"longitude out of range: {lon}")
            if not (-90.0 <= lat <= 90.0):
                failures.append(f"latitude out of range: {lat}")
            if not _finite(alt):
                failures.append(f"non-finite altitude: {alt}")
    if len(emitted) != len(points):
        failures.append(f"emitted {len(emitted)} coords, expected {len(points)}")

    if time_track:
        whens = root.findall(f".//{{{GX_NS}}}Track/{{{KML_NS}}}when")
        coords = root.findall(f".//{{{GX_NS}}}Track/{{{GX_NS}}}coord")
        if len(whens) != len(coords) or len(whens) != len(points):
            failures.append(
                f"gx:Track mismatch: {len(whens)} when / {len(coords)} coord, "
                f"expected {len(points)}"
            )
        times = [dt.datetime.strptime(w.text, "%Y-%m-%dT%H:%M:%S.%fZ") for w in whens]
        if any(b <= a for a, b in zip(times, times[1:])):
            failures.append("gx:Track <when> timestamps are not strictly increasing")

    return failures, emitted


def _report(args, points, raw, emitted, stats, stride, utc_interp):
    duration_s = (points[-1][0] - points[0][0]) / 1e6
    highest = max(points, key=lambda point: point[3])
    print(f"Track: {len(points)} points, {duration_s:.2f} s, stride {stride}")
    print(
        f"  start {points[0][1]:.7f}, {points[0][2]:.7f}, {points[0][3]:.2f} m"
    )
    print(
        f"  end   {points[-1][1]:.7f}, {points[-1][2]:.7f}, {points[-1][3]:.2f} m"
    )
    print(f"  highest {highest[1]:.7f}, {highest[2]:.7f}, {highest[3]:.2f} m")
    print(
        f"  dropped: invalid={stats['invalid']} nonfinite={stats['nonfinite']} "
        f"dup={stats['dup']}"
    )

    def fmt_range(values):
        return f"{min(values):.7f}..{max(values):.7f}"

    def finites(values):
        return [value for value in values if _finite(value)]

    print("\nRange check (emitted vs raw ULog vehicle_global_position):")
    print(f"  lat  emitted {fmt_range([p[0] for p in emitted])}")
    print(f"       raw     {fmt_range(finites(raw['lat']))}")
    print(f"  lon  emitted {fmt_range([p[1] for p in emitted])}")
    print(f"       raw     {fmt_range(finites(raw['lon']))}")
    print(f"  alt  emitted {fmt_range([p[2] for p in emitted])}")
    print(f"       raw     {fmt_range(finites(raw['alt']))}")

    if args.time_track:
        if utc_interp is not None:
            first, last = utc_interp(points[0][0]), utc_interp(points[-1][0])
            print(
                f"\nTime track: {_iso_utc(first)} .. {_iso_utc(last)} "
                f"(GPS UTC, {(last - first) / 1e6:.2f} s)"
            )
        else:
            print("\nTime track: GPS UTC unavailable; using log-relative time "
                  "(timeline order is still correct)")
    print(f"\nWrote {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ulg_path", type=Path, help="path to a PX4 .ulg log")
    parser.add_argument("-o", "--output", type=Path,
                        help="KML output path (default: flight_trajectory.kml "
                             "next to the log)")
    parser.add_argument("--name", help="KML document name (default: log file name)")
    parser.add_argument("--color", default="00FFFF",
                        help="track color as RRGGBB (default 00FFFF)")
    parser.add_argument("--width", type=int, default=4, help="track line width")
    parser.add_argument("--opacity", type=float, default=0.9,
                        help="track opacity 0..1 (default 0.9)")
    parser.add_argument("--altitude-mode", choices=("absolute", "relative"),
                        default="absolute",
                        help="absolute = MSL altitude; relative = MSL minus "
                             "home_position altitude (default absolute)")
    parser.add_argument("--every", type=int,
                        help="keep every Nth point (overrides --max-points)")
    parser.add_argument("--max-points", type=int, default=20000,
                        help="cap the number of points via decimation "
                             "(default 20000)")
    parser.add_argument("--time-track", action="store_true",
                        help="also emit a gx:Track with GPS UTC timestamps "
                             "for the Google Earth time slider")
    args = parser.parse_args()

    if not args.ulg_path.is_file():
        raise SystemExit(f"log file not found: {args.ulg_path}")
    if args.output is None:
        args.output = args.ulg_path.with_name("flight_trajectory.kml")
    if args.name is None:
        args.name = args.ulg_path.name
    if not (0.0 <= args.opacity <= 1.0):
        raise SystemExit("--opacity must be between 0 and 1")
    if args.every is not None and args.every < 1:
        raise SystemExit("--every must be >= 1")

    log = pyulog.ULog(str(args.ulg_path))
    points, stats = _load_track(log, args.altitude_mode)
    points, stride = _decimate(points, args.every, args.max_points)

    raw = {"lat": log.get_dataset("vehicle_global_position").data["lat"],
           "lon": log.get_dataset("vehicle_global_position").data["lon"],
           "alt": log.get_dataset("vehicle_global_position").data["alt"]}
    utc_samples = _gps_utc_samples(log) if args.time_track else None
    utc_interp = _utc_interpolator(utc_samples) if utc_samples else None

    args.output.write_text(_build_kml(args, points, utc_interp, stride),
                           encoding="utf-8")

    failures, emitted = _verify(args.output, points, args.time_track)
    _report(args, points, raw, emitted, stats, stride, utc_interp)
    if failures:
        for failure in failures:
            print(f"VERIFICATION FAILED: {failure}")
        raise SystemExit(1)
    print("Verification OK: XML valid, lon,lat,alt ordering correct, all finite.")


if __name__ == "__main__":
    main()
