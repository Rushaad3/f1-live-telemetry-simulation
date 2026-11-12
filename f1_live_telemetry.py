#!/usr/bin/env python3
"""
F1 TELEMETRY SIMULATION  -  synthetic telemetry generator + analysis dashboard in ONE file.

Run:            Open this file in Python IDLE / VS Code / PyCharm and press Run.
                Missing Python packages are checked and installed automatically.

Options:
  --rate 60       simulation update rate (samples per second)
  --seed 1        repeatable lap-to-lap variation
  --track monaco  start on this circuit (any part of the name); pick others from the dropdown
  --reverse       drive the circuit the other way round
  --driver NAME   driving style (Benchmark, Late Braker, Smooth Operator, ...); edit traits with the TRAITS button

Layout (like the F1 / AWS qualifying analysis graphic):
  RED    = last completed lap        GREEN = best lap        BLUE = live lap
  Cards  = lap time, gap, % of lap at full throttle / heavy braking / cornering
  Map    = coloured by who is faster in each corner section
  Zoom   = mouse wheel / drag on any trace or the map; RESET ZOOM button restores
  Traces = speed, throttle, brake and cumulative delta vs distance, with detected corners
  Lap history = click LAP HISTORY to inspect laps and export Excel session logs
"""
import argparse, math, struct, threading, time, datetime, queue
import importlib.util
import subprocess
import sys

# ---------------------------------------------------------------------
# ONE-FILE STARTUP: install the small set of required packages if absent.
# This means the user can press Run without manually opening a terminal.
# An internet connection is needed only the first time a missing package
# has to be downloaded. The application itself remains a single .py file.
# ---------------------------------------------------------------------
def ensure_dependencies():
    packages = {
        "numpy": "numpy",
        "pyqtgraph": "pyqtgraph",
        "PyQt5": "PyQt5",
        "openpyxl": "openpyxl",
    }
    missing = [package for module, package in packages.items()
               if importlib.util.find_spec(module) is None]
    if not missing:
        return

    print("\nF1 TELEMETRY - FIRST-TIME SETUP")
    print("Installing required components: " + ", ".join(missing))
    print("This may take a few minutes. Please keep this window open.\n")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing
        ])
    except Exception as exc:
        raise RuntimeError(
            "Automatic setup could not install the required packages.\n"
            "Check your internet connection and make sure this Python installation "
            "allows package installation, then press Run again.\n\n"
            f"Details: {exc}"
        ) from exc

ensure_dependencies()

import numpy as np

G = 9.81
UID = 0xF1F1F1F1F1F1F1F1

HDR = struct.Struct("<HBBBBQfIBB")                    # 24 bytes
TEL = struct.Struct("<HfffBbHBBH4H4B4BH4f4B")         # 60 bytes per car
LAP = struct.Struct("<IIHHfff14BHHB")                 # 43 bytes per car
MOT = struct.Struct("<6f6h6f")                        # 60 bytes per car
assert (HDR.size, TEL.size, LAP.size, MOT.size) == (24, 60, 43, 60)


def fmt(ms):
    return "-:--.---" if not ms else f"{int(ms) // 60000}:{(int(ms) % 60000) / 1000:06.3f}"


# =====================================================================
#  PART 1: FAKE TELEMETRY GENERATOR  (Monza-style circuit + car physics)
# =====================================================================
def _catmull(pts, n_per=80):
    """Closed centripetal Catmull-Rom spline through the control points."""
    P = np.array(pts, float); N = len(P); out = []
    for i in range(N):
        p0, p1, p2, p3 = P[(i - 1) % N], P[i], P[(i + 1) % N], P[(i + 2) % N]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
        t = np.linspace(t1, t2, n_per, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    return np.vstack(out)


def _from_xy(X, Y, ds, pos_m=30.0, k_m=60.0, kmax=1 / 12):
    """Uniformly resample a closed path, smooth it, and compute heading + signed curvature."""
    C = np.column_stack([np.append(X, X[0]), np.append(Y, Y[0])])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    L = s[-1]; n = int(L // ds); S = np.linspace(0, L, n, endpoint=False); ds = L / n
    X, Y = np.interp(S, s, C[:, 0]), np.interp(S, s, C[:, 1])

    def sm(a, m, passes=1):
        w = max(3, int(m / ds))
        for _ in range(passes):
            a = np.convolve(np.pad(a, w, mode="wrap"), np.ones(w) / w, mode="valid")[:n]
        return a

    X, Y = sm(X, pos_m, 2), sm(Y, pos_m, 2)
    dx = (np.roll(X, -1) - np.roll(X, 1)) / (2 * ds); dy = (np.roll(Y, -1) - np.roll(Y, 1)) / (2 * ds)
    ddx = (np.roll(X, -1) - 2 * X + np.roll(X, 1)) / ds ** 2
    ddy = (np.roll(Y, -1) - 2 * Y + np.roll(Y, 1)) / ds ** 2
    k = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    k = np.clip(sm(k, k_m, 2), -kmax, kmax)
    return S, X, Y, np.arctan2(dy, dx), k, L, ds


def _signed_area(xy):
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))


def _real_builder(coords, length, want):
    """coords = [[lon, lat], ...]; want = 'CW' / 'CCW' / None (keep the dataset's order)."""
    c = np.array(coords, float)
    lon0, lat0 = c[:, 0].mean(), c[:, 1].mean()
    xy = np.column_stack([(c[:, 0] - lon0) * math.cos(math.radians(lat0)) * 111320.0,
                          (c[:, 1] - lat0) * 110540.0])
    pts = [xy[0]]
    for p in xy[1:]:
        if np.hypot(*(p - pts[-1])) > 1.0:                      # drop duplicate points
            pts.append(p)
    if np.hypot(*(pts[0] - pts[-1])) <= 1.0:
        pts.pop()                                                # closed ring: drop repeated end point
    xy = np.array(pts)
    if want and (_signed_area(xy) > 0) == (want == "CW"):        # area > 0 means anticlockwise
        xy = xy[::-1].copy()

    def build(ds=2.0, reverse=False):
        P = xy[::-1] if reverse else xy
        C = _catmull(P.tolist())
        C = np.vstack([C, C[:1]])
        sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        C = C * (length / sl[-1])                                # scale to the official lap length
        return _from_xy(C[:-1, 0], C[:-1, 1], ds)
    return build


def _build_oval(ds=2.0, reverse=False, straight=700.0, R=230.0):
    a = np.arange(0, math.pi, 0.5 / R)
    x1 = np.arange(0, straight, 1.0)
    X = np.concatenate([-x1, -straight - R * np.sin(a), -straight + x1, R * np.sin(a)])
    Y = np.concatenate([np.zeros(len(x1)), R - R * np.cos(a), np.full(len(x1), 2 * R), R + R * np.cos(a)])
    if reverse:
        X, Y = X[::-1], Y[::-1]
    return _from_xy(X, Y, ds, pos_m=10.0, k_m=10.0)


# Direction of travel where I'm confident of it (the dataset's own order is not always the racing direction).
# Anything not listed keeps the dataset order; tick "Reverse" in the GUI if a track runs backwards.
CW_IDS = {"mc-1929", "at-1969", "it-1922", "gb-1948", "be-1925", "au-1953", "bh-2002", "cn-2004", "es-1991",
          "ca-1978", "hu-1986", "mx-1962", "nl-1948", "de-1927", "de-1932", "qa-2004", "my-1999", "fr-1960",
          "it-1914", "pt-1972", "za-1961", "us-1956"}
CCW_IDS = {"sg-2008", "ru-2014", "us-2012", "br-1940", "ae-2009", "it-1953", "az-2016", "sa-2021",
           "us-2022", "us-2023", "tr-2005", "us-1909"}
# (CdA, ClA[, power W]) per circuit: low-downforce Monza ... high-downforce Monaco
AERO = {"it-1922": (1.10, 2.6), "be-1925": (1.15, 2.9), "at-1969": (1.20, 3.1), "mc-1929": (1.40, 3.8, 420e3),
        "hu-1986": (1.35, 3.6), "sg-2008": (1.40, 3.7, 480e3), "az-2016": (1.10, 2.7), "sa-2021": (1.15, 2.9),
        "us-2023": (1.10, 2.7)}
DEFAULT_AERO = (1.20, 3.1)

TRACKS = {}            # display name -> (builder(ds, reverse), aero)


def register_circuits():
    for r in sorted(_load_circuits(), key=lambda r: r["name"].lower()):
        want = "CW" if r["id"] in CW_IDS else "CCW" if r["id"] in CCW_IDS else None
        TRACKS[f"{r['name']} ({r['loc']})"] = (_real_builder(r["c"], r["length"], want),
                                               AERO.get(r["id"], DEFAULT_AERO))
    TRACKS["Test Oval (simple)"] = (_build_oval, (1.00, 2.4))


DEFAULT_TRACK = "Red Bull Ring (Spielberg)"


def find_track(text):
    """Exact name, else first name containing `text` (case-insensitive)."""
    if text in TRACKS:
        return text
    t = text.lower()
    for n in TRACKS:
        if t in n.lower():
            return n
    raise SystemExit("Unknown track '%s'. Choose from:\n  %s" % (text, "\n  ".join(TRACKS)))


def build_track(name=DEFAULT_TRACK, ds=2.0, reverse=False):
    return TRACKS[name][0](ds, reverse)


TYRE_COMPOUNDS = {
    # grip multiplier, wear percentage per lap at benchmark driving, ideal temperature °C
    "Soft":         {"grip": 1.045, "wear": 3.2, "ideal": 100, "life": "Short"},
    "Medium":       {"grip": 1.000, "wear": 2.0, "ideal": 95,  "life": "Medium"},
    "Hard":         {"grip": 0.965, "wear": 1.25,"ideal": 90,  "life": "Long"},
    "Intermediate": {"grip": 0.955, "wear": 2.4, "ideal": 85,  "life": "Wet"},
    "Wet":          {"grip": 0.925, "wear": 2.0, "ideal": 80,  "life": "Very wet"},
}

WEATHER_PRESETS = {
    # air temperature (°C), track temperature (°C), rain intensity (%) and wind (m/s)
    "Dry":         {"code": 0, "air_temp": 24, "track_temp": 32, "rain": 0,   "wind": 1.0},
    "Cloudy":      {"code": 2, "air_temp": 18, "track_temp": 23, "rain": 0,   "wind": 3.0},
    "Light Rain":  {"code": 3, "air_temp": 16, "track_temp": 19, "rain": 25,  "wind": 4.0},
    "Heavy Rain":  {"code": 4, "air_temp": 13, "track_temp": 15, "rain": 75,  "wind": 6.0},
    "Storm":       {"code": 5, "air_temp": 11, "track_temp": 13, "rain": 100, "wind": 10.0},
}


class Car:
    m, rho, CdA, ClA, mu, P, vmax = 798.0, 1.2, 1.1, 2.6, 1.9, 600e3, 110.0
    cu = bu = tu = 1.0          # driver: share of grip used in corners / braking / traction

    def __init__(self, aero=None, weather=None, compound="Medium"):
        if aero:
            self.CdA, self.ClA = aero[:2]
            if len(aero) > 2:
                self.P = aero[2]
        wx = dict(WEATHER_PRESETS["Dry"] if weather is None else weather)
        air_temp = float(wx.get("air_temp", 24))
        track_temp = float(wx.get("track_temp", 32))
        rain = float(np.clip(wx.get("rain", 0), 0, 100)) / 100.0

        # Ideal-gas density approximation at constant atmospheric pressure.
        # Density changes both aerodynamic drag and downforce in the vehicle model.
        self.rho = 1.225 * 288.15 / (273.15 + air_temp)

        # Tyre compound and weather establish the starting grip. Wear and temperature
        # are updated by the lap generator and feed back into this same friction model.
        self.base_mu = float(type(self).mu)
        self.compound = compound if compound in TYRE_COMPOUNDS else "Medium"
        self.compound_grip_factor = TYRE_COMPOUNDS[self.compound]["grip"]
        self.wear_grip_factor = 1.0
        self.tyre_temp_factor = 1.0
        self.wet_grip_factor = 1.0 - 0.48 * rain ** 0.85
        self.track_temp_factor = float(np.clip(1.0 - 0.0012 * abs(track_temp - 32.0), 0.90, 1.0))
        self.weather_grip_factor = self.wet_grip_factor * self.track_temp_factor
        self.mu = self.base_mu * self.compound_grip_factor * self.weather_grip_factor

        # Mild ambient-temperature correction to available engine power. This is
        # deliberately modest because the model represents a modern race power unit.
        self.P *= float(np.clip(1.0 - 0.0005 * (air_temp - 20.0), 0.96, 1.04))
        self.weather = wx

    def drag(self, v): return 0.5 * self.rho * self.CdA * v * v / self.m
    def down(self, v): return 0.5 * self.rho * self.ClA * v * v / self.m
    def grip(self, v): return self.mu * (G + self.down(v))
    def corner_speed(self, k):
        kd = 0.5 * self.rho * self.ClA / self.m
        mu = self.mu * self.cu
        den = np.abs(k) - mu * kd
        v = np.sqrt(mu * G / np.maximum(den, 1e-9))
        return np.where(den > 1e-9, np.minimum(v, self.vmax), self.vmax)
    def circle(self, v, k):
        return math.sqrt(max(0.0, 1 - (v * v * abs(k) / self.grip(v)) ** 2))
    def a_acc(self, v, k):
        drive = min(self.P / (self.m * max(v, 5.0)), 0.8 * self.grip(v) * self.tu)
        return self.circle(v, k) * drive - self.drag(v)
    def a_dec(self, v, k):
        return self.circle(v, k) * self.grip(v) * self.bu + self.drag(v)

# ---------------------------------------------------------------------
#  DRIVERS: nine traits, each 0..1, change how the car is driven.
#  0.75 is the "Benchmark" level (the calibrated reference driver).
# ---------------------------------------------------------------------
TRAITS = [
    ("pace", "Raw pace", "Overall speed. High = quicker everywhere."),
    ("corner", "Corner speed", "How close to the grip limit the car is held in corners."),
    ("brake", "Braking aggression", "How late and hard the driver brakes. High = later braking points."),
    ("trail", "Trail braking", "How long the brake stays on while turning in. Low = abrupt release."),
    ("throttle", "Throttle attack", "How fast the throttle goes on at the exit. High = snappy."),
    ("traction", "Traction use", "How much power is used on exit. Low = cautious, lifts early."),
    ("smooth", "Smoothness", "Low = jerky steering and twitchy pedals."),
    ("consistency", "Consistency", "How alike the laps are. Low = big lap-to-lap swings."),
    ("mistakes", "Mistake rate", "How often lock-ups / missed apexes happen."),
]
DRIVERS = {
    "Benchmark": dict(pace=.75, corner=.75, brake=.70, trail=.60, throttle=.60, traction=.75, smooth=.75,
                      consistency=.70, mistakes=.10,
                      desc="Balanced reference driver. Solid pace, tidy inputs, rare errors."),
    "Late Braker": dict(pace=.80, corner=.65, brake=.95, trail=.25, throttle=.75, traction=.70, smooth=.55,
                        consistency=.55, mistakes=.35,
                        desc="Dives deep into braking zones and releases the brake abruptly. "
                             "Quick on a good lap, ragged on a bad one."),
    "Smooth Operator": dict(pace=.78, corner=.92, brake=.55, trail=.90, throttle=.45, traction=.90, smooth=.95,
                            consistency=.95, mistakes=.05,
                            desc="Brakes earlier, carries lots of corner speed, very smooth and repeatable."),
    "Aggressive Charger": dict(pace=.85, corner=.85, brake=.90, trail=.40, throttle=.95, traction=.65,
                               smooth=.35, consistency=.40, mistakes=.60,
                               desc="Attacks everything: late braking, snappy throttle. Fast but messy."),
    "Qualifying Hero": dict(pace=.97, corner=.95, brake=.85, trail=.70, throttle=.70, traction=.85, smooth=.70,
                            consistency=.35, mistakes=.25,
                            desc="Huge one-lap pace, but lap times swing around a lot."),
    "Cautious Tyre Saver": dict(pace=.45, corner=.45, brake=.45, trail=.70, throttle=.35, traction=.55,
                                smooth=.85, consistency=.90, mistakes=.05,
                                desc="Protects the car and tyres. Slow but extremely consistent."),
    "Rookie": dict(pace=.35, corner=.50, brake=.40, trail=.30, throttle=.55, traction=.50, smooth=.30,
                   consistency=.25, mistakes=.65,
                   desc="Still learning: early braking, jerky inputs, frequent mistakes."),
}
DRIVERS["Custom"] = dict(DRIVERS["Benchmark"], desc="Your own settings. Press TRAITS to edit them.")
DEFAULT_DRIVER = "Benchmark"


def find_driver(text):
    if text in DRIVERS:
        return text
    for n in DRIVERS:
        if text.lower() in n.lower():
            return n
    raise SystemExit("Unknown driver '%s'. Choose from: %s" % (text, ", ".join(DRIVERS)))


def driver_params(tr):
    """Turn the 0..1 traits into physical / behavioural numbers."""
    lerp = lambda a, b, x: a + (b - a) * x
    return dict(
        cu=1.0 + (tr["corner"] - 0.75) * 0.16,            # grip used in corners (corner speed)
        bu=1.0 + (tr["brake"] - 0.70) * 0.22,             # grip used under braking (braking point)
        tu=1.0 + (tr["traction"] - 0.75) * 0.20,         # grip used for traction on exit
        pace_add=(0.75 - tr["pace"]) * 0.07,               # overall time added (or removed)
        thr_up=lerp(2.0, 7.0, tr["throttle"]),             # throttle application rate (1/s)
        brk_down=lerp(6.0, 1.2, tr["trail"]),              # brake release rate (1/s): low = trail braking
        jitter=(1 - tr["smooth"]) * 0.05,                  # pedal / steering noise
        cs=lerp(2.0, 0.25, tr["consistency"]) / 0.775,     # lap-to-lap variation scale (1 = Benchmark)
        mistakes=tr["mistakes"] * 2.0,                     # expected mistakes per lap
    )


def speed_profile(car, k, ds):
    N = len(k); v = car.corner_speed(k).copy()
    for _ in range(2):
        for i in range(N):
            j = (i + 1) % N
            v[j] = min(v[j], math.sqrt(max(0.0, v[i] ** 2 + 2 * car.a_acc(v[i], k[i]) * ds)))
        for i in range(N - 1, -1, -1):
            j = (i + 1) % N
            v[i] = min(v[i], math.sqrt(max(0.0, v[j] ** 2 + 2 * car.a_dec(v[j], k[j]) * ds)))
    w = 7                                                # round off the sharp corners of the profile
    return np.convolve(np.pad(v, w, mode="wrap"), np.ones(w) / w, mode="valid")[:N]

def driver_pedals(car, v, k, ds):
    N = len(v); vn = np.roll(v, -1)
    a = (vn ** 2 - v ** 2) / (2 * ds)
    thr, brk = np.zeros(N), np.zeros(N)
    for i in range(N):
        need = a[i] + car.drag(v[i])
        if need >= 0:
            drive = min(car.P / (car.m * max(v[i], 5.0)), 0.8 * car.grip(v[i]) * car.tu)
            cap = max(1e-6, car.circle(v[i], k[i]) * drive)      # grip left after cornering
            thr[i] = min(1.0, need / cap) if cap > 1e-3 else 0.0
        else:
            brk[i] = min(1.0, -need / car.grip(v[i]))
    sm = lambda x, w: np.convolve(np.pad(x, w, mode="wrap"), np.ones(w) / w, mode="valid")[:N]
    for _ in range(2):                                   # analogue-looking pedals, no square edges
        thr, brk = sm(thr, 9), sm(brk, 7)
    return np.clip(thr, 0, 1), np.clip(brk, 0, 1), a


TOPS = np.array([85, 125, 165, 200, 235, 268, 300, 345.0])   # km/h at 12,000 rpm per gear


def header(pid, t, frame, uid=UID):
    return HDR.pack(2022, 1, 0, 1, pid, uid, t, frame, 0, 255)


def smooth_noise(rng, n, corr_m, ds):
    """Smooth random curve along the lap (wraps round), zero mean, unit std."""
    w = max(3, int(corr_m / ds))
    k = np.hanning(2 * w + 1); k /= k.sum()
    z = np.convolve(np.pad(rng.normal(size=n), w, mode="wrap"), k, mode="valid")
    return (z - z.mean()) / (z.std() + 1e-9)


def frames(track, rate, seed, uid, reverse=False, driver=None, max_laps=0, log=True, weather=None, compound="Medium"):
    """Generate internal telemetry samples for one simulated car, frame by frame."""
    tr = driver or DRIVERS[DEFAULT_DRIVER]
    P = driver_params(tr)
    rng = np.random.default_rng(seed)
    S, X, Y, h, k, L, ds = build_track(track, 2.0, reverse)
    N = len(S)
    wx = dict(WEATHER_PRESETS["Dry"] if weather is None else weather)
    car = Car(TRACKS[track][1], wx, compound)
    car.cu, car.bu, car.tu = P["cu"], P["bu"], P["tu"]
    v = speed_profile(car, k, ds)
    thr, brk, a_arr = driver_pedals(car, v, k, ds)
    hx, hy = np.cos(h), np.sin(h)
    at = lambda arr, s: float(np.interp(s, S, arr, period=L))
    Sx = np.append(S, L)
    brake_zones = np.where(brk > 0.4)[0]

    if log:
        print(f"[generator] {track} | {wx.get('condition', 'weather preset')}: {L:.0f} m, "
              f"top {v.max() * 3.6:.0f} km/h, min {v.min() * 3.6:.0f} km/h | "
              f"effective grip {car.weather_grip_factor * 100:.1f}%")

    tyre_temp = np.full(4, max(55.0, float(wx.get("track_temp", 32)) + 40.0))
    tyre_wear = np.zeros(4, dtype=float)  # percentage, independently tracked for FL/FR/RL/RR
    wear_stress_sum = 0.0
    wear_samples = 0
    compound_data = TYRE_COMPOUNDS[car.compound]
    driver_stress = 1.0 + 0.30 * max(0.0, P["cs"]) + 0.20 * max(0.0, P["jitter"])
    tyre_packet_code = list(TYRE_COMPOUNDS).index(car.compound)

    def new_lap():
        """Recalculate the lap speed profile using current tyre wear and temperature."""
        nonlocal v, thr, brk, a_arr, brake_zones, wear_stress_sum, wear_samples
        temp_error = float(np.mean(np.abs(tyre_temp - compound_data["ideal"])))
        car.tyre_temp_factor = float(np.clip(1.0 - 0.0018 * temp_error, 0.78, 1.0))
        car.wear_grip_factor = float(np.clip(1.0 - 0.0025 * float(np.mean(tyre_wear)), 0.75, 1.0))
        car.mu = (car.base_mu * car.compound_grip_factor * car.weather_grip_factor
                  * car.tyre_temp_factor * car.wear_grip_factor)
        v = speed_profile(car, k, ds)
        thr, brk, a_arr = driver_pedals(car, v, k, ds)
        brake_zones = np.where(brk > 0.4)[0]
        z = smooth_noise(rng, N, 220.0, ds)
        sd = 1.0 + P["pace_add"] + P["cs"] * (rng.uniform(0.004, 0.02) + 0.008 * np.abs(z)
                                              + 0.004 * np.clip(z, 0, None))
        events = []
        for _ in range(rng.poisson(P["mistakes"])):                 # lock-ups / missed apexes
            idx = int(rng.choice(brake_zones)) if len(brake_zones) else int(rng.integers(N))
            sm_, amp = float(S[idx]), rng.uniform(0.04, 0.10)
            dd = (S - sm_ + L / 2) % L - L / 2
            sd = sd + amp * np.exp(-0.5 * (dd / 25.0) ** 2)
            events.append(sm_)
        sd = np.maximum(sd, 0.985)
        vl = v / sd
        dts = ds / (0.5 * (vl + np.roll(vl, -1)))
        Tx = np.append(np.concatenate([[0], np.cumsum(dts)[:-1]]), dts.sum())
        return sd, Tx, float(dts.sum()), events

    dt = 1.0 / rate
    # Synthetic weather is fed into the same weather-aware vehicle model used to
    # calculate the speed profile, braking limits and traction limits.
    weather_code = int(wx.get("code", 0))
    track_temp = int(wx.get("track_temp", 32))
    air_temp = int(wx.get("air_temp", 24))
    rain_pct = int(wx.get("rain", 0))
    wind_speed = float(wx.get("wind", 1.0))
    sim_weather_body = (
        struct.pack("<Bbb", weather_code, track_temp, air_temp) +
        struct.pack("<BHBbBHHBBBBBB", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) +
        struct.pack("<BBB", 0, 0, 1) +  # zero marshal zones, one forecast sample
        struct.pack("<BBBbbBfh", 0, 0, 0, track_temp, air_temp, rain_pct, wind_speed, 0)
    )
    sim_weather = header(1, 0.0, 0, uid) + sim_weather_body
    yield (sim_weather,)
    frame, lap, lap_t, last_ms = 0, 1, 0.0, 0
    sd, Tx, lap_total, events = new_lap()
    brake_t = np.full(4, 300.0)
    thr_p = brk_p = 0.0                                       # actual pedal positions (slew limited)

    while True:
        if lap_t >= lap_total:
            last_ms = int(round(lap_total * 1000))
            if log:
                extra = f"   (mistake at {', '.join(f'{e:.0f} m' for e in events)})" if events else ""
                print(f"[generator] Lap {lap}: {fmt(last_ms)}{extra}")
            if max_laps and lap >= max_laps:
                return
            lap_t -= lap_total
            # Wear per lap is compound-specific and rises with aggressive driving.
            # Front tyres carry a little extra load from braking; rear tyres from traction.
            mean_load_stress = wear_stress_sum / max(1, wear_samples)
            base_wear = float(compound_data["wear"]) * driver_stress * mean_load_stress
            tyre_wear += base_wear * np.array([1.08, 1.08, 0.96, 0.96])
            tyre_wear = np.clip(tyre_wear, 0.0, 100.0)
            wear_stress_sum = 0.0
            wear_samples = 0
            lap += 1
            sd, Tx, lap_total, events = new_lap()
        s = float(np.interp(lap_t, Tx, Sx))

        f = at(sd, s)
        vv, kk = at(v, s) / f, at(k, s)
        acc = at(a_arr, s) / f ** 2
        t_thr, t_brk = at(thr, s), at(brk, s)
        if events and min(abs((s - e + L / 2) % L - L / 2) for e in events) < 12:
            t_brk, t_thr = max(t_brk, 0.98), 0.0                   # lock-up: stamps on the brake
        if t_brk > 0.02:
            t_thr = 0.0
        if 0.02 < t_thr < 0.98:
            t_thr = float(np.clip(t_thr + rng.normal(0, 0.012 + P["jitter"]), 0, 1))
        if t_brk > 0.05 and P["jitter"] > 0:
            t_brk = float(np.clip(t_brk * (1 + rng.normal(0, P["jitter"])), 0, 1))
        # pedal dynamics: the foot can't teleport. Throttle attack and brake release depend on the driver.
        thr_p += float(np.clip(t_thr - thr_p, -9.0 * dt, P["thr_up"] * dt))
        brk_p += float(np.clip(t_brk - brk_p, -P["brk_down"] * dt, 9.0 * dt))
        tb, bb = float(np.clip(thr_p, 0, 1)), float(np.clip(brk_p, 0, 1))

        kmh = vv * 3.6
        gear = int(min(np.searchsorted(TOPS, kmh) + 1, 8))
        rpm = int(np.clip(12000 * kmh / TOPS[gear - 1], 4000, 12500))
        rev_pct = int(np.clip((rpm - 10500) / 1500 * 100, 0, 100))
        steer = float(-np.clip(kk * 3.6 / 0.12, -1, 1) + rng.normal(0, P["jitter"] * 0.6))   # -1 = left
        steer = max(-1.0, min(1.0, steer))
        drs = int(tb > 0.98 and abs(kk) < 0.002 and vv > 50)
        lat_g, lon_g = vv * vv * kk / G, acc / G

        w = np.array([0.4, 1.0, 0.4, 1.0]) if kk > 0 else np.array([1.0, 0.4, 1.0, 0.4])
        # Thermal response: cornering and braking heat the loaded tyres; low-load
        # running lets them cool towards ambient/track temperature.
        track_ambient = float(wx.get("track_temp", 32))
        heat_target = (compound_data["ideal"] + 5.0 * abs(lat_g) * w
                       + 9.0 * bb * np.array([0.35, 0.35, 1.0, 1.0]))
        tyre_temp += (heat_target - tyre_temp) * dt / 18.0
        tyre_temp += (track_ambient - tyre_temp) * dt / 180.0
        # Accumulate a lap-level wear severity: lateral load, braking, traction
        # demand and operation outside the compound's temperature window.
        temp_excess = float(np.mean(np.maximum(0.0, np.abs(tyre_temp - compound_data["ideal"]) - 10.0)))
        wear_stress_sum += 1.0 + 0.035 * abs(lat_g) + 0.025 * bb + 0.015 * tb + 0.0015 * temp_excess
        wear_samples += 1
        btgt = 250 + 750 * bb
        brake_t += (btgt - brake_t) * dt / (0.6 if btgt > brake_t.mean() else 3.0)
        brakes = [int(x) for x in brake_t * np.array([0.8, 0.8, 1, 1])]
        surf = [int(np.clip(x, 0, 255)) for x in tyre_temp]
        inner = [int(np.clip(x - 5, 0, 255)) for x in tyre_temp]
        press = [22.5 + 0.05 * (x - 85) for x in tyre_temp]

        car_tel = TEL.pack(int(kmh), tb, steer, bb, 0, gear, rpm, drs, rev_pct, 0,
                           *brakes, *surf, *inner, int(100 + 8 * rpm / 12000), *press, 0, 0, 0, 0)
        p_tel = (header(6, frame * dt, frame, uid) + car_tel + bytes(60) * 21
                 + struct.pack("<BBb", 255, 255, gear))

        sec = 0 if s < L / 3 else (1 if s < 2 * L / 3 else 2)
        s1 = int(np.interp(L / 3, Sx, Tx) * 1000) if sec >= 1 else 0
        s2 = int((np.interp(2 * L / 3, Sx, Tx) - np.interp(L / 3, Sx, Tx)) * 1000) if sec >= 2 else 0
        car_lap = LAP.pack(last_ms, int(lap_t * 1000), s1, s2, s, (lap - 1) * L + s, 0.0,
                           1, lap, 0, 0, sec, 0, 0, 0, 0, 0, 1, 4, 2, 0, 0, 0, 0)
        p_lap = header(2, frame * dt, frame, uid) + car_lap + bytes(43) * 21 + struct.pack("<BB", 255, 255)

        hd = math.atan2(at(hy, s), at(hx, s))
        c, sn = math.cos(hd), math.sin(hd)
        car_mot = MOT.pack(at(X, s), 0.0, at(Y, s), vv * c, 0.0, vv * sn,
                           int(c * 32767), 0, int(sn * 32767), int(sn * 32767), 0, int(-c * 32767),
                           lat_g, lon_g, 1.0, hd, 0.0, 0.0)
        p_mot = header(0, frame * dt, frame, uid) + car_mot + bytes(60) * 21 + bytes(120)

        # Private extension packet for the standalone dashboard's tyre panel.
        tyre_state = struct.pack("<B4f4f", tyre_packet_code, *[float(x) for x in tyre_temp],
                                 *[float(x) for x in tyre_wear])
        p_tyre = header(255, frame * dt, frame, uid) + tyre_state
        yield p_mot, p_tyre, p_lap, p_tel
        frame += 1
        lap_t += dt


def run_generator(stop, output_queue, rate, seed, speedup=1.0, max_laps=0, track=DEFAULT_TRACK, uid=UID,
                  reverse=False, driver=None, weather=None, pause_gate=None, compound="Medium"):
    """Pace synthetic telemetry samples into an in-process queue; no network is used."""
    dt = 1.0 / rate
    next_t = None
    for samples in frames(track, rate, seed, uid, reverse, driver, max_laps,
                          weather=weather, compound=compound):
        if stop.is_set():
            break
        if pause_gate is not None and not pause_gate.is_set():
            while not pause_gate.wait(0.05):
                if stop.is_set():
                    break
            if stop.is_set():
                break
            next_t = time.perf_counter()
        if next_t is None:
            next_t = time.perf_counter()
        for sample in samples:
            if stop.is_set():
                break
            output_queue.put(sample)
        next_t += dt / speedup
        delay = next_t - time.perf_counter()
        if delay > 0 and stop.wait(delay):
            break


def simulate_ghost(track, reverse, driver, laps=2, seed=777, weather=None, compound="Medium"):
    """Run a driver independently and return their best lap as a reference ('ghost') lap."""
    tel = Telemetry()
    for pk in frames(track, 30.0, seed, 1, reverse, driver, max_laps=laps + 1, log=False, weather=weather, compound=compound):
        for p in pk:
            tel.handle(p)
    return min(tel.laps, key=lambda l: l["ms"]) if tel.laps else None


# =====================================================================
#  PART 2: TELEMETRY SAMPLE DECODER + LAP ANALYSIS (no GUI code)
# =====================================================================
CHANNELS = ("d", "t", "spd", "thr", "brk", "str", "x", "z", "gear", "rpm")


class Telemetry:
    """Decodes internal telemetry samples, keeps the live lap, and stores completed laps."""

    def __init__(self):
        self.x = self.z = 0.0
        self.lap = None
        self.sector = 0
        self.cur_ms = self.last_ms = 0
        self.speed = self.gear = self.rpm = 0
        self.dist = 0.0
        self.buf = {k: [] for k in CHANNELS}
        self.laps = []                 # completed laps (dicts of numpy arrays)
        self.best = None
        self.version = 0               # bumps whenever a lap completes
        self.s1ms = self.s2ms = 0      # sector times of the lap in progress
        self.b1 = self.b2 = None       # lap distance where sector 2 / sector 3 began
        self.weather = None            # latest conditions from the synthetic session sample
        self.tyres = None              # synthetic compound, four temperatures and four wear values

    def handle(self, data):
        if len(data) < 24:
            return
        pid = HDR.unpack_from(data, 0)[4]
        if pid == 255 and len(data) >= 24 + struct.calcsize("<B4f4f"):
            values = struct.unpack_from("<B4f4f", data, 24)
            compound_names = list(TYRE_COMPOUNDS)
            self.tyres = {"compound": compound_names[min(int(values[0]), len(compound_names)-1)],
                          "temperatures": list(values[1:5]), "wear": list(values[5:9])}
        elif pid == 1 and len(data) >= 27:  # Synthetic session sample: current weather + temperatures
            weather_code, track_temp, air_temp = struct.unpack_from("<Bbb", data, 24)
            weather_names = {
                0: "Clear", 1: "Light cloud", 2: "Overcast",
                3: "Light rain", 4: "Heavy rain", 5: "Storm"
            }
            conditions = {
                "condition": weather_names.get(weather_code, f"Unknown ({weather_code})"),
                "track_temp": int(track_temp), "air_temp": int(air_temp),
                "rain": None, "wind": None, "source": "SIMULATED"
            }
            # The synthetic forecast sample carries rain percentage and wind
            # speed as well as air and track temperatures.
            try:
                zones_count = int(data[42])
                sample_count_offset = 43 + zones_count * 5 + 2
                sample_count = int(data[sample_count_offset])
                sample_offset = sample_count_offset + 1
                if sample_count and len(data) >= sample_offset + 12:
                    sample = struct.unpack_from("<BBBbbBfh", data, sample_offset)
                    conditions["rain"] = int(sample[5])
                    conditions["wind"] = float(sample[6])
            except (IndexError, struct.error, ValueError):
                pass
            self.weather = conditions
        elif pid == 0 and len(data) >= 24 + MOT.size:
            m = MOT.unpack_from(data, 24)
            self.x, self.z = m[0], m[2]
        elif pid == 2 and len(data) >= 24 + LAP.size:
            l = LAP.unpack_from(data, 24)
            if self.lap is not None and l[8] != self.lap:              # new lap started
                if len(self.buf["d"]) > 50 and l[0] > 0:
                    done = {k: np.array(v, dtype=float) for k, v in self.buf.items()}
                    done["ms"], done["lap"] = l[0], self.lap
                    done["s1"], done["s2"] = self.s1ms / 1000.0, self.s2ms / 1000.0
                    done["s3"] = l[0] / 1000.0 - done["s1"] - done["s2"]
                    done["b1"], done["b2"] = self.b1, self.b2
                    done["weather"] = dict(self.weather) if self.weather else None
                    done["tyres"] = dict(self.tyres) if self.tyres else None
                    self.laps.append(done)
                    if self.best is None or l[0] < self.best["ms"]:
                        self.best = done
                    self.version += 1
                for v in self.buf.values():
                    v.clear()
                self.b1 = self.b2 = None
                self.sector = 0
            if l[11] != self.sector:                                   # crossed a sector line
                if l[11] == 1: self.b1 = l[4]
                elif l[11] == 2: self.b2 = l[4]
            self.lap, self.sector = l[8], l[11]
            self.s1ms, self.s2ms = l[2], l[3]
            self.cur_ms, self.last_ms, self.dist = l[1], l[0], l[4]
        elif pid == 6 and len(data) >= 24 + TEL.size and self.lap is not None:
            t = TEL.unpack_from(data, 24)
            self.speed, self.gear, self.rpm = t[0], t[5], t[6]
            b = self.buf
            b["d"].append(self.dist); b["t"].append(self.cur_ms / 1000.0); b["spd"].append(t[0])
            b["thr"].append(t[1]); b["brk"].append(t[3]); b["str"].append(t[2])
            b["x"].append(self.x); b["z"].append(self.z); b["gear"].append(t[5]); b["rpm"].append(t[6])


def _smooth(v, n=5):
    return np.convolve(np.pad(v, n // 2, mode="edge"), np.ones(n) / n, mode="valid")


def find_corners(v, ds, win_m=50.0, prom=10.0):
    """Corner apexes = speed minima that stand out by at least `prom` km/h."""
    n, W = len(v), int(win_m / ds)
    vs = _smooth(v, 5)
    apex = []
    for i in range(0, n):
        if vs[i] <= vs[max(0, i - W):i + W + 1].min():
            left = vs[max(0, i - 6 * W):i + 1].max(); right = vs[i:i + 6 * W].max()
            if min(left, right) - vs[i] >= prom and (not apex or i - apex[-1] > W):
                apex.append(i)
    k = 1
    while k < len(apex):                       # merge "twin" minima with no real hump between them
        i, j = apex[k - 1], apex[k]
        if vs[i:j + 1].max() - max(vs[i], vs[j]) < prom * 2:
            apex.pop(k if vs[j] >= vs[i] else k - 1)
        else:
            k += 1
    return apex, vs


def lap_pcts(lap):
    """% of lap time at full throttle / heavy braking / everything else (cornering)."""
    w = np.clip(np.diff(lap["t"], prepend=lap["t"][0]), 0, None)
    tot = max(w.sum(), 1e-9)
    ft = 100 * (w * (lap["thr"] >= 0.99)).sum() / tot
    hb = 100 * (w * (lap["brk"] >= 0.5)).sum() / tot
    return ft, hb, max(0.0, 100 - ft - hb)


def analyse(A, B, ds=2.0):
    """Compare lap A (red) with reference lap B (yellow) on a common distance grid."""
    Lmax = math.floor(min(A["d"].max(), B["d"].max()))
    grid = np.arange(0.0, Lmax, ds)

    def res(lap):
        d = np.maximum.accumulate(lap["d"]) + np.arange(len(lap["d"])) * 1e-6
        o = {k: np.interp(grid, d, lap[k]) for k in CHANNELS if k != "d"}
        o["t"] = o["t"] - o["t"][0]
        o["gear"] = np.round(o["gear"])
        return o

    a, b = res(A), res(B)
    apex, vs = find_corners(b["spd"], ds)
    bounds = [0]
    for i, j in zip(apex[:-1], apex[1:]):
        bounds.append(i + int(np.argmax(vs[i:j + 1])))
    bounds.append(len(grid) - 1)
    segs = []
    for n, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        gain = (b["t"][e] - b["t"][s]) - (a["t"][e] - a["t"][s])       # +ve = A faster
        vmin = vs[apex[n]] if n < len(apex) else vs[s:e + 1].min()
        cls = "LOW" if vmin < 140 else "MEDIUM" if vmin < 230 else "HIGH"
        ap = apex[n] if n < len(apex) else (s + e) // 2
        steer = float(np.mean(b["str"][max(0, ap - 12):ap + 13]))          # F1: negative = left
        turn = "LEFT" if steer < -0.03 else "RIGHT" if steer > 0.03 else ""
        segs.append(dict(s=s, e=e, apex=ap, gain=gain, cls=cls, vmin=float(vs[ap]), turn=turn))
    b1 = B.get("b1") if B.get("b1") is not None else Lmax / 3
    b2 = B.get("b2") if B.get("b2") is not None else 2 * Lmax / 3
    for sg in segs:
        x = grid[sg["apex"]]
        sg["sector"] = 1 if x < b1 else 2 if x < b2 else 3
    return dict(bounds=(b1, b2), A=A, B=B, grid=grid, a=a, b=b, segs=segs, delta=b["t"] - a["t"],
                gap=(A["ms"] - B["ms"]) / 1000.0, pa=lap_pcts(A), pb=lap_pcts(B),
                same=A is B)


# =====================================================================
#  PART 3: DASHBOARD (PyQt + pyqtgraph)
# =====================================================================
BG, PANEL, GRID = "#15151e", "#1d1d29", (255, 255, 255, 28)
RED, GRN, BLU = (255, 72, 60), (60, 210, 90), (70, 150, 255)
SESSION_BEST, GHOST = "Session best lap", "Ghost: "
RED_H, GRN_H, BLU_H = "#ff483c", "#3cd25a", "#4696ff"


def _align(name):
    from pyqtgraph.Qt import QtCore
    return getattr(QtCore.Qt, name, None) or getattr(QtCore.Qt.AlignmentFlag, name)


def make_card_class():
    from pyqtgraph.Qt import QtWidgets

    class Card(QtWidgets.QFrame):
        def __init__(self, colour_hex, right=False):
            super().__init__()
            al = _align("AlignRight") if right else _align("AlignLeft")
            edge = "border-right" if right else "border-left"
            self.setObjectName("card")
            self.setStyleSheet(f"QFrame#card{{background:{PANEL};{edge}:5px solid {colour_hex};}}"
                               f"QLabel{{background:transparent;color:#ffffff;}}")
            v = QtWidgets.QVBoxLayout(self)
            v.setContentsMargins(14, 8, 14, 8); v.setSpacing(2)
            self.title = QtWidgets.QLabel("")
            self.title.setStyleSheet(f"color:{colour_hex};font-size:13px;font-weight:bold;")
            self.title.setAlignment(al)
            self.time = QtWidgets.QLabel("-:--.---")
            self.time.setStyleSheet("font-size:30px;font-weight:bold;")
            self.gap = QtWidgets.QLabel("")
            self.gap.setStyleSheet("font-size:20px;font-weight:bold;color:#d8d8e0;")
            row = QtWidgets.QHBoxLayout()
            if right:
                self.time.setAlignment(_align("AlignRight")); self.gap.setAlignment(_align("AlignLeft"))
                row.addWidget(self.gap); row.addStretch(1); row.addWidget(self.time)
            else:
                self.time.setAlignment(_align("AlignLeft")); self.gap.setAlignment(_align("AlignRight"))
                row.addWidget(self.time); row.addStretch(1); row.addWidget(self.gap)
            self.sec = QtWidgets.QLabel("")
            self.sec.setAlignment(al)
            v.addWidget(self.title); v.addLayout(row); v.addWidget(self.sec)
            self.bars = []
            for name in ("FULL THROTTLE", "HEAVY BRAKING", "CORNERING"):
                lab = QtWidgets.QLabel(name); lab.setStyleSheet("font-size:10px;color:#9a9aa8;")
                lab.setAlignment(al)
                bar = QtWidgets.QProgressBar(); bar.setRange(0, 100); bar.setTextVisible(False)
                bar.setFixedHeight(6)
                bar.setStyleSheet("QProgressBar{background:#2d2d3a;border:none;}"
                                  f"QProgressBar::chunk{{background:{colour_hex};}}")
                pct = QtWidgets.QLabel("--"); pct.setStyleSheet("font-size:13px;font-weight:bold;")
                pct.setFixedWidth(44)
                br = QtWidgets.QHBoxLayout()
                if right:
                    bar.setInvertedAppearance(True); pct.setAlignment(_align("AlignLeft"))
                    br.addWidget(pct); br.addWidget(bar)
                else:
                    pct.setAlignment(_align("AlignRight")); br.addWidget(bar); br.addWidget(pct)
                v.addWidget(lab); v.addLayout(br)
                self.bars.append((bar, pct))

        def set(self, title, time_s, gap_s, pcts, sec_html=""):
            self.title.setText(title); self.time.setText(time_s); self.gap.setText(gap_s)
            self.sec.setText(sec_html)
            for (bar, lab), p in zip(self.bars, pcts):
                bar.setValue(int(round(p)) if p is not None else 0)
                lab.setText("--" if p is None else f"{p:.0f}%")

    return Card


def step_xy(x, y):
    """Turn sampled data into a staircase (for gears)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return x, y
    return np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1]


def sector_html(mine, other):
    """S1 S2 S3 times; the faster of the two laps is bright, the slower one dim."""
    out = []
    for i in (1, 2, 3):
        a, b = mine[f"s{i}"], other[f"s{i}"]
        col = "#ffffff" if a <= b + 1e-9 else "#7d7d90"
        out.append(f"<span style='color:#9a9aa8;font-size:10px'>S{i}</span> "
                   f"<span style='color:{col};font-size:14px;font-weight:bold'>{a:.3f}</span>")
    return "&nbsp;&nbsp;&nbsp;".join(out)


class Dashboard:
    def __init__(self, args):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
        self.pg, self.QtCore, self.QtGui, self.QtWidgets = pg, QtCore, QtGui, QtWidgets
        self.args = args

        # All synthetic telemetry is passed directly between threads in memory.
        # The application does not open network sockets or receive game data.
        self.data_queue = queue.Queue()

        self.stop = threading.Event()                      # stops the current generator (and the app)
        self.uid = 0
        self.track = getattr(args, "track", DEFAULT_TRACK)
        self.reverse = bool(getattr(args, "reverse", False))
        self.driver = getattr(args, "driver", DEFAULT_DRIVER)
        # Set the weather before building the controls, because the weather
        # dropdown reads this value during _build_controls().
        requested_weather = getattr(args, "weather", "Dry")
        self.weather_name = requested_weather if requested_weather in WEATHER_PRESETS else "Dry"
        self.compound_name = "Medium"
        self.ref_mode = SESSION_BEST                       # or "Ghost: <driver name>"
        self.ghost_cache, self.ghost = {}, None
        self.an_key = None

        self.tel = Telemetry()
        self.an = None
        self.seen_version = 0
        self.decor = []                                   # (plot, item) pairs rebuilt per analysis
        self.is_fullscreen = False
        self.simulation_running = False
        self.simulation_paused = False
        self.pause_gate = threading.Event()
        self.pause_gate.set()

        pg.setConfigOptions(antialias=True, background=BG, foreground=(190, 190, 200))
        self.app = pg.mkQApp("F1 Telemetry Simulation")
        self._build_ui()

        self.start_generator()
        self.t_poll = QtCore.QTimer(); self.t_poll.timeout.connect(self.poll); self.t_poll.start(5)
        self.t_draw = QtCore.QTimer(); self.t_draw.timeout.connect(self.redraw); self.t_draw.start(33)
        def cleanup():
            self.stop.set()
            self.simulation_running = False
        self.app.aboutToQuit.connect(cleanup)

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        pg, QtWidgets, QtGui = self.pg, self.QtWidgets, self.QtGui
        Card = make_card_class()
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("F1 Telemetry Simulation")
        self.win.setStyleSheet(f"background:{BG};color:#ffffff;")
        root = QtWidgets.QVBoxLayout(self.win)
        root.setContentsMargins(12, 8, 12, 8); root.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("<span style='font-size:20px;font-weight:bold;letter-spacing:1px'>"
                                 "F1 TELEMETRY SIMULATION</span> <span style='color:#9a9aa8;font-size:14px'>"
                                 "&nbsp;&nbsp;SYNTHETIC DATA</span>")
        self.weather_label = QtWidgets.QLabel("WEATHER  |  Waiting for session data")
        self.weather_label.setStyleSheet("background:#20202b;color:#d7d7e2;font-size:11px;font-weight:bold;padding:6px 9px;")
        self.weather_label.setToolTip(
            "Select a simulated weather preset to change tyre grip, braking, cornering, "
            "aerodynamic density and engine power in the vehicle simulation model."
        )
        self.btn_tyre_monitor = QtWidgets.QPushButton("TYRE MONITOR")
        self.btn_tyre_monitor.setToolTip("Open the live tyre monitor. Temperatures, wear and grip update while the simulation runs.")
        self.btn_tyre_monitor.setStyleSheet("QPushButton{background:#245b78;color:#fff;border:none;padding:6px 12px;font-weight:bold;}QPushButton:hover{background:#31789d;}")
        self.btn_tyre_monitor.clicked.connect(self.open_tyre_monitor)
        self.tyre_window = None
        self.tyre_temp_bars = []
        self.tyre_wear_bars = []
        self.tyre_temp_values = []
        self.tyre_wear_values = []
        self.tyre_corner_titles = []
        self.live = QtWidgets.QLabel("Preparing simulation...")
        self.live.setStyleSheet("font-size:15px;font-weight:bold;")
        self.live.setAlignment(_align("AlignRight"))
        self.btn_laps = QtWidgets.QPushButton("LAP HISTORY")
        self.btn_laps.setStyleSheet("QPushButton{background:#176b50;color:#fff;border:none;padding:6px 12px;font-weight:bold;}"
                                    "QPushButton:hover{background:#218363;}")
        self.btn_laps.clicked.connect(self.open_lap_history)
        self.btn_sim_start = QtWidgets.QPushButton("START SIM")
        self.btn_sim_pause = QtWidgets.QPushButton("PAUSE SIM")
        self.btn_sim_stop = QtWidgets.QPushButton("STOP SIM")
        # Fixed dimensions prevent the toolbar/window from resizing when button
        # labels change between START/SIM RUNNING and PAUSE/RESUME.
        for sim_button in (self.btn_sim_start, self.btn_sim_pause, self.btn_sim_stop):
            sim_button.setFixedSize(104, 32)
            sim_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sim_button_css = ("QPushButton{background:#176b50;color:#fff;border:none;padding:6px 12px;font-weight:bold;}"
                          "QPushButton:hover{background:#218363;} QPushButton:disabled{background:#292936;color:#777;}" )
        stop_button_css = ("QPushButton{background:#8b3038;color:#fff;border:none;padding:6px 12px;font-weight:bold;}"
                           "QPushButton:hover{background:#a83b45;} QPushButton:disabled{background:#292936;color:#777;}")
        self.btn_sim_start.setStyleSheet(sim_button_css)
        self.btn_sim_pause.setStyleSheet("QPushButton{background:#a87522;color:#fff;border:none;padding:6px 12px;font-weight:bold;}"
                                         "QPushButton:hover{background:#c18a2b;} QPushButton:disabled{background:#292936;color:#777;}")
        self.btn_sim_stop.setStyleSheet(stop_button_css)
        tooltip_css = "QToolTip{background:#20202b;color:#ffffff;border:1px solid #555568;padding:6px 8px;font-size:11px;}"
        for sim_button in (self.btn_sim_start, self.btn_sim_pause, self.btn_sim_stop):
            sim_button.setStyleSheet(sim_button.styleSheet() + tooltip_css)
        self.btn_sim_start.setToolTip("START SIM\nBegin a new simulated session and reset lap data.")
        self.btn_sim_pause.setToolTip("PAUSE / RESUME\nFreeze the simulation, then continue from the same point.")
        self.btn_sim_stop.setToolTip("STOP SIM\nEnd the current simulation. Start again for a fresh session.")
        self.btn_sim_start.clicked.connect(self.start_simulation)
        self.btn_sim_pause.clicked.connect(self.toggle_pause_simulation)
        self.btn_sim_stop.clicked.connect(self.stop_simulation)
        self.btn_fullscreen = QtWidgets.QPushButton("FULLSCREEN")
        self.btn_fullscreen.setToolTip("Toggle fullscreen mode (F11)")
        self.btn_fullscreen.setStyleSheet("QPushButton{background:#2d2d3a;color:#fff;border:none;padding:6px 12px;font-weight:bold;}"
                                           "QPushButton:hover{background:#3d3d4e;}")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        self.btn = QtWidgets.QPushButton("RESET ZOOM")
        self.btn.setStyleSheet("QPushButton{background:#2d2d3a;color:#fff;border:none;padding:6px 12px;font-weight:bold;}"
                               "QPushButton:hover{background:#3d3d4e;}")
        self.btn.clicked.connect(self.reset_zoom)
        top.addWidget(title)
        top.addWidget(self.weather_label)
        top.addWidget(self.live, 1)
        top.addWidget(self.btn_sim_start); top.addWidget(self.btn_sim_pause); top.addWidget(self.btn_sim_stop)
        top.addWidget(self.btn_laps); top.addWidget(self.btn_fullscreen); top.addWidget(self.btn)
        root.addLayout(top)
        self.update_simulation_buttons()
        root.addLayout(self._build_controls())

        mid = QtWidgets.QHBoxLayout(); mid.setSpacing(10)
        self.card_a, self.card_b = Card(RED_H), Card(GRN_H, right=True)
        self.map = pg.PlotWidget(background=BG)
        self.map.setAspectLocked(True); self.map.hideAxis("bottom"); self.map.hideAxis("left")
        self.map.setMouseEnabled(True, True); self.map.setMenuEnabled(False); self.map.hideButtons()
        self.map.setMinimumHeight(250)
        self.map_outline = self.map.plot(pen=pg.mkPen((70, 70, 85), width=9))
        self.map_dot = self.map.plot(symbol="o", symbolSize=13, symbolBrush=BLU,
                                     symbolPen=pg.mkPen((255, 255, 255), width=2), pen=None)
        mid.addWidget(self.card_a, 3); mid.addWidget(self.map, 4); mid.addWidget(self.card_b, 3)
        root.addLayout(mid, 4)

        self.live_bar = QtWidgets.QFrame(); self.live_bar.setObjectName("live")
        self.live_bar.setStyleSheet(f"QFrame#live{{background:{PANEL};border-left:5px solid {BLU_H};}}"
                                    "QLabel{background:transparent;color:#ffffff;}")
        lh = QtWidgets.QHBoxLayout(self.live_bar); lh.setContentsMargins(14, 4, 14, 4); lh.setSpacing(26)
        self.lv_title = QtWidgets.QLabel("LIVE")
        self.lv_title.setStyleSheet(f"color:{BLU_H};font-size:13px;font-weight:bold;")
        self.lv_time = QtWidgets.QLabel("-:--.---")
        self.lv_time.setStyleSheet(f"font-size:30px;font-weight:bold;color:{BLU_H};")
        self.lv_time.setMinimumWidth(150)
        self.lv_delta = QtWidgets.QLabel(""); self.lv_delta.setStyleSheet("font-size:20px;font-weight:bold;")
        self.lv_delta.setMinimumWidth(260)
        self.lv_sec = QtWidgets.QLabel("")
        self.lv_car = QtWidgets.QLabel(""); self.lv_car.setAlignment(_align("AlignRight"))
        for w in (self.lv_title, self.lv_time, self.lv_delta, self.lv_sec):
            lh.addWidget(w)
        lh.addStretch(1); lh.addWidget(self.lv_car)
        root.addWidget(self.live_bar)
        self.readout = QtWidgets.QLabel("Hover over the graphs to read speed, throttle, brake and gear at any "
                                        "point of the lap.  Click to lock the cursor, click again to release it.")
        self.readout.setStyleSheet(f"background:{PANEL};color:#c8c8d6;font-size:13px;padding:5px 12px;")
        root.addWidget(self.readout)
        gl = pg.GraphicsLayoutWidget()
        gl.setBackground(BG)
        self.p_spd = gl.addPlot(row=0, col=0)
        self.p_num = gl.addPlot(row=1, col=0)
        self.p_thr = gl.addPlot(row=2, col=0)
        self.p_brk = gl.addPlot(row=3, col=0)
        self.p_gear = gl.addPlot(row=4, col=0)
        self.p_del = gl.addPlot(row=5, col=0)
        for r, f in enumerate((10, 1, 3, 3, 3, 4)):
            gl.ci.layout.setRowStretchFactor(r, f)
        gl.ci.layout.setSpacing(0)
        gl.ci.setContentsMargins(0, 0, 0, 0)
        font = QtGui.QFont(); font.setPixelSize(10)
        labels = {self.p_spd: "SPEED km/h", self.p_num: "GAIN s", self.p_thr: "THR",
                  self.p_brk: "BRK", self.p_gear: "GEAR", self.p_del: "DELTA s"}
        for p, lab in labels.items():
            p.setMouseEnabled(x=(p is not self.p_num), y=False); p.setMenuEnabled(False); p.hideButtons()
            ax = p.getAxis("left"); ax.setWidth(78); ax.setStyle(tickFont=font); ax.enableAutoSIPrefix(False)
            ax.setPen(pg.mkPen(60, 60, 75)); ax.setLabel(lab)
            if p is not self.p_del:
                p.hideAxis("bottom")
            p.showGrid(x=True, y=(p is not self.p_num), alpha=0.12)
            if p is not self.p_spd:
                p.setXLink(self.p_spd)
        self.p_del.getAxis("bottom").setStyle(tickFont=font)
        self.p_del.setLabel("bottom", "Lap distance (m)")
        self.p_num.setYRange(0, 1, padding=0); self.p_num.setMouseEnabled(x=True, y=False)
        self.p_num.getAxis("left").setStyle(showValues=False, tickLength=0)
        self.p_thr.setYRange(0, 1.05, padding=0); self.p_brk.setYRange(0, 1.05, padding=0)
        self.p_del.setYRange(-0.3, 0.3, padding=0)
        self.p_gear.setYRange(0.4, 8.6, padding=0)
        self.p_gear.getAxis("left").setTicks([[(i, str(i)) for i in range(1, 9)]])

        mk = pg.mkPen
        self.c = {}
        for name, plot, key in (("spd", self.p_spd, "spd"), ("thr", self.p_thr, "thr"), ("brk", self.p_brk, "brk")):
            self.c[name + "B"] = plot.plot(pen=mk(GRN, width=2.4))
            self.c[name + "A"] = plot.plot(pen=mk(RED, width=2.4))
            self.c[name + "L"] = plot.plot(pen=mk(BLU, width=1.8))
        for nm, col in (("B", GRN), ("A", RED), ("L", BLU)):
            self.c["gear" + nm] = self.p_gear.plot(pen=mk(col, width=2.2 if nm != "L" else 1.8))
        self.p_del.addItem(pg.InfiniteLine(pos=0, angle=0, pen=mk(GRN, width=2.4)), ignoreBounds=True)
        self.c["delA"] = self.p_del.plot(pen=mk(RED, width=2.4))
        self.c["delL"] = self.p_del.plot(pen=mk(BLU, width=1.8))
        root.addWidget(gl, 9)
        self._build_cursor(gl)
        # Start maximised to fit the available display area instead of opening
        # at a fixed size that can place the lower controls off-screen.
        self.win.showMaximized()
        try:
            self.fullscreen_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("F11"), self.win)
            self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        except Exception:
            pass

    def toggle_fullscreen(self):
        """Toggle true fullscreen while retaining a one-click way back."""
        if self.win.isFullScreen():
            self.win.showMaximized()
            self.btn_fullscreen.setText("FULLSCREEN")
        else:
            self.win.showFullScreen()
            self.btn_fullscreen.setText("EXIT FULLSCREEN")

    # ------------------------------------------------------------ data
    def _build_controls(self):
        QtWidgets, QtCore = self.QtWidgets, self.QtCore
        css = ("QComboBox{background:#2d2d3a;color:#fff;border:none;padding:5px 10px;font-weight:bold;}"
               "QComboBox QAbstractItemView{background:#2d2d3a;color:#fff;selection-background-color:#4a4a62;}")
        lab = lambda t: QtWidgets.QLabel(f"<span style='color:#9a9aa8;font-weight:bold'>{t}</span>")
        row = QtWidgets.QHBoxLayout(); row.setSpacing(8)
        self.combo = QtWidgets.QComboBox(); self.combo.addItems(list(TRACKS))
        self.combo.setCurrentText(self.track); self.combo.setMinimumWidth(300); self.combo.setMaxVisibleItems(25)
        self.combo.setStyleSheet(css); self.combo.currentTextChanged.connect(self.on_track_changed)
        self.chk = QtWidgets.QCheckBox("Reverse")
        self.chk.setStyleSheet("QCheckBox{color:#d0d0dc;font-weight:bold;}")
        self.chk.setChecked(self.reverse); self.chk.toggled.connect(self.on_reverse_toggled)
        self.drv_combo = QtWidgets.QComboBox(); self.drv_combo.setMinimumWidth(190); self.drv_combo.setStyleSheet(css)
        self.weather_combo = QtWidgets.QComboBox(); self.weather_combo.addItems(list(WEATHER_PRESETS))
        self.weather_combo.setCurrentText(self.weather_name); self.weather_combo.setMinimumWidth(125)
        self.weather_combo.setStyleSheet(css)
        self.weather_combo.currentTextChanged.connect(self.on_weather_changed)
        self.btn_traits = QtWidgets.QPushButton("TRAITS")
        self.btn_traits.setStyleSheet("QPushButton{background:#2d2d3a;color:#fff;border:none;padding:6px 12px;"
                                      "font-weight:bold;}QPushButton:hover{background:#3d3d4e;}")
        self.btn_traits.clicked.connect(self.open_traits)
        self.ref_combo = QtWidgets.QComboBox(); self.ref_combo.setMinimumWidth(230); self.ref_combo.setStyleSheet(css)
        self.fill_driver_combos()
        self.drv_combo.currentTextChanged.connect(self.on_driver_changed)
        self.ref_combo.currentTextChanged.connect(self.on_ref_changed)
        for w in (lab("TRACK"), self.combo, self.chk, lab("DRIVER"), self.drv_combo, self.btn_traits,
                  lab("WEATHER"), self.weather_combo,
                  lab("COMPARE TO"), self.ref_combo):
            row.addWidget(w)
        row.addWidget(self.btn_tyre_monitor)
        row.addStretch(1)
        return row

    def fill_driver_combos(self):
        """(Re)fill the driver and compare dropdowns, keeping the current choices."""
        role = getattr(self.QtCore.Qt, "ToolTipRole", None) or self.QtCore.Qt.ItemDataRole.ToolTipRole
        for cb in (self.drv_combo, self.ref_combo):
            cb.blockSignals(True); cb.clear()
        self.ref_combo.addItem(SESSION_BEST)
        for n, tr in DRIVERS.items():
            self.drv_combo.addItem(n)
            self.drv_combo.setItemData(self.drv_combo.count() - 1, tr["desc"], role)
            self.ref_combo.addItem(GHOST + n)
            self.ref_combo.setItemData(self.ref_combo.count() - 1, tr["desc"], role)
        self.drv_combo.setCurrentText(self.driver)
        self.ref_combo.setCurrentText(self.ref_mode)
        for cb in (self.drv_combo, self.ref_combo):
            cb.blockSignals(False)

    def open_traits(self):
        QtWidgets, QtCore = self.QtWidgets, self.QtCore
        dlg = QtWidgets.QDialog(self.win)
        dlg.setWindowTitle("Driver traits")
        dlg.setStyleSheet(f"QDialog{{background:{BG};}} QLabel{{color:#fff;}}")
        lay = QtWidgets.QGridLayout(dlg)
        cur = DRIVERS[self.driver]
        lay.addWidget(QtWidgets.QLabel(f"<b>Editing from: {self.driver}</b><br>"
                                       f"<span style='color:#9a9aa8'>{cur['desc']}</span>"), 0, 0, 1, 3)
        sliders = {}
        for r, (key, label, tip) in enumerate(TRAITS, start=1):
            name = QtWidgets.QLabel(f"<b>{label}</b><br><span style='color:#8c8ca0;font-size:10px'>{tip}</span>")
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal if hasattr(QtCore.Qt, "Horizontal")
                                   else QtCore.Qt.Orientation.Horizontal)
            sl.setRange(0, 100); sl.setValue(int(round(cur[key] * 100))); sl.setMinimumWidth(240)
            val = QtWidgets.QLabel(str(sl.value())); val.setFixedWidth(30)
            sl.valueChanged.connect(lambda v, lb=val: lb.setText(str(v)))
            lay.addWidget(name, r, 0); lay.addWidget(sl, r, 1); lay.addWidget(val, r, 2)
            sliders[key] = sl
        lay.addWidget(QtWidgets.QLabel("<span style='color:#8c8ca0'>75 = Benchmark level. Applying restarts "
                                       "the session with a driver called Custom.</span>"), len(TRAITS) + 1, 0, 1, 3)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                                          if hasattr(QtWidgets.QDialogButtonBox, "Ok") else
                                          QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                          QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        lay.addWidget(btns, len(TRAITS) + 2, 0, 1, 3)
        if dlg.exec():
            DRIVERS["Custom"] = {k: sl.value() / 100.0 for k, sl in sliders.items()}
            DRIVERS["Custom"]["desc"] = "Your own settings. Press TRAITS to edit them."
            self.ghost_cache.clear()
            self.driver = "Custom"
            if self.ref_mode == GHOST + "Custom":
                self.ref_mode = SESSION_BEST
            self.fill_driver_combos()
            self.change_session()

    def update_simulation_buttons(self):
        """Keep the simulation controls consistent with the generator state."""
        running = bool(self.simulation_running)
        paused = bool(self.simulation_paused)
        self.btn_sim_start.setEnabled(not running)
        self.btn_sim_pause.setEnabled(running)
        self.btn_sim_stop.setEnabled(running)
        # Keep labels short and the same fixed width to avoid layout jumps.
        self.btn_sim_start.setText("START SIM" if not running else "RUNNING")
        self.btn_sim_pause.setText("RESUME" if paused else "PAUSE SIM")

    def start_generator(self):
        import random
        self.stop.set()                                    # stop the old generator, if any
        self.stop = threading.Event()
        self.data_queue = queue.Queue()
        self.pause_gate = threading.Event()
        self.pause_gate.set()
        self.simulation_paused = False
        self.uid = random.getrandbits(62)
        self.simulation_running = True
        threading.Thread(target=run_generator, daemon=True,
                         args=(self.stop, self.data_queue, self.args.rate, self.args.seed,
                               getattr(self.args, "speedup", 1.0), 0, self.track, self.uid, self.reverse,
                               dict(DRIVERS[self.driver]), dict(WEATHER_PRESETS[self.weather_name]),
                               self.pause_gate, self.compound_name)).start()
        self.update_simulation_buttons()

    def start_simulation(self):
        """Start a fresh simulated run after a user-requested stop."""
        if self.simulation_running:
            return
        self.reset_session()
        self.start_generator()
        self.live.setText("SIMULATION STARTED")

    def toggle_pause_simulation(self):
        """Pause or resume without resetting the frame/lap progression."""
        if not self.simulation_running:
            return
        if self.simulation_paused:
            self.pause_gate.set()
            self.simulation_paused = False
            self.live.setText("SIMULATION RESUMED")
        else:
            self.simulation_paused = True
            self.pause_gate.clear()
            self.live.setText("SIMULATION PAUSED")
        self.update_simulation_buttons()

    def stop_simulation(self):
        """Stop generating simulated telemetry without closing the dashboard."""
        if not self.simulation_running:
            return
        self.stop.set()
        self.pause_gate.set()  # release a paused generator so it can exit
        self.simulation_running = False
        self.simulation_paused = False
        self.update_simulation_buttons()
        self.live.setText("SIMULATION STOPPED")

    def reset_session(self):
        """Throw away all laps and plots, ready for a new track."""
        self.tel = Telemetry()
        self.an = None
        self.seen_version = 0
        self.an_key = None
        self.ghost = None
        self._clear_decor()
        for curve in self.c.values():
            curve.setData([], [])
        self.map_outline.setData([], []); self.map_dot.setData([], [])
        self.card_a.set("CURRENT LAP", "-:--.---", "LIVE", [None] * 3, "")
        self.card_b.set("BEST LAP", "-:--.---", "", [None] * 3, "")
        self.p_spd.enableAutoRange(x=True, y=True)
        self.map.enableAutoRange()
        self.live.setText("Loading track...")

    def change_session(self, track=None, reverse=None, driver=None):
        if track is not None:
            self.track = track
        if reverse is not None:
            self.reverse = reverse
        if driver is not None:
            self.driver = driver
        was_running = self.simulation_running
        was_paused = self.simulation_paused
        self.reset_session()
        if was_running:
            self.start_generator()
            if was_paused:
                self.pause_gate.clear()
                self.simulation_paused = True
                self.update_simulation_buttons()
        self.request_ghost()
        self.win.setWindowTitle(f"F1 Telemetry Simulation - {self.track}" + (" (reversed)" if self.reverse else "")
                                + f" - {self.driver}")

    def on_track_changed(self, name):
        if name in TRACKS and name != self.track:
            self.change_session(track=name)

    def on_driver_changed(self, name):
        if name in DRIVERS and name != self.driver:
            self.change_session(driver=name)

    def on_weather_changed(self, name):
        if name in WEATHER_PRESETS and name != self.weather_name:
            self.weather_name = name
            self.ghost_cache.clear()
            self.change_session()

    def on_compound_changed(self, name):
        if name in TYRE_COMPOUNDS and name != self.compound_name:
            self.compound_name = name
            self.ghost_cache.clear()
            self.change_session()

    def on_ref_changed(self, text):
        if text and text != self.ref_mode:
            self.ref_mode = text
            self.an_key = None
            self.request_ghost()

    # --- ghost laps: another driver's best lap, simulated offline in the background ---
    def request_ghost(self):
        self.ghost = None
        if not self.ref_mode.startswith(GHOST):
            return
        name = self.ref_mode[len(GHOST):]
        key = (self.track, self.reverse, name, self.weather_name, self.compound_name,
               tuple(sorted((k, v) for k, v in DRIVERS[name].items() if k != "desc")))
        if key in self.ghost_cache:
            self.ghost = self.ghost_cache[key]
            return
        traits = dict(DRIVERS[name])

        def work():
            lap = simulate_ghost(self.track_at(key), key[1], traits,
                                 weather=WEATHER_PRESETS[key[3]], compound=key[4])
            if lap is not None:
                lap["label"] = f"GHOST: {name.upper()}"
                self.ghost_cache[key] = lap
        threading.Thread(target=work, daemon=True).start()
        self.live.setText(f"Simulating {name} ghost lap...")

    @staticmethod
    def track_at(key):
        return key[0]

    def current_ref(self):
        if self.ref_mode.startswith(GHOST):
            if self.ghost is None:                                   # ghost may have just finished
                name = self.ref_mode[len(GHOST):]
                key = (self.track, self.reverse, name, self.weather_name, self.compound_name,
                       tuple(sorted((k, v) for k, v in DRIVERS[name].items() if k != "desc")))
                self.ghost = self.ghost_cache.get(key)
            if self.ghost is not None:
                return self.ghost
        return self.tel.best

    def on_reverse_toggled(self, on):
        if on != self.reverse:
            self.change_session(reverse=bool(on))

    def poll(self):
        """Consume internally generated telemetry samples from the simulation queue."""
        for _ in range(900):
            try:
                data = self.data_queue.get_nowait()
            except queue.Empty:
                break
            self.tel.handle(data)
            if len(data) >= 24 and HDR.unpack_from(data, 0)[4] == 1 and self.tel.weather:
                self.tel.weather["source"] = "SIMULATED"
        self._update_weather_display()
        self._update_tyre_display()

    def open_tyre_monitor(self):
        """Show a separate, non-modal tyre monitor linked to live simulation data."""
        if self.tyre_window is None:
            self._build_tyre_monitor()
        self._update_tyre_display()
        self.tyre_window.show()
        self.tyre_window.raise_()
        self.tyre_window.activateWindow()

    def _build_tyre_monitor(self):
        QtWidgets = self.QtWidgets
        dlg = QtWidgets.QDialog(self.win)
        dlg.setWindowTitle("F1 Simulation - Tyre Monitor")
        window_flag = getattr(self.QtCore.Qt, "Window", None) or self.QtCore.Qt.WindowType.Window
        dlg.setWindowFlag(window_flag, True)
        dlg.setMinimumSize(560, 430)
        dlg.resize(650, 470)
        dlg.setStyleSheet(f"QDialog{{background:{BG};color:#fff;}} QLabel{{color:#fff;}} "
                          "QProgressBar{background:#292936;border:1px solid #48485a;border-radius:4px;"
                          "text-align:center;color:#fff;height:18px;} "
                          "QProgressBar::chunk{background:#2c91b5;border-radius:3px;}")
        root = QtWidgets.QVBoxLayout(dlg)
        root.setContentsMargins(16, 14, 16, 14); root.setSpacing(10)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("LIVE TYRE MONITOR")
        title.setStyleSheet("font-size:19px;font-weight:bold;color:#65c8ed;")
        header.addWidget(title); header.addStretch(1)
        header.addWidget(QtWidgets.QLabel("COMPOUND"))
        self.tyre_compound_combo = QtWidgets.QComboBox()
        self.tyre_compound_combo.addItems(list(TYRE_COMPOUNDS))
        self.tyre_compound_combo.setCurrentText(self.compound_name)
        self.tyre_compound_combo.setStyleSheet("QComboBox{background:#2d2d3a;color:#fff;padding:5px 9px;min-width:125px;} QComboBox QAbstractItemView{background:#2d2d3a;color:#fff;}")
        self.tyre_compound_combo.currentTextChanged.connect(self.on_compound_changed)
        header.addWidget(self.tyre_compound_combo)
        root.addLayout(header)
        self.tyre_monitor_status = QtWidgets.QLabel("Waiting for simulated tyre data...")
        self.tyre_monitor_status.setStyleSheet("color:#a9a9b8;font-size:11px;padding:2px;")
        root.addWidget(self.tyre_monitor_status)
        grid = QtWidgets.QGridLayout(); grid.setSpacing(10)
        for idx, corner in enumerate(("FRONT LEFT", "FRONT RIGHT", "REAR LEFT", "REAR RIGHT")):
            card = QtWidgets.QFrame(); card.setStyleSheet("QFrame{background:#20202b;border:1px solid #383849;border-radius:7px;}")
            lay = QtWidgets.QVBoxLayout(card); lay.setContentsMargins(12,10,12,10); lay.setSpacing(6)
            heading = QtWidgets.QLabel(corner); heading.setStyleSheet("font-weight:bold;color:#65c8ed;border:none;")
            lay.addWidget(heading); self.tyre_corner_titles.append(heading)
            tval = QtWidgets.QLabel("-- °C"); tval.setStyleSheet("font-size:20px;font-weight:bold;border:none;")
            lay.addWidget(tval); self.tyre_temp_values.append(tval)
            tbar = QtWidgets.QProgressBar(); tbar.setRange(0,140); tbar.setValue(0); tbar.setFormat("Temperature %v °C")
            lay.addWidget(tbar); self.tyre_temp_bars.append(tbar)
            wval = QtWidgets.QLabel("Wear: --%"); wval.setStyleSheet("font-size:12px;color:#d7d7e2;border:none;")
            lay.addWidget(wval); self.tyre_wear_values.append(wval)
            wbar = QtWidgets.QProgressBar(); wbar.setRange(0,100); wbar.setValue(0); wbar.setFormat("Wear %p%")
            wbar.setStyleSheet("QProgressBar::chunk{background:#c58b38;border-radius:3px;}")
            lay.addWidget(wbar); self.tyre_wear_bars.append(wbar)
            grid.addWidget(card, idx//2, idx%2)
        root.addLayout(grid)
        self.tyre_grip_label = QtWidgets.QLabel("Estimated tyre grip: --")
        self.tyre_grip_label.setStyleSheet("background:#20202b;padding:10px;font-size:13px;font-weight:bold;border-radius:5px;")
        root.addWidget(self.tyre_grip_label)
        note = QtWidgets.QLabel("Temperatures and wear are read from the running vehicle model. Grip estimate is indicative and feeds into the simulated vehicle dynamics.")
        note.setWordWrap(True); note.setStyleSheet("color:#9999aa;font-size:10px;")
        root.addWidget(note)
        self.tyre_window = dlg

    def _update_tyre_display(self):
        state = self.tel.tyres
        if not state or self.tyre_window is None:
            return
        if self.tyre_compound_combo.currentText() != self.compound_name:
            self.tyre_compound_combo.blockSignals(True)
            self.tyre_compound_combo.setCurrentText(self.compound_name)
            self.tyre_compound_combo.blockSignals(False)
        temps = state["temperatures"]
        wear = state["wear"]
        names = ("FL", "FR", "RL", "RR")
        for i, (temp, used) in enumerate(zip(temps, wear)):
            self.tyre_temp_values[i].setText(f"{temp:.1f} °C")
            self.tyre_temp_bars[i].setValue(int(np.clip(temp, 0, 140)))
            self.tyre_wear_values[i].setText(f"Wear: {used:.1f}%  |  Remaining: {max(0.0,100.0-used):.1f}%")
            self.tyre_wear_bars[i].setValue(int(np.clip(used, 0, 100)))
        compound = state.get("compound", self.compound_name)
        data = TYRE_COMPOUNDS.get(compound, TYRE_COMPOUNDS["Medium"])
        temp_error = float(np.mean(np.abs(np.asarray(temps, dtype=float) - data["ideal"])))
        temp_factor = float(np.clip(1.0 - 0.0018 * temp_error, 0.78, 1.0))
        wear_factor = float(np.clip(1.0 - 0.0025 * float(np.mean(wear)), 0.75, 1.0))
        grip = data["grip"] * temp_factor * wear_factor
        self.tyre_grip_label.setText(f"ESTIMATED GRIP  |  {grip:.3f}  |  Temperature factor {temp_factor:.3f}  |  Wear factor {wear_factor:.3f}")
        self.tyre_monitor_status.setText(f"{compound.upper()} COMPOUND  ·  {data['life'].upper()} LIFE  ·  IDEAL TEMPERATURE {data['ideal']} °C  ·  LIVE UPDATES")

    def _update_weather_display(self):
        weather = self.tel.weather
        if not weather:
            return
        parts = [weather["condition"].upper(),
                 f"AIR {weather['air_temp']}°C", f"TRACK {weather['track_temp']}°C"]
        if weather.get("rain") is not None:
            parts.append(f"RAIN {weather['rain']}%")
        if weather.get("wind") is not None:
            parts.append(f"WIND {weather['wind']:.1f} m/s")
        source = "SIMULATED"
        self.weather_label.setText(f"{source} WEATHER  |  " + "  ·  ".join(parts))

    def _clear_decor(self):
        for plot, item in self.decor:
            plot.removeItem(item)
        self.decor = []

    def _add(self, plot, item):
        plot.addItem(item, ignoreBounds=True)
        self.decor.append((plot, item))

    def rebuild_analysis(self):
        pg = self.pg
        tel = self.tel
        A, B = tel.laps[-1], self.current_ref()
        first = self.an is None
        an = self.an = analyse(A, B)
        g, a, b = an["grid"], an["a"], an["b"]
        self._clear_decor()
        for key, ch in (("spd", "spd"), ("thr", "thr"), ("brk", "brk")):
            self.c[key + "A"].setData(g, a[ch]); self.c[key + "B"].setData(g, b[ch])
        self.c["delA"].setData(g, an["delta"])
        self.c["gearA"].setData(*step_xy(g, a["gear"])); self.c["gearB"].setData(*step_xy(g, b["gear"]))

        vmin, vmax = float(min(a["spd"].min(), b["spd"].min())), float(max(a["spd"].max(), b["spd"].max()))
        self.p_spd.setYRange(vmin - 30, vmax + 70, padding=0)
        if first:
            self.p_spd.setXRange(0, g[-1], padding=0)
        dm = max(0.15, 1.25 * float(np.abs(an["delta"]).max()))
        self.dm = dm

        for sg in an["segs"]:
            x0, x1 = g[sg["s"]], g[sg["e"]]
            faster_a = sg["gain"] >= 0
            col = RED if faster_a else GRN
            brush = pg.mkBrush(col[0], col[1], col[2], 22)
            for plot in (self.p_spd, self.p_num, self.p_thr, self.p_brk, self.p_gear, self.p_del):
                reg = pg.LinearRegionItem(values=(x0, x1), movable=False, brush=brush)
                for ln in reg.lines:
                    ln.setPen(pg.mkPen(255, 255, 255, 14))
                self._add(plot, reg)
        # corner labels, per-section gain numbers, map colouring
        for n, sg in enumerate(an["segs"]):
            gx = g[sg["apex"]]
            arrow = {"LEFT": "&#9664; ", "RIGHT": " &#9654;"}.get(sg["turn"], "")
            ttl = f"{arrow}T{n + 1}" if sg["turn"] == "LEFT" else f"T{n + 1}{arrow}"
            txt = pg.TextItem(html=f"<div style='text-align:center'><span style='font-size:12px;"
                                   f"font-weight:bold;color:#ffffff'>{ttl}</span><br>"
                                   f"<span style='font-size:9px;color:#d0d0dc'>{sg['vmin']:.0f} km/h</span><br>"
                                   f"<span style='font-size:8px;color:#8c8ca0'>{sg['cls']} - S{sg['sector']}</span></div>",
                              anchor=(0.5, 0))
            txt.setPos(gx, vmax + 66); self._add(self.p_spd, txt)
            gain = sg["gain"]
            colh = RED_H if gain >= 0 else GRN_H
            num = pg.TextItem(html=f"<span style='font-size:12px;font-weight:bold;color:{colh}'>"
                                   f"{'-' if abs(gain) >= 0.0005 else ''}{abs(gain):.3f}</span>",
                              anchor=(0.5, 0.5))
            num.setPos((g[sg["s"]] + g[sg["e"]]) / 2, 0.5); self._add(self.p_num, num)
            col = RED if gain >= 0 else GRN
            seg = self.map.plot(b["x"][sg["s"]:sg["e"] + 2], b["z"][sg["s"]:sg["e"] + 2],
                                pen=pg.mkPen(col, width=5))
            self.decor.append((self.map, seg))
            lab = pg.TextItem(str(n + 1), color=(255, 255, 255), anchor=(0.5, 0.5))
            lab.setPos(b["x"][sg["apex"]], b["z"][sg["apex"]]); self._add(self.map, lab)
        self.map_outline.setData(b["x"], b["z"])
        b1, b2 = an["bounds"]
        for xb in (b1, b2):                                            # sector boundary lines
            for plot in (self.p_spd, self.p_thr, self.p_brk, self.p_gear, self.p_del):
                ln = pg.InfiniteLine(pos=xb, angle=90, pen=pg.mkPen(200, 200, 215, 90, width=1.2,
                                     style=self.QtCore.Qt.DashLine))
                self._add(plot, ln)
        for i, (xa, xe) in enumerate(((0, b1), (b1, b2), (b2, g[-1])), start=1):
            st = pg.TextItem(html=f"<span style='font-size:14px;font-weight:bold;color:#9a9ab0'>SECTOR {i}</span>"
                                  f"<span style='font-size:11px;color:#c8c8d6'> &nbsp;{A[f's{i}']:.3f} / {B[f's{i}']:.3f}</span>",
                             anchor=(0, 1))
            st.setPos(xa + 15, vmin - 10); self._add(self.p_spd, st)
        self.wm = []
        for txt in ("FASTER", "SLOWER"):
            w = pg.TextItem(html=f"<span style='font-size:22px;font-weight:bold;color:#3b3b4c'>{txt}</span>",
                            anchor=(0, 0.5))
            self._add(self.p_del, w); self.wm.append(w)
        self.set_delta_range(dm)

        if an["same"]:
            gap_a = gap_b = "0.000s"
        else:
            gap_a, gap_b = f"{an['gap']:+.3f}s", f"{-an['gap']:+.3f}s"
        who = self.driver.upper()
        # The left card is reserved for the in-progress lap; update_live()
        # refreshes its time, sector timings and delta on every redraw.
        self.card_a.set(f"{who}  -  CURRENT LAP {self.tel.lap}", fmt(self.tel.cur_ms), "LIVE",
                        [None] * 3, "")
        self.card_b.set(B.get("label") or f"{who}  -  BEST LAP {B['lap']}", fmt(B["ms"]), "GAP " + gap_b,
                        an["pb"], sector_html(B, A))

    # ------------------------------------------------------------ cursor
    def _build_cursor(self, gl):
        pg = self.pg
        self.cursor_x, self.cursor_lock = None, False
        self.cursor_lines = {}
        for p in (self.p_spd, self.p_num, self.p_thr, self.p_brk, self.p_gear, self.p_del):
            ln = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(255, 255, 255, 190, width=1))
            ln.setVisible(False); p.addItem(ln, ignoreBounds=True); self.cursor_lines[p] = ln
        self.map_cursor = self.map.plot(symbol="o", symbolSize=11, symbolBrush=(255, 255, 255),
                                        symbolPen=pg.mkPen(0, 0, 0), pen=None)
        self.gl = gl
        self.tip_y = 120.0
        QtCore, QtWidgets = self.QtCore, self.QtWidgets
        self.tip = QtWidgets.QLabel(gl.viewport())
        self.tip.setStyleSheet("background:rgba(24,24,36,235);color:#ffffff;border:1px solid #6a6a85;"
                               "border-radius:4px;padding:6px 9px;font-size:12px;")
        wa = getattr(QtCore.Qt, "WA_TransparentForMouseEvents", None) or QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        self.tip.setAttribute(wa, True)
        self.tip.hide()
        dash = self

        class Leave(QtCore.QObject):
            def eventFilter(self, obj, ev):
                leave = getattr(QtCore.QEvent, "Leave", None) or QtCore.QEvent.Type.Leave
                if ev.type() == leave and not dash.cursor_lock:
                    dash.tip.hide()
                    for ln in dash.cursor_lines.values():
                        ln.setVisible(False)
                    dash.cursor_x = None
                return False
        self._leave = Leave()
        gl.viewport().installEventFilter(self._leave)
        self.proxy = pg.SignalProxy(gl.scene().sigMouseMoved, rateLimit=60, slot=self.on_mouse)
        gl.scene().sigMouseClicked.connect(self.on_click)

    def _plot_at(self, pos):
        for p in self.cursor_lines:
            if p.sceneBoundingRect().contains(pos):
                return p
        return None

    def on_mouse(self, evt):
        if self.cursor_lock:
            return
        pos = evt[0]
        p = self._plot_at(pos)
        if p is not None:
            self.tip_y = self.gl.mapFromScene(pos).y()
            self.set_cursor(p.vb.mapSceneToView(pos).x())

    def on_click(self, ev):
        p = self._plot_at(ev.scenePos())
        if p is None:
            return
        self.cursor_lock = not self.cursor_lock
        if self.cursor_lock:
            self.tip_y = self.gl.mapFromScene(ev.scenePos()).y()
            self.set_cursor(p.vb.mapSceneToView(ev.scenePos()).x())
        else:
            self.update_readout()

    def set_cursor(self, x):
        self.cursor_x = x
        for ln in self.cursor_lines.values():
            ln.setVisible(True); ln.setPos(x)
        self.update_readout()

    def update_readout(self):
        x = self.cursor_x
        if x is None:
            return
        sq = lambda col: f"<span style='color:{col}'>&#9632;</span>"
        b, an = self.tel.buf, self.an
        d = np.array(b["d"])

        def live(ch):
            if len(d) > 2 and d[0] <= x <= d[-1]:
                return float(np.interp(x, d, b[ch]))
            return None

        def trio(chan, f):
            """[(colour, text), ...] for red lap, green lap, live lap at the cursor distance."""
            vals = []
            if an and 0 <= x <= an["grid"][-1]:
                g = an["grid"]
                vals += [(RED_H, float(np.interp(x, g, an["a"][chan]))),
                         (GRN_H, float(np.interp(x, g, an["b"][chan])))]
            vals.append((BLU_H, live(chan)))
            return [(c, "--" if v is None else f(v)) for c, v in vals]

        corner = ""
        if an and 0 <= x <= an["grid"][-1]:
            g = an["grid"]
            for n, sg in enumerate(an["segs"]):
                if g[sg["s"]] <= x <= g[sg["e"]]:
                    corner = f"T{n + 1} - S{sg['sector']}"
                    break
        rows = [("SPEED", trio("spd", lambda v: f"{v:.0f}"), "km/h"),
                ("THROTTLE", trio("thr", lambda v: f"{v * 100:.0f}"), "%"),
                ("BRAKE", trio("brk", lambda v: f"{v * 100:.0f}"), "%"),
                ("GEAR", trio("gear", lambda v: f"{v:.0f}"), "")]
        gap = None
        if an and 0 <= x <= an["grid"][-1]:
            gap = -float(np.interp(x, an["grid"], an["delta"]))
            self.map_cursor.setData([float(np.interp(x, an["grid"], an["b"]["x"]))],
                                    [float(np.interp(x, an["grid"], an["b"]["z"]))])
        elif len(d) > 2 and d[0] <= x <= d[-1]:
            self.map_cursor.setData([float(np.interp(x, d, b["x"]))], [float(np.interp(x, d, b["z"]))])

        # --- bar under the live strip ---
        bar = [f"<b>DIST {x:.0f} m</b>" + (f" &nbsp;<span style='color:#9a9ab0'>{corner}</span>" if corner else "")]
        for name, vals, unit in rows:
            bar.append(f"{name} " + " ".join(f"{sq(c)}{t}" for c, t in vals) + (f" {unit}" if unit else ""))
        if gap is not None:
            bar.append(f"GAP {gap:+.3f}s")
        if self.cursor_lock:
            bar.append("<span style='color:#e0c060'>LOCKED</span>")
        self.readout.setText("&nbsp;&nbsp;|&nbsp;&nbsp;".join(bar))

        # --- floating box next to the cursor line ---
        html = [f"<div style='font-size:13px'><b>{x:.0f} m</b>"
                + (f" &nbsp;<span style='color:#9a9ab0'>{corner}</span>" if corner else "")
                + ("&nbsp;&nbsp;<span style='color:#e0c060'>LOCKED</span>" if self.cursor_lock else "") + "</div>",
                "<table cellspacing='0' cellpadding='1'>"]
        for name, vals, unit in rows:
            cells = "".join(f"<td align='right' style='color:{c};font-weight:bold'>&nbsp;&nbsp;{t}</td>"
                            for c, t in vals)
            html.append(f"<tr><td style='color:#9a9aa8'>{name}</td>{cells}"
                        f"<td style='color:#7d7d90'>&nbsp;{unit}</td></tr>")
        if gap is not None:
            html.append(f"<tr><td style='color:#9a9aa8'>GAP</td><td colspan='{len(rows[0][1])}' align='right' "
                        f"style='font-weight:bold'>{gap:+.3f} s</td><td></td></tr>")
        html.append("</table>")
        self.tip.setText("".join(html))
        self.tip.adjustSize()
        self.place_tip()

    def place_tip(self):
        """Put the box beside the cursor line, flipping to the left near the right-hand edge."""
        if self.cursor_x is None:
            return
        QPointF = self.QtCore.QPointF
        sp = self.p_spd.vb.mapViewToScene(QPointF(self.cursor_x, 0))
        px = self.gl.mapFromScene(sp).x()
        vw, vh = self.gl.viewport().width(), self.gl.viewport().height()
        w, h = self.tip.width(), self.tip.height()
        bx = px + 16
        if bx + w > vw - 4:
            bx = px - 16 - w
        by = int(min(max(self.tip_y - h / 2, 4), vh - h - 4))
        self.tip.move(int(max(bx, 4)), by)
        self.tip.show(); self.tip.raise_()

    def update_live(self):
        tel, an, b = self.tel, self.an, self.tel.buf
        self.lv_time.setText(fmt(tel.cur_ms))
        if an is not None and len(b["d"]) > 2:
            tref = float(np.interp(b["d"][-1], an["grid"], an["b"]["t"]))
            dl = (b["t"][-1] - b["t"][0]) - tref
            col = "#5fe08a" if dl <= 0 else "#ff6b5e"
            ref_name = (an["B"].get("label") or "BEST LAP").upper()
            self.lv_delta.setText(f"<span style='color:{col}'>{dl:+.3f}s</span> "
                                  f"<span style='font-size:11px;color:#9a9aa8'>vs {ref_name}</span>")
        else:
            self.lv_delta.setText("<span style='color:#9a9aa8;font-size:12px'>complete a lap to see the live gap</span>")
        sec, cur, s1, s2 = tel.sector, tel.cur_ms, tel.s1ms, tel.s2ms
        times = [s1 if sec >= 1 else cur,
                 s2 if sec >= 2 else (cur - s1 if sec == 1 else None),
                 (cur - s1 - s2) if sec == 2 else None]
        out = []
        for i, t in enumerate(times):
            ref = an["B"].get(f"s{i + 1}") if an else None
            if t is None:
                out.append(f"<span style='color:#6a6a7c;font-size:10px'>S{i + 1}</span> "
                           f"<span style='color:#6a6a7c'>--</span>")
                continue
            running = (i == sec)
            col = BLU_H if running else "#ffffff"
            diff = ""
            if ref is not None and not running:
                diff = f" <span style='color:#9a9aa8;font-size:11px'>{t / 1000 - ref:+.3f}</span>"
            out.append(f"<span style='color:#9a9aa8;font-size:10px'>S{i + 1}</span> "
                       f"<span style='color:{col};font-size:15px;font-weight:bold'>{t / 1000:.3f}</span>{diff}")
        self.lv_sec.setText("&nbsp;&nbsp;&nbsp;".join(out))
        self.lv_car.setText(f"<span style='font-size:24px;font-weight:bold'>{tel.speed}</span> km/h &nbsp;&nbsp;"
                            f"<span style='font-size:24px;font-weight:bold'>G{tel.gear}</span> &nbsp;&nbsp;"
                            f"<span style='font-size:15px'>{tel.rpm} rpm</span>")
        self.lv_title.setText(f"CURRENT LAP<br>LAP {tel.lap}")

        # Keep the large left-hand card live, not frozen on the previous lap.
        # The right-hand card continues to show the session best lap.
        if hasattr(self, "card_a"):
            if an is not None and len(b["d"]) > 2:
                live_delta = (b["t"][-1] - b["t"][0]) - float(
                    np.interp(b["d"][-1], an["grid"], an["b"]["t"]))
                gap_text = f"DELTA {live_delta:+.3f}s"
                gap_col = "#a855f7" if live_delta < -0.001 else (
                    "#5fe08a" if live_delta <= 0.001 else "#ff6b5e")
            else:
                gap_text, gap_col = "REFERENCE PENDING", "#9a9aa8"
            sector_text = "&nbsp;&nbsp;&nbsp;".join(out)
            # Recalculate the current-lap driving metrics from the samples
            # accumulated so far.  Passing [None] here leaves all three
            # progress bars blank, even though the live telemetry is updating.
            if len(b["t"]) > 1:
                live_pcts = lap_pcts({
                    "t": np.asarray(b["t"], dtype=float),
                    "thr": np.asarray(b["thr"], dtype=float),
                    "brk": np.asarray(b["brk"], dtype=float),
                })
            else:
                live_pcts = [0.0, 0.0, 0.0]
            self.card_a.set(f"CURRENT LAP  {tel.lap}", fmt(cur),
                            f"<span style='color:{gap_col}'>{gap_text}</span>",
                            live_pcts, sector_text)
            self.card_a.time.setStyleSheet(f"font-size:30px;font-weight:bold;color:{gap_col};")

    def set_delta_range(self, lim):
        self.p_del.setYRange(-lim, lim, padding=0)
        x0 = float(self.an["grid"][0]) + 20 if self.an else 0.0
        for w, sgn in zip(getattr(self, "wm", []), (1, -1)):
            w.setPos(x0, sgn * lim * 0.6)


    def _lap_metrics(self, lap):
        """Return useful derived metrics for a completed lap."""
        ft, hb, corner = lap_pcts(lap)
        speed = np.asarray(lap["spd"], dtype=float)
        rpm = np.asarray(lap.get("rpm", []), dtype=float)
        return {
            "max_speed": float(np.max(speed)) if speed.size else 0.0,
            "avg_speed": float(np.mean(speed)) if speed.size else 0.0,
            "max_rpm": float(np.max(rpm)) if rpm.size else 0.0,
            "ft": ft, "hb": hb, "corner": corner,
        }

    def open_lap_history(self):
        """Open a selectable lap-by-lap analysis window with Excel export."""
        QtWidgets = self.QtWidgets
        if not self.tel.laps:
            QtWidgets.QMessageBox.information(self.win, "Lap history",
                                               "No completed laps yet. Complete a lap to populate the history.")
            return

        dlg = QtWidgets.QDialog(self.win)
        dlg.setWindowTitle(f"Lap History & Analysis — {self.track}")
        dlg.resize(1180, 820)
        dlg.setStyleSheet(f"QDialog{{background:{BG};color:#fff;}}"
                          "QLabel{color:#fff;} QHeaderView::section{background:#30303d;color:#fff;padding:6px;border:0;}"
                          "QTableWidget{background:#1d1d29;color:#fff;gridline-color:#393947;"
                          "selection-background-color:#285b86;selection-color:#ffffff;}"
                          "QTableWidget::item{padding:5px;background-color:#1d1d29;color:#ffffff;}"
                          "QTableWidget::item:alternate{background-color:#242432;color:#ffffff;}"
                          "QTableWidget::item:selected{background-color:#285b86;color:#ffffff;}"
                          "QTableWidget::item:selected:!active{background-color:#285b86;color:#ffffff;}")
        layout = QtWidgets.QVBoxLayout(dlg)

        heading = QtWidgets.QLabel(
            f"<b style='font-size:17px'>SESSION LAP HISTORY</b>"
            f"<span style='color:#a0a0b0'> &nbsp; {self.track} &nbsp;|&nbsp; "
            f"{self.driver} &nbsp;|&nbsp; "
            f"{len(self.tel.laps)} completed laps</span>")
        layout.addWidget(heading)

        cols = ["Lap", "Lap time", "S1", "S2", "S3", "Δ Best", "Max km/h",
                "Avg km/h", "Full throttle", "Heavy braking"]
        table = QtWidgets.QTableWidget(len(self.tel.laps), len(cols))
        table.setHorizontalHeaderLabels(cols)
        view = QtWidgets.QAbstractItemView
        table.setSelectionBehavior(getattr(view, "SelectRows", None) or view.SelectionBehavior.SelectRows)
        table.setSelectionMode(getattr(view, "SingleSelection", None) or view.SelectionMode.SingleSelection)
        table.setEditTriggers(getattr(view, "NoEditTriggers", None) or view.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setAutoFillBackground(True)
        palette = table.palette()
        palette.setColor(self.QtGui.QPalette.Base, self.QtGui.QColor("#1d1d29"))
        palette.setColor(self.QtGui.QPalette.AlternateBase, self.QtGui.QColor("#242432"))
        palette.setColor(self.QtGui.QPalette.Text, self.QtGui.QColor("#ffffff"))
        palette.setColor(self.QtGui.QPalette.Highlight, self.QtGui.QColor("#285b86"))
        palette.setColor(self.QtGui.QPalette.HighlightedText, self.QtGui.QColor("#ffffff"))
        table.setPalette(palette)
        table.verticalHeader().setVisible(False)
        table.setSortingEnabled(False)
        header = table.horizontalHeader()
        header_view = QtWidgets.QHeaderView
        header.setSectionResizeMode(getattr(header_view, "ResizeToContents", None)
                                    or header_view.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

        best_ms = min(float(lap["ms"]) for lap in self.tel.laps)
        for row, lap in enumerate(self.tel.laps):
            metrics = self._lap_metrics(lap)
            values = [
                str(lap["lap"]), fmt(lap["ms"]),
                f'{lap["s1"]:.3f}', f'{lap["s2"]:.3f}', f'{lap["s3"]:.3f}',
                f'{(lap["ms"] - best_ms) / 1000:+.3f}',
                f'{metrics["max_speed"]:.1f}', f'{metrics["avg_speed"]:.1f}',
                f'{metrics["ft"]:.1f}%', f'{metrics["hb"]:.1f}%'
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(_align("AlignCenter"))
                table.setItem(row, col, item)
            if float(lap["ms"]) == best_ms:
                for col in range(len(cols)):
                    table.item(row, col).setForeground(self.QtGui.QColor("#55dc85"))
        table.setMinimumHeight(135)
        table.setMaximumHeight(245)
        table.setToolTip("Click any lap row to load that lap's telemetry below.")
        layout.addWidget(table)

        detail = QtWidgets.QLabel("Select a lap above to inspect its telemetry.")
        detail.setStyleSheet("background:#252532;padding:9px;font-size:13px;")
        layout.addWidget(detail)

        pg = self.pg
        graph = pg.GraphicsLayoutWidget()
        graph.setBackground(BG)
        p_speed = graph.addPlot(row=0, col=0, title="SPEED (km/h)")
        p_rpm = graph.addPlot(row=1, col=0, title="ENGINE RPM")
        p_inputs = graph.addPlot(row=2, col=0, title="THROTTLE / BRAKE")
        p_gear = graph.addPlot(row=3, col=0, title="GEAR")
        p_steer = graph.addPlot(row=4, col=0, title="STEERING")
        for p in (p_speed, p_rpm, p_inputs, p_gear, p_steer):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setLabel("bottom", "Distance (m)")
            p.setMouseEnabled(x=True, y=False)
        p_inputs.setYRange(0, 1.05, padding=0)
        speed_curve = p_speed.plot(pen=pg.mkPen("#4696ff", width=2))
        rpm_curve = p_rpm.plot(pen=pg.mkPen("#f2c14e", width=1.8))
        thr_curve = p_inputs.plot(pen=pg.mkPen("#3cd25a", width=2))
        brk_curve = p_inputs.plot(pen=pg.mkPen("#ff483c", width=2))
        gear_curve = p_gear.plot(pen=pg.mkPen("#b78cff", width=2))
        steer_curve = p_steer.plot(pen=pg.mkPen("#e0e0e8", width=1.8))
        graph.setMinimumHeight(260)
        layout.addWidget(graph, 1)

        # Synchronized distance cursor for the selected historical lap.
        # Hover to inspect values; click once to lock and click again to release.
        history_plots = (p_speed, p_rpm, p_inputs, p_gear, p_steer)
        history_cursor_lines = {}
        for plot in history_plots:
            line = pg.InfiniteLine(angle=90, movable=False,
                                   pen=pg.mkPen(255, 255, 255, 205, width=1.2))
            line.setVisible(False)
            plot.addItem(line, ignoreBounds=True)
            history_cursor_lines[plot] = line

        # Compact floating telemetry box, positioned beside the vertical cursor
        # at the mouse pointer's Y position on the speed plot.
        cursor_box = pg.TextItem(
            text="", color="#f4f4f7", anchor=(0, 0),
            fill=pg.mkBrush(25, 28, 38, 235),
            border=pg.mkPen(70, 150, 255, 220, width=1.2))
        cursor_box.setZValue(100)
        cursor_box.setVisible(False)
        p_speed.addItem(cursor_box, ignoreBounds=True)

        history_cursor = {"x": None, "locked": False, "lap": None}

        def history_plot_at(scene_pos):
            for plot in history_plots:
                if plot.sceneBoundingRect().contains(scene_pos):
                    return plot
            return None

        def update_history_cursor(distance, mouse_y=None):
            lap = history_cursor["lap"]
            if lap is None:
                return
            d = np.asarray(lap.get("d", []), dtype=float)
            if len(d) < 2:
                return
            x = float(np.clip(distance, d[0], d[-1]))
            history_cursor["x"] = x
            for line in history_cursor_lines.values():
                line.setPos(x)
                line.setVisible(True)

            def value(key, default=0.0):
                values = np.asarray(lap.get(key, []), dtype=float)
                if len(values) == 0:
                    return default
                n = min(len(d), len(values))
                return float(np.interp(x, d[:n], values[:n]))

            gear_value = int(round(value("gear", 0)))
            locked = "<span style='color:#f2c14e'>LOCKED</span>" if history_cursor["locked"] else ""
            cursor_box.setHtml(
                "<div style='font-size:10pt; line-height:145%; white-space:nowrap;'>"
                "<span style='color:#9bbcff;font-weight:bold'>LAP TELEMETRY</span> " + locked + "<br/>"
                f"<b>Distance</b> &nbsp; {x:.1f} m<br/>"
                f"<b>Speed</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {value('spd'):.1f} km/h<br/>"
                f"<b>RPM</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {value('rpm'):.0f}<br/>"
                f"<b>Throttle</b> &nbsp; {value('thr') * 100:.1f}%<br/>"
                f"<b>Brake</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {value('brk') * 100:.1f}%<br/>"
                f"<b>Gear</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {gear_value}<br/>"
                f"<b>Steering</b> &nbsp; {value('str'):.3f}</div>")
            # Keep the box beside the cursor and flip it to the left near the
            # right edge so it remains inside the plotting area.
            (xr, yr) = p_speed.viewRange()
            x_span = max(float(xr[1] - xr[0]), 1.0)
            y_span = max(float(yr[1] - yr[0]), 1.0)
            if x > xr[0] + 0.72 * x_span:
                cursor_box.setAnchor((1, 1))
                box_x = x - 0.018 * x_span
            else:
                cursor_box.setAnchor((0, 1))
                box_x = x + 0.018 * x_span
            # Follow the pointer vertically instead of pinning the telemetry
            # box to the top of the speed graph where it can obscure the trace.
            if mouse_y is None:
                box_y = 0.5 * (yr[0] + yr[1])
            else:
                box_y = float(np.clip(mouse_y, yr[0], yr[1]))
            if box_y >= 0.5 * (yr[0] + yr[1]):
                cursor_box.setAnchor((0 if box_x > x else 1, 1))
            else:
                cursor_box.setAnchor((0 if box_x > x else 1, 0))
            cursor_box.setPos(box_x, box_y)
            cursor_box.setVisible(True)

        def history_mouse(event):
            if history_cursor["locked"]:
                return
            scene_pos = event[0]
            plot = history_plot_at(scene_pos)
            if plot is not None:
                # Map the same scene position into the speed plot's coordinate
                # system. This keeps the vertical cursor synchronized by
                # distance while placing the readout at the mouse's Y position.
                speed_pointer = p_speed.vb.mapSceneToView(scene_pos)
                update_history_cursor(speed_pointer.x(), speed_pointer.y())

        def history_click(event):
            plot = history_plot_at(event.scenePos())
            if plot is None:
                return
            history_cursor["locked"] = not history_cursor["locked"]
            if history_cursor["locked"]:
                speed_pointer = p_speed.vb.mapSceneToView(event.scenePos())
                update_history_cursor(speed_pointer.x(), speed_pointer.y())
            elif history_cursor["x"] is not None:
                update_history_cursor(history_cursor["x"])

        history_mouse_proxy = pg.SignalProxy(
            graph.scene().sigMouseMoved, rateLimit=60, slot=history_mouse)
        graph.scene().sigMouseClicked.connect(history_click)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        export_btn = QtWidgets.QPushButton("EXPORT SESSION TO EXCEL")
        export_btn.setStyleSheet("QPushButton{background:#176b50;color:#fff;border:0;padding:9px 16px;font-weight:bold;}"
                                 "QPushButton:hover{background:#218363;}")
        history_fullscreen_btn = QtWidgets.QPushButton("FULLSCREEN")
        history_fullscreen_btn.setStyleSheet("QPushButton{background:#2d2d3a;color:#fff;border:0;padding:9px 16px;font-weight:bold;}")
        close_btn = QtWidgets.QPushButton("CLOSE")
        close_btn.setStyleSheet("QPushButton{background:#343443;color:#fff;border:0;padding:9px 16px;}")
        footer.addWidget(export_btn)
        footer.addWidget(history_fullscreen_btn)
        footer.addWidget(close_btn)
        def toggle_history_fullscreen():
            if dlg.isFullScreen():
                dlg.showMaximized()
                history_fullscreen_btn.setText("FULLSCREEN")
            else:
                dlg.showFullScreen()
                history_fullscreen_btn.setText("EXIT FULLSCREEN")
        history_fullscreen_btn.clicked.connect(toggle_history_fullscreen)
        layout.addLayout(footer)

        def show_lap(row, _column=0):
            if row < 0 or row >= len(self.tel.laps):
                return
            lap = self.tel.laps[row]
            m = self._lap_metrics(lap)
            delta = (float(lap["ms"]) - best_ms) / 1000.0
            detail.setText(
                f"<b>LAP {lap['lap']} — {fmt(lap['ms'])}</b> &nbsp; "
                f"S1: {lap['s1']:.3f}s &nbsp; S2: {lap['s2']:.3f}s &nbsp; S3: {lap['s3']:.3f}s"
                f"&nbsp; | &nbsp; Δ Best: {delta:+.3f}s<br>"
                f"Max speed: {m['max_speed']:.1f} km/h &nbsp; Average speed: {m['avg_speed']:.1f} km/h"
                f" &nbsp; Max RPM: {m['max_rpm']:.0f}<br>"
                f"Full throttle: {m['ft']:.1f}% &nbsp; Heavy braking: {m['hb']:.1f}%"
                f" &nbsp; Cornering / other: {m['corner']:.1f}%")
            d = np.asarray(lap["d"], dtype=float)
            history_cursor["lap"] = lap
            history_cursor["locked"] = False
            history_cursor["x"] = None
            for line in history_cursor_lines.values():
                line.setVisible(False)
            cursor_box.setVisible(False)
            speed_curve.setData(d, lap["spd"])
            rpm_curve.setData(d, lap.get("rpm", []))
            thr_curve.setData(d, lap["thr"])
            brk_curve.setData(d, lap["brk"])
            gear_curve.setData(*step_xy(d, lap["gear"]))
            steer_curve.setData(d, lap["str"])
            if len(d):
                # Scale each channel to its own lap data. Fixed wide ranges made
                # normal variation look almost flat, especially RPM and steering.
                def fit_y(plot, values, floor_zero=False, min_span=1.0, pad=0.08):
                    a = np.asarray(values, dtype=float)
                    a = a[np.isfinite(a)]
                    if not a.size:
                        plot.setYRange(0, min_span, padding=0)
                        return
                    lo, hi = float(np.min(a)), float(np.max(a))
                    span = hi - lo
                    if span < min_span:
                        centre = 0.5 * (hi + lo)
                        lo, hi = centre - 0.5 * min_span, centre + 0.5 * min_span
                    margin = max((hi - lo) * pad, min_span * 0.025)
                    lo, hi = lo - margin, hi + margin
                    if floor_zero:
                        lo = max(0.0, lo)
                    plot.setYRange(lo, hi, padding=0)

                fit_y(p_speed, lap["spd"], floor_zero=True, min_span=8.0, pad=0.12)
                fit_y(p_rpm, lap.get("rpm", []), floor_zero=True, min_span=500.0, pad=0.10)
                p_inputs.setYRange(0, 1.05, padding=0)
                gears = np.asarray(lap.get("gear", []), dtype=float)
                if gears.size:
                    gmin = max(0, int(np.nanmin(gears)))
                    gmax = max(gmin + 1, int(np.nanmax(gears)))
                    p_gear.setYRange(max(-0.5, gmin - 0.5), gmax + 0.5, padding=0)
                    p_gear.getAxis("left").setTicks([[(i, str(i)) for i in range(gmin, gmax + 1)]])
                fit_y(p_steer, lap["str"], min_span=0.05, pad=0.15)
                for p in (p_speed, p_rpm, p_inputs, p_gear, p_steer):
                    p.setXRange(0, float(d[-1]), padding=0)

        # currentCellChanged also catches keyboard navigation and row selection;
        # cellClicked makes a single mouse click immediately load that lap.
        table.cellClicked.connect(show_lap)
        table.currentCellChanged.connect(
            lambda current_row, _current_col, _previous_row, _previous_col: show_lap(current_row)
        )
        table.selectRow(len(self.tel.laps) - 1)
        show_lap(len(self.tel.laps) - 1)

        def export_excel():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Export telemetry session", "F1_Telemetry_Session.xlsx",
                "Excel Workbook (*.xlsx)")
            if not path:
                return
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            try:
                from openpyxl import Workbook
                from openpyxl.styles import Font, PatternFill, Alignment
                from openpyxl.utils import get_column_letter

                wb = Workbook()
                ws = wb.active
                ws.title = "Lap Summary"
                ws.append(["F1 TELEMETRY — SESSION LAP SUMMARY"])
                ws.append(["Track", self.track])
                ws.append(["Driver / Source", self.driver])
                ws.append(["Exported", datetime.datetime.now().astimezone().isoformat(timespec="seconds")])
                ws.append([])
                headers = ["Lap", "Lap time (s)", "Lap time", "Sector 1 (s)", "Sector 2 (s)",
                           "Sector 3 (s)", "Delta to best (s)", "Max speed (km/h)",
                           "Average speed (km/h)", "Max RPM", "Full throttle (%)",
                           "Heavy braking (%)", "Cornering / other (%)", "Weather",
                           "Air temp (°C)", "Track temp (°C)", "Rain (%)", "Wind (m/s)"]
                ws.append(headers)
                for lap in self.tel.laps:
                    m = self._lap_metrics(lap)
                    weather = lap.get("weather") or {}
                    ws.append([lap["lap"], lap["ms"] / 1000.0, fmt(lap["ms"]),
                               lap["s1"], lap["s2"], lap["s3"],
                               (lap["ms"] - best_ms) / 1000.0, m["max_speed"],
                               m["avg_speed"], m["max_rpm"], m["ft"], m["hb"], m["corner"],
                               weather.get("condition", "N/A"), weather.get("air_temp"),
                               weather.get("track_temp"), weather.get("rain"), weather.get("wind")])

                # One Excel row per recorded telemetry packet/sample. In addition to
                # raw channels, compare each sample with the best lap at the same
                # distance. Positive delta means this lap is behind the reference.
                samples = wb.create_sheet("Telemetry Samples")
                samples.append(["Lap", "Sample #", "Lap time at point (s)", "Distance (m)",
                                "Speed (km/h)", "RPM", "Throttle (0-1)", "Brake (0-1)",
                                "Steering", "Gear", "X position", "Z position",
                                "Reference lap", "Reference lap time at distance (s)",
                                "Time delta vs best at distance (s)",
                                "Change in delta since previous point (s)",
                                "Speed difference vs best (km/h)"])

                reference = self.tel.best if self.tel.best is not None else min(self.tel.laps, key=lambda item: item["ms"])
                ref_d = np.asarray(reference.get("d", []), dtype=float)
                ref_t = np.asarray(reference.get("t", []), dtype=float)
                ref_spd = np.asarray(reference.get("spd", []), dtype=float)
                valid = np.isfinite(ref_d) & np.isfinite(ref_t)
                ref_d, ref_t = ref_d[valid], ref_t[valid]
                if ref_spd.size == valid.size:
                    ref_spd = ref_spd[valid]
                if ref_d.size:
                    order = np.argsort(ref_d)
                    ref_d, ref_t = ref_d[order], ref_t[order]
                    if ref_spd.size == order.size:
                        ref_spd = ref_spd[order]
                    # np.interp expects increasing x; collapse repeated distance samples.
                    unique_d, unique_idx = np.unique(ref_d, return_index=True)
                    ref_d, ref_t = unique_d, ref_t[unique_idx]
                    if ref_spd.size == order.size:
                        ref_spd = ref_spd[unique_idx]

                point_analysis = wb.create_sheet("Distance Time Loss")
                point_analysis.append(["Lap", "Sample #", "Distance (m)", "Lap time (s)",
                                       "Best lap time at same distance (s)",
                                       "Time delta vs best (s)",
                                       "Time gained/lost since previous point (s)",
                                       "Speed (km/h)", "Best lap speed at same distance (km/h)",
                                       "Speed difference (km/h)"])

                for lap in self.tel.laps:
                    count = len(lap["d"])
                    previous_delta = None
                    for i in range(count):
                        get = lambda key, default=None: lap[key][i] if key in lap and len(lap[key]) > i else default
                        distance = float(get("d", 0.0) or 0.0)
                        elapsed = float(get("t", 0.0) or 0.0)
                        speed = float(get("spd", 0.0) or 0.0)
                        if ref_d.size:
                            ref_elapsed = float(np.interp(distance, ref_d, ref_t))
                            delta = elapsed - ref_elapsed
                            ref_speed = float(np.interp(distance, ref_d, ref_spd)) if ref_spd.size == ref_d.size else None
                        else:
                            ref_elapsed, delta, ref_speed = None, None, None
                        delta_step = (delta - previous_delta) if delta is not None and previous_delta is not None else None
                        speed_diff = (speed - ref_speed) if ref_speed is not None else None
                        samples.append([lap["lap"], i + 1, elapsed, distance, speed, get("rpm"),
                                        get("thr"), get("brk"), get("str"), get("gear"),
                                        get("x"), get("z"), reference.get("lap"), ref_elapsed,
                                        delta, delta_step, speed_diff])
                        point_analysis.append([lap["lap"], i + 1, distance, elapsed,
                                               ref_elapsed, delta, delta_step, speed,
                                               ref_speed, speed_diff])
                        if delta is not None:
                            previous_delta = delta

                # Give all three sheets a consistent report-style layout.
                from openpyxl.styles import Border, Side
                from openpyxl.formatting.rule import ColorScaleRule

                navy = "15151E"
                blue = "285B86"
                light_blue = "EAF2F8"
                pale = "F5F7FA"
                white = "FFFFFF"
                grey = "64748B"
                thin_grey = Side(style="thin", color="D9E1E8")

                # The summary already has a title and metadata in rows 1-4.
                # Add matching title bands to the two detailed-data sheets.
                for sheet, title, subtitle in (
                    (samples, "TELEMETRY SAMPLES", "Generated telemetry samples with same-distance comparison against the reference lap."),
                    (point_analysis, "DISTANCE-BASED TIME ANALYSIS", "Positive time delta = behind the reference; negative = ahead. Change in delta shows gain/loss over each interval."),
                ):
                    sheet.insert_rows(1, 3)
                    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=sheet.max_column)
                    sheet.cell(1, 1, title)
                    sheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=sheet.max_column)
                    sheet.cell(2, 1, subtitle)

                ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ws.max_column)

                layouts = [
                    (ws, 6, 7),
                    (samples, 4, 5),
                    (point_analysis, 4, 5),
                ]
                for sheet, header_row, first_data_row in layouts:
                    sheet.sheet_view.showGridLines = False
                    sheet.sheet_view.zoomScale = 90 if sheet is not ws else 95
                    sheet.freeze_panes = f"A{first_data_row}"
                    sheet.sheet_properties.pageSetUpPr.fitToPage = True
                    sheet.page_setup.fitToWidth = 1
                    sheet.page_setup.fitToHeight = 0
                    sheet.page_setup.orientation = "landscape"
                    sheet.sheet_properties.tabColor = blue if sheet is not ws else "16A085"
                    sheet.auto_filter.ref = (f"A{header_row}:{get_column_letter(sheet.max_column)}{sheet.max_row}")

                    # Main title band.
                    sheet.row_dimensions[1].height = 29
                    for cell in sheet[1]:
                        cell.fill = PatternFill("solid", fgColor=navy)
                        cell.font = Font(name="Aptos Display", bold=True, color=white, size=15)
                        cell.alignment = Alignment(vertical="center")
                    if sheet is not ws:
                        sheet.row_dimensions[2].height = 25
                        c = sheet.cell(2, 1)
                        c.font = Font(name="Aptos", italic=True, color=grey, size=10)
                        c.alignment = Alignment(vertical="center", wrap_text=True)
                    else:
                        # Style the summary metadata block.
                        for r in range(2, 5):
                            sheet.cell(r, 1).font = Font(name="Aptos", bold=True, color=grey)
                            sheet.cell(r, 2).font = Font(name="Aptos", color=navy)

                    sheet.row_dimensions[header_row].height = 34
                    for cell in sheet[header_row]:
                        cell.fill = PatternFill("solid", fgColor=blue)
                        cell.font = Font(name="Aptos", bold=True, color=white, size=10)
                        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                        cell.border = Border(bottom=Side(style="medium", color=navy))

                    # Alternating row shading, readable alignment, and subtle separators.
                    for r in range(first_data_row, sheet.max_row + 1):
                        sheet.row_dimensions[r].height = 19
                        for c in range(1, sheet.max_column + 1):
                            cell = sheet.cell(r, c)
                            cell.font = Font(name="Aptos", size=10, color="243447")
                            cell.fill = PatternFill("solid", fgColor=white if (r-first_data_row) % 2 == 0 else pale)
                            cell.border = Border(bottom=thin_grey)
                            cell.alignment = Alignment(vertical="center", horizontal="right" if isinstance(cell.value, (int, float)) else "left")

                    # Widths based on headers and a sample of values, with useful limits.
                    for col in range(1, sheet.max_column + 1):
                        header = str(sheet.cell(header_row, col).value or "")
                        vals = [len(str(sheet.cell(r, col).value or "")) for r in range(first_data_row, min(sheet.max_row, first_data_row + 300) + 1)]
                        ideal = max([len(header) + 2] + vals + [10])
                        sheet.column_dimensions[get_column_letter(col)].width = min(max(ideal, 12), 25)
                    sheet.sheet_view.selection[0].activeCell = f"A{first_data_row}"
                    sheet.sheet_view.selection[0].sqref = f"A{first_data_row}"

                # Summary: lap times and performance metrics.
                for r in range(7, ws.max_row + 1):
                    for c in (2, 4, 5, 6, 7): ws.cell(r, c).number_format = "0.000"
                    for c in (8, 9): ws.cell(r, c).number_format = "0.0"
                    ws.cell(r, 10).number_format = "#,##0"
                    for c in (11, 12, 13): ws.cell(r, c).number_format = "0.0"
                ws.column_dimensions["A"].width = 10
                ws.column_dimensions["C"].width = 14

                # Detailed telemetry number formats.
                for r in range(5, samples.max_row + 1):
                    for c in (3, 14, 15, 16): samples.cell(r, c).number_format = "0.000"
                    for c in (4, 5, 17): samples.cell(r, c).number_format = "0.0"
                    for c in (6,): samples.cell(r, c).number_format = "#,##0"
                    for c in (7, 8, 9): samples.cell(r, c).number_format = "0.000"
                    for c in (11, 12): samples.cell(r, c).number_format = "0.000"
                for r in range(5, point_analysis.max_row + 1):
                    for c in (3, 4, 5, 6, 7): point_analysis.cell(r, c).number_format = "0.000"
                    for c in (8, 9, 10): point_analysis.cell(r, c).number_format = "0.0"

                # Heatmaps make gains/losses immediately visible.
                if samples.max_row >= 5:
                    for col in (15, 16):
                        letter = get_column_letter(col)
                        samples.conditional_formatting.add(
                            f"{letter}5:{letter}{samples.max_row}",
                            ColorScaleRule(start_type="min", start_color="63BE7B",
                                           mid_type="percentile", mid_value=50, mid_color="FFEB84",
                                           end_type="max", end_color="F8696B"))
                if point_analysis.max_row >= 5:
                    for col in (6, 7):
                        letter = get_column_letter(col)
                        point_analysis.conditional_formatting.add(
                            f"{letter}5:{letter}{point_analysis.max_row}",
                            ColorScaleRule(start_type="min", start_color="63BE7B",
                                           mid_type="percentile", mid_value=50, mid_color="FFEB84",
                                           end_type="max", end_color="F8696B"))

                wb.save(path)
                QtWidgets.QMessageBox.information(dlg, "Export complete",
                                                  f"Session exported successfully:\n{path}")
            except ImportError:
                QtWidgets.QMessageBox.critical(
                    dlg, "Excel library missing",
                    "Install openpyxl first:\n\npython -m pip install openpyxl")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(dlg, "Export failed", str(exc))

        export_btn.clicked.connect(export_excel)
        close_btn.clicked.connect(dlg.accept)
        # Maximised history window keeps the graph and Excel export button
        # inside the usable screen area, including on smaller laptop displays.
        dlg.showMaximized()
        dlg.exec()

    def reset_zoom(self):
        if self.an is not None:
            self.p_spd.setXRange(0, self.an["grid"][-1], padding=0)
        self.map.autoRange()

    def redraw(self):
        tel = self.tel
        if tel.lap is None:
            return
        ref = self.current_ref()
        if tel.laps and ref is not None and (tel.version, id(ref)) != self.an_key:
            self.an_key = (tel.version, id(ref))
            self.rebuild_analysis()
        b = tel.buf
        d = np.array(b["d"])
        for key, ch in (("spd", "spd"), ("thr", "thr"), ("brk", "brk")):
            self.c[key + "L"].setData(d, b[ch])
        self.c["gearL"].setData(*step_xy(d, b["gear"]))
        if self.an is None:                                   # first lap: draw the track as we go
            self.map_outline.setData(b["x"], b["z"])
            if len(d) > 2:
                self.p_spd.enableAutoRange(x=True, y=True)
                self.p_spd.setXRange(0, max(float(d.max()), 100), padding=0)
                self.p_spd.setYRange(0, 360, padding=0)
        elif len(d) > 2:                                      # live delta vs best lap (+ = faster)
            gb, tb = self.an["grid"], self.an["b"]["t"]
            ydel = np.interp(d, gb, tb) - (np.array(b["t"]) - b["t"][0])
            self.c["delL"].setData(d, ydel)
            lim = max(getattr(self, "dm", 0.3), 1.15 * float(np.abs(ydel).max()))
            self.set_delta_range(lim)
        self.map_dot.setData([tel.x], [tel.z])
        self.update_live()
        if self.cursor_x is not None:
            self.update_readout()
        self.live.setText(f"{self.driver.upper()}   LAP {tel.lap}   S{tel.sector + 1}")

    def run(self):
        self.pg.exec()
        self.stop.set()


def main():
    ap = argparse.ArgumentParser(description="F1 telemetry simulation and analysis dashboard")
    ap.add_argument("--rate", type=float, default=60, help="simulation samples per second")
    ap.add_argument("--seed", type=int, default=None, help="generator random seed")
    ap.add_argument("--track", default=None, help='track name or part of it, e.g. "monaco" or "red bull"')
    ap.add_argument("--driver", default=None, help='driver style, e.g. "late braker" (see the DRIVER dropdown)')
    ap.add_argument("--weather", choices=list(WEATHER_PRESETS), default="Dry",
                    help="initial simulated weather preset")
    ap.add_argument("--reverse", action="store_true", help="run the track in the opposite direction")
    ap.add_argument("--speedup", type=float, default=1.0, help=argparse.SUPPRESS)
    # Ignore IDE-specific launch arguments so the file runs with one click.
    args, _unknown_args = ap.parse_known_args()
    args.track = find_track(args.track) if args.track else DEFAULT_TRACK
    args.driver = find_driver(args.driver) if args.driver else DEFAULT_DRIVER
    Dashboard(args).run()


# =====================================================================
#  CIRCUIT DATA (embedded)
#  Circuit centre-lines (40 F1 circuits, with official lap lengths) come from the open
#  "f1-circuits" dataset by Tomislav Bacinger, MIT licence:
#  https://github.com/bacinger/f1-circuits   (gzip + base64 below, so no download is needed)
# =====================================================================
_CIRCUITS_B64 = (
    "H4sIAJ1X9GoC/4y9T69muZHm91USWqsu+DdIzk6SZzzdGDUaLcCA3ZjFVdXtqrRKmZqszMa0jPkwhldeeOWtd/piPmQ8T0S8"
    "77nwuDaF/N0gGQxGBPmew0P+8//2q48//Orf/er123d59fqrX//q0+uf3y7wm5//+Pbl64d/fP3ypw+/+/jl+28fv15//Pnz"
    "99fffv/28x8/f/vy6W2Tt08/fv3pV/+ulzF//avrr//8z7m1lyVT2q+/q+NltjXkP/8adEgFncup5AI6RneaKDtkOiXr6x3W"
    "3isdq5yELYfmrYISKi2mVA7Vlg4qq79HZ6iXFUjUK1MviSqYtlKG0b4yafbGulm2L6+3V7bWh9fbJjvcq9fbKltrK8hmtnbZ"
    "wWjtJht0qIU61Ok6lGW0eWulDtASxrGYeUvzektmDSUMRck0b8mBWgW5hsaMptDYqEZD19oCDbXWTovl6X2oZvMc+lvFZR1a"
    "W1lCBcMq6DVUQA1SDcYVGKyuMDwmW9cMrc1OOrzeRibBQWjaOtcKbsNaZ+iudGpwha8HbyqkNYR0M5q9hkEr1JlcrzHFqPds"
    "rPEOncmp22aWbK25FWY12eA2s1vfQqhd/Xynb6uYvmF8Vjd9QwivaZbs1uOR6KR1ijh164xAh1EP4ZEWdVjuuyNno963kenn"
    "dfXutJnsLIGal6VALbBTDvWaR9f3oNtm5GHUE85FbyE1smSLvqirWKjXe79acR+7bGCyUQPL5kVCvclaGyvY1mrwvD2SMLFc"
    "VQRKHS4cqKVHye/RMQOlbXsUHZaLgwqDinXxrqVpNNjRJoNgA0uYEirNlZYRzwDXiJvsDDUwg7TLSEYLXWYER6rJJsoVqE+U"
    "nlhGtTlmJleh9m40yJoNfPYcjdnmQbSV9ypobb5HzeAPlNF7NeadsFqD33abEGfxLvRqtPqIS2Nby6fJYel9Na9gWLUhPY9R"
    "xNZLXu1gQLcVbDvodD1VN9hgQPbkWWkyr/Uc/P76F2gZLVAh9cXZsCzaa6zXaY41FKNBlirUEvtLbWv3xmxS7yH92JLpgj5i"
    "Ysx9uXNsHyS7WFPTm+pmgzpc196c3of80jW6F01QfKK/BsxM6xOUu+1FS6CEId+3bKLB3i3Z2CTvw7VAMBpDz0f3HqV7pe40"
    "Gw1xXrzaYLDCifOiXkMZ71Lp79XAMH1srZqsL9Uv6p0Ik/fDr4r//N9+jd8vf/zpu3JNbv775bevP315/fjpw999+vr25dPr"
    "14+fP73+fPsl84fXP/308Uv8GdMug+jPmJ5eek5XyizycoXZ8RFlV3I/rE1D13x+kIzABGx60SvXgi1jy5iQ5cT6jp2UXRlG"
    "WTPEFqqXHJXMWr3mTDATK6awWANlLrJMdiVOZd1qmwV69DPpKqvQpAe5yqItyLFsbc4yWCneBlBQhCVJBtVt02y+k6iyZl0V"
    "jkNdZvNrVgTzskIr1WF6CHWrxfp1/cQAy84y5UxhNpqso31SKFkDvbTngv1a1TyhtgpLWpPXZEvmcvTM4t26zAMP7maSymHI"
    "zeqrJZOZiSu7delpjAOR3ZsqVc6hXbpEdUe8hhjDUx01jthyhmZbaILe30Zogh6wzARlmTdNd+xOr3O5MSm3nNWbJ5ZOj0oh"
    "njgc1o3SUDQvL0orJ08UBQ51qemxWNUEV/z15/qugQyqTLDhcvCCi1nXWHK4kRGyFxvPhr9acMMPyC33Wkit4oQtLPeUxqFI"
    "oSQDKrmJrxGlTVxuwgWSJ4+eMGRJrOw1+YJ5//fUDmb27HSB5DbZ6wq04ckimX5m45HZj2xlr/kU/V1elmbxnD0ahmK54Z15"
    "St2zPsy3PG9haJcHy0gsW6rPFWhXcvM5BepJ8jlF+p31wbJmvsSi4lMUVRGPZWPD02BCfF/dyN5EmC19cv7+056cm0/Of/jp"
    "9dOPP71+/O/NzhB7mJ+v3IDHjCW/XEus65dOzS9XttHnMgdeayhA/Uml8MrbgNnhNdHe4PXriFAcDrARBDsFZ4ANbDmq0Gek"
    "IFcgiMeLCjO6M0poOqPt8Z5gD4LozND16FMzIypEyRlgQ2/GCsWvgD1wptDOnHc4rpBV6CizmRUGx3ozR4DszoiQqsdhTGaM"
    "cm/7WuaG4jRldIK7za+fnxydoHoXSoYaxWDQ0uosEbI/NVhosj9tBTjf6eS1trkN2n54BmsGyWHjE5RHh2ZUk6MTxkcWx3H2"
    "4NSAoZX9fFtZCJOOMGkpeHUfYG61johoqXqN1xSrMAzEXvkoG176WugrLKkFKIBhdHrKhG60DhdspQTJ+o5gW2wnQOnQKITE"
    "XscpnMFCCZ3MEs2GhnI0OuzbUvR/Fk953oNCJx56K6J0hdY5uEHL/ezhNmSDrLstZ86EzqrcmflFLDzeaWUl1hjGcRXC6RGx"
    "Kt0l+MtVP/3FG1o0eppBcmFw07I690wDo6cAM0Y3u78VE3TXKPt5HmB1ONY7kou+4VFWrhmajS+H1rj38nJcuoHHmeWnK3yy"
    "w0mYA2TrKXnrhQGUQuuFrafsDRVh7EoJkJLjPehzXCmMi9hL+nVpwe4L5gzTwjUYkGwtmDMZ7GGA75KXKxDWIImcdwkGiHzd"
    "mnc9r04YMuZqjTD4HKbtCwaXNeVzgKlSJQ+DySGKyiOxX510jaZ1KKwZpkmGhc2kw7eQESYzZM8hBhHVrYdZaSGNX43Lveu9"
    "BSOJwVB8kvlY3kmyRuLotnJruaS+qGMoTm/v0bfMvm6MUhlBnsUv13snAjqXg3g0e6DYcsWjgilzhDDnRC6hQpuJ+4oQFfYZ"
    "IKfiHpoZ0t+BdbFO7+LgvCspSCaqFAJ6/zjCnO8GvsaeMFgjYwKZIcgX109DwiQQlta+mn/75bsrLrOv5rFs//DD24ffvn75"
    "/u3nazH/3e9ev77+/O3Tv73aat7+FpbzTXrHcr68XD9MLn9q+aWPdLp7oX79mDpI5lkQb3QFxiHDyZXJD5IT/BvtteNB/WTn"
    "ja5uKDrOesggOQnvoGtsFAlr3y9DgRpRY0GxgleyAKLugu5ciGrtR9lAVKJPSk0W7MMQlbj+I1pELd/QvVxaRBmoFepQ2eB+"
    "9a+oUK39aBmIylfTtLCu6ycSUKYO1ZRwdHnSMyr5Vlfqzy3uJzdQlXa+ooeIA1tMicYB2s9KgNhiqZQabPEKBhrH6soY7J7E"
    "UKcrWV0cRn2Tp4gO1/OtLrGCBUr0ZdVXSIlX36C9PtjVDmE4zJUaR3bamF0rUiBTqzF8aJkrLZMYEhZr5hAzP5db7bnuThkb"
    "Lw79tKDb21I0Wpc5KQdiTCtHGwwLMZcye3pdNvT7+SmSQX32+JGKmQqa9mG+vCbHRgxxuDpVtSG1iugKbjyGTq8+DIuoPQ2D"
    "PoFUQgXM5rReN2dvfT7XJAycblFv2cgyFgNCsqUnZk0xbxnsjNiwDw6DueIwZDKTYN5kbKj2rw8MKDXYv6gVWfxdv5jpCc3y"
    "dqa/GEr0qsSCsujW2dCcz76338kqMkvREZYl/M5RX54zgUayBjuy4Uhu0cTJylxPoPxI1RP+BDIjS61A3QasUYr+L/uB00HF"
    "EbLhKLSg7J8BQMsQVC3skEBR3fmgMqi8JkfQveZq7RUiqxzjOionX8nUvVYriNi9kFWfTWoQpUoET5LrJ/GzVBrUiyFw/dRm"
    "XUw81/LopmpKVDXRkxarL6vY+oLGsSlgtWcD9lVp5kmnX+xQMUddCKirxWprFXmyKpcvQ1+GHzLNNraiMUO0bOXYxW6aAtj0"
    "sjJNOsykHIvWb/az1Ytcfkdk9huUYuhfS+BB5ON6R7U9F8zeIvXKVr3phcw2mvlDqehiKybU2WCmB1bzLY6h1GqGYH+uH6ZE"
    "jJ7a6Vtcs53tf7CyIWFdzQhHp4wbys/FzEEuFYQxbWaIS15fZP/5+2uRXda7i+zfX2vo7z/7blz7JxbV+40bFtXjpZX9SqPV"
    "l+v3xVZDySBZRAXkvB5RQiExoT2wB51fCQftZelBi2jufl+o5ZM9DxokrGru9cVB2YRWf5ZaqRFVooyCybqzcgGajqBDMk1X"
    "Rl3J+rNgmQtRibXXZwedOf+gPVEf1EyJRqkzUx8E5VOxcp0NFhNiMQfzJjOoejHVR781N2iGkyi1LpqhmVS1Pk9DNIzZqtAK"
    "LmMGdSNQhWnWa/M2EtadZXWN9ewNa3HsEwrWlMxDJlFpT+5wZVXzLDFEk67+LJWqSdG7R39qsK5qVl6UckMIpKa7EbwBm0fV"
    "cQUoWwxALbxfOQhgDIumDNTpVzIRTcOGUBirg0KdsTrM1faza6BiiFXlcUdWV6tA1md47UVMiDL8d0bnZLFz+9E7SlGnRnM6"
    "2juYUZAes3cOKJqsvvRMRI8pNKcMtnh+uimi5iWzrkErlMSCnUOTOVpSWVeeLFiqIVZv+SkjZdW+HIH0cRNq87mqboOaaZtu"
    "3lfokPosTPtjyAxRWDA3QxiwbgNWGpRoTtDpZt5Q2iAqN5RcqhBZXcgO+0m+6VCJXC2QuZ51rz6IleUs25aG7tTqDkFk3m3z"
    "VU3WIDJwLeZJFXNFNZOyvWLZb+8hAaKRK1XQbX2KKGV+VGksi8LK1KAPwNW9qYE5GxNB8aFnlgmIXS71Nl6llefB0S8oHlH3"
    "sWdvnqOpiLvazXhm4P4cJTWba2NiupbmtGd2t3JEb7Qo4UKi2cSbudxoy6TM2c2PHQ1HlWE5nusSC5zMkRdXojE1WFgyG+qm"
    "OEWWZoyw3Cx3ZCmlzedUl7ulTatrMLnb5GjpyZEFznBn75xOPFI5BVjCqj5RcMCazyYs1/p6nnMaR3rYsHabq2wW4jxhyXZ/"
    "w4RJj4SGGeZWMli3rSoZpcMXqKzIJpjR0Zdpi8PB7unm7IMmCk6LpDFv8/WYnNWlGmrPE/3g2NiYDsbpslw4ODcus/pgQlnF"
    "EVcWlisG1pn6kv15mR62zrxevwPGvP8O+B8//vzz2y/f/U/7f5/evv3rW/w98PXL2+vP8TH7NY3hF8F3o770K4Vfbt1frmn7"
    "OLHC/fXWgek8iAbcVj4wO9su01/aWmfeI5yAOVS5jbihji1hA5zOBGiEZjLYmRzAiIycJyCHnRRLyHaLw/2+V2EOMFMyh5Yr"
    "6zwLTXZwvAMnNfLStVHxEuAkrA4bCzevsXWaQnwUOqucOcBF2HwYzhOuA3uEFVD6O3C0UCdsNNc7Dem2QMLK8Y6QnlHcCWTB"
    "2VIovrcGwa9cpdEoGYw0hG7ZvM79Bk6hBKij/ujAM0Eypx7gAgx6zjQVlp7fgaGhmaFncRvvbdFgATZCd/U5rHBQSE28odvo"
    "/NZV6MWXtTO9+KroTwnDRrPr0ygyWKOGKF1CySA4KlgPghMK1aDPhGCLzTCV9JA21oJk76GP7LhO36wTWkoLkgN9lJCzzu/g"
    "DWP4rgLfmmEgzyOBAyXUmSgpnhL2m3eFIagnG5oj+EEfhBLGnA3NULzAjeaKboTi68E1CYNrDk2ue2NngBUwh+E4c82BK4RQ"
    "0R7lEhKNw2AQK15CCJ2H+Rvq7wLAARbj1wRLjF+yoGVrgKGVShaiahQ2E62B2SvrrxlCZTpfM8lAsIYMycyTa3D38y7kwFCl"
    "DFrINd8vrGFKF+wIv6yPHACpZUpe/LwcPdESJqbKtNeCH1SmjhoCo3ZGaigtk0EZBMe4RWq16B01QLDgBOfXjua8AOOSwRco"
    "//LlWqDIOw8q//H1288f/unj969ffrClyX96+/C711++vl1rlq9xU+9s9tHNy1h7D+G1EroMqN9JXKjp76HS9XHXRqI/7Utb"
    "nVKiv2IupDurN9JfP1uKRB+7bFINsVy32usAEpOiCse3lRRq5UKdyFTolNJvUjYSom5qDQHSz3s20md1pWMf+4WwVt6IBYcu"
    "/Ddi9aPfhMSESCYrtw6OJc9azcxylZVP67TZYVrtyaTWhP0Wu7MKrTxZ/epEg9ZakwM2tOC8VggcC/0wZyPYNCI2iE7Pa8Jn"
    "uUlSWbl+krWR0B2WlZu0X6IK553aQfpZwIVKph0yC+KZ1EZsscCPun5jfZAJUYciHIvM3uAn0NbBap+mqRWctJ9VPmmGaVUt"
    "GuYZdK+almqOaOHaTCdaTz9IOJqzoH7OcuyiZEox4xE1Q+hem4X2zAjxNharyg0hPqwz+Il+oVENofoh7SbVvS5oOpp5Q6dU"
    "MyWEqLqqVKK4N1BVH3p4Q9MHqTr0JGZA0yHVp6GPiOM8ktl0sSAVrdaeqVCZ6obZtDKx4buNjRqlqhV0O1CHyqjQE1UUUQcx"
    "qfE8FnjksEfMEWtf5VlqzfWEekpPtXfdtqoqTKYeR+UxQe3uWBqzPrOcydT5mFq3+RiX5myVyXbZ2GDke04WXqjJE4MIiXkH"
    "LLxXZU/+uD/uYwZDV/SEkZhF904iy+2oqnR5muN6WTYTJhZcLNg5m9TEhN+Y6KrNe02IxBDtWW06abSV7v89CEm6t2xSiVJU"
    "q1IHXc0dhHwfEeevhjR6IbHaaZtaqNakISp1qJNTU02GQMqSZ6E8JmuHaebE6qJ3Nnj9dCai5Wdh7fjs90KZozg41c6E/NcH"
    "fORCk1KZdSUqgRltEwoVE6JvDa44ZmIEDCqa6NxjWnuVQoi44fPQWNQ9Yf3UZzJEO0yOxTVFE1GtvZoAqiZFoWYEmadPLi/G"
    "4hhOyURpEaE/l8qGWHBiwopS9Bqva9Jak/59jjgAMqmeqSr7Mxmc02rnQogDfREOBQcsIJm3clxmXV4DO+gJVgfZ8LiizFmj"
    "mwETC1qnV7mRKs8Nrv7c3urUqpsQI2w0H9UnzxprUKgWQ+vJSSN69o+RrT0ucGSuO/IG+1OAeV3i1TOPCZOWO6mIxQCJhU7m"
    "Ek4yYy7TMp2J8wpy+Id5w8QT+b1R11IBF/5drOBgvoDbzsrk0PnrYNZ2SzQMX/vNMhtVYDDNxkTajThij1m1bovdpNP3LLXO"
    "buXMCJ1GaJaw+qjPKbJPy8k0AtsrlBFGs/7cPSize2YDoaVas5o4+zf+upvCiGhmT8Z3s4wsNnc4Yvp1NGxmstpHpmXMxsOn"
    "BUpNrtQrU9G0WKq0zErMKEVsiom/af3X8+vXp1/P//T2w4fffvt5/3L+9KN/CPuXj2/7PL4fH57pZz9w72XI/tCtjfMhjR6t"
    "cbGzyXWcLzeM5Z3kD8NpUhfbnzKBTWOzgOEcps0ECMctvVwpZVv3NGvV9ZVVrmQc53Wx86jsMC879xPozYqXHVCv8KSVi+2f"
    "lwfhrKyNOoriG6DN+kKzDacbvZzdSeN8AcJ+nQM+DpPkLEE5yeJsgpXmZY152QYDSLXO7k8swIax2cm8Y0Jk1Z1dIcqs2WnV"
    "le5Dwa4tH9oBQ/VJVaQVdK2PZqwZMwRNujhhA90R9OjWBWkc1+6u0ypM3JO3SbO3aXL72CdlYorsvWvKuvVrH0yirFm/ism5"
    "e+bJNtye5wGxtmHDsxqdx31ssWg33znrDm3Wh8KbcNemJu48+5gauOxyhlZ5YuJhKFtlenWU6x5ltFRxrziP0zV6zCoJ2vE0"
    "oI16JuvPsV14wtDJC1C5ZDdotSacMfJKq+54rM/FaNDifnfetR02TZUzbx62ho83c0Cy+s5y/7BszUqnlbOVXYlyxfxnsRvV"
    "LXXStpY1XZAtS3W3XYsq06Aj0Z7DiKCv2Xx2WHzmVu5yNop7DzhYqK+S2VCMs5tZc6rZ5KyOTu5drjDLJkuCwvDBZ5232cLn"
    "ox//eM1HLbxu/sPHn//17csvXz9/ersfz+B/e3iYu+xdc35J59SOXl6unzfnGd9he0e6Mn0tfNjlf2CFbD97UTZN7rIx63NW"
    "Klh3lthuJbpMqMSAgOibuIOuVYYiI+yA6MsTZUD61ukg6qCJEExR615bRm14AXCxtL9yVZaqMYGc/hBXxq5X0yQwL9pRHd7x"
    "nWZpDd1QBAY5zQKHKcizW2UTw5KDHhUFs3U1XSsHsOwIo5emi1VUp7tw0VV0Qc/JgUkGWPU+NNaXfbioHvYnnJHXXsiyXuT9"
    "jFCZO0jB6Ehwrg71pCcbxH0sxmH6abCyAVbdqQWjKLo2PGy/vlLWzXeuOZ7MVN7fjoKZt+5fk2A2PPvhO9owXaawrHvxtPqW"
    "17dgAyu5UFlz0tlXr2vNRZtQ7vpJBDk9KUkZTdeyoQmT6JHEh+39GoeZlc4uTSBrYe/+AHO5anJW3X5aqqyIsUU5165mk6vG"
    "SiMbxjjaUp3BQYPKdWIQLSpKowFkmXb7qdhhw0KxNPrE8BauZADWAkMvRh/3+sR6W2lQ3WSlhmdiS838c0FuZA/3ZUE7n13g"
    "0tiLFiaGGtyJ1VV3xb7Iirs7WXFWWLZ49LCJ7BHF3BaKCqIi5J7cp6liZSunjmLumPcXGWAWjZVJpXRro2I0RvForNnkDGHA"
    "h66ENKkwb+leTsx27Fr3GbCTWc7LdO+hSwbMirSed4NGMVfJeXIcPc1kInHEOSUvT5adnfCx2Jse0Ynq7GaSy8ZgYSam2PJW"
    "G8R0NfeoSfXRZiaXNW/+pHvIkdogp8+S1D85qerjVGU2kWdPb0gWY3gq38totGvRnTgRYquJpjwuA6zd4nP+WsYqdO4xHBuZ"
    "Ifp7t5XGNQJk3mxu1oSn0A5HkeRMyKy7+5wmMM9vmTEgnqZthEQ8w6M6124/Q1bkfbU1VkgpVGR6gNqsP92z57I1W7mN9/RI"
    "4dQ4pk+rD2tKX73+8HatXms4BPQ/fv7+T2+ffnr7+Ocv8XGK4/g8Ze9o1XXrPF/XXgv/9VKLWuCg/fx1o3neOEz9Ul2RHsaj"
    "qAItI0IyifYvgQvVdGaygybIsMoXkZXrGUjPnDmoDqCThw8aFUhYu+xfgRvpWfnzfBQLHXS/7kZnO9RBkwXPsxxFhagMFmQX"
    "z/4zRSbVWVe36tlr/ahREVTVL+keURJDHdovR2ix2GCc584HDfZxkQhso7+jDuqGBF0sFcqPzC7qD+mDUC5PEox95VCMs+Pu"
    "EHZ5VNpdT9Xc6Dxs0gEz1FmwsCCtl6lSpw1s6Mf5NkgR+yLC9jjOY7jPsL0xGhE7M+kyRUgWUUWDM2XWLqh9VjppYZ9no6sJ"
    "Qck3mWLFiiGOTPOCHBk6zGwcB51gNyqDWtH59lYghE5lB81FdbPOQbkRmZUZOvS9/eDumViomgv1xnhuVhPrLk4sMwyrnGgY"
    "gmOXZU50Phk+yNpbyE6LfWGKKU//ztmKMFUlQ7MzoY2n/pc5ng1Q9FoDdU40P+lRFUFbxrCAEZQTBu15tquEAKXEKjrvZBRZ"
    "fC5DrCgN1jRppNStOdokwROZWsb5pvCAbOkAdQ9GnkzrnFiqzGYVkIHMUvQpxEHThDqRjJtUv1VFt/D2zlZIta+VK2ywWe2F"
    "A2OZ/zw9PoheJ/SEITetdGP9QYOIsXf1kCPK8JeJ3FkG4+UapE77mbGEJrU5anK4BtOpTPq0LEv87LV0QxxBsV6f18gHVZei"
    "GxWbkCjkkxbbyz61sXJGhJxXZhv1YYjzeXd3SJMom7VuZJD40JfnykeFYbrNrgPJsnSb6wbbc90flie+AvppX+M0JayAvn36"
    "8fXL54flz2+//fD6l7dfvj68TJp2rOp6Ka3rI8M+9SPkw2qpyvRh60bnY72Nip48dpg+mN9sOUNRPDRVxrJVnKFZPFvWNlif"
    "vsM5rJN1b1efQW/mZVe7sarvhDYrxhZZM/2a6VytH83aKMa6Pg/e/TX9zmvWw5LVNxrk0jKdB/uRptlgsA08wD1szhubt5Iz"
    "NzLTbpbOVrsx9gzHESpbZNaL2VAWj6GVCViZd1a9DfY267sTbQO9zT4aY9DyxeSGTFrerSfjNkJj0PJd7vXJdOtRbrjOEPMW"
    "Fu3eA6P/4GqWpRs5wczyHe9Y9taT7GySTWOCNvAiTxna0H0JYECtGErQWI/KAcNAjtVvcrg0SRlUntlZRhszqFJg5BlUaVA5"
    "NEuv1Q0zyjgW08anW6uOcrKiVl0ulKtmKLxk2to5Y1GzSbZukZRcaBErd7a4q+WIzi78Y+A1nKF+sejZt32QFWcc1yH3+ry6"
    "hh7gAp7DKsZGujdRWbQHOZZ1A9dKn6jOxqDLmgXqhFzPrsuqZC636MbJ6sMr781MrtHIusdYWTVmZTsTDd7vHkbfaZa2eu9w"
    "9+bD0aWSuRwdqrn9zjenh3Vvd0Dn1t5hNTDq5/5uutQS9IMuNQU5JpXlusDMxT0Dr/PPDxIyYQqdTphp3VlYMPeALIG+U9Sm"
    "pC7dEvK6NVpcD8vH2braForqnlt1C+TesQLDbIbzupfudQES98ZkzCzHCTjKFXjo0G+ulMGT8VIvRvfQ8+WRAjpY94xCsRkS"
    "D2pby9PxZNGQK+FjYy15TtsPcvDj4SGQKpGPfypswjuRKOaDnaxj3SeGQk3afM7Gw50uVbZQPQkKTSdup4XqQobOCx0LabbY"
    "kBUfnsThcU85X84elvKtvrHu7Q5PyZliI6hMMY/tTBsPz43na39lPoPcpNK8VcZ5dgxLKI9L2nBx0H5mWPq7By394S+v3/2H"
    "L6+fvv/85fufXv/8l1/ChqzXD7e/YC099gHU/JppyXaBnl4aDibu595JIMFW7I1AsONuk0E0iYTlsFd6f6a0KLWA9gVUqJ1S"
    "e0PfQV3Y4D5sS1FlwX36jKLEFgc1xa0r5wrHCYTNemucJ8Mbcb/bvvYQLbLTI1HTwgavX//Qgdvb9xV10IHffu0bBWGbPA1N"
    "2CZj3+v+pBsFcyMqtHxOw5AoSsuk9qvMg6z6UlB9Emq/PyHYqK5GqVZUrzpNidYEqLJgg+nrxPbAtU9KUzRo1LHP9DmoW10k"
    "pVJoP1I6KIsRtId7fA5CuVzZ3v7loYhuM/ZdGIqyoUSpRN33xpGD0hqGQKaRgarSqIagaOrLalepws+zImJ3zhv0Q5q3V4Aq"
    "FRVhOevO3hx00Bg033k5d5DQfmXWG5LMgjb4cPlig7OP1lCyqiHUPs1YBSYt3CkfCs6arXY0OJvV1dHr2c214KZlZXM22qEY"
    "ohmqk0b7ma9VluvUqo7MATP7mdMwi1z2w9gXMUUR+bW2/mTSWm2kCx232liwh7Wa1+ynBAc9h2Ft+SnIazN/z0KyLKugNe45"
    "3tloElk2SlYVc/DeUacIHwXty16h5TKCJIlXJAcldo+5TvZZdooqCw54ci2W4S3kEraR7/t0oVYSsRzMAOuWqZH1eWXSRhXK"
    "J0/euTyjbnU1mwiELTZ2qFcWFJOigzBO9tHU9Bn2ujH34IamMPNUfsN6IboRvym9pqwlz/3hLMabovaNzQzDyaATD6fJukqm"
    "1GDBs6tFEY2aB8kg6ayJlSdG4SoslixBYUf92sdbIGUVksFQZXN9wciFn7dtRMswX3SzQrHKhWmzWeWWSAfRZDlzyfMWVqOX"
    "hNGVm3WGgZKtXC4oVxhgYlFYbJz352yKOAkIh7nmZk60GOHmfRaZFgKSLManeTKlLHb2ceuKzGU467RhoZOhQ7dODwZFZ3qS"
    "AV+oYoM6Gb9i3rcf5yqyCJh0W9zJeBDGgh927TuuQazcvpxFZ/tineYkOpfF6uBiogwLctS1LK3Q7msVswxXL24/zALNhkKw"
    "BmnZQnxvllfUTUq4XOryvPxzqcHlUkDludxg5ba2HFwPZrNUYUU3nYpV9LBS9hX6x/2NRAlv9X/z7evnH758/vPnD//w+tdz"
    "c9g5E/Wvr/EIpL/Gawb2DyYsya+fAjPr+SCyb8642j5IT664cszJCAfpySRXJjr+pKgCnala0SKiVNEjYzYSojxvUnqyxQOq"
    "ckN6RJBgq7oiFpRqiNWfjP2IRjEluqKgFsCZkh50l+RCkPLmCswwmvcG5YZL6XkeFzo/BlUpFDQtO2SmKSmoyFWaJEKhmmD0"
    "yYrOZwOb6NvNg4SIxVpCT/STloOsXGe5NoiEvescPl29HiToyjJFJaMzi0N1zmFXQqEJR6uJBl4ZtdeUDdVKNIgaSDEwSDqR"
    "Hg9yoeaI7bGDKwuleiMahtheSSwoLFgapcSkFpUYGIq9f/AQ/QFySCdh/zqtENAcRNRKCPjvzKqtx1JYd15ElXUnqiSNFl4m"
    "1WnPaVLd+kKlxHzIC06hOw5DCKzZTa9Z6MYmRb+aLRsCMSvgeBmxNLR6YrAxkFdlrBUKVQoJHW0VjHLRM1MPYtxKZ58to+mr"
    "BnUPaKWT5kapojfdAn5ZrjL/Hwx4iy7JhVnV4mZBLd2kc9Bk7vWAY8wXahpQW1YXqy+O0B99G/eIkrXI1J4tLjuHNdsE0Af0"
    "yiNb1MM22fXqUCKbbTqit5grewbJNOp+GE9EkmAbPRT7JCNOL3rD2UEIlJJKs5Slls/6+PBkPz2/6xoBmxIqgi7r21VFHaha"
    "bkUU6MOCQ5qAmFoVDpFLMikkqOw2xYGC++x/m7z0GK2NbH7phoZNHZV12SRUSPrT/Bx0yIOaJhbM1h2rK1eianVxReDOlVm7"
    "sDtp0X42WyW4bra0rJ/2H2QGPJ/2H1SXIZhG7+RWhE7XVZ5R4zQwEw3YxJESm8FMpneXgaLiZsBUn2duz2bYCyNfgf3y476/"
    "NXwV9PvXLx8/vX747eu/ffjD1y9vb1/f+Tbo04+vf/n8JX4Z1PY3IthkkOrLlGuJ+ut8Zp/zlFbZQVP3QxEVZfrC/kGsObl8"
    "5iB9D6LsmvA301dSB+1fnweNEpgi8ZL7qqMbu4LlmV2xh+pck/14W1v1Puwt1Yfpw3JllSzIXavC0wt9KweW0bMamKK6bqgF"
    "lFGbfkmoDMTrT7M8s/NZ72Fi+u77DzA23VlC2WGt7u1SyqbL7UsvD1vexk6me/h1ugUrU10ircCKstycZbJQVkf7QW6fQq71"
    "TWdTwIozqNIDUlMt/doIDK6iq0xldalZxD3qYmoW/TwEbC6wGhjkuqu3jz9RFsqKvMNY1k21t5vedGlgM5RFsOg+eLCi/cCF"
    "t4ftLzEPay5XEAb6qxeso74R5ODy2LCgrKpD6rU9ZAg/d+9eoPOsoY2GyG1RTtsNnlYqURDLiLRgqv3D7Sma9yStgetFM4M+"
    "OO5eU6sXjMDGc5Du28kRpUFuMLDqs8orxVGDg5ccR0gdt9Qw4mq9VXoc8QYmz96iT7foaBNiwW+BqoRwQVh5Yun7hGhlITYW"
    "snnz3nYlwbs7+oqPlpUN9Gt5bV0T6z7AxxmCuXrRfSSBirmZBNOK/g5TNhtiXryJiSZCfK/83C2By67slts/OhSFiQDJLNcg"
    "RiNFtKCHTz+FrYY5j3K6XR9sgMWyGvB6SBuQxsSqPVRXVRW9+oZMe1HD7EjWgpgaqfUg1bQ2bOGAIqqc3kVIm4AFBJ9rsdE5"
    "UF2w06pgAaGoR87+qQbtfMBqAQuDWBv61cJsLkBhIUDtalgwTGgSWqXCeTyvKy5HDMuPsL7x1dSXb9dqKrfwjfXn73/6+IFP"
    "tXwJtfHjKZlcPtV1HkieI0Wavpw7SPTEnn1V7zLWwPRcPWV6nszFzsrrsK5nMV6sm5zeerDvislEVY9vuViyZnEE1X5dSrY/"
    "HwA7T7yVjUwmxvRYTVzBAwRyFtCK9NSzpi8WFDVr1MTElDMxnFO/LwKqxjL7NYlap5WmFa2TtXl1tVZayVhZtGaxrhaBNcWs"
    "3nG1RUv6nFlZtc56WYjp7bCKODh6kAlUgcp6N+ajes0coOOgtIslZwkW0PMpNuNYF7N6wylb+zJqk5oyyaYxjo5u4lTGzuo3"
    "hIeNVtmqlR2msRmg4Waci7Ugh/ra8PrQMb1eE4iGGsPYIJvZGMc2GDkXumdxtG6+mOsgM2/P7JlUr848ewXGIEten6Ab+kHj"
    "k1wxlgTqrdrvrAU5lNVHmspGJTP/SRP6LQ+qTLPoA8sHF83JdcYZlZu5T8EGWU8xVaYHV26WPdQWyw6PSGvX6hMOh17MqIxD"
    "rm/bNBEwDlZIIYv9bc2yVAKbwxmqsxYEZ4Ltq6+KMQz4NHeUwrq6kcm6LAlKZcLTT0KUMaj0F6gyqqtb5zT3Wtq2VN4s+Uxn"
    "vT0nLmljkFl1nTnKSvZGRZKV7JMKlzBZTHbM5IRZYOUw+XC8istxirJpRjD6SRfFaME80VhfZCMwOklxZM5pXRU4dtZb4w4b"
    "mKJy9llw6gmI7VrZOatwzuIz4+yDcma8Rcf2lDcwqWbPjAMnKjbdiQlEMas/TOS+TPhf/7KPBguvvf7w7a/f/vT64e8+fX37"
    "8un163nx9eGfXr//+OnHD7/7/O3LL34FiIo+rB0SrwS8Fqgvve1jS2t7mU1v1VS4H1wcqE9KFe4P3w9MuhZTWMF0KQ6GKlP2"
    "KvdWtw2rfvv6DGdop4GNUGWhoP5GgUIVMIfSQMmbvlL3YfreFWxkQnHY2ztw36euVbo++wCjW9P7h6TC6kruh2tqDGf7k29l"
    "uoBXOGhe8cZTp33Fdd9HcSq0cbgMWMi6Q+mAM0Cr0geiXvOgwpydFZTGzxGFdIK8qsNU6C5e5fV7GjAHSI1wQJTC1gitk/u+"
    "PEBv51q5kTWHje00133Qq/FrWGFm4+Ia7a03CmeoM6Gh+g7KUQ5t65eFgIwJ3eREiGbqCn1koODAMEBo2YKJrD/4kYL+EHYf"
    "XlloCEepKRxQCT/UFDZC10gySwc1+0I73bNB7Qig1oMX7ttp7pB14gODA/eWc0DvUGN/VhDsNEewZjNzBG+v1vPuI1St59V1"
    "3zvTkPMClHEfy/2rGtDrzKnRFbz4tcC4+8deWaF4YERu4f1eAAp52/tAMHhHaKbQbmHMGX447U9ZRvz01W6CEuJ0fwqrsAQl"
    "YTV8YwIGJfXQHUBhlWHM9pdMB47gMInBq59+so+oU08vJ8yEQRJTBQ41p0r1HVgIQ1Qlxv5wJyzLOjScdXaoOMOUdNlyOjQD"
    "+9y1nxsDDrP6fnjJMVsOO4u3ABn73TNPGZwt9OhqQAr6AJlXRzmxtlt32Ewh19Kj3LPjvl+d/XEtezNzZIcl07e8ocaQFM/i"
    "ex8KYHPdLcxFAmQ6EY/9c7qSjmTQszLQpntcuaYGwOzFK+IH35Mpoz2mzyv7iEBCL12Yth4gOxSLX30ndFYGG3fnKkwI0+fe"
    "c22ZdtLDah/ACjhLgGaOFiAa9/R4ztZSFlWnEw+fbM7Nm4iV2EmmiemSNdmotbs1I5t0hByGt9JhV/AOLmT0Ykc6l7zjsObv"
    "LXg25xpPHed+I8wrbrdpMKi5bF4JIbSW5eH1Tmb3GAgTQ5irMpdR1Q23t2OyuDObgELGLqyyhcmz2Lqh3Kev5ovPvVPznSmR"
    "GU4vAuGMuu7zSmOS8cVnbWIZzueQy02ZC8PMPW0GCtP+PXXU/boOpWdYNTAdDa+SS9cu7605xDXvpnkozB81wbMq55+ouNyS"
    "fRWbJMMwSH9nOhVO5RKWQLZOlLDaGT6rhBVlpwPH5SijJ6Tm/RkJ3T9AM0YJvyziTzz/Xfntl/34udw/ePr8Lx++/vT24Td/"
    "fvvy8ftX/9bpN99++frxU/wx2TM/cPpujRfRHxzpZd+ddY7fOnA/gjkw6aFXCncAHKiHyylMCzAbk0VBPakLkKw6m/0dwWGS"
    "I0BWmUM7O78plABZZ3HJfW2YdjJ7Q2eBfmD3hvbjS5gjMJae+R3B9U6VJei+Px5QqCfhQSOoqftYaEz0EveTwXAs3lewHJj4"
    "+OxvDRQOh52wVm/8TPwHdte900b6cQRggomrK7SPtgKbAbL0DHBScoXiYyjUn4mEYEHL/Q5fYXOFWicLzUBH3dgLlmHK5q5R"
    "qXgLVqvUEadmArLp6VVWepbOSYQYnR782py9hxHfW1oVhhGvAt176M/+cl1hj5ANReVpDhxEp7CyoRn6Xii5QusZ3ZQQf/sg"
    "LIXZlS8MSv2gEZDKSwi1Qj0fYHsPMlgkxNp+aI8ezQDJXE2q/sBSuZsjTRoueCYDLXY8mZKhj+fH24E1wMyOuy33E2owEyyL"
    "VpPijL6FA/sU0jkeYC+E2aE13Z0xF4nnjbI44tJCadZYQzNMRTiFVWHi4ATNb8O1t0zc4XDzeMvD7eNN20whnoLLPk30ZrW9"
    "w/hAH9nSWFo/UAdk1lg+tPvwF8ASinM+W7UGOAl9GBsHYgVT7sPzAUNDUgl7gOsdONhQ89aFeV23msHGQJ7+9ztzwBKiJKMZ"
    "PT/jKchGzGQcyhAmQncZIdvunV8Ka5hh2cdRw6zNZcQIiwNhgsDBq4BkjigWFhF9vJMyerNYDswyQX1HMIRopxfEvLpfnQO2"
    "+6zZw5TSudzoPVjI8lUPXawmGQxcKRnmV+HcFfUU+mCPNrJJYc27gSUFq49yz2PD4jksN/I9JKsV7mE5WO5hWodYmAa27swG"
    "MhbmIkB6DXDd0tgF5ZbGvDc9ZGXGTlthhckoazO/A+W+omoPq6zK1UaATE96iwQVgtkelER3WgrrPlt6hXl40F9w/y3W4JQU"
    "t8ZMtkjz4pPjWH3yqNPqDGMxu0m63SaHvIYAmlx1BgOvbLo7K2wmuOWqd2vsl24YCRPcr6W4/KkObaXTgiS9LQTffutOGIrP"
    "d9iyeDa2jzNCM+KQcq75frhMd/EaK6O5eZw0X4m6v7TKaUZ/+j9L+sI61FmDZKUP+tTVilk9pKKHX3L+m/HP//XpXeRvvn39"
    "2/+t3+D9x7cvf3799PmXD//0+Ycvf/u/fvz29lf/Du/tv378/vOH3338+m8Ph80l//24XvbehfbrvM6OCrXOhtveB3IVd8G5"
    "33cphJtuuN8uKewBXpGjsHnxnesUFoftcskDGfQb7vWzwhkhmaO9r/Aw/PRVWAnfYUHz/UxGoZ5vDUh9QuGpiBF/YIFcxbR8"
    "YEYPa7BapeAodyZB74wOVkzA2gwUx5HeCiu6XR56gzpx9rlad++L3DD08GzaXC9XuDcvLdo43gmTURATuMIGmGNpsOSa76fY"
    "B+o2CMJOmANElbgIBHAQOhsUDP0WHcULhsals/HYcRHCUHxYcTfb/spMYQ7F5wQs77AaYK8KV3BpwGskUjBm5/CkGCZwyxQb"
    "p6A3sx+JKAvjuK+8wIC73Qb9JQV3Gw3uloID70NODszBbvsNsMLsFh6DksFbx0RDOfjgnvEUNjfHYpXi1thniwJ644t2y8Hs"
    "i3bLwTuWIAD0BlfAMRgV5Z7HSgtwMX569jTI7FTEc2NO6GXNLrm/GUX0BjgIvXRbTActsEFodlvMBtNJYnUziCXmnNUCZAId"
    "rk3PwvwbJDOSW3cVO5MOH3QoZEqvw2G10kGjNtjOctgpmUNDNiOk0MvBWSaF1mcmDA1xyMKICW3UPXqWMC/37O3sn2o3c8gw"
    "3b2XYo3XwNq95/txKKa9KMn+9FAlw6eHEfcJchWHdBg+zTmwFs7ZsZdgcdRYOIz4IAqu0VhfcLeHhUE4s+vLvqUqvbsc+fvP"
    "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"
    "OSWZs9NBQw6VyrnMcoM0IjOx6VCzaQrSqmUG9KbbkK5p074FaiLyaapCdcsMubIqZgFBcsJPA0Uwu7AzjXOpPtxQIdjTksA+"
    "kABoeVUwlR7/cVAhYhJogzrocSIHMd2bo7VV0MFhai0aNCB0UH+tHMKpa5it1qIOzRGrqvmGsunA2Utf6uoKjCbt8v+FwjrN"
    "l4F/+fp0CULYdIoz+b7XM/l++PzhNz//+PrlX/1Avn/8/OXrxz//7f942G66N+Rju+k8HxnnX9fxsk+gOW8JN8tkTff7bHbO"
    "BtkMnzEp62AuRjC8sgWmr10P25/GHabvfQ/LKCuG0v607aBmRc83VsqmsZaf2b4H4Km6QkWw//Og/XHuYclLTiHzohO1Ydul"
    "VjfYCSu731LDSoYWlMOXWUfhRKYv2ZXRwMUNkFhdKIpW28qmCYYh2TiUvflNWXXG0UnW6rmM68aIqtuJyumljmCsrlvRQVX0"
    "VhWwSg8rztCL4vXtx1XKwpDRw8r03k7o0lK+jU+LY4tx1IkVjJ7dvKzQosF8iQ7q9Q0b7zxu/cAG+8MGBzdbfXsnjbLktqIn"
    "d+/HsP5Wb4N21p9Bh8lqz1G7F1uw/bI22DV8GnJQ4xANK9poUnz+chiK6p3Fijq9YJh2zRzDlNvPBJ8dyMRKQLBxcnvu7bNg"
    "pm9R0xVs1NU+EFVDKFmwu1V7X8HcmjIpl91KSy28k4ZZHemojODJOYONMDrQRV9EoL5K5t1AyivDo+Wce/Isx35gC9M891/B"
    "BNPb2G8VlQ2Xy4PM+nGOGlDmlkeWKtPbbUI5z6HnRILD6nweIVw4/MSSOwG7saqjQZO64y1axUp2ZO6iR6zDCaCJeDoTpLiC"
    "b0HUeqhPwugiJxXJIc4o53l/NCIfXDqa22lMa8GzRa43TWYdN433qhM98+xDH9Wz6DCzsLfme3pq44n45NNDnLuf3orEo/79"
    "rQiWDLy/8oe3nz/8/tuPbz/rFyn6jcr3r1/+8vbl4+uHtw9/eP304R8/vn35/LCn50og9rJkL9LOOcIL2+uU6TH8e+/xICuF"
    "bAoZDoTfu0ybMX3mmLCf/LCmT8/3bi1j59qyzXStrCwPZTWZLp1l9cmPMn1yerFi7e63eGAut25i51zqp+rOmeHarKknldoV"
    "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"
    "dUfDHJ1aiRmGVhB368xyxaQKdU+OaBmLrP09DNLEsgZvdqdjiZ7F9KBpM982qSyO4O7ZpaCDvgo8hDpkzwAZ5ZIFgCOxtJBh"
    "m+RpoaCulDxT0LmLKQ+f7PpDUVMMpbJVDzv3ZQPbp9VlqbcJkOfezKAzo+67+Z7isKIgHk4dZJbINH1hos1UvnCs3aaJqJhz"
    "JTp4seFP9JtidTEb6Tb3Q+jgxaIgYVj1q8FNzrftKsTe7M8jgKh6LoZYe+5scNLwD/Pbw4cleY33L1f54fOHf//L189fPv5s"
    "U2n4t51b3uzc8n3pQ90fJsyXsdfS5yj1i+09mod1cbY3yx6G+2c2K4OsGdt3yjzL7cu6tb5qbIHhs47NzkVfh4mxHZyHDSP7"
    "oqFDcnE2wUp1JmA4JP7cmgGNJ47bP61C46kb8JXRKFb0bMA/LChcKOeqSGbR7n2dbMHtuVDdHG6nRU2aN1vYs16doRe8Ou0w"
    "WqU6y2apddMFh0oeNjuZle00QZTjOI7ictasd62yumaW2ttA1SxmlTZoPDNAM7uLKZzhUL2s6kydolc3srMcmIAVa+Ncq3yY"
    "u8U+Ak2Za5wH5bw6jFmvPhb7wJEbq/m56Lnq+6mF8z2EaiLPoddxkrOGHrRLbryOIOgpuFQzOR8yKWTexihsw8tO1JdzcFHK"
    "lZsr93sQ9FrmnQWxSqOEWKHdPeD314bKQmAMWm9Otwo0bslSw35tpcw1rkvIrI2KoOrN47umOyu0QPMgKNSlFfPRcwfpYdNH"
    "vKJv3UeyIIl2fIt0nGqiPl7hcfyRch7MGQEe5RKSaKxvH7GFds3OqdFW0/qRqLKYynt/hzKz1H7Aht66FAfS3THRHXlr1PPM"
    "8nQfxxj/Pz6N/Ie3n3/5vL9k+C/f3vzR+d+/fv/65e0vrz9++9v/Hl8U749ieR9HPZPTPl21vKyBQTm0nDNTlQ6n50TbQ5fD"
    "c3DshhhBpZnV4oM/pWTNWT7nUx+KU6+f6LqX500S2tIgxeHOqhXL40JWdGCS4j6AQ0c22ozWtYx6v/QaB6XeWvPWAmwGVw2U"
    "Koz8Dpy9BMr+zoe2aNk539NrBntRASxatF+zGHV7nxsNFLppa6vviZrBZ+hXNiPqw1zQbLS57D4NBTRCqjDKOxAXcSqdyxzJ"
    "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"
)


def _load_circuits():
    import base64, gzip, json
    return json.loads(gzip.decompress(base64.b64decode("".join(_CIRCUITS_B64))).decode("utf-8"))


register_circuits()


if __name__ == "__main__":
    main()