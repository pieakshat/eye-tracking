"""
gaze_classifier.py — Feature-based dyslexia risk scorer

Computes five clinically-grounded reading metrics from the raw gaze CSV
and produces a weighted risk score in [0, 1].

No model weights needed — thresholds are sourced from reading research
literature and work directly on the (frame, x, y, fixation, saccade) data
recorded by the tracker.
"""

import csv
import numpy as np

# ─── Thresholds ───────────────────────────────────────────────────────────────
# Each entry: (typical_boundary, dyslexic_boundary)
# Risk score ramps linearly 0→1 between the two boundaries.
# Values beyond the dyslexic boundary are clamped to 1.
#
# Sources:
#   Rayner (1998), Hutzler & Wimmer (2004), Rello & Baeza-Yates (2013)

THRESHOLDS = {
    #                                typical   dyslexic
    "regression_rate":            (  0.15,     0.38  ),  # fraction; ↑ = more dyslexic
    "mean_fixation_duration_ms":  (  210,      330   ),  # ms;       ↑ = more dyslexic
    "fixation_per_min":           (  190,      300   ),  # count;    ↑ = more dyslexic
    "saccade_to_fix_ratio":       (  0.85,     1.60  ),  # ratio;    ↑ = more dyslexic
    "ltr_score_norm":             (  0.04,    -0.04  ),  # frac sw;  ↓ = more dyslexic
}

WEIGHTS = {
    "regression_rate":            0.35,  # direction-based → robust to position drift
    "mean_fixation_duration_ms":  0.25,  # purely temporal → not affected by drift
    "fixation_per_min":           0.15,  # count-based
    "saccade_to_fix_ratio":       0.10,  # relative count
    "ltr_score_norm":             0.15,  # mean position → more stable than std
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1"

# ─── Detection parameters ─────────────────────────────────────────────────────

FPS = 30            # assumed webcam frame rate

MIN_GAZE_FRAMES = 90          # ~3 s minimum — shorter sessions are inconclusive

MIN_FIX_FRAMES  = 3           # ~100 ms  — shorter runs are noise
MIN_SAC_FRAMES  = 2           # ~67 ms

# Regression = leftward saccade within this fraction-of-screen-width range.
# Below REG_MIN_FRAC  → positional noise, ignore.
# Above REG_MAX_FRAC  → line-return sweep, not a regression.
REG_MIN_FRAC = 0.015
REG_MAX_FRAC = 0.35


# ─── Feature computation ──────────────────────────────────────────────────────

def _parse_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "x":        float(r["x"]),
                "y":        float(r["y"]),
                "fixation": r["fixation"] == "True",
                "saccade":  r["saccade"]  == "True",
            })
    return rows


def _group_events(boolean_array, min_length):
    """Return list of (start, end) index pairs for runs of True >= min_length."""
    events = []
    in_run = False
    start = 0
    for i, v in enumerate(boolean_array):
        if v and not in_run:
            in_run = True; start = i
        elif not v and in_run:
            in_run = False
            if i - start >= min_length:
                events.append((start, i))
    if in_run and len(boolean_array) - start >= min_length:
        events.append((start, len(boolean_array)))
    return events


def compute_features(csv_path, screen_w=1920, screen_h=1080):
    """
    Parse the gaze CSV and return a dict of raw feature values plus metadata.
    Returns None if the session is too short to score reliably.
    """
    rows = _parse_csv(csv_path)
    n = len(rows)

    if n < MIN_GAZE_FRAMES:
        return None

    xs  = np.array([r["x"]        for r in rows], dtype=float)
    ys  = np.array([r["y"]        for r in rows], dtype=float)
    fix = np.array([r["fixation"] for r in rows], dtype=bool)
    sac = np.array([r["saccade"]  for r in rows], dtype=bool)

    duration_s   = n / FPS
    duration_min = duration_s / 60.0

    # ── fixation events ───────────────────────────────────────────────────────
    fix_events = _group_events(fix, MIN_FIX_FRAMES)

    fixation_durations_ms = [
        (end - start) / FPS * 1000
        for start, end in fix_events
    ]
    mean_fixation_duration_ms = float(np.mean(fixation_durations_ms)) if fixation_durations_ms else 0.0
    fixation_per_min          = len(fix_events) / max(duration_min, 1 / 60)

    # ── saccade events ────────────────────────────────────────────────────────
    sac_events = _group_events(sac, MIN_SAC_FRAMES)

    reg_min_px = REG_MIN_FRAC * screen_w   # e.g. ~29 px at 1920
    reg_max_px = REG_MAX_FRAC * screen_w   # e.g. ~672 px at 1920

    regression_count  = 0
    forward_amps      = []
    backward_amps     = []

    for start, end in sac_events:
        dx = float(xs[end - 1] - xs[start])   # net horizontal displacement
        abs_dx = abs(dx)

        if dx > reg_min_px:
            # forward saccade
            forward_amps.append(abs_dx)
        elif dx < -reg_min_px and abs_dx < reg_max_px:
            # regression (small-to-medium leftward move, not a line-return sweep)
            regression_count += 1
            backward_amps.append(abs_dx)
        # else: noise or line-return — ignored

    total_classifiable = len(forward_amps) + len(backward_amps)
    regression_rate = (
        regression_count / total_classifiable
        if total_classifiable > 0 else 0.0
    )

    saccade_to_fix_ratio = (
        len(sac_events) / len(fix_events)
        if fix_events else 0.0
    )

    # ── left-to-right progression (normalised by screen width) ───────────────
    mid = n // 2
    ltr_score_norm = (
        float(np.mean(xs[mid:]) - np.mean(xs[:mid])) / screen_w
        if mid > 10 else 0.0
    )

    return {
        # scored features
        "regression_rate":            round(regression_rate,           4),
        "mean_fixation_duration_ms":  round(mean_fixation_duration_ms, 2),
        "fixation_per_min":           round(fixation_per_min,          2),
        "saccade_to_fix_ratio":       round(saccade_to_fix_ratio,      4),
        "ltr_score_norm":             round(ltr_score_norm,            4),
        # extra context (not scored)
        "fixation_count":             len(fix_events),
        "saccade_count":              len(sac_events),
        "regression_count":           regression_count,
        "mean_forward_amp_px":        round(float(np.mean(forward_amps)),  1) if forward_amps  else 0.0,
        "mean_backward_amp_px":       round(float(np.mean(backward_amps)), 1) if backward_amps else 0.0,
        "duration_s":                 round(duration_s, 1),
        "n_frames":                   n,
    }


# ─── Scoring ──────────────────────────────────────────────────────────────────

def _feature_score(value, typical, dyslexic):
    """
    Map a single feature value to a risk score in [0, 1].
    0 = clearly typical, 1 = clearly dyslexic.
    Handles both orientations (higher = more dyslexic, lower = more dyslexic).
    """
    span = dyslexic - typical
    if abs(span) < 1e-9:
        return 0.0
    raw = (value - typical) / span
    return float(np.clip(raw, 0.0, 1.0))


def score_features(features):
    """
    Return (risk_score, per_feature_scores).
    risk_score is in [0, 1]; per_feature_scores is a dict of the same.
    """
    per_feature = {}
    for feat, (typ, dys) in THRESHOLDS.items():
        per_feature[feat] = _feature_score(features[feat], typ, dys)

    risk_score = sum(WEIGHTS[f] * per_feature[f] for f in WEIGHTS)
    return round(risk_score, 4), per_feature


# ─── Classification ───────────────────────────────────────────────────────────

# Thresholds for labelling
DYSLEXIC_THRESHOLD    = 0.55   # above → Dyslexic
NONDYSLEXIC_THRESHOLD = 0.40   # below → Non-Dyslexic
# gap between them → Borderline


def _confidence(risk_score, label):
    """
    Translate risk_score → confidence percentage for the given label.
    Uses distance from the nearest decision boundary, scaled to 50–99%.
    """
    if label == "Dyslexic":
        # distance above DYSLEXIC_THRESHOLD, mapped to [50, 99]
        d = (risk_score - DYSLEXIC_THRESHOLD) / (1.0 - DYSLEXIC_THRESHOLD)
    elif label == "Non-Dyslexic":
        # distance below NONDYSLEXIC_THRESHOLD, mapped to [50, 99]
        d = (NONDYSLEXIC_THRESHOLD - risk_score) / NONDYSLEXIC_THRESHOLD
    else:
        # Borderline: low confidence by definition
        d = 0.0
    return round(50.0 + np.clip(d, 0.0, 1.0) * 49.0, 1)


def _interpret(features, per_feature_scores, label):
    """Build a human-readable summary of the most influential features."""
    lines = []

    r = features["regression_rate"]
    if r > 0.30:
        lines.append(f"High regression rate ({r:.0%}) — frequent backward re-reads.")
    elif r < 0.12:
        lines.append(f"Low regression rate ({r:.0%}) — smooth forward progression.")

    fd = features["mean_fixation_duration_ms"]
    if fd > 280:
        lines.append(f"Long fixations ({fd:.0f} ms avg) — extended word processing time.")
    elif fd < 180:
        lines.append(f"Short fixations ({fd:.0f} ms avg) — rapid word recognition.")

    fpm = features["fixation_per_min"]
    if fpm > 260:
        lines.append(f"Many fixations ({fpm:.0f}/min) — high re-fixation rate.")

    ltr = features["ltr_score_norm"]
    if ltr < 0:
        lines.append(f"Negative L→R score ({ltr:+.3f}) — gaze drifted leftward overall.")
    elif ltr > 0.06:
        lines.append(f"Positive L→R score ({ltr:+.3f}) — consistent forward reading direction.")

    if not lines:
        lines.append("Features are within typical range.")

    return " ".join(lines)


def classify(csv_path, screen_w=1920, screen_h=1080):
    """
    Full pipeline: parse CSV → compute features → score → label.

    Returns a dict with:
        label       — "Dyslexic" | "Non-Dyslexic" | "Borderline" | "Inconclusive"
        risk_score  — float [0, 1]
        confidence  — float [0, 100]
        features    — per-feature values, individual scores, and weights
        meta        — session metadata
        summary     — plain-English explanation
    """
    features = compute_features(csv_path, screen_w, screen_h)

    if features is None:
        return {
            "label":      "Inconclusive",
            "risk_score": None,
            "confidence": 0.0,
            "features":   {},
            "meta":       {},
            "summary":    "Session too short — need at least 3 seconds of gaze data.",
        }

    risk_score, per_feature_scores = score_features(features)

    if risk_score >= DYSLEXIC_THRESHOLD:
        label = "Dyslexic"
    elif risk_score <= NONDYSLEXIC_THRESHOLD:
        label = "Non-Dyslexic"
    else:
        label = "Borderline"

    feature_breakdown = {
        feat: {
            "value":  features[feat],
            "score":  per_feature_scores[feat],
            "weight": WEIGHTS[feat],
        }
        for feat in WEIGHTS
    }

    meta = {
        "duration_s":         features["duration_s"],
        "n_frames":           features["n_frames"],
        "fixation_count":     features["fixation_count"],
        "saccade_count":      features["saccade_count"],
        "regression_count":   features["regression_count"],
        "mean_forward_amp_px":  features["mean_forward_amp_px"],
        "mean_backward_amp_px": features["mean_backward_amp_px"],
    }

    return {
        "label":      label,
        "risk_score": risk_score,
        "confidence": _confidence(risk_score, label),
        "features":   feature_breakdown,
        "meta":       meta,
        "summary":    _interpret(features, per_feature_scores, label),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python gaze_classifier.py <csv_path> [screen_w] [screen_h]")
        sys.exit(1)

    csv_path = sys.argv[1]
    sw = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
    sh = int(sys.argv[3]) if len(sys.argv) > 3 else 1080

    result = classify(csv_path, sw, sh)
    print(json.dumps(result, indent=2))
