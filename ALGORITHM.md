# Dyslexia Screening Algorithm

Feature-based risk scorer implemented in `gaze_classifier.py`.  
Replaces the ResNet18 heatmap classifier as the default method (`--algo` flag).

---

## Overview

Instead of classifying an image of a heatmap, this algorithm extracts five
interpretable reading metrics directly from the raw gaze time-series recorded
during the reading task. Each metric is scored against thresholds sourced from
reading research literature, and a weighted sum produces a final **risk score**
between 0 and 1.

```
CSV (frame, x, y, fixation, saccade)
        │
        ▼
  Feature Extraction
        │
        ▼
  Per-Feature Scoring  (0 = typical, 1 = dyslexic)
        │
        ▼
  Weighted Sum  →  risk_score ∈ [0, 1]
        │
        ▼
  Label:  ≥ 0.55 → Dyslexic
          ≤ 0.40 → Non-Dyslexic
          between → Borderline
```

---

## Input Data

The webcam tracker records at ~30 fps and writes one row per frame to
`data/predict_gaze.csv`:

| Column | Type | Description |
|---|---|---|
| `frame` | int | Frame number |
| `x` | float | Gaze X position in screen pixels |
| `y` | float | Gaze Y position in screen pixels |
| `fixation` | bool | True when gaze is stable (std < 40 px over last 6 frames) |
| `saccade` | bool | True when gaze is moving fast (displacement > 120 px) |

A minimum of 90 frames (~3 seconds) is required; shorter sessions return
**Inconclusive**.

---

## Feature Extraction

### Step 1 — Group raw frames into events

Consecutive `fixation = True` frames are grouped into **fixation events**.  
Consecutive `saccade = True` frames are grouped into **saccade events**.  
Events shorter than a minimum length are discarded as noise:

| Event | Min length | Reason |
|---|---|---|
| Fixation | 3 frames (~100 ms) | Below this is a detection artifact |
| Saccade | 2 frames (~67 ms) | Single-frame spikes are noise |

### Step 2 — Classify saccades by direction

For each saccade event the net horizontal displacement `dx = x_end − x_start`
determines its type:

| Condition | Classification |
|---|---|
| `dx > +1.5% screen_w` | Forward saccade (reading direction) |
| `−1.5% screen_w > dx > −35% screen_w` | **Regression** (re-reading) |
| `dx < −35% screen_w` | Line-return sweep — excluded from regression count |
| `|dx| < 1.5% screen_w` | Noise — ignored |

The lower bound (1.5%) filters out positional noise.  
The upper bound (35%) excludes normal end-of-line return sweeps, which are
large leftward jumps that are not a dyslexic marker.

---

## The Five Features

### 1. Regression Rate (weight 0.35)

```
regression_rate = regressions / (forward_saccades + regressions)
```

The fraction of meaningful saccades that are backward (re-reading a word or
phrase). This is the **strongest single marker** of dyslexic reading — dyslexic
readers frequently re-read words because they fail to decode them on the first
pass.

| Range | Interpretation |
|---|---|
| < 15% | Typical |
| 15–38% | Transitional |
| > 38% | Dyslexic |

Robust to position drift because it uses direction, not absolute position.

---

### 2. Mean Fixation Duration (weight 0.25)

```
mean_fixation_duration_ms = mean(fixation_event_lengths) × (1000 / FPS)
```

The average time the eye rests on a single point. Dyslexic readers fixate
longer because word decoding takes more effort — they stare at a word while
their brain works harder to recognise it.

| Range | Interpretation |
|---|---|
| < 210 ms | Typical |
| 210–330 ms | Transitional |
| > 330 ms | Dyslexic |

Purely temporal — not affected by positional tracking error.

---

### 3. Fixation Rate (weight 0.15)

```
fixation_per_min = fixation_event_count / duration_minutes
```

How many times per minute the eye stops. Dyslexic readers make more fixations
because they re-read words and struggle to chunk multiple words into a single
fixation the way fluent readers do.

| Range | Interpretation |
|---|---|
| < 190 / min | Typical |
| 190–300 / min | Transitional |
| > 300 / min | Dyslexic |

---

### 4. Saccade-to-Fixation Ratio (weight 0.10)

```
saccade_to_fix_ratio = saccade_event_count / fixation_event_count
```

The balance between moving and looking. A high ratio means the eye is jumping
around more than it is settling — associated with erratic, unsystematic scanning
rather than a clean left-to-right sweep through the text.

| Range | Interpretation |
|---|---|
| < 0.85 | Typical |
| 0.85–1.60 | Transitional |
| > 1.60 | Dyslexic |

---

### 5. Left-to-Right Progression Score (weight 0.15)

```
ltr_score_norm = (mean_x[second_half] − mean_x[first_half]) / screen_width
```

Splits the session in two and measures whether the average gaze position moved
rightward over time. A positive value means the reader progressed through the
text. A near-zero or negative value means the gaze stayed in the same horizontal
position or drifted back — indicating heavy re-reading or an inability to
advance through the passage.

| Range | Interpretation |
|---|---|
| > 0.04 | Typical (clear forward movement) |
| −0.04 to 0.04 | Transitional |
| < −0.04 | Dyslexic (no net forward progress) |

---

## Scoring

Each feature is mapped to a score in **[0, 1]** using linear interpolation
between its typical and dyslexic thresholds:

```
score = clip((value − typical_boundary) / (dyslexic_boundary − typical_boundary), 0, 1)
```

For features where a **lower** value is the dyslexic indicator (ltr_score_norm),
the formula direction is reversed automatically.

The five scores are combined into a single **risk score**:

```
risk_score = 0.35 × regression_rate_score
           + 0.25 × fixation_duration_score
           + 0.15 × fixation_rate_score
           + 0.10 × saccade_ratio_score
           + 0.15 × ltr_score
```

---

## Classification

| Risk Score | Label |
|---|---|
| ≥ 0.55 | **Dyslexic** |
| 0.41 – 0.54 | **Borderline** |
| ≤ 0.40 | **Non-Dyslexic** |

**Confidence** is computed as the distance from the nearest decision boundary,
scaled to 50–99%. A score of exactly 0.55 (just crossed the threshold) gives
50% confidence; a score of 1.0 gives 99% confidence.

---

## Why Not ResNet?

The ResNet18 approach classifies a screenshot of the gaze heatmap. Problems:

1. It was trained on heatmaps that look different from ours (different rendering,
   colormap, text layout) — so it generalises poorly.
2. It has no concept of *time* — fixation sequences, regression direction, and
   reading speed are all invisible to it.
3. It collapses spatial information to a 224×224 thumbnail, losing most detail.
4. It requires a large labelled training set to retrain properly.

The feature-based algorithm uses the same signals a human clinician would look
at (regression frequency, fixation duration, reading progression) and requires
no training data.

---

## Limitations

- **Tracking accuracy**: the gaze tracker has ~5–10% vertical offset. Features
  that rely on absolute Y position (e.g. line-by-line analysis) are not used.
  All five features are either direction-based or temporal, making them
  significantly more robust to drift than position-based metrics.

- **Session length**: short sessions produce noisy estimates. 60+ seconds of
  active reading gives reliable feature values.

- **Calibration quality**: a poor calibration shifts gaze systematically.
  The drift-correction step reduces this but does not eliminate it entirely.

- **This is a screening tool, not a diagnosis.** A positive result should be
  followed up with a formal assessment by a qualified specialist.

---

## References

- Rayner, K. (1998). Eye movements in reading and information processing.
  *Psychological Bulletin, 124*(3), 372–422.
- Hutzler, F., & Wimmer, H. (2004). Eye movements of dyslexic children when
  reading in a regular orthography. *Brain and Language, 89*(1), 235–242.
- Rello, L., & Baeza-Yates, R. (2013). Good fonts for dyslexia. *ASSETS '13.*
