import os, csv, time, threading, traceback
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from flask import Flask, Response, jsonify, send_file, render_template_string, request
from gaze_tracker_mp import MPGazeTracker
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

SCREEN_W, SCREEN_H = 1920, 1080
CSV_PATH     = "data/predict_gaze.csv"
HEATMAP_PATH = "outputs/predict_heatmap.png"
OVERLAY_PATH = "outputs/predict_overlay.png"
MODEL_PATH   = "models/best_model.pt"

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def load_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print("ResNet18 loaded from", MODEL_PATH)
    else:
        print("WARNING: model not found at", MODEL_PATH)
    model.eval()
    return model

resnet_model = load_resnet18()
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_heatmap(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        t   = resnet_transform(img).unsqueeze(0)
        with torch.no_grad():
            out   = resnet_model(t)
            probs = torch.softmax(out, dim=1)[0]
            pred  = torch.argmax(probs).item()
            conf  = float(probs[pred]) * 100
        return ("Dyslexic" if pred == 1 else "Non-Dyslexic"), round(conf, 1)
    except Exception as e:
        print(f"Classification error: {e}")
        return "Unknown", 0.0

state = {
    "running": False, "gaze_points": [], "frame_count": 0,
    "fixation": False, "saccade": False, "current_xy": (0, 0),
    "heatmap_ready": False, "status": "idle", "child_name": "Child",
    "classification": None, "confidence": 0.0,
    "calib_point_idx": -1, "calib_total": 9,
    "calib_done": False, "calib_collecting": False,
    "drift_done": False, "drift_collecting": False, "drift_target": None,
}
lock        = threading.Lock()
frame_buf   = {"jpg": None}
stop_event  = threading.Event()
tracker_ref = {"tracker": None}

def run_tracker():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with lock: state["status"] = "error"
        return

    with lock:
        sw = state.get("screen_w", SCREEN_W)
        sh = state.get("screen_h", SCREEN_H)
    tracker = MPGazeTracker(screen_w=sw, screen_h=sh,
                            smoothing=6, fixation_threshold=40, fixation_frames=6)
    tracker_ref["tracker"] = tracker

    with lock:
        state.update(running=True, heatmap_ready=False, frame_count=0,
                     gaze_points=[], status="calibrating",
                     calib_point_idx=-1, calib_done=False, calib_collecting=False,
                     classification=None, confidence=0.0,
                     drift_done=False, drift_collecting=False, drift_target=None)

    # Auto-start calibration immediately
    tracker.start_calibration()
    print("Calibration auto-started")

    csvfile = open(CSV_PATH, "w", newline="")
    writer  = csv.writer(csvfile)
    writer.writerow(["frame", "x", "y", "fixation", "saccade"])

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        with lock:
            state["frame_count"] += 1
            fc = state["frame_count"]

        gx, gy, fix, sac = tracker.process(frame)

        with lock:
            state["calib_point_idx"]  = tracker.calib_point_idx
            state["calib_collecting"] = tracker.calib_collecting
            state["calib_done"]       = tracker.calib_done
            if tracker.calib_done:
                state["drift_done"]       = tracker._drift_done
                state["drift_collecting"] = tracker._drift_collecting
                if tracker._drift_target:
                    state["drift_target"] = list(tracker._drift_target)
                state["status"] = "recording" if tracker._drift_done else "drift"
            else:
                state["status"] = "calibrating"

        if gx is not None and tracker.calib_done:
            with lock:
                state.update(fixation=fix, saccade=sac, current_xy=(gx, gy))
                state["gaze_points"].append((gx, gy))
            writer.writerow([fc, gx, gy, fix, sac])

        frame = tracker.draw_debug(frame, gx, gy, fix, sac)
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_buf["jpg"] = jpg.tobytes()

    cap.release(); tracker.release(); csvfile.close()
    with lock:
        state["running"] = False
        state["status"]  = "analyzing"
        n = len(state["gaze_points"])
    print(f"Stopped. {n} gaze points collected.")
    gen_heatmap()

def gen_heatmap():
    try:
        with lock:
            pts = list(state["gaze_points"])
            scr_w = state.get("screen_w", SCREEN_W)
            scr_h = state.get("screen_h", SCREEN_H)
        if len(pts) < 5:
            with lock: state["status"] = "done"
            return
        W, H = 1280, 720
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)

        # ── DIAGNOSTICS ──────────────────────────────────────────
        print(f"=== HEATMAP DIAGNOSTICS ===")
        print(f"  Screen from state : {scr_w} x {scr_h}")
        print(f"  Gaze points       : {len(pts)}")
        print(f"  X  min/mean/max   : {xs.min():.0f} / {xs.mean():.0f} / {xs.max():.0f}")
        print(f"  Y  min/mean/max   : {ys.min():.0f} / {ys.mean():.0f} / {ys.max():.0f}")
        print(f"  Pts clipped to 0  : x={int((xs<=0).sum())}  y={int((ys<=0).sum())}")
        print(f"  First 8 pts       : {pts[:8]}")
        print(f"===========================")
        # ─────────────────────────────────────────────────────────

        xn = (xs / scr_w * W).astype(int).clip(0, W-1)
        yn = (ys / scr_h * H).astype(int).clip(0, H-1)
        hm = np.zeros((H, W))
        for x, y in zip(xn, yn): hm[y, x] += 1
        hm = gaussian_filter(hm, sigma=18)

        fig, ax = plt.subplots(figsize=(16, 9), facecolor="#fffef9")
        ax.set_facecolor("#fffef9"); ax.set_xlim(0, W); ax.set_ylim(H, 0)

        lines = [
            "When Mary Lennox was sent to Misselthwaite Manor to live with her uncle,",
            "everybody said she was the most disagreeable-looking child ever seen.",
            "It was true, too. She had a little thin face and a little thin body,",
            "thin light hair and a sour expression.", "",
            "Her father had held a position under the English Government and had always",
            "been busy and ill himself, and her mother had been a great beauty who cared",
            "only to go to parties and amuse herself with gay people.",
            "She had not wanted a little girl at all.", "",
            "When Mary was born she handed her over to the care of an Ayah, who was",
            "made to understand that if she wished to please the Mem Sahib she must",
            "keep the child out of sight as much as possible.",
        ]
        ax.text(W//2, 55, "The Secret Garden", fontsize=22, fontweight="bold",
                color="#1a1612", ha="center", va="center", fontfamily="serif")
        ax.text(W//2, 90, "— Frances Hodgson Burnett", fontsize=12, color="#8a7f74",
                ha="center", va="center", fontstyle="italic", fontfamily="serif")
        ax.plot([W//2-40, W//2+40], [108, 108], color="#c4432a", linewidth=2)
        for i, line in enumerate(lines):
            if line:
                ax.text(120, 145+i*38, line, fontsize=11, color="#3d3530",
                        va="center", fontfamily="serif")

        hm_norm = hm / (hm.max() + 1e-6)
        ax.imshow(hm_norm, extent=[0, W, H, 0], cmap="inferno", alpha=0.55,
                  interpolation="bilinear", origin="upper", vmin=0.02)
        sm = plt.cm.ScalarMappable(cmap="inferno",
                                   norm=plt.Normalize(vmin=0, vmax=hm.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("Gaze Density", color="#3d3530", fontsize=9)
        ax.set_title("Gaze Heatmap — Attention Density on Reading Task",
                     fontsize=13, color="#1a1612", pad=10, fontfamily="serif")
        ax.axis("off")
        plt.tight_layout(pad=0.5)
        plt.savefig(HEATMAP_PATH, dpi=120, bbox_inches="tight", facecolor="#fffef9")
        plt.close()

        # RGBA overlay at exact screen resolution — no text baked in.
        # When displayed fullscreen in the browser over the reading layout,
        # gaze pixels align 1:1 with screen coordinates.
        hm_scr = np.zeros((scr_h, scr_w))
        xi = xs.astype(int).clip(0, scr_w - 1)
        yi = ys.astype(int).clip(0, scr_h - 1)
        for px, py in zip(xi, yi):
            hm_scr[py, px] += 1
        sigma_px = max(scr_w, scr_h) * 0.025
        hm_scr = gaussian_filter(hm_scr, sigma=sigma_px)
        hm_scr_n = hm_scr / (hm_scr.max() + 1e-6)
        rgba_f = plt.cm.inferno(hm_scr_n)           # (H, W, 4) float [0,1]
        rgba_f[:, :, 3] = np.where(
            hm_scr_n > 0.015,
            np.minimum(hm_scr_n * 0.80, 0.88),
            0.0
        )
        ov_uint8 = (rgba_f * 255).astype(np.uint8)
        Image.fromarray(ov_uint8, mode="RGBA").save(OVERLAY_PATH)
        print("Overlay saved:", OVERLAY_PATH)

        label, conf = classify_heatmap(HEATMAP_PATH)
        print(f"Classification: {label} ({conf}%)")
        with lock:
            state["heatmap_ready"]  = True
            state["status"]         = "done"
            state["classification"] = label
            state["confidence"]     = conf
        print("Heatmap saved:", HEATMAP_PATH)

    except Exception as e:
        print(f"Heatmap error: {e}"); traceback.print_exc()
        with lock: state["status"] = "done"

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>NeuroScan</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--cream:#faf7f2;--warm:#f0ebe0;--ink:#1a1612;--ink2:#3d3530;--rust:#c4432a;--teal:#1a7a6e;--gold:#d4922a;--muted:#8a7f74;--border:#e0d8cc;--shadow:rgba(26,22,18,.08)}
body{background:var(--cream);color:var(--ink);font-family:'Plus Jakarta Sans',sans-serif;min-height:100vh}
#calib-overlay{display:none;position:fixed;inset:0;background:rgba(13,13,13,0.82);z-index:999;align-items:center;justify-content:center}
#calib-overlay.active{display:flex}
#calib-title{font-family:'DM Mono',monospace;font-size:.85rem;color:rgba(255,255,255,.5);letter-spacing:.12em;text-transform:uppercase;position:fixed;top:32px;left:50%;transform:translateX(-50%)}
#calib-progress{font-family:'DM Mono',monospace;font-size:.75rem;color:rgba(255,255,255,.35);position:fixed;bottom:32px;left:50%;transform:translateX(-50%)}
.calib-dot{position:fixed;width:24px;height:24px;border-radius:50%;background:#d4922a;transform:translate(-50%,-50%);animation:pdot 1s ease-out infinite;transition:left .05s,top .05s}
.calib-dot.collecting{background:#1a7a6e;animation:pdotc .6s ease-out infinite}
@keyframes pdot{0%{box-shadow:0 0 0 0 rgba(212,146,42,.6)}70%{box-shadow:0 0 0 18px rgba(212,146,42,0)}100%{box-shadow:0 0 0 0 rgba(212,146,42,0)}}
@keyframes pdotc{0%{box-shadow:0 0 0 0 rgba(26,122,110,.7)}70%{box-shadow:0 0 0 22px rgba(26,122,110,0)}100%{box-shadow:0 0 0 0 rgba(26,122,110,0)}}
#calib-dot{display:none}
#calib-complete{display:none;position:fixed;inset:0;background:#0d0d0d;z-index:1000;align-items:center;justify-content:center;flex-direction:column;gap:16px}
#calib-complete.show{display:flex}
#calib-complete h2{font-family:'DM Serif Display',serif;font-size:2.5rem;color:#1a7a6e}
#calib-complete p{font-family:'DM Mono',monospace;font-size:.8rem;color:rgba(255,255,255,.4)}
#screening-view{display:none;position:fixed;inset:0;background:#fffef9;z-index:100;flex-direction:column}
#screening-view.active{display:flex}
.screening-bar{display:flex;align-items:center;justify-content:space-between;background:var(--ink);padding:0 32px;height:52px;flex-shrink:0}
.screening-bar-title{font-family:'DM Mono',monospace;font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.7)}
.screening-status{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--gold);letter-spacing:.06em}
.timer-bar{height:4px;background:var(--border);flex-shrink:0}
.timer-fill{height:100%;background:linear-gradient(90deg,var(--teal),var(--gold));width:0%}
#task-reading{flex:1;overflow:hidden;padding:20px 120px;background:#fffef9;display:flex;flex-direction:column;justify-content:flex-start}
.reading-meta{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px}
.reading-title{font-family:'DM Serif Display',serif;font-size:2rem;color:var(--ink);line-height:1.2;margin-bottom:6px}
.reading-subtitle{font-family:'DM Serif Display',serif;font-style:italic;font-size:1rem;color:var(--muted);margin-bottom:16px}
.reading-divider{width:60px;height:3px;background:var(--rust);margin-bottom:16px}
.reading-body p{font-family:'DM Serif Display',serif;font-size:1rem;line-height:1.7;color:var(--ink2);margin-bottom:12px;max-width:820px}
#cam-pip{position:fixed;bottom:20px;left:20px;width:200px;border-radius:12px;overflow:hidden;border:2px solid var(--ink);box-shadow:0 8px 32px rgba(0,0,0,.3);z-index:200;display:none}
#cam-pip.show{display:block}
#cam-feed{width:100%;display:block}
.pip-badge{position:absolute;top:8px;right:8px;background:var(--rust);color:#fff;border-radius:100px;font-family:'DM Mono',monospace;font-size:.6rem;font-weight:500;padding:2px 8px;display:none}
.pip-badge.show{display:block}
.pip-pts{position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,.6);color:#fff;border-radius:6px;font-family:'DM Mono',monospace;font-size:.6rem;padding:2px 7px}
#btn-stop-float{position:fixed;bottom:24px;right:24px;z-index:300;padding:12px 24px;background:var(--rust);color:#fff;border:none;border-radius:100px;font-family:'DM Mono',monospace;font-size:.78rem;font-weight:500;cursor:pointer;box-shadow:0 4px 20px rgba(196,67,42,.4);display:none;transition:all .2s;letter-spacing:.06em}
#btn-stop-float:hover{transform:scale(1.04)}
#btn-stop-float.show{display:block}
#dashboard{padding:40px 48px;max-width:1300px;margin:0 auto}
.brand{display:flex;align-items:baseline;gap:10px;margin-bottom:48px}
.brand-name{font-family:'DM Serif Display',serif;font-size:2rem;color:var(--ink)}
.brand-tag{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--muted);background:var(--warm);padding:3px 10px;border-radius:100px;border:1px solid var(--border)}
.dash-grid{display:grid;grid-template-columns:1fr 380px;gap:28px;align-items:start}
.child-card{background:#fff;border:1px solid var(--border);border-radius:16px;padding:28px;box-shadow:0 2px 12px var(--shadow)}
.child-card h2{font-family:'DM Serif Display',serif;font-size:1.3rem;margin-bottom:20px}
.field-label{font-family:'DM Mono',monospace;font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.field-input{width:100%;padding:11px 14px;border:1.5px solid var(--border);border-radius:9px;font-family:'Plus Jakarta Sans',sans-serif;font-size:.95rem;color:var(--ink);background:var(--cream);outline:none;transition:border .2s;margin-bottom:20px}
.field-input:focus{border-color:var(--teal)}
.btn-start{width:100%;padding:14px;background:var(--ink);color:#fff;border:none;border-radius:10px;font-family:'DM Mono',monospace;font-size:.8rem;font-weight:500;letter-spacing:.08em;cursor:pointer;transition:all .2s;text-transform:uppercase}
.btn-start:hover{background:var(--rust)}
.btn-start:disabled{opacity:.4;cursor:not-allowed;background:var(--ink)}
.metrics-card{background:#fff;border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:0 2px 12px var(--shadow)}
.metrics-header{padding:18px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.metrics-title{font-family:'DM Mono',monospace;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
.status-dot.calibrating{background:var(--gold);animation:blink 1s infinite}
.status-dot.recording{background:var(--teal);animation:blink 1s infinite}
.status-dot.analyzing{background:var(--rust);animation:blink .5s infinite}
.status-dot.done{background:var(--teal)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.metric-row{padding:12px 20px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.metric-row:last-child{border-bottom:none}
.metric-key{font-family:'DM Mono',monospace;font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.metric-val{font-family:'DM Mono',monospace;font-size:.8rem;color:var(--ink);font-weight:500}
.event-pill{padding:2px 9px;border-radius:100px;font-size:.62rem;font-family:'DM Mono',monospace;font-weight:500}
.pill-on{background:rgba(26,122,110,.12);color:var(--teal);border:1px solid rgba(26,122,110,.25)}
.pill-off{background:var(--warm);color:var(--muted);border:1px solid var(--border)}
.heatmap-card{margin-top:16px;background:#fff;border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:0 2px 12px var(--shadow)}
.heatmap-head{padding:14px 18px;border-bottom:1px solid var(--border);font-family:'DM Mono',monospace;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);display:flex;justify-content:space-between;align-items:center}
.heatmap-body{padding:14px}
#heatmap-img{width:100%;border-radius:8px;display:none}
.heatmap-ph{aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:var(--muted);font-family:'DM Mono',monospace;font-size:.72rem;border:1.5px dashed var(--border);border-radius:8px;background:var(--warm)}
.heatmap-analyzing{aspect-ratio:16/9;display:none;align-items:center;justify-content:center;flex-direction:column;gap:12px;background:var(--warm);border-radius:8px}
.heatmap-analyzing.show{display:flex}
.spinner{width:28px;height:28px;border:3px solid var(--border);border-top-color:var(--teal);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div id="calib-overlay">
  <div id="calib-title">Follow the dot with your eyes</div>
  <div id="calib-dot" class="calib-dot"></div>
  <div id="calib-progress">Point 0 / 9</div>
</div>
<div id="calib-complete">
  <h2>✓ Ready to Read</h2>
  <p>Starting reading session…</p>
</div>
<div id="screening-view">
  <div class="screening-bar">
    <div class="screening-bar-title">NeuroScan · Reading Task</div>
    <div class="screening-status" id="nav-status">Recording…</div>
  </div>
  <div class="timer-bar"><div class="timer-fill" id="timer-fill"></div></div>
  <div id="task-reading">
    <div class="reading-meta">Reading Task · Read naturally</div>
    <div class="reading-title">The Secret Garden</div>
    <div class="reading-subtitle">— Frances Hodgson Burnett</div>
    <div class="reading-divider"></div>
    <div class="reading-body">
      <p>When Mary Lennox was sent to Misselthwaite Manor to live with her uncle, everybody said she was the most disagreeable-looking child ever seen. It was true, too. She had a little thin face and a little thin body, thin light hair and a sour expression.</p>
      <p>Her father had held a position under the English Government and had always been busy and ill himself, and her mother had been a great beauty who cared only to go to parties and amuse herself with gay people. She had not wanted a little girl at all.</p>
      <p>When Mary was born she handed her over to the care of an Ayah, who was made to understand that if she wished to please the Mem Sahib she must keep the child out of sight as much as possible. So when she was a scolding, fretful, ugly little baby she was kept out of the way.</p>
      <p>She never remembered seeing familiarly anything but the dark faces of her Ayah and the other native servants. They always obeyed her and gave her her own way in everything, because the Mem Sahib would be angry if she was disturbed by her crying.</p>
      <p>By the time she was six years old she was as tyrannical and selfish a little pig as ever lived. The young Ayah, who loved her, had died of cholera. Mary had been sent away to live with an English clergyman and his wife.</p>
    </div>
  </div>
</div>
<div id="cam-pip">
  <img id="cam-feed" src="/video_feed" alt="cam"/>
  <div class="pip-badge" id="pip-badge">● REC</div>
  <div class="pip-pts" id="pip-pts">0 pts</div>
</div>
<button id="btn-stop-float" onclick="stopTracking()">■ Stop &amp; Generate Heatmap</button>
<div id="dashboard">
  <div class="brand">
    <div class="brand-name">NeuroScan</div>
    <div class="brand-tag">Eye-Tracking Gaze Analysis</div>
  </div>
  <div class="dash-grid">
    <div>
      <div class="child-card">
        <h2>New Screening Session</h2>
        <div class="field-label">Child's Name</div>
        <input class="field-input" id="child-name" type="text" placeholder="Enter child's name…"/>
        <button class="btn-start" id="btn-start" onclick="startScreening()">▶ Begin Eye-Tracking</button>
      </div>
      <div class="heatmap-card">
        <div class="heatmap-head">
          <span>🔥 Gaze Heatmap — Reading Task</span>
          <div style="display:flex;align-items:center;gap:12px;">
            <span id="heatmap-pts" style="color:var(--teal);display:none"></span>
            <button id="btn-overlay" onclick="document.getElementById('overlay-modal').style.display='block'"
              style="display:none;padding:5px 14px;background:var(--teal);color:#fff;border:none;border-radius:100px;font-family:'DM Mono',monospace;font-size:.65rem;cursor:pointer;letter-spacing:.06em;">
              ⌖ View Aligned
            </button>
          </div>
        </div>
        <div class="heatmap-body">
          <div class="heatmap-ph" id="heatmap-ph">Heatmap appears after session ends</div>
          <div class="heatmap-analyzing" id="heatmap-analyzing">
            <div class="spinner"></div>
            <span style="font-family:'DM Mono',monospace;font-size:.72rem;color:var(--muted)">Generating heatmap…</span>
          </div>
          <img id="heatmap-img" src="" alt="Gaze Heatmap"/>
          <div id="classification-result" style="display:none;margin-top:14px;padding:16px;border-radius:10px;text-align:center;font-family:'DM Mono',monospace;">
            <div style="font-size:.75rem;color:var(--muted);margin-bottom:6px;letter-spacing:2px;">RESNET18 CLASSIFICATION</div>
            <div id="class-label" style="font-size:1.6rem;font-weight:800;margin-bottom:4px;"></div>
            <div id="class-conf" style="font-size:.8rem;color:var(--muted);"></div>
          </div>
        </div>
      </div>
    </div>
    <div>
      <div class="metrics-card">
        <div class="metrics-header">
          <div class="metrics-title">Live Metrics</div>
          <div class="status-dot" id="status-dot"></div>
        </div>
        <div class="metric-row"><span class="metric-key">Status</span><span class="metric-val" id="m-status">Idle</span></div>
        <div class="metric-row"><span class="metric-key">Gaze Points</span><span class="metric-val" id="m-pts">0</span></div>
        <div class="metric-row"><span class="metric-key">Fixation</span><span class="event-pill pill-off" id="m-fix">OFF</span></div>
        <div class="metric-row"><span class="metric-key">Saccade</span><span class="event-pill pill-off" id="m-sac">OFF</span></div>
        <div class="metric-row"><span class="metric-key">Gaze X / Y</span><span class="metric-val" id="m-xy">— / —</span></div>
        <div class="metric-row"><span class="metric-key">Calib Point</span><span class="metric-val" id="m-calib">—</span></div>
      </div>
      <div style="background:var(--warm);border:1px solid var(--border);border-radius:16px;padding:20px;margin-top:16px;">
        <div style="font-family:'DM Mono',monospace;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--gold);margin-bottom:12px;">📌 Calibration Info</div>
        <div style="font-family:'DM Mono',monospace;font-size:.72rem;color:var(--ink2);line-height:1.9;">
          9 dots appear on screen<br/>Follow each dot with your eyes<br/>
          <span style="color:var(--gold)">● Orange</span> = look here<br/>
          <span style="color:var(--teal)">● Teal</span> = recording<br/>
          Reading starts automatically
        </div>
      </div>
    </div>
  </div>
</div>
<!-- Aligned overlay modal: exact reading layout + RGBA heatmap on top -->
<div id="overlay-modal" style="display:none;position:fixed;inset:0;z-index:9000;background:#fffef9;overflow:hidden;">
  <!-- same top bars as screening view -->
  <div class="screening-bar">
    <div class="screening-bar-title">NeuroScan · Gaze Overlay</div>
    <button onclick="document.getElementById('overlay-modal').style.display='none'"
      style="padding:6px 18px;background:var(--rust);color:#fff;border:none;border-radius:100px;font-family:'DM Mono',monospace;font-size:.72rem;cursor:pointer;letter-spacing:.06em">✕ Close</button>
  </div>
  <div class="timer-bar"></div>
  <div id="overlay-task-reading" style="flex:1;overflow:hidden;padding:20px 120px;background:#fffef9;display:flex;flex-direction:column;justify-content:flex-start;position:absolute;top:56px;left:0;right:0;bottom:0;">
    <div class="reading-meta">Reading Task · Read naturally</div>
    <div class="reading-title">The Secret Garden</div>
    <div class="reading-subtitle">— Frances Hodgson Burnett</div>
    <div class="reading-divider"></div>
    <div class="reading-body">
      <p>When Mary Lennox was sent to Misselthwaite Manor to live with her uncle, everybody said she was the most disagreeable-looking child ever seen. It was true, too. She had a little thin face and a little thin body, thin light hair and a sour expression.</p>
      <p>Her father had held a position under the English Government and had always been busy and ill himself, and her mother had been a great beauty who cared only to go to parties and amuse herself with gay people. She had not wanted a little girl at all.</p>
      <p>When Mary was born she handed her over to the care of an Ayah, who was made to understand that if she wished to please the Mem Sahib she must keep the child out of sight as much as possible. So when she was a scolding, fretful, ugly little baby she was kept out of the way.</p>
      <p>She never remembered seeing familiarly anything but the dark faces of her Ayah and the other native servants. They always obeyed her and gave her her own way in everything, because the Mem Sahib would be angry if she was disturbed by her crying.</p>
      <p>By the time she was six years old she was as tyrannical and selfish a little pig as ever lived. The young Ayah, who loved her, had died of cholera. Mary had been sent away to live with an English clergyman and his wife.</p>
    </div>
  </div>
  <!-- RGBA overlay: fills exactly 100vw × 100vh, object-fit:fill keeps pixel alignment -->
  <img id="overlay-img" src="" alt=""
    style="position:fixed;inset:0;width:100vw;height:100vh;object-fit:fill;pointer-events:none;z-index:9001;"/>
</div>

<script>
let polling=false, calibInterval=null;
const CPOS=[[.05,.05],[.50,.05],[.95,.05],[.05,.50],[.50,.50],[.95,.50],[.05,.95],[.50,.95],[.95,.95]];

function _doStart(name){
  fetch("/start?name="+encodeURIComponent(name)+"&sw="+window.innerWidth+"&sh="+window.innerHeight).then(r=>r.json()).then(d=>{
    if(!d.ok){alert(d.msg);document.getElementById("btn-start").disabled=false;if(document.fullscreenElement)document.exitFullscreen();return;}
    document.getElementById("dashboard").style.display="none";
    // Show reading text DURING calibration so the user sits in reading posture,
    // not a forced-stare posture — eliminates the systematic vertical bias.
    document.getElementById("screening-view").classList.add("active");
    document.getElementById("calib-overlay").classList.add("active");
    document.getElementById("calib-dot").style.display="block";
    setTimeout(()=>{calibInterval=setInterval(pollCalib,150);}, 800);
  });
}

function startScreening(){
  const name=document.getElementById("child-name").value.trim()||"Child";
  document.getElementById("btn-start").disabled=true;
  if(document.fullscreenEnabled && !document.fullscreenElement){
    document.documentElement.requestFullscreen().then(()=>{
      // Give browser ~150ms to finish the fullscreen transition so
      // window.innerWidth/innerHeight reflect the true screen size.
      setTimeout(()=>_doStart(name), 150);
    }).catch(()=>_doStart(name));
  } else {
    _doStart(name);
  }
}

function pollCalib(){
  fetch("/state").then(r=>r.json()).then(d=>{
    const dot=document.getElementById("calib-dot");
    const prog=document.getElementById("calib-progress");
    const title=document.getElementById("calib-title");

    // Phase 3: both calibration and drift done → start reading
    if(d.calib_done && d.drift_done){
      clearInterval(calibInterval);
      dot.style.display="none";
      document.getElementById("calib-overlay").classList.remove("active");
      const cc=document.getElementById("calib-complete");
      cc.classList.add("show");
      setTimeout(()=>{
        cc.classList.remove("show");
        document.getElementById("screening-view").classList.add("active");
        document.getElementById("cam-pip").classList.add("show");
        document.getElementById("pip-badge").classList.add("show");
        document.getElementById("btn-stop-float").classList.add("show");
        polling=true; poll();
      },1200);
      return;
    }

    // Phase 2: drift correction dot
    if(d.calib_done && !d.drift_done){
      const t=d.drift_target;
      if(t){
        dot.style.display="block";
        dot.style.left=t[0]+"px"; dot.style.top=t[1]+"px";
        dot.className="calib-dot"+(d.drift_collecting?" collecting":"");
        title.textContent="Fine-tune · keep looking at this dot";
        prog.textContent=d.drift_collecting?"Hold still…":"Look at the dot";
        document.getElementById("m-calib").textContent="Fine-tuning";
      }
      return;
    }

    // Phase 1: 9-point calibration
    const idx=d.calib_point_idx;
    if(idx>=0&&idx<9){
      const p=CPOS[idx];
      dot.style.left=(p[0]*window.innerWidth)+"px";
      dot.style.top=(p[1]*window.innerHeight)+"px";
      dot.className="calib-dot"+(d.calib_collecting?" collecting":"");
      prog.textContent="Point "+(idx+1)+" / 9 — "+(d.calib_collecting?"Hold still...":"Move eyes to dot");
      document.getElementById("m-calib").textContent=(idx+1)+" / 9";
    } else if(idx===-1){
      dot.style.left=(0.5*window.innerWidth)+"px";
      dot.style.top=(0.5*window.innerHeight)+"px";
      prog.textContent="Starting calibration...";
    }
  }).catch(()=>{});
}

function stopTracking(){
  fetch("/stop").then(r=>r.json()).then(()=>{
    if(document.fullscreenElement) document.exitFullscreen();
    document.getElementById("screening-view").classList.remove("active");
    document.getElementById("dashboard").style.display="block";
    document.getElementById("btn-stop-float").classList.remove("show");
    document.getElementById("cam-pip").classList.remove("show");
    document.getElementById("pip-badge").classList.remove("show");
    document.getElementById("heatmap-ph").style.display="none";
    document.getElementById("heatmap-analyzing").classList.add("show");
    setTimeout(checkHeatmap,2000);
  });
}

// If user presses Escape to exit fullscreen mid-session, stop tracking gracefully.
document.addEventListener("fullscreenchange",()=>{
  if(!document.fullscreenElement && polling){stopTracking();}
});

function checkHeatmap(){
  fetch("/state").then(r=>r.json()).then(d=>{
    if(d.heatmap_ready){
      document.getElementById("heatmap-analyzing").classList.remove("show");
      document.getElementById("heatmap-img").src="/heatmap?t="+Date.now();
      document.getElementById("heatmap-img").style.display="block";
      document.getElementById("heatmap-pts").textContent=d.gaze_count+" gaze pts";
      document.getElementById("heatmap-pts").style.display="block";
      document.getElementById("btn-start").disabled=false;
      document.getElementById("btn-overlay").style.display="inline-block";
      document.getElementById("overlay-img").src="/overlay?t="+Date.now();
      if(d.classification){
        const box=document.getElementById("classification-result");
        const lbl=document.getElementById("class-label");
        const conf=document.getElementById("class-conf");
        box.style.display="block";
        lbl.textContent=d.classification;
        lbl.style.color=d.classification==="Dyslexic"?"#c4432a":"#2a7f4f";
        box.style.background=d.classification==="Dyslexic"?"#fff0ee":"#f0fff4";
        box.style.border="1.5px solid "+(d.classification==="Dyslexic"?"#c4432a":"#2a7f4f");
        conf.textContent="Confidence: "+d.confidence+"%";
      }
    }else{setTimeout(checkHeatmap,1500);}
  }).catch(()=>setTimeout(checkHeatmap,2000));
}

function poll(){
  if(!polling)return;
  fetch("/state").then(r=>r.json()).then(d=>{
    document.getElementById("status-dot").className="status-dot "+d.status;
    document.getElementById("nav-status").textContent=d.status==="recording"?"Recording · "+d.gaze_count+" pts":d.status;
    document.getElementById("m-status").textContent=d.status.charAt(0).toUpperCase()+d.status.slice(1);
    document.getElementById("m-pts").textContent=d.gaze_count;
    document.getElementById("pip-pts").textContent=d.gaze_count+" pts";
    document.getElementById("m-xy").textContent=d.x+" / "+d.y;
    const fix=document.getElementById("m-fix");
    fix.textContent=d.fixation?"ON":"OFF";fix.className="event-pill "+(d.fixation?"pill-on":"pill-off");
    const sac=document.getElementById("m-sac");
    sac.textContent=d.saccade?"ON":"OFF";sac.className="event-pill "+(d.saccade?"pill-on":"pill-off");
    if(d.running)setTimeout(poll,200);else polling=false;
  }).catch(()=>{if(polling)setTimeout(poll,500);});
}
</script>
</body>
</html>"""

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/start")
def start():
    if state["running"]: return jsonify(ok=False, msg="Already running")
    name = request.args.get("name", "Child")
    sw = int(request.args.get("sw", SCREEN_W))
    sh = int(request.args.get("sh", SCREEN_H))
    with lock:
        state["child_name"] = name
        state["screen_w"] = sw
        state["screen_h"] = sh
        # Reset calibration state synchronously so the first pollCalib() call
        # never sees stale calib_done=True from the previous session.
        state.update(calib_done=False, calib_point_idx=-1, calib_collecting=False,
                     heatmap_ready=False, gaze_points=[], status="calibrating",
                     classification=None, confidence=0.0,
                     drift_done=False, drift_collecting=False, drift_target=None)
    stop_event.clear()
    threading.Thread(target=run_tracker, daemon=True).start()
    return jsonify(ok=True)

@app.route("/begin_calib")
def begin_calib():
    t = tracker_ref.get("tracker")
    if t: t.start_calibration()
    return jsonify(ok=True)

@app.route("/stop")
def stop():
    stop_event.set()
    return jsonify(ok=True)

@app.route("/state")
def get_state():
    with lock:
        xy = state["current_xy"]
        return jsonify(
            running=bool(state["running"]), status=str(state["status"]),
            gaze_count=int(len(state["gaze_points"])),
            fixation=bool(state["fixation"]), saccade=bool(state["saccade"]),
            x=float(xy[0]), y=float(xy[1]),
            heatmap_ready=bool(state["heatmap_ready"]),
            child_name=str(state["child_name"]),
            classification=str(state["classification"]) if state["classification"] else None,
            confidence=float(state["confidence"]),
            calib_point_idx=int(state["calib_point_idx"]),
            calib_total=int(state["calib_total"]),
            calib_done=bool(state["calib_done"]),
            calib_collecting=bool(state["calib_collecting"]),
            drift_done=bool(state["drift_done"]),
            drift_collecting=bool(state["drift_collecting"]),
            drift_target=state["drift_target"],
        )

@app.route("/heatmap")
def heatmap():
    if os.path.exists(HEATMAP_PATH):
        return send_file(HEATMAP_PATH, mimetype="image/png")
    return "", 404

@app.route("/overlay")
def overlay():
    if os.path.exists(OVERLAY_PATH):
        return send_file(OVERLAY_PATH, mimetype="image/png")
    return "", 404

def gen_frames():
    while True:
        jpg = frame_buf.get("jpg")
        if jpg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.033)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

REALTIME_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>NeuroScan · Realtime Gaze</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d0d;overflow:hidden}

/* ── dark calibration / start phases ── */
#calib-overlay{display:none;position:fixed;inset:0;background:rgba(13,13,13,0.80);z-index:900;align-items:center;justify-content:center}
#calib-overlay.active{display:flex}
#calib-title{position:fixed;top:32px;left:50%;transform:translateX(-50%);font-family:'DM Mono',monospace;font-size:.8rem;color:rgba(255,255,255,.42);letter-spacing:.14em;text-transform:uppercase;white-space:nowrap}
#calib-progress{position:fixed;bottom:32px;left:50%;transform:translateX(-50%);font-family:'DM Mono',monospace;font-size:.72rem;color:rgba(255,255,255,.28);letter-spacing:.06em}
.calib-dot{position:fixed;width:22px;height:22px;border-radius:50%;background:#d4922a;transform:translate(-50%,-50%);transition:left .05s,top .05s;animation:pdot 1s ease-out infinite}
.calib-dot.collecting{background:#1a7a6e;animation:pdotc .6s ease-out infinite}
@keyframes pdot{0%{box-shadow:0 0 0 0 rgba(212,146,42,.7)}70%{box-shadow:0 0 0 18px rgba(212,146,42,0)}100%{box-shadow:0 0 0 0 rgba(212,146,42,0)}}
@keyframes pdotc{0%{box-shadow:0 0 0 0 rgba(26,122,110,.8)}70%{box-shadow:0 0 0 22px rgba(26,122,110,0)}100%{box-shadow:0 0 0 0 rgba(26,122,110,0)}}

#start-screen{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:22px;background:#0d0d0d}
#start-screen h1{font-family:'Plus Jakarta Sans',sans-serif;color:#fff;font-size:1.6rem;font-weight:600;letter-spacing:.04em}
.start-sub{font-family:'DM Mono',monospace;color:rgba(255,255,255,.38);font-size:.73rem;letter-spacing:.06em;text-align:center;line-height:2;max-width:420px}
#btn-start{padding:13px 40px;background:#1a7a6e;color:#fff;border:none;border-radius:100px;font-family:'DM Mono',monospace;font-size:.8rem;font-weight:500;letter-spacing:.1em;cursor:pointer;transition:transform .15s,background .15s}
#btn-start:hover{transform:scale(1.05);background:#23a596}

/* ── reading / gaze view ── */
#gaze-view{display:none;position:fixed;inset:0;background:#f5f1ea}

/* reading passage */
#reading-pane{
  position:absolute;
  /* sits below the 48px HUD with a comfortable top margin */
  top:48px; left:50%; transform:translateX(-50%);
  width:min(760px,86vw);
  padding:44px 0 60px;
}
.r-label{font-family:'DM Mono',monospace;font-size:.62rem;color:#9a8f82;letter-spacing:.16em;text-transform:uppercase;margin-bottom:18px}
.r-title{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.6rem;font-weight:600;color:#1a1612;margin-bottom:4px}
.r-byline{font-family:'Plus Jakarta Sans',sans-serif;font-style:italic;font-size:.95rem;color:#9a8f82;margin-bottom:18px}
.r-rule{width:52px;height:3px;background:#c4432a;margin-bottom:36px}
/* Each paragraph is one block; large font + generous line-height
   makes it visually obvious which line the cursor sits on.
   Research baseline: 22px sans-serif, 2.2 line-height for dyslexia screening. */
.r-body p{
  font-family:'Plus Jakarta Sans',sans-serif;
  font-size:1.22rem;        /* ≈ 19.5px — readable without crowding */
  line-height:2.35;         /* ≈ 46px per line — clearly distinct lines */
  color:#2a2420;
  margin-bottom:32px;
  letter-spacing:0.025em;
  word-spacing:0.06em;
}
.r-body p:last-child{margin-bottom:0}

/* trail canvas — sits above text but cursor ring stays on top */
#trail-canvas{position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:10}

/* gaze cursor — hollow ring so text beneath is readable
   Teal = moving/saccade  |  Gold = fixation */
#gaze-dot{
  position:fixed;
  width:30px;height:30px;border-radius:50%;
  transform:translate(-50%,-50%);
  pointer-events:none;z-index:11;
  background:rgba(0,168,150,.18);
  border:2.5px solid #00a896;
  box-shadow:0 0 10px rgba(0,168,150,.45);
  transition:background .08s,border-color .08s,box-shadow .08s;
}
#gaze-dot.fixating{
  background:rgba(212,146,42,.22);
  border-color:#d4922a;
  box-shadow:0 0 14px rgba(212,146,42,.55);
}

/* HUD bar */
#hud{position:fixed;top:0;left:0;right:0;height:48px;display:flex;align-items:center;justify-content:space-between;padding:0 24px;background:rgba(20,16,12,.85);backdrop-filter:blur(6px);z-index:20}
.hud-l{display:flex;align-items:center;gap:16px}
.hud-badge{font-family:'DM Mono',monospace;font-size:.63rem;color:#1fd6c2;letter-spacing:.12em;text-transform:uppercase}
#hud-coords{font-family:'DM Mono',monospace;font-size:.68rem;color:rgba(255,255,255,.38)}
.hud-flags{display:flex;gap:7px}
.flag{font-family:'DM Mono',monospace;font-size:.58rem;padding:2px 8px;border-radius:100px;letter-spacing:.06em;border:1px solid rgba(255,255,255,.12);color:rgba(255,255,255,.28)}
.flag.on.fix{color:#d4922a;border-color:#d4922a;background:rgba(212,146,42,.12)}
.flag.on.sac{color:#1fd6c2;border-color:#1fd6c2;background:rgba(31,214,194,.1)}
#btn-stop{padding:5px 16px;background:rgba(196,67,42,.85);color:#fff;border:none;border-radius:100px;font-family:'DM Mono',monospace;font-size:.63rem;cursor:pointer;letter-spacing:.06em}

/* webcam PiP */
#pip{position:fixed;bottom:20px;right:20px;width:160px;border-radius:10px;overflow:hidden;border:1px solid rgba(0,0,0,.12);z-index:20;box-shadow:0 4px 18px rgba(0,0,0,.15)}
#pip img{width:100%;display:block}
#pip-label{position:absolute;top:6px;left:6px;background:rgba(196,67,42,.85);color:#fff;font-family:'DM Mono',monospace;font-size:.52rem;padding:1px 6px;border-radius:100px;letter-spacing:.06em}
</style>
</head>
<body>

<!-- dark calibration overlay -->
<div id="calib-overlay">
  <div id="calib-title">Keep head still — move eyes only</div>
  <div class="calib-dot" id="calib-dot" style="display:none"></div>
  <div id="calib-progress">Starting…</div>
</div>

<!-- start screen (dark) -->
<div id="start-screen">
  <h1>NeuroScan · Realtime Gaze</h1>
  <div class="start-sub">
    Rest keyboard edge on chest · ~13 inches from screen<br/>
    Look straight ahead · keep head still during calibration<br/>
    9-point calibration runs first, then the reading text appears
  </div>
  <button id="btn-start" onclick="beginSession()">▶ Begin</button>
</div>

<!-- reading + gaze view (cream) -->
<div id="gaze-view">

  <!-- reading passage -->
  <div id="reading-pane">
    <div class="r-label">Eye-Tracking Active · Read naturally</div>
    <div class="r-title">Alice's Adventures in Wonderland</div>
    <div class="r-byline">— Lewis Carroll</div>
    <div class="r-rule"></div>
    <div class="r-body">
      <p>Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, and what is the use of a book without pictures or conversations?</p>
      <p>So she was considering in her own mind whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.</p>
      <p>There was nothing so very remarkable in that, nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" But when the Rabbit actually took a watch out of its waistcoat-pocket and looked at it, Alice started to her feet.</p>
      <p>She had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and, burning with curiosity, she ran across the field after it and was just in time to see it pop down a large rabbit-hole under the hedge.</p>
    </div>
  </div>

  <!-- trail canvas (over text) -->
  <canvas id="trail-canvas"></canvas>

  <!-- gaze cursor -->
  <div id="gaze-dot"></div>

  <!-- HUD -->
  <div id="hud">
    <div class="hud-l">
      <span class="hud-badge">● Tracking</span>
      <span id="hud-coords">— / —</span>
      <div class="hud-flags">
        <span class="flag fix" id="flag-fix">FIX</span>
        <span class="flag sac" id="flag-sac">SAC</span>
      </div>
    </div>
    <button id="btn-stop" onclick="stopSession()">■ Stop</button>
  </div>

  <!-- webcam PiP -->
  <div id="pip">
    <img src="/video_feed" alt="cam"/>
    <div id="pip-label">● CAM</div>
  </div>
</div>

<script>
const CPOS = [[.05,.05],[.50,.05],[.95,.05],[.05,.50],[.50,.50],[.95,.50],[.05,.95],[.50,.95],[.95,.95]];
let calibInterval = null;
let gazeInterval  = null;
let trail = [];
const TRAIL_MAX = 90;
const TRAIL_MS  = 1100;
let canvas, ctx;

function _doBegin() {
  // Show reading text before calibration starts — user calibrates in reading posture.
  document.getElementById("gaze-view").style.display = "block";
  document.getElementById("calib-overlay").classList.add("active");
  document.getElementById("calib-dot").style.display = "block";
  fetch("/start?name=realtime&sw=" + window.innerWidth + "&sh=" + window.innerHeight)
    .then(r => r.json()).then(d => {
      if (!d.ok) { alert(d.msg); location.reload(); return; }
      setTimeout(() => { calibInterval = setInterval(pollCalib, 120); }, 800);
    });
}

function beginSession() {
  document.getElementById("start-screen").style.display = "none";
  if (document.fullscreenEnabled && !document.fullscreenElement) {
    document.documentElement.requestFullscreen()
      .then(() => setTimeout(_doBegin, 150))
      .catch(() => _doBegin());
  } else {
    _doBegin();
  }
}

function pollCalib() {
  fetch("/state").then(r => r.json()).then(d => {
    const dot  = document.getElementById("calib-dot");
    const prog = document.getElementById("calib-progress");
    const title = document.getElementById("calib-title");

    // Phase 3: done
    if (d.calib_done && d.drift_done) {
      clearInterval(calibInterval);
      document.getElementById("calib-overlay").classList.remove("active");
      startGazeView();
      return;
    }

    // Phase 2: drift correction
    if (d.calib_done && !d.drift_done) {
      const t = d.drift_target;
      if (t) {
        dot.style.display = "block";
        dot.style.left = t[0] + "px"; dot.style.top = t[1] + "px";
        dot.className = "calib-dot" + (d.drift_collecting ? " collecting" : "");
        title.textContent = "Fine-tune · keep looking at this dot";
        prog.textContent = d.drift_collecting ? "Hold still…" : "Look at the dot";
      }
      return;
    }

    // Phase 1: 9-point calibration
    const idx = d.calib_point_idx;
    if (idx >= 0 && idx < 9) {
      const p = CPOS[idx];
      dot.style.left  = (p[0] * window.innerWidth)  + "px";
      dot.style.top   = (p[1] * window.innerHeight) + "px";
      dot.className   = "calib-dot" + (d.calib_collecting ? " collecting" : "");
      prog.textContent = "Point " + (idx+1) + " / 9 — " + (d.calib_collecting ? "Hold still…" : "Move eyes to dot");
    } else {
      dot.style.left = "50%"; dot.style.top = "50%";
      prog.textContent = "Starting…";
    }
  }).catch(() => {});
}

function startGazeView() {
  document.getElementById("gaze-view").style.display = "block";
  canvas = document.getElementById("trail-canvas");
  ctx    = canvas.getContext("2d");
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  requestAnimationFrame(drawTrail);
  gazeInterval = setInterval(fetchGaze, 40);
}

function resizeCanvas() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}

function fetchGaze() {
  fetch("/state").then(r => r.json()).then(d => {
    if (!d.running && d.status !== "recording") return;
    const x = d.x, y = d.y, fix = d.fixation;
    const dot = document.getElementById("gaze-dot");
    dot.style.left = x + "px";
    dot.style.top  = y + "px";
    dot.className  = fix ? "fixating" : "";
    document.getElementById("hud-coords").textContent = Math.round(x) + " / " + Math.round(y);
    document.getElementById("flag-fix").className = "flag fix" + (fix       ? " on" : "");
    document.getElementById("flag-sac").className = "flag sac" + (d.saccade ? " on" : "");
    trail.push({ x, y, fix, ts: Date.now() });
    if (trail.length > TRAIL_MAX) trail.shift();
  }).catch(() => {});
}

function drawTrail() {
  requestAnimationFrame(drawTrail);
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const now = Date.now();
  trail.forEach(pt => {
    const age   = (now - pt.ts) / TRAIL_MS;
    const alpha = Math.max(0, 1 - age);
    if (alpha <= 0) return;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, pt.fix ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = pt.fix
      ? `rgba(212,146,42,${alpha * 0.55})`
      : `rgba(0,168,150,${alpha * 0.45})`;
    ctx.fill();
  });
}

function stopSession() {
  clearInterval(gazeInterval);
  fetch("/stop");
  if (document.fullscreenElement) document.exitFullscreen();
  setTimeout(() => location.reload(), 400);
}
</script>
</body>
</html>"""

@app.route("/realtime")
def realtime():
    return render_template_string(REALTIME_HTML)

if __name__ == "__main__":
    print("=" * 55)
    print("NeuroScan — Eye-Tracking Gaze Analysis")
    print("Open http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=3000, debug=False, threaded=True)