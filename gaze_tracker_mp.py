"""
gaze_tracker_mp.py — 9-point calibration + drift correction
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque
import os, time

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

LEFT_IRIS=468; RIGHT_IRIS=473
LEFT_EYE_L=33; LEFT_EYE_R=133
RIGHT_EYE_L=362; RIGHT_EYE_R=263
LEFT_EYE_TOP=159; LEFT_EYE_BOTTOM=145
RIGHT_EYE_TOP=386; RIGHT_EYE_BOTTOM=374

FACE_OUTLINE=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
LEFT_EYE_PTS=[33,160,158,133,153,144,33]
RIGHT_EYE_PTS=[362,385,387,263,373,380,362]

class MPGazeTracker:
    def __init__(self, screen_w=1920, screen_h=1080,
                 smoothing=6, fixation_threshold=40, fixation_frames=6):
        self.screen_w=screen_w; self.screen_h=screen_h
        self.smoothing=smoothing
        self.fixation_threshold=fixation_threshold
        self.fixation_frames=fixation_frames

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = mp_vision.FaceLandmarker.create_from_options(options)
        self._start_time = time.time()

        # Horizontal 5%: text spans full width, extreme corners work fine.
        # Vertical 14%: reading text starts ~22% from top. Using 5% forced extreme
        # upward gaze during calibration → large Y coefficients → 20-30% upward bias.
        # 14% keeps the top dot just above the text region, minimising extrapolation.
        mx=int(screen_w*0.05); my=int(screen_h*0.14)
        cx=screen_w//2; cy=screen_h//2
        self.calib_points=[
            (mx,my),(cx,my),(screen_w-mx,my),
            (mx,cy),(cx,cy),(screen_w-mx,cy),
            (mx,screen_h-my),(cx,screen_h-my),(screen_w-mx,screen_h-my),
        ]
        self.calib_done=False; self.calib_point_idx=-1
        self.calib_collecting=False; self.calib_buffer=[]
        self.calib_frames_per_pt=45; self.calib_wait_frames=30
        self.calib_wait_count=0; self.calib_iris=[]
        self._last_landmarks=None; self._last_fw=0; self._last_fh=0

        self._coeff_x = np.zeros(3)
        self._coeff_y = np.zeros(3)

        # Drift correction — one fixation dot shown after the 9-point calibration.
        # Measures the offset between where the model predicts and where the dot
        # actually is, then adds that offset to every subsequent prediction.
        # Handles the posture shift between "staring at calibration dots" and
        # "naturally reading", which survives the feature-engineering fix.
        self._drift_x = 0.0
        self._drift_y = 0.0
        self._drift_target = None    # (tx, ty) screen coords of correction dot
        self._drift_done = False
        self._drift_collecting = False
        self._drift_buffer = []
        self._drift_wait = 0

        self.gaze_history=deque(maxlen=smoothing)
        self.recent_points=deque(maxlen=fixation_frames)
        self.prev_gaze=None

    # ── calibration ──────────────────────────────────────────────────────────

    def start_calibration(self):
        self.calib_point_idx=0; self.calib_collecting=False
        self.calib_wait_count=0; self.calib_buffer=[]
        self.calib_iris=[]; self.calib_done=False
        self._drift_done=False; self._drift_target=None
        self._drift_x=0.0; self._drift_y=0.0
        self.gaze_history.clear()
        print("Active 9-point calibration started")

    def get_current_calib_point(self):
        if self.calib_done or self.calib_point_idx<0: return None
        if self.calib_point_idx>=len(self.calib_points): return None
        return self.calib_points[self.calib_point_idx]

    def get_calib_progress(self):
        return self.calib_point_idx, len(self.calib_points)

    def _get_iris_ratio(self, landmarks, fw, fh):
        def lm(i):
            l=landmarks[i]; return np.array([l.x*fw, l.y*fh])

        li=lm(LEFT_IRIS); ri=lm(RIGHT_IRIS)
        lec=(lm(LEFT_EYE_L)+lm(LEFT_EYE_R))/2.0
        rec=(lm(RIGHT_EYE_L)+lm(RIGHT_EYE_R))/2.0
        eye_mid=(lec+rec)/2.0
        iod=np.linalg.norm(rec-lec)+1e-6
        avg_iris=(li+ri)/2.0

        xr=(avg_iris[0]-eye_mid[0])/iod*100.0
        yr=(avg_iris[1]-eye_mid[1])/iod*100.0

        return float(xr), float(yr)

    def _step_calibration(self, xr, yr):
        if self.calib_point_idx>=len(self.calib_points): return True
        if not self.calib_collecting:
            self.calib_wait_count+=1
            if self.calib_wait_count>=self.calib_wait_frames:
                self.calib_collecting=True; self.calib_buffer=[]; self.calib_wait_count=0
            return False
        self.calib_buffer.append((xr,yr))
        if len(self.calib_buffer)>=self.calib_frames_per_pt:
            xs=[p[0] for p in self.calib_buffer]
            ys=[p[1] for p in self.calib_buffer]
            mx=float(np.median(xs)); my=float(np.median(ys))
            self.calib_iris.append((mx,my))
            pt=self.calib_points[self.calib_point_idx]
            print(f"  Pt{self.calib_point_idx} screen=({pt[0]},{pt[1]}) iris=({mx:.4f},{my:.4f})")
            self.calib_point_idx+=1; self.calib_collecting=False; self.calib_buffer=[]
            if self.calib_point_idx>=len(self.calib_points):
                self._finish_calibration(); return True
        return False

    def _finish_calibration(self):
        iris  = np.array(self.calib_iris)
        screen= np.array(self.calib_points, dtype=float)
        x, y = iris[:,0], iris[:,1]
        A = np.column_stack([x, y, np.ones(len(x))])

        lam = 0.5
        ATA = A.T @ A
        reg = lam * np.eye(3)
        reg[2, 2] = 0.0
        res_x = np.linalg.solve(ATA + reg, A.T @ screen[:, 0])
        res_y = np.linalg.solve(ATA + reg, A.T @ screen[:, 1])
        self._coeff_x = res_x
        self._coeff_y = res_y
        self.calib_done = True

        pred_x = A @ res_x; pred_y = A @ res_y
        errs = np.sqrt((pred_x-screen[:,0])**2 + (pred_y-screen[:,1])**2)
        print(f"Calibration done (ridge affine)!")
        print(f"  iris X range : {x.min():.3f} to {x.max():.3f}  (spread={x.max()-x.min():.3f})")
        print(f"  iris Y range : {y.min():.3f} to {y.max():.3f}  (spread={y.max()-y.min():.3f})")
        print(f"  coeff_x      : {np.round(res_x,2)}")
        print(f"  coeff_y      : {np.round(res_y,2)}")
        print(f"  Residuals    : {errs.round(1)}  mean={errs.mean():.1f}px")

        # Auto-start drift correction at the vertical center of the reading area.
        # 35% from top puts the dot just above the first paragraph on both pages.
        self.start_drift_correction(self.screen_w // 2, int(self.screen_h * 0.35))

    # ── drift correction ──────────────────────────────────────────────────────

    def start_drift_correction(self, tx, ty):
        self._drift_target = (tx, ty)
        self._drift_done = False
        self._drift_collecting = False
        self._drift_buffer = []
        self._drift_wait = 0
        print(f"Drift correction started, target=({tx},{ty})")

    def _step_drift_correction(self, xr, yr):
        if self._drift_done: return True
        if not self._drift_collecting:
            self._drift_wait += 1
            if self._drift_wait >= 30:
                self._drift_collecting = True
                self._drift_buffer = []
            return False
        # Predict without drift offset applied yet
        feat = np.array([xr, yr, 1.0])
        gx_raw = float(np.dot(self._coeff_x, feat))
        gy_raw = float(np.dot(self._coeff_y, feat))
        self._drift_buffer.append((gx_raw, gy_raw))
        if len(self._drift_buffer) >= 45:
            med_x = float(np.median([p[0] for p in self._drift_buffer]))
            med_y = float(np.median([p[1] for p in self._drift_buffer]))
            tx, ty = self._drift_target
            # Clamp to ±25% of screen to reject wild measurements
            self._drift_x = float(np.clip(tx - med_x, -self.screen_w*0.25, self.screen_w*0.25))
            self._drift_y = float(np.clip(ty - med_y, -self.screen_h*0.25, self.screen_h*0.25))
            self._drift_done = True
            print(f"Drift correction: dx={self._drift_x:.0f}px  dy={self._drift_y:.0f}px")
            return True
        return False

    # ── mapping & smoothing ───────────────────────────────────────────────────

    def _map_to_screen(self, xr, yr):
        feat = np.array([xr, yr, 1.0])
        gx   = float(np.dot(self._coeff_x, feat)) + self._drift_x
        gy   = float(np.dot(self._coeff_y, feat)) + self._drift_y
        return (int(np.clip(gx, 0, self.screen_w-1)),
                int(np.clip(gy, 0, self.screen_h-1)))

    def _smooth(self, gx, gy):
        self.gaze_history.append((gx,gy))
        return (int(np.mean([p[0] for p in self.gaze_history])),
                int(np.mean([p[1] for p in self.gaze_history])))

    def _detect_fixation_saccade(self, gx, gy):
        self.recent_points.append((gx,gy))
        fix=False; sac=False
        if len(self.recent_points)>=self.fixation_frames:
            xs=[p[0] for p in self.recent_points]
            ys=[p[1] for p in self.recent_points]
            fix=np.sqrt(np.std(xs)**2+np.std(ys)**2)<self.fixation_threshold
        if self.prev_gaze is not None:
            sac=np.sqrt((gx-self.prev_gaze[0])**2+(gy-self.prev_gaze[1])**2)>self.fixation_threshold*3
        self.prev_gaze=(gx,gy)
        return fix,sac

    def process(self, frame):
        h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self.face_mesh.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            self._last_landmarks=None
            return None,None,False,False
        lms = result.face_landmarks[0]
        self._last_landmarks=lms
        self._last_fw=w; self._last_fh=h
        xr,yr=self._get_iris_ratio(lms,w,h)

        if not self.calib_done:
            if self.calib_point_idx>=0: self._step_calibration(xr,yr)
            return None,None,False,False

        # Drift correction phase — hold until done, return no gaze output
        if not self._drift_done:
            self._step_drift_correction(xr,yr)
            return None,None,False,False

        gx,gy=self._map_to_screen(xr,yr)
        gx,gy=self._smooth(gx,gy)
        fix,sac=self._detect_fixation_saccade(gx,gy)
        return gx,gy,fix,sac

    def draw_debug(self, frame, gx, gy, fix, sac, status=""):
        h,w=frame.shape[:2]

        if self._last_landmarks is not None:
            lms=self._last_landmarks
            fw=self._last_fw; fh=self._last_fh

            def pt(i):
                return (int(lms[i].x*fw), int(lms[i].y*fh))

            for lm in lms:
                x=int(lm.x*fw); y=int(lm.y*fh)
                cv2.circle(frame,(x,y),1,(80,160,255),-1)

            for i in range(len(FACE_OUTLINE)-1):
                cv2.line(frame,pt(FACE_OUTLINE[i]),pt(FACE_OUTLINE[i+1]),(60,200,100),1,cv2.LINE_AA)

            for ep in [LEFT_EYE_PTS,RIGHT_EYE_PTS]:
                for i in range(len(ep)-1):
                    cv2.line(frame,pt(ep[i]),pt(ep[i+1]),(0,230,210),1,cv2.LINE_AA)

            for iris_idx in [LEFT_IRIS, RIGHT_IRIS]:
                ip=pt(iris_idx)
                cv2.circle(frame,ip,5,(0,255,180),-1)
                cv2.circle(frame,ip,11,(0,255,180),1)

        if not self.calib_done:
            color=(0,165,255); label="CALIBRATING"
        elif not self._drift_done:
            color=(212,146,42); label="DRIFT CORRECTION"
        else:
            color=(0,220,80); label="RECORDING"

        cv2.rectangle(frame,(0,0),(380,90),(10,10,10),-1)
        cv2.putText(frame,label,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        if not self.calib_done and self.calib_point_idx>=0:
            idx,total=self.get_calib_progress()
            phase="Hold still..." if self.calib_collecting else "Look at dot"
            cv2.putText(frame,f"Point {idx+1}/{total} — {phase}",
                        (10,55),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)

        if gx is not None:
            cv2.putText(frame,f"Gaze:({gx},{gy})",
                        (10,55),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
            cv2.putText(frame,f"Fix:{fix}  Sac:{sac}",
                        (10,75),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1)
            dx=int(gx/self.screen_w*w)
            dy=int(gy/self.screen_h*h)
            cv2.circle(frame,(dx,dy),10,(0,255,255) if fix else (0,165,255),-1)
            cv2.circle(frame,(dx,dy),14,(255,255,255),1)

        return frame

    def release(self):
        self.face_mesh.close()
