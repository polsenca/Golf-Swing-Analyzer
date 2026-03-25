#!/usr/bin/env python3
"""
Golf Swing Analyzer
Uses MediaPipe Pose Landmarker (Tasks API) to detect body landmarks and
analyze golf swing mechanics frame-by-frame.

Usage:
    python golf_swing_analyzer.py swing.mp4
    python golf_swing_analyzer.py swing.mp4 --output annotated.mp4 --report results.json
    python golf_swing_analyzer.py swing.mp4 --handed left --model heavy
"""

import argparse
import json
import math
import sys
import urllib.request
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Tasks API imports (mediapipe >= 0.10)
# ---------------------------------------------------------------------------
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
import mediapipe as mp

# ---------------------------------------------------------------------------
# BlazePose 33-point landmark indices (same in new Tasks API)
# ---------------------------------------------------------------------------
NOSE            = 0
L_EYE_INNER     = 1;  L_EYE = 2;  L_EYE_OUTER = 3
R_EYE_INNER     = 4;  R_EYE = 5;  R_EYE_OUTER = 6
L_EAR           = 7;  R_EAR = 8
L_SHOULDER      = 11; R_SHOULDER = 12
L_ELBOW         = 13; R_ELBOW    = 14
L_WRIST         = 15; R_WRIST    = 16
L_HIP           = 23; R_HIP      = 24
L_KNEE          = 25; R_KNEE     = 26
L_ANKLE         = 27; R_ANKLE    = 28
L_HEEL          = 29; R_HEEL     = 30
L_FOOT          = 31; R_FOOT     = 32

# Skeleton connections for drawing
POSE_CONNECTIONS = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),    (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP),      (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT),
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
]

# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------

MODEL_URLS = {
    "lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

def ensure_model(model_variant: str = "full") -> str:
    """Download the MediaPipe pose landmarker .task model if not already cached."""
    model_dir  = Path.home() / ".cache" / "golf_swing_analyzer"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"pose_landmarker_{model_variant}.task"

    if model_path.exists():
        print(f"Using cached model: {model_path}")
        return str(model_path)

    url = MODEL_URLS[model_variant]
    print(f"Downloading pose model ({model_variant}) from MediaPipe...")
    print(f"  URL: {url}")
    urllib.request.urlretrieve(url, model_path)
    print(f"  Saved to: {model_path}")
    return str(model_path)


# ---------------------------------------------------------------------------
# Swing phase enum
# ---------------------------------------------------------------------------

class SwingPhase(Enum):
    UNKNOWN        = auto()
    ADDRESS        = auto()
    TAKEAWAY       = auto()
    BACKSWING      = auto()
    TOP            = auto()
    DOWNSWING      = auto()
    IMPACT         = auto()
    FOLLOW_THROUGH = auto()
    FINISH         = auto()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b (rays b→a and b→c), in degrees [0, 180]."""
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos_val, -1.0, 1.0)))


def line_angle_to_horizontal(p1: np.ndarray, p2: np.ndarray) -> float:
    """Signed angle of line p1→p2 relative to horizontal, in degrees."""
    d = p2 - p1
    return math.degrees(math.atan2(-d[1], d[0]))  # flip y because image y-axis is down


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2.0


def lm_xy(landmarks: list, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=float)


def lm_vis(landmarks: list, idx: int) -> float:
    return getattr(landmarks[idx], "visibility", 0.0) or 0.0


# ---------------------------------------------------------------------------
# Biomechanics calculator
# ---------------------------------------------------------------------------

class BiomechanicsCalculator:
    """Computes per-frame biomechanical measurements from landmark lists."""

    def __init__(self, handed: str = "right"):
        if handed == "right":
            # right-handed: lead side = left body side
            self.lead_sh    = L_SHOULDER; self.trail_sh    = R_SHOULDER
            self.lead_el    = L_ELBOW;    self.trail_el    = R_ELBOW
            self.lead_wr    = L_WRIST;    self.trail_wr    = R_WRIST
            self.lead_hip   = L_HIP;      self.trail_hip   = R_HIP
            self.lead_knee  = L_KNEE;     self.trail_knee  = R_KNEE
            self.lead_ankle = L_ANKLE;    self.trail_ankle = R_ANKLE
        else:
            self.lead_sh    = R_SHOULDER; self.trail_sh    = L_SHOULDER
            self.lead_el    = R_ELBOW;    self.trail_el    = L_ELBOW
            self.lead_wr    = R_WRIST;    self.trail_wr    = L_WRIST
            self.lead_hip   = R_HIP;      self.trail_hip   = L_HIP
            self.lead_knee  = R_KNEE;     self.trail_knee  = L_KNEE
            self.lead_ankle = R_ANKLE;    self.trail_ankle = L_ANKLE

        self.address_head: Optional[np.ndarray] = None
        self.address_shoulder_angle: Optional[float] = None
        self.address_hip_angle: Optional[float] = None
        # Rolling 3-frame buffer: median of last 3 ADDRESS frames gives a stable
        # baseline even if the final address frame has already started rotating.
        self._addr_sh_buf: deque = deque(maxlen=3)
        self._addr_hip_buf: deque = deque(maxlen=3)

    @staticmethod
    def _angle_delta(current: float, baseline: float) -> float:
        """Signed angular difference handling ±180° wraparound."""
        delta = current - baseline
        return ((delta + 180.0) % 360.0) - 180.0

    def compute(self, landmarks: list, w: int, h: int, phase: SwingPhase) -> dict:
        def xy(idx):
            return lm_xy(landmarks, idx, w, h)

        l_sh  = xy(self.lead_sh);    t_sh  = xy(self.trail_sh)
        l_el  = xy(self.lead_el);    t_el  = xy(self.trail_el)
        l_wr  = xy(self.lead_wr);    t_wr  = xy(self.trail_wr)
        l_hip = xy(self.lead_hip);   t_hip = xy(self.trail_hip)
        l_kn  = xy(self.lead_knee);  t_kn  = xy(self.trail_knee)
        l_an  = xy(self.lead_ankle); t_an  = xy(self.trail_ankle)
        nose  = xy(NOSE)

        sh_mid  = midpoint(l_sh, t_sh)
        hip_mid = midpoint(l_hip, t_hip)

        # --- Spine tilt: angle of hip_mid→sh_mid from vertical (0° = perfectly upright) ---
        spine_vec = sh_mid - hip_mid
        vert      = np.array([0.0, -1.0])
        cos_s     = np.dot(spine_vec, vert) / (np.linalg.norm(spine_vec) + 1e-9)
        spine_angle = math.degrees(math.acos(np.clip(cos_s, -1.0, 1.0)))

        # --- Raw line angles (noisy in face-on view due to MediaPipe label orientation) ---
        shoulder_angle_raw = line_angle_to_horizontal(l_sh, t_sh)
        hip_angle_raw      = line_angle_to_horizontal(l_hip, t_hip)

        # --- Knee flex (joint angle; 180° = straight leg) ---
        lead_knee_flex  = angle_between(l_hip, l_kn, l_an)
        trail_knee_flex = angle_between(t_hip, t_kn, t_an)

        # --- Elbow angles ---
        lead_elbow  = angle_between(l_sh, l_el, l_wr)
        trail_elbow = angle_between(t_sh, t_el, t_wr)

        # --- Wrist height (average, normalised 0=top, 1=bottom) ---
        wrist_height = (l_wr[1] + t_wr[1]) / 2.0 / h

        # --- Capture address baseline for angle deltas ---
        # Buffer the last 3 ADDRESS frames and use their median so that a single
        # rotating frame at the end of ADDRESS doesn't corrupt the baseline.
        if phase == SwingPhase.ADDRESS:
            # At true static address the shoulder line is near-horizontal (angle ≈ 0°).
            # Early pre-shot frames often have high-magnitude angles (-90° to -120°) due
            # to low landmark visibility. Accept only frames where |angle| < 20° so the
            # buffer settles on the genuine static setup position, not the waggle noise.
            if abs(shoulder_angle_raw) < 20.0:
                self._addr_sh_buf.append(shoulder_angle_raw)
            # Hip: near ±180° at address in face-on view, so use stability within ±15°
            # of the last accepted hip angle rather than an absolute gate.
            if not self._addr_hip_buf or abs(self._angle_delta(hip_angle_raw, self._addr_hip_buf[-1])) < 15.0:
                self._addr_hip_buf.append(hip_angle_raw)
            if len(self._addr_sh_buf) >= 2:
                self.address_shoulder_angle = sorted(self._addr_sh_buf)[len(self._addr_sh_buf) // 2]
            if len(self._addr_hip_buf) >= 2:
                self.address_hip_angle = sorted(self._addr_hip_buf)[len(self._addr_hip_buf) // 2]
        if self.address_shoulder_angle is None:
            self.address_shoulder_angle = shoulder_angle_raw
            self.address_hip_angle      = hip_angle_raw

        # Signed deltas from address: positive = lead shoulder dropping (backswing turn),
        # negative = lead shoulder rising (follow-through). Handles ±180° wraparound.
        shoulder_turn = self._angle_delta(shoulder_angle_raw, self.address_shoulder_angle)
        hip_rotation  = self._angle_delta(hip_angle_raw,      self.address_hip_angle)

        # --- Head position drift from address ---
        if phase == SwingPhase.ADDRESS or self.address_head is None:
            self.address_head = nose.copy()
        drift_x = (nose[0] - self.address_head[0]) / w
        drift_y = (nose[1] - self.address_head[1]) / h

        return dict(
            spine_angle       = round(spine_angle,       1),
            hip_rotation      = round(hip_rotation,      1),
            shoulder_turn     = round(shoulder_turn,     1),
            lead_knee_flex    = round(lead_knee_flex,    1),
            trail_knee_flex   = round(trail_knee_flex,   1),
            lead_elbow_angle  = round(lead_elbow,        1),
            trail_elbow_angle = round(trail_elbow,       1),
            wrist_height      = round(wrist_height,      3),
            head_drift_x      = round(drift_x,           3),
            head_drift_y      = round(drift_y,           3),
        )


# ---------------------------------------------------------------------------
# Phase detector (state machine)
# ---------------------------------------------------------------------------

class PhaseDetector:
    """
    Detects swing phase from a rolling window of wrist_height and shoulder_turn.
    wrist_height is normalised [0=top of frame, 1=bottom].

    State flow:
        UNKNOWN → ADDRESS → TAKEAWAY → BACKSWING → TOP →
        DOWNSWING → IMPACT → FOLLOW_THROUGH → FINISH
    """

    def __init__(self, fps: float):
        self.fps            = max(fps, 1.0)
        self.phase          = SwingPhase.UNKNOWN
        self._win           = deque(maxlen=3)   # 3-frame window for fast response
        self._addr_wrist_y: Optional[float] = None
        self._top_wrist_y:  Optional[float] = None  # instantaneous minimum

    def update(self, metrics: dict) -> SwingPhase:
        wh  = metrics.get("wrist_height", 0.6)
        sht = abs(metrics.get("shoulder_turn", 0.0))
        self._win.append(wh)

        if len(self._win) < 2:
            return self.phase

        avg_wh = sum(self._win) / len(self._win)
        curr   = self._win[-1]   # most recent instantaneous value

        if self.phase == SwingPhase.UNKNOWN:
            # Rely on shoulder_turn delta being small (near 0 at address).
            # shoulder_turn is 0 at address (it's a delta), so avg_sht will be
            # near 0 only when the baseline has been captured at ADDRESS phase.
            # Use wrist stability + reasonable height instead.
            wh_range = max(self._win) - min(self._win)
            if 0.40 < avg_wh < 0.82 and wh_range < 0.015:
                self.phase         = SwingPhase.ADDRESS
                self._addr_wrist_y = avg_wh
            return self.phase

        if self.phase == SwingPhase.ADDRESS:
            addr = self._addr_wrist_y or 0.65
            if curr < addr - 0.025:          # wrists lifting: takeaway begun
                self.phase = SwingPhase.TAKEAWAY
            return self.phase

        if self.phase == SwingPhase.TAKEAWAY:
            if curr < 0.48:
                self.phase = SwingPhase.BACKSWING
            return self.phase

        if self.phase == SwingPhase.BACKSWING:
            # Track the true instantaneous minimum wrist height
            if self._top_wrist_y is None or curr < self._top_wrist_y:
                self._top_wrist_y = curr
            # TOP: wrists have risen at least 2.5% above their tracked minimum
            if curr > (self._top_wrist_y or 0.0) + 0.025:
                self.phase = SwingPhase.TOP
            return self.phase

        if self.phase == SwingPhase.TOP:
            # DOWNSWING: wrists back down significantly from the peak
            if curr > (self._top_wrist_y or 0.0) + 0.08:
                self.phase = SwingPhase.DOWNSWING
            return self.phase

        if self.phase == SwingPhase.DOWNSWING:
            addr = self._addr_wrist_y or 0.65
            if curr > addr - 0.04:
                self.phase = SwingPhase.IMPACT
            return self.phase

        if self.phase == SwingPhase.IMPACT:
            if curr < 0.48:
                self.phase = SwingPhase.FOLLOW_THROUGH
            return self.phase

        if self.phase == SwingPhase.FOLLOW_THROUGH:
            if curr < 0.32:
                self.phase = SwingPhase.FINISH
            return self.phase

        return self.phase


# ---------------------------------------------------------------------------
# Feedback engine
# ---------------------------------------------------------------------------

FEEDBACK_MSGS = {
    "addr_spine_low":    "Spine too upright at address — add more forward tilt by hinging from the hips.",
    "addr_spine_high":   "Too much spine tilt at address — stand taller and hinge less aggressively.",
    "top_turn_low":      "Shoulder turn is short at the top — rotate your lead shoulder under your chin for a full backswing.",
    "top_turn_high":     "Shoulder turn is overswinging — aim for a controlled, compact backswing.",
    "imp_spine_low":     "Early extension at impact (lost spine angle) — hold your forward tilt through the ball.",
    "imp_hip_low":       "Hips not clearing enough at impact — rotate your lead hip toward the target aggressively.",
    "imp_hip_high":      "Excessive hip slide at impact — rotate hips without lateral sway.",
    "imp_elbow_low":     "Lead arm bending at impact — keep a firm, extended lead arm through the hitting zone.",
    "knee_flex_low":     "Lead knee too straight at address — add a slight flex (≈15–20°) for an athletic base.",
    "knee_flex_high":    "Excessive knee bend at address — straighten slightly for better balance.",
    "trail_knee_low":    "Trail knee losing flex in backswing — maintain trail knee flex to coil properly.",
    "head_x_target":     "Head drifting toward target during swing — keep your head still and behind the ball through impact.",
    "head_x_away":       "Head swaying away from target — minimize lateral movement and stay centered.",
    "head_y_drop":       "Head dropping during swing — maintain your height for consistent ball striking.",
    "head_y_rise":       "Head rising during swing — stay down through impact and don't come up early.",
}


class FeedbackEngine:
    def __init__(self):
        self._data: dict[SwingPhase, list[dict]] = {}

    def record(self, phase: SwingPhase, metrics: dict):
        self._data.setdefault(phase, []).append(metrics)

    def _avg(self, phase: SwingPhase, key: str) -> Optional[float]:
        frames = self._data.get(phase, [])
        vals   = [f[key] for f in frames if f.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    def generate(self) -> list[dict]:
        items = []

        def check(key, val, lo, hi, msg_lo, msg_hi, sev="warning"):
            if val is None:
                return
            if val < lo:
                items.append({"key": key, "value": round(val, 2), "severity": sev, "message": FEEDBACK_MSGS[msg_lo]})
            elif val > hi:
                items.append({"key": key, "value": round(val, 2), "severity": sev, "message": FEEDBACK_MSGS[msg_hi]})

        # ADDRESS: spine tilt (forward bend, 25–45° = good)
        check("addr_spine",    self._avg(SwingPhase.ADDRESS, "spine_angle"),
              25, 45, "addr_spine_low", "addr_spine_high")

        # ADDRESS: lead knee flex (joint angle 140–165° = 15–40° of bend = good)
        check("lead_knee",     self._avg(SwingPhase.ADDRESS, "lead_knee_flex"),
              140, 165, "knee_flex_high", "knee_flex_low")

        check("trail_knee",    self._avg(SwingPhase.ADDRESS, "trail_knee_flex"),
              140, 165, "trail_knee_low", "knee_flex_low")

        # TOP: shoulder turn delta from address (face-on 2D tilt; 20–60° = good)
        # The 2D shoulder tilt at the top correlates with actual 3D shoulder rotation.
        # A full 90° shoulder turn in 3D typically shows as 30–55° tilt in a face-on view.
        sht_top = self._avg(SwingPhase.TOP, "shoulder_turn")
        check("top_shoulder_turn", abs(sht_top) if sht_top is not None else None,
              20, 65, "top_turn_low", "top_turn_high")

        # IMPACT: spine tilt held (20–40° = good)
        check("imp_spine",     self._avg(SwingPhase.IMPACT, "spine_angle"),
              20, 40, "imp_spine_low", "addr_spine_high")

        # IMPACT: hip rotation delta from address (5–80° = adequate clearing).
        # Note: hip rotation is noisy at impact due to fast motion; use wide bounds.
        hip_imp = self._avg(SwingPhase.IMPACT, "hip_rotation")
        check("imp_hip",       abs(hip_imp) if hip_imp is not None else None,
              5, 80, "imp_hip_low", "imp_hip_high")

        # IMPACT: lead elbow (155–185° = straight)
        check("imp_elbow",     self._avg(SwingPhase.IMPACT, "lead_elbow_angle"),
              155, 185, "imp_elbow_low", "imp_elbow_low")

        # HEAD STABILITY across mid-swing phases
        drift_xs, drift_ys = [], []
        for ph in [SwingPhase.BACKSWING, SwingPhase.TOP, SwingPhase.DOWNSWING,
                   SwingPhase.IMPACT, SwingPhase.FOLLOW_THROUGH]:
            dx = self._avg(ph, "head_drift_x")
            dy = self._avg(ph, "head_drift_y")
            if dx is not None: drift_xs.append(dx)
            if dy is not None: drift_ys.append(dy)

        if drift_xs:
            worst_x = max(drift_xs, key=abs)
            if abs(worst_x) > 0.06:
                key = "head_x_target" if worst_x > 0 else "head_x_away"
                items.append({"key": "head_drift_x", "value": round(worst_x, 3),
                               "severity": "warning", "message": FEEDBACK_MSGS[key]})

        if drift_ys:
            worst_y = max(drift_ys, key=abs)
            if abs(worst_y) > 0.06:
                key = "head_y_drop" if worst_y > 0 else "head_y_rise"
                items.append({"key": "head_drift_y", "value": round(worst_y, 3),
                               "severity": "tip", "message": FEEDBACK_MSGS[key]})

        if not items:
            items.append({"key": "overall", "value": None, "severity": "ok",
                           "message": "Great swing mechanics! Technique looks consistent."})

        return items

    def phase_averages(self) -> dict:
        keys = ["spine_angle", "hip_rotation", "shoulder_turn", "lead_knee_flex",
                "trail_knee_flex", "lead_elbow_angle", "trail_elbow_angle",
                "wrist_height", "head_drift_x", "head_drift_y"]
        out = {}
        for phase, frames in self._data.items():
            avgs = {"frame_count": len(frames)}
            for k in keys:
                vals = [f[k] for f in frames if f.get(k) is not None]
                avgs[k] = round(sum(vals) / len(vals), 2) if vals else None
            out[phase.name] = avgs
        return out


# ---------------------------------------------------------------------------
# Frame annotator
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    SwingPhase.UNKNOWN:        (140, 140, 140),
    SwingPhase.ADDRESS:        (50,  220, 50),
    SwingPhase.TAKEAWAY:       (50,  200, 160),
    SwingPhase.BACKSWING:      (50,  140, 255),
    SwingPhase.TOP:            (30,  60,  255),
    SwingPhase.DOWNSWING:      (255, 140, 30),
    SwingPhase.IMPACT:         (255, 50,  50),
    SwingPhase.FOLLOW_THROUGH: (200, 30,  220),
    SwingPhase.FINISH:         (160, 30,  255),
}


def draw_skeleton(frame: np.ndarray, landmarks: list, w: int, h: int):
    pts = [lm_xy(landmarks, i, w, h) for i in range(len(landmarks))]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame,
                     (int(pts[a][0]), int(pts[a][1])),
                     (int(pts[b][0]), int(pts[b][1])),
                     (100, 230, 100), 2)
    for i, pt in enumerate(pts):
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), -1)


def annotate_frame(frame: np.ndarray, landmarks: Optional[list],
                   metrics: dict, phase: SwingPhase,
                   frame_idx: int, fps: float) -> np.ndarray:
    out  = frame.copy()
    h, w = out.shape[:2]

    if landmarks:
        draw_skeleton(out, landmarks, w, h)

    color = PHASE_COLORS.get(phase, (180, 180, 180))

    # Top banner
    cv2.rectangle(out, (0, 0), (w, 44), (20, 20, 20), -1)
    label = phase.name.replace("_", " ")
    cv2.putText(out, f"Phase: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    t_str = f"t={frame_idx/fps:.2f}s  f={frame_idx}"
    cv2.putText(out, t_str, (w - 230, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Metrics panel
    if metrics:
        lines = [
            f"Spine tilt    : {metrics.get('spine_angle', 0):.1f} deg",
            f"Hip rotation  : {metrics.get('hip_rotation', 0):.1f} deg",
            f"Shoulder turn : {metrics.get('shoulder_turn', 0):.1f} deg",
            f"Lead knee     : {metrics.get('lead_knee_flex', 0):.1f} deg",
            f"Trail knee    : {metrics.get('trail_knee_flex', 0):.1f} deg",
            f"Lead elbow    : {metrics.get('lead_elbow_angle', 0):.1f} deg",
            f"Head drift    : ({metrics.get('head_drift_x', 0)*100:.1f}%, {metrics.get('head_drift_y', 0)*100:.1f}%)",
        ]
        lh  = 22
        bh  = len(lines) * lh + 12
        cv2.rectangle(out, (6, 50), (296, 50 + bh), (20, 20, 20), -1)
        for i, txt in enumerate(lines):
            cv2.putText(out, txt, (12, 68 + i * lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)

    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(feedback: list[dict], phase_avgs: dict,
                 video: str, fps: float, total_frames: int) -> dict:
    return {
        "source_video": str(video),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(total_frames / fps, 2) if fps else 0,
        "phase_averages": phase_avgs,
        "feedback": feedback,
        "summary": {sev: [x for x in feedback if x["severity"] == sev]
                    for sev in ("error", "warning", "tip", "ok")},
    }


def print_report(report: dict):
    div = "=" * 62
    sub = "-" * 58

    print(f"\n{div}")
    print("  GOLF SWING ANALYSIS REPORT")
    print(div)
    print(f"  Video    : {report['source_video']}")
    print(f"  Duration : {report['duration_seconds']:.2f}s  "
          f"({report['total_frames']} frames @ {report['fps']:.1f} fps)\n")

    print("  PER-PHASE AVERAGES")
    print(f"  {sub}")
    metric_labels = {
        "spine_angle":      "Spine tilt (deg)",
        "shoulder_turn":    "Shoulder turn (deg)",
        "hip_rotation":     "Hip rotation (deg)",
        "lead_knee_flex":   "Lead knee flex (deg)",
        "trail_knee_flex":  "Trail knee flex (deg)",
        "lead_elbow_angle": "Lead elbow (deg)",
        "head_drift_x":     "Head drift X (%)",
        "head_drift_y":     "Head drift Y (%)",
    }
    order = ["ADDRESS","TAKEAWAY","BACKSWING","TOP","DOWNSWING","IMPACT","FOLLOW_THROUGH","FINISH"]
    for ph in order:
        avgs = report["phase_averages"].get(ph)
        if not avgs:
            continue
        print(f"\n  [{ph}]  ({avgs.get('frame_count', 0)} frames)")
        for k, lbl in metric_labels.items():
            v = avgs.get(k)
            if v is not None:
                val_str = f"{v*100:.1f}%" if "drift" in k else f"{v:.1f}"
                print(f"    {lbl:<30} {val_str}")

    print(f"\n  FEEDBACK")
    print(f"  {sub}")
    icons = {"error": "✖ ERROR", "warning": "⚠ WARNING", "tip": "◎ TIP", "ok": "✔ OK"}
    for sev in ("error", "warning", "tip", "ok"):
        for item in report["summary"][sev]:
            tag  = icons[sev]
            val  = f" [{item['value']}]" if item.get("value") is not None else ""
            print(f"\n  {tag}{val}")
            words, line, wrapped = item["message"].split(), "", []
            for word in words:
                if len(line) + len(word) + 1 > 68:
                    wrapped.append(line); line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                wrapped.append(line)
            for ln in wrapped:
                print(f"    {ln}")

    print(f"\n{div}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def analyze(video_path: str, output_path: Optional[str], report_path: Optional[str],
            handed: str = "right", model_variant: str = "full", skip: int = 0,
            progress_callback=None):
    """
    progress_callback(current_frame, total_frames, phase_name) — called every 30 frames.
    Raises ValueError instead of sys.exit so callers (e.g. Streamlit) can catch it.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: '{video_path}'")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo : {video_path}")
    print(f"  {W}x{H}  {fps:.1f} fps  ~{total_frames} frames")
    print(f"  Handedness: {handed}-handed golfer\n")

    model_path = ensure_model(model_variant)

    options = PoseLandmarkerOptions(
        base_options       = BaseOptions(model_asset_path=model_path),
        running_mode       = VisionTaskRunningMode.VIDEO,
        num_poses          = 1,
        min_pose_detection_confidence = 0.2,
        min_pose_presence_confidence  = 0.2,
        min_tracking_confidence       = 0.2,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        print(f"Annotated video → {output_path}")

    bio        = BiomechanicsCalculator(handed=handed)
    detector   = PhaseDetector(fps=fps)
    feedback_e = FeedbackEngine()

    frame_idx = 0
    processed = 0
    print("Processing frames…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Optional frame-skip for speed
        if skip > 0 and (frame_idx % (skip + 1)) != 1:
            if writer:
                writer.write(frame)
            continue

        # MediaPipe Tasks API requires an mp.Image
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(frame_idx * 1000 / fps)  # milliseconds

        result: PoseLandmarkerResult = landmarker.detect_for_video(mp_image, timestamp)

        landmarks = None
        metrics: dict = {}
        phase = detector.phase  # carry over last known phase

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]  # first (only) pose

            # Check visibility of key joints
            key_vis = [lm_vis(landmarks, i) for i in
                       [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_WRIST, R_WRIST]]
            if min(key_vis) > 0.05:
                metrics = bio.compute(landmarks, W, H, phase)
                phase   = detector.update(metrics)
                feedback_e.record(phase, metrics)

        annotated = annotate_frame(frame, landmarks, metrics, phase, frame_idx, fps)
        if writer:
            writer.write(annotated)

        processed += 1
        if processed % 30 == 0:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"  [{pct:5.1f}%] frame {frame_idx:5d}  phase: {phase.name}")
            if progress_callback:
                progress_callback(frame_idx, total_frames, phase.name)

    cap.release()
    if writer:
        writer.release()
    landmarker.close()

    print(f"\nDone. Processed {processed} frames.")

    feedback    = feedback_e.generate()
    phase_avgs  = feedback_e.phase_averages()
    report      = build_report(feedback, phase_avgs, video_path, fps, frame_idx)

    print_report(report)

    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report saved → {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze golf swing mechanics from a video using MediaPipe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python golf_swing_analyzer.py swing.mp4
  python golf_swing_analyzer.py swing.mp4 -o annotated.mp4 -r report.json
  python golf_swing_analyzer.py swing.mp4 --handed left --model heavy
  python golf_swing_analyzer.py swing.mp4 --skip 1   # process every other frame (faster)
        """,
    )
    parser.add_argument("video",   help="Input video file")
    parser.add_argument("-o", "--output", default=None, help="Annotated output video path")
    parser.add_argument("-r", "--report", default=None, help="JSON report output path")
    parser.add_argument("--handed", choices=["right", "left"], default="right",
                        help="Golfer handedness (default: right)")
    parser.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                        help="MediaPipe model variant — lite=fast, heavy=most accurate (default: full)")
    parser.add_argument("--skip", type=int, default=0, metavar="N",
                        help="Skip N frames between processed frames (0=all, 1=every other, …)")
    args = parser.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"Error: video file not found: {args.video}")

    analyze(
        video_path   = args.video,
        output_path  = args.output,
        report_path  = args.report,
        handed       = args.handed,
        model_variant= args.model,
        skip         = args.skip,
    )


if __name__ == "__main__":
    main()
