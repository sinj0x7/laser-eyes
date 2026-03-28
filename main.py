
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import subprocess
import time
import math
import sys
import os

# ─── Config ──────────────────────────────────────────────────────────────────
VIRTUAL_CAM   = '/dev/video4'
MODEL_PATH    = 'face_landmarker.task'
CHARGE_TIME   = 1.5
FIRE_DURATION = 0.4
SHAKE_MAX     = 12

LASER_CORE  = (255, 255, 255)
LASER_GLOW  = (255,  60, 180)
LASER_OUTER = (180,  10,  80)

LEFT_IRIS  = 468
RIGHT_IRIS = 473

# ─── Init mediapipe ───────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    sys.exit(
        f"❌  Model not found: {MODEL_PATH}\n"
        "    wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = mp_vision.FaceLandmarker.create_from_options(options)

# ─── Init webcam ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("❌  Could not open webcam.")

W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS)) or 30
print(f"📷  Webcam: {W}×{H} @ {FPS}fps")

# ─── Init ffmpeg pipe to virtual cam ─────────────────────────────────────────
ffmpeg_cmd = [
    'ffmpeg', '-loglevel', 'error',
    '-f', 'rawvideo',
    '-pixel_format', 'bgr24',
    '-video_size', f'{W}x{H}',
    '-framerate', str(FPS),
    '-i', 'pipe:0',
    '-f', 'v4l2',
    '-pixel_format', 'yuv420p',
    VIRTUAL_CAM
]

try:
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    print(f"✅  ffmpeg → {VIRTUAL_CAM}")
except FileNotFoundError:
    sys.exit("❌  ffmpeg not found. Run: sudo dnf install -y ffmpeg")

# ─── State ────────────────────────────────────────────────────────────────────
charging     = False
charge_lvl   = 0.0
charge_start = None
firing       = False
fire_start   = None

# ─── Helpers ──────────────────────────────────────────────────────────────────
def to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def draw_glow_line(img, p1, p2):
    tmp = img.copy()
    cv2.line(tmp, p1, p2, LASER_OUTER, 32)
    cv2.addWeighted(tmp, 0.28, img, 0.72, 0, img)
    tmp = img.copy()
    cv2.line(tmp, p1, p2, LASER_GLOW, 14)
    cv2.addWeighted(tmp, 0.55, img, 0.45, 0, img)
    cv2.line(img, p1, p2, LASER_CORE, 2)

def draw_charge_ring(img, center, charge, t):
    x, y = center
    glow_r = int(5 + charge * 22)
    tmp = img.copy()
    cv2.circle(tmp, (x, y), glow_r, LASER_GLOW, -1)
    cv2.addWeighted(tmp, charge * 0.75, img, 1.0 - charge * 0.75, 0, img)
    bright = min(255, int(charge * 255))
    cv2.circle(img, (x, y), 4, (bright, bright, bright), -1)
    pulse  = math.sin(t * 12) * 0.5 + 0.5
    ring_r = int(10 + charge * 18 + pulse * 4 * charge)
    for angle in range(0, int(charge * 360), 5):
        rad = math.radians(angle - 90)
        cv2.circle(img, (x + int(ring_r * math.cos(rad)), y + int(ring_r * math.sin(rad))), 2, LASER_GLOW, -1)

def beam_endpoint(face_cx, face_cy, eye_x, eye_y, w, h):
    dx = eye_x - face_cx or 1
    dy = eye_y - face_cy
    target_x = w if dx > 0 else 0
    target_y = int(eye_y + dy * (target_x - eye_x) / dx)
    return target_x, max(0, min(h, target_y))

def apply_shake(img, intensity):
    if intensity <= 0:
        return img
    dx = np.random.randint(-intensity, intensity + 1)
    dy = np.random.randint(-intensity, intensity + 1)
    return cv2.warpAffine(img, np.float32([[1,0,dx],[0,1,dy]]), (img.shape[1], img.shape[0]))

def apply_edge_pulse(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    overlay = np.zeros_like(img)
    cv2.rectangle(overlay, (0,0), (w,h), LASER_GLOW, max(1, int(60*strength))*2)
    overlay = cv2.GaussianBlur(overlay, (81,81), 0)
    cv2.addWeighted(overlay, strength*0.7, img, 1.0, 0, img)
    return img

# ─── Main loop ────────────────────────────────────────────────────────────────
print("👁  Laser Eyes ready!")
print("    SPACE → charge & fire | Q → quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    t     = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' ') and not charging and not firing:
        charging, charge_start, charge_lvl = True, t, 0.0

    if charging:
        charge_lvl = min((t - charge_start) / CHARGE_TIME, 1.0)
        if charge_lvl >= 1.0:
            charging, firing, fire_start = False, True, t

    fire_progress = 0.0
    if firing:
        fire_progress = (t - fire_start) / FIRE_DURATION
        if fire_progress >= 1.0:
            firing, charge_lvl = False, 0.0

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_img)

    left_eye = right_eye = None
    face_cx, face_cy = w // 2, h // 2
    if result.face_landmarks:
        lms = result.face_landmarks[0]
        left_eye  = to_px(lms[LEFT_IRIS],  w, h)
        right_eye = to_px(lms[RIGHT_IRIS], w, h)
        face_cx, face_cy = to_px(lms[1], w, h)

    if (charging or firing) and left_eye and right_eye:
        ring_charge = 1.0 if firing else charge_lvl
        draw_charge_ring(frame, left_eye,  ring_charge, t)
        draw_charge_ring(frame, right_eye, ring_charge, t)

    if firing and left_eye and right_eye:
        draw_glow_line(frame, left_eye,  beam_endpoint(face_cx, face_cy, *left_eye,  w, h))
        draw_glow_line(frame, right_eye, beam_endpoint(face_cx, face_cy, *right_eye, w, h))
        for eye in (left_eye, right_eye):
            cv2.circle(frame, eye, 10, LASER_CORE, -1)
            cv2.circle(frame, eye, int(18 + math.sin(t*25)*4), LASER_GLOW, 2)
        frame = apply_shake(frame, int(SHAKE_MAX * max(0.0, 1.0 - fire_progress * 2)))
        frame = apply_edge_pulse(frame, max(0.0, 1.0 - fire_progress))

    if charging:
        cv2.rectangle(frame, (0, h-10), (int(w*charge_lvl), h), LASER_GLOW, -1)
        cv2.putText(frame, f"CHARGING {int(charge_lvl*100)}%", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LASER_GLOW, 2)
    elif firing:
        cv2.putText(frame, ">>> FIRING <<<", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, LASER_CORE, 3)
    else:
        cv2.putText(frame, "[SPACE] to charge", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)

    cv2.imshow("👁  Laser Eyes  —  Q to quit", frame)

    try:
        ffmpeg.stdin.write(frame.tobytes())
    except BrokenPipeError:
        print("⚠️  ffmpeg pipe broke, restarting...")
        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

cap.release()
ffmpeg.stdin.close()
ffmpeg.wait()
cv2.destroyAllWindows()
print("✅  Done.")