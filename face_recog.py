from typing import Dict, Any
from pathlib import Path

import cv2
import numpy as np

# ---- Paths ----
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "lbph_face_model.yml"
LABELS_PATH = BASE_DIR / "models" / "lbph_labels.txt"

# ---- Face detector ----
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---- Load recognizer + labels once ----
if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    raise RuntimeError(
        "LBPH model not found. Run: python train_lbph.py "
        "to create models/lbph_face_model.yml and models/lbph_labels.txt"
    )

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    ID2LABEL = [line.strip() for line in f if line.strip()]

RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()
RECOGNIZER.read(str(MODEL_PATH))

# ---- Threshold (tune if needed) ----
# Lower distance = better match. Typical decent range ~30-80 depending on data.
UNKNOWN_THRESHOLD = 65.0


def recognize_face(image_bytes: bytes) -> Dict[str, Any]:
    if not image_bytes or len(image_bytes) < 1000:
        return {"ok": False, "reason": "empty_image"}

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "reason": "decode_failed"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Camera covered / too dark
    if float(np.mean(gray)) < 15.0:
        return {"ok": False, "reason": "frame_too_dark"}

    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return {"ok": False, "reason": "no_face"}

    # Use largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (200, 200))
    roi = cv2.equalizeHist(roi)

    pred_id, dist = RECOGNIZER.predict(roi)
    dist = float(dist)

    if pred_id < 0 or pred_id >= len(ID2LABEL):
        return {"ok": False, "reason": "unknown_face"}

    name = ID2LABEL[pred_id]

    # Reject unknowns by distance
    if dist > UNKNOWN_THRESHOLD:
        return {"ok": False, "reason": "unknown_face", "distance": dist}

    # Convert distance to a simple confidence-ish score (0..1)
    confidence = float(max(0.0, min(0.99, 1.0 - (dist / UNKNOWN_THRESHOLD))))

    return {"ok": True, "name": name, "confidence": confidence, "distance": dist}