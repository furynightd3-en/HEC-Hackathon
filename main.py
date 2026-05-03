import os
import base64
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from firebase_admin_init import init_firebase, verify_id_token
from face_recog import recognize_face
from classifier import DistilBertOnnxClassifier, ClassifierConfig

load_dotenv()

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]

# ---- Status constants ----
STATUS_PENDING = "pending"
STATUS_DISPLAYED = "displayed"
STATUS_REJECTED = "rejected"
STATUS_ARCHIVED = "archived"
ALLOWED_STATUSES = {STATUS_PENDING, STATUS_DISPLAYED, STATUS_REJECTED, STATUS_ARCHIVED}

app = FastAPI(title="Smart Notice Board Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = init_firebase()

labels = [x.strip() for x in os.getenv("LABELS", "Low,Medium,High").split(",") if x.strip()]
classifier = DistilBertOnnxClassifier(
    ClassifierConfig(
        onnx_model_path=os.getenv("ONNX_MODEL_PATH", "./models/distilbert_priority.onnx"),
        tokenizer_name=os.getenv("TOKENIZER_NAME", "distilbert-base-uncased"),
        labels=labels,
    )
)


class SubmitNoticeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=400)
    image_base64: str = Field(..., description="Webcam JPEG/PNG as base64 string (no data: prefix)")


class SubmitNoticeResponse(BaseModel):
    ok: bool
    doc_id: str
    priority: str
    score: float
    recognized_as: str


def _get_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    if not auth_header:
        return None
    parts = auth_header.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.post("/api/submit_notice", response_model=SubmitNoticeResponse)
def submit_notice(payload: SubmitNoticeRequest, authorization: Optional[str] = Header(default=None)):
    """
    Demo flow:
    - Verify Firebase Auth token (anonymous auth from web)
    - Face recognition on webcam snapshot
    - ONNX classification of text into priority
    - Write the final notice document to Firestore
    """
    token = _get_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <Firebase ID token>")

    try:
        decoded = verify_id_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Firebase ID token")

    uid = decoded.get("uid", "unknown")

    try:
        img_bytes = base64.b64decode(payload.image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image_base64 (must be raw base64)")

    face_result = recognize_face(img_bytes)
    if not face_result.get("ok"):
        raise HTTPException(status_code=403, detail=f"Face not recognized: {face_result.get('reason','unknown')}")

    recognized_as = face_result.get("name", "unknown")
    face_conf = float(face_result.get("confidence", 0.0) or 0.0)

    # Prefer recognized face name for demo, else token display name/email, else "Anonymous"
    display_name = decoded.get("name") or decoded.get("email") or "Anonymous"
    user_name = recognized_as if recognized_as and recognized_as != "unknown" else display_name

    # ---- ONNX prediction ----
    pred = classifier.predict(payload.text)

    raw_priority = pred["label"]
    probs = pred.get("probs", {})
    priority = raw_priority
    score = float(pred["score"])

    # ---- Demo calibration: downgrade weak predictions ----
    # (This makes "kinda Medium" stuff behave more like Low in demo.)
    if priority == "High" and score < 0.75:
        priority = "Medium"
        score = float(probs.get("Medium", score))
    elif priority == "Medium" and score < 0.60:
        priority = "Low"
        score = float(probs.get("Low", score))

    doc = {
        "text": payload.text.strip(),
        "priority": priority,
        "priority_score": score,
        "createdAt": datetime.now(timezone.utc),
        "userId": uid,
        "userName": user_name,
        "recognizedAs": recognized_as,
        "faceConfidence": face_conf,
        "status": STATUS_PENDING,
        "source": "web",
    }

    ref = db.collection("notices").document()
    ref.set(doc)

    return {
        "ok": True,
        "doc_id": ref.id,
        "priority": priority,
        "score": score,
        "recognized_as": recognized_as,
    }