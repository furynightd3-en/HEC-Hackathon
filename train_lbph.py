import os
from pathlib import Path
import cv2
import numpy as np

DB_DIR = Path(__file__).parent / "face_db"
OUT_DIR = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUT_DIR / "lbph_face_model.yml"
LABELS_PATH = OUT_DIR / "lbph_labels.txt"

def main():
    if not DB_DIR.exists():
        raise RuntimeError(f"Missing {DB_DIR}. Create face_db/<PersonName>/ with images first.")

    X = []
    y = []
    label2id = {}

    people = sorted([p for p in DB_DIR.iterdir() if p.is_dir()])
    if len(people) < 1:
        raise RuntimeError("No people folders found in face_db/")

    print("[INFO] People:", [p.name for p in people])

    next_id = 0
    for person_dir in people:
        name = person_dir.name
        if name not in label2id:
            label2id[name] = next_id
            next_id += 1
        pid = label2id[name]

        imgs = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
        if len(imgs) < 10:
            print(f"[WARN] {name} has only {len(imgs)} images. Use 30-50 for best results.")

        for img_path in imgs:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            X.append(img)
            y.append(pid)

    if len(X) < 10:
        raise RuntimeError("Not enough training images found.")

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(X, y)
    recognizer.write(str(MODEL_PATH))

    # Save labels in order of id
    id2label = {v: k for k, v in label2id.items()}
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for i in range(len(id2label)):
            f.write(id2label[i] + "\n")

    print("[DONE] Saved:", MODEL_PATH)
    print("[DONE] Labels:", LABELS_PATH)

if __name__ == "__main__":
    main()