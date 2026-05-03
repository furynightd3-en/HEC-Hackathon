import os
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Person name / label (e.g., Sajjal)")
    ap.add_argument("--count", type=int, default=40, help="How many face images to collect")
    ap.add_argument("--cam", type=int, default=0, help="Camera index")
    ap.add_argument("--out", default="face_db", help="Output folder inside backend/")
    args = ap.parse_args()

    out_dir = Path(__file__).parent / args.out / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Camera not opening. Try --cam 1 or close other apps using camera.")

    saved = 0
    last_save_t = 0.0

    print(f"[INFO] Collecting for: {args.name}")
    print("[INFO] Press 's' to save a face, 'q' to quit.")
    print("[TIP] Keep face centered, good lighting, no fast movement.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Hard reject near-black frames (covered camera)
        if float(np.mean(gray)) < 15:
            cv2.putText(frame, "Too dark / camera covered", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        face_crop = None
        if len(faces) > 0:
            # pick largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
            roi = cv2.equalizeHist(roi)
            face_crop = roi

        cv2.putText(frame, f"Saved: {saved}/{args.count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Collect Faces", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break

        if k == ord("s") and face_crop is not None:
            # prevent accidental double-save spam
            if time.time() - last_save_t > 0.25:
                fname = out_dir / f"{args.name}_{int(time.time()*1000)}.jpg"
                cv2.imwrite(str(fname), face_crop)
                saved += 1
                last_save_t = time.time()

        if saved >= args.count:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {saved} images in: {out_dir}")

if __name__ == "__main__":
    main()