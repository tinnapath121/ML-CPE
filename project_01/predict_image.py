"""
Real-world inference: feed ANY photo (not just the training/test set) and get
a Mask / No-Mask prediction. This is the "applied usage" layer on top of the
trained SVM — it adds automatic face detection (Haar Cascade) so the system
works on a normal photo, not just a pre-cropped face image like the dataset.

Usage:
    python predict_image.py path/to/photo.jpg
"""
import sys
import os
import json

import cv2
import joblib
import numpy as np

OUTPUT_DIR = "outputs"
IMG_SIZE = 100

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_pipeline():
    model = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)
    return model, scaler, classes


def classify_face(face_gray, model, scaler, classes):
    face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    feature = face_resized.reshape(1, -1).astype(np.float32) / 255.0
    feature_scaled = scaler.transform(feature)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_scaled)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx]) * 100
    else:
        idx = int(model.predict(feature_scaled)[0])
        confidence = None

    return classes[idx], confidence


def main(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    model, scaler, classes = load_pipeline()
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        # Fallback: no face detector match (e.g. mask covers too much of the
        # face for Haar Cascade, or the photo is already a tight face crop).
        # Treat the whole image as the "face" region instead of failing.
        print("No face auto-detected by Haar Cascade — classifying the full image instead.")
        h, w = gray.shape[:2]
        faces = [(0, 0, w, h)]

    img_h, img_w = gray.shape[:2]

    for (x, y, w, h) in faces:
        # Haar's box is a tight square around eyes/nose and often crops out
        # the chin/mouth — the exact region the classifier needs to tell
        # mask vs no-mask apart. Pad ~20% on each side (clipped to image
        # bounds) so the crop matches the fuller-face framing the model was
        # trained on.
        pad_x, pad_y = int(w * 0.20), int(h * 0.25)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)

        face_gray = gray[y0:y1, x0:x1]
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
        label, confidence = classify_face(face_gray, model, scaler, classes)

        color = (0, 200, 0) if label == "with_mask" else (0, 0, 220)
        text = f"{label} ({confidence:.0f}%)" if confidence is not None else label

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        # Put the label above the box normally; if there isn't room (box
        # starts near the top of the image), draw it just inside the box
        # instead so the text never gets clipped by the image edge.
        text_y = y - 10 if y - 10 > 22 else y + 24
        cv2.putText(image, text, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        print(f"Face at ({x},{y},{w},{h}) -> {text}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "live_prediction.jpg")
    cv2.imwrite(out_path, image)
    print(f"Saved annotated result: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <path_to_photo.jpg>")
        sys.exit(1)

    main(sys.argv[1])
