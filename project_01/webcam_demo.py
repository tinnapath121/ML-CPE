"""
Live webcam demo: real-time Mask / No-Mask detection using the trained SVM.

*** ต้องรันบนเครื่องที่มีกล้องเว็บแคมจริง (โน้ตบุ๊ก/พีซีของนาย) ***
ไม่สามารถรันบน remote sandbox / cloud ได้ เพราะไม่มีกล้อง

Usage:
    python webcam_demo.py
    (กด 'q' เพื่อออก)
"""
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


def main(camera_index=1):
    model, scaler, classes = load_pipeline()
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("เปิดกล้องไม่ได้ ลองเปลี่ยน camera_index เป็น 1, 2, ...")
        return

    print("กด 'q' เพื่อออก")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        img_h, img_w = gray.shape[:2]

        for (x, y, w, h) in faces:
            # Same padding as predict_image.py: Haar's box is a tight square
            # around eyes/nose and crops out the chin/mouth — the exact
            # region the model needs to tell mask vs no-mask apart.
            pad_x, pad_y = int(w * 0.20), int(h * 0.25)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)

            face_gray = gray[y0:y1, x0:x1]
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            label, confidence = classify_face(face_gray, model, scaler, classes)

            color = (0, 200, 0) if label == "with_mask" else (0, 0, 220)
            text = f"{label} ({confidence:.0f}%)" if confidence is not None else label

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text_y = y - 10 if y - 10 > 22 else y + 24
            cv2.putText(frame, text, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("SVM Mask Detection - Live Demo (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        input("\n(กด Enter เพื่อปิดหน้าต่างนี้)")
