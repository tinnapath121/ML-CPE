"""
Quick visual test of the trained SVM.

เงื่อนไข:
  - ถ้ามีรูปภาพอยู่ใน `test_photos/`  -> ใช้รูปเหล่านั้นทดสอบก่อน (ผ่าน Haar Cascade
    auto-crop เหมือน predict_image.py จริง ไม่ใช่ภาพที่ preprocess ไว้แล้วเหมือน test set)
  - ถ้าโฟลเดอร์ `test_photos/` ว่างเปล่า/ไม่มี -> สุ่มภาพจาก test set ของ dataset
    (X_test.npy) มาทดสอบแทน เหมือนพฤติกรรมเดิม

วิธีใช้:
    python test_svm.py
    (แค่ก็อปรูปที่อยากลองใส่ folder `test_photos/` แล้วรันใหม่ — ไม่ต้องพิมพ์ argument)

ถ้าตั้งชื่อไฟล์ขึ้นต้นด้วยชื่อคลาส เช่น `with_mask_1.jpg` หรือ `without_mask_test.png`
สคริปต์จะรู้ ground truth จากชื่อไฟล์เองและขึ้นสีเขียว/แดงบอกถูก-ผิดให้ด้วย
(ถ้าตั้งชื่ออย่างอื่น จะโชว์แค่ผลทำนาย ไม่บอกถูก/ผิด เพราะไม่รู้ ground truth จริง)
"""
import glob
import json
import os

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs"
TEST_PHOTOS_DIR = "test_photos"
IMG_SIZE = 100
N_SAMPLES = 4
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")
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


def detect_and_crop_face(gray):
    """Same auto-crop logic as predict_image.py: Haar Cascade + ~20-25% padding.
    Returns (face_gray, face_found)."""
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return gray, False

    img_h, img_w = gray.shape[:2]
    x, y, w, h = faces[0]
    pad_x, pad_y = int(w * 0.20), int(h * 0.25)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)

    return gray[y0:y1, x0:x1], True


def guess_ground_truth(filename, classes):
    """If the filename starts with a class name, use it as ground truth."""
    name = os.path.basename(filename).lower()
    for cls in classes:
        if name.startswith(cls.lower()):
            return cls
    return None


def find_test_photos(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in glob.glob(os.path.join(folder, "*"))
        if f.lower().endswith(VALID_EXT)
    )


def test_from_folder(files):
    """Test using real photos dropped in test_photos/ (auto face-crop + classify)."""
    model, scaler, classes = load_pipeline()
    n = len(files)
    print(f"พบภาพใน '{TEST_PHOTOS_DIR}/' จำนวน {n} ภาพ -> ใช้ภาพเหล่านี้ทดสอบก่อน (ไม่ใช้ dataset)")

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    correct, known = 0, 0
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        path = files[i]
        image = cv2.imread(path)
        if image is None:
            ax.axis("off")
            print(f"[{i + 1}] {os.path.basename(path)} -> อ่านไฟล์ไม่ได้ ข้าม")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_gray, found = detect_and_crop_face(gray)
        label, confidence = classify_face(face_gray, model, scaler, classes)
        truth = guess_ground_truth(path, classes)

        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])

        pred_line = f"Pred: {label} ({confidence:.0f}%)"
        if not found:
            pred_line += "\n(no face auto-detected)"

        if truth is not None:
            known += 1
            is_correct = (truth == label)
            correct += int(is_correct)
            color = "green" if is_correct else "red"
            title = f"{os.path.basename(path)}\nTrue: {truth}\n{pred_line}"
        else:
            color = "black"
            title = f"{os.path.basename(path)}\n{pred_line}"

        ax.set_title(title, color=color, fontsize=9)

        status = ""
        if truth is not None:
            status = "OK" if truth == label else "WRONG"
        print(f"[{i + 1}] {os.path.basename(path):<28} -> {label} ({confidence:.0f}%) "
              f"{'' if found else '[no face found]'} {status}")

    if known:
        fig.suptitle(f"Test photos from '{TEST_PHOTOS_DIR}/' — {correct}/{known} correct "
                      f"(known ground truth from filename)")
        print(f"\nCorrect: {correct}/{known} (เฉพาะไฟล์ที่ตั้งชื่อบอก ground truth)")
    else:
        fig.suptitle(f"Test photos from '{TEST_PHOTOS_DIR}/' ({n} images, ground truth unknown)")

    fig.tight_layout()
    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def test_from_dataset(n_samples=N_SAMPLES):
    """Original behaviour: random samples from the held-out test set."""
    model, scaler, classes = load_pipeline()
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")

    print(f"ไม่พบภาพใน '{TEST_PHOTOS_DIR}/' -> สุ่มภาพจาก test set ของ dataset มาทดสอบแทน")

    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    X_sample_scaled = scaler.transform(X_sample)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_sample_scaled)
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1) * 100
    else:
        predictions = model.predict(X_sample_scaled)
        confidences = None

    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"

        ax.imshow(X_sample[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

        if confidences is not None:
            title_text = f"Pred: {pred} ({int(confidences[i])}%)\nTrue: {true}"
        else:
            title_text = f"Pred: {pred}\nTrue: {true}"

        ax.set_title(title_text, color=color)

        print(f"[{i + 1}] Pred: {pred:<14} True: {true:<14} "
              f"{'OK' if correct else 'WRONG'}")

    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def test_svm(n_samples=N_SAMPLES):
    photos = find_test_photos(TEST_PHOTOS_DIR)
    if photos:
        test_from_folder(photos)
    else:
        test_from_dataset(n_samples)


if __name__ == "__main__":
    test_svm()
