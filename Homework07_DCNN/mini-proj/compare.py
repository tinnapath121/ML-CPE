"""
ML-LAB-08: DCNN on CIFAR-10 -- เปรียบเทียบโครงสร้าง DCNN และจำนวน epoch
=====================================================================
รัน:   python compare.py            # เต็มรูปแบบ (แนะนำให้มี GPU)
       python compare.py --quick    # ทดสอบเร็ว (ชุดข้อมูลย่อย, epoch น้อย)

ผลลัพธ์ทั้งหมดบันทึกในโฟลเดอร์ outputs_cifar10/
  results.csv             accuracy ของแต่ละโครงสร้าง x จำนวน epoch (ตารางหลักของใบงาน)
  accuracy_by_config.png  กราฟแท่งเทียบ accuracy ของแต่ละโครงสร้าง
  accuracy_by_epoch.png   กราฟเส้นเทียบ accuracy ตามจำนวน epoch
  history_<config>.png    กราฟ train/validation accuracy และ loss
  predictions.png         ตัวอย่างการทำนายบนชุดทดสอบ (โมเดลที่ดีที่สุด)
  summary.txt             สรุปผลเป็นข้อความ + classification report

หลักการเปรียบเทียบ epoch: เทรนแต่ละโครงสร้าง "ครั้งเดียว" จนถึง epoch สูงสุด
(learning rate คงที่ ไม่ใช้ early stopping) แล้ววัด accuracy บนชุดทดสอบ ณ epoch
ที่กำหนด ผลเทียบเท่ากับการเทรนแยกครั้งด้วยจำนวน epoch นั้น แต่ใช้เวลาน้อยกว่ามาก
ชุดทดสอบใช้ "วัดผลเท่านั้น" ไม่ถูกนำไปปรับโมเดล
"""
import argparse
import csv
import os
import random
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")  # ไม่ต้องมีหน้าจอ
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# โครงสร้างที่เปรียบเทียบ: blocks = [(จำนวน filter, จำนวนชั้น Conv ต่อ block)]
# dense = จำนวน neurons ในชั้น Dense ก่อนชั้นสุดท้าย
CONFIGS = {
    "A_1block_32":      {"blocks": [(32, 1)],                      "dense": 64},
    "B_2block_32-64":   {"blocks": [(32, 1), (64, 1)],             "dense": 128},
    "C_3block_32-64-128": {"blocks": [(32, 1), (64, 1), (128, 1)], "dense": 256},
    "D_VGG_3block_x2":  {"blocks": [(32, 2), (64, 2), (128, 2)],   "dense": 256},
}


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)


def load_cifar10(quick):
    """โหลด CIFAR-10 และแบ่ง train 40,000 / validation 10,000 / test 10,000"""
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
    y_tr, y_te = y_tr.ravel(), y_te.ravel()

    # สลับลำดับด้วย seed คงที่ แล้วแยก validation 10,000 ภาพออกจากชุดฝึก
    idx = np.random.RandomState(SEED).permutation(len(x_tr))
    x_tr, y_tr = x_tr[idx], y_tr[idx]
    x_val, y_val = x_tr[:10000], y_tr[:10000]
    x_tr, y_tr = x_tr[10000:], y_tr[10000:]

    if quick:  # โหมดทดสอบเร็ว
        x_tr, y_tr = x_tr[:4000], y_tr[:4000]
        x_val, y_val = x_val[:1000], y_val[:1000]
        x_te, y_te = x_te[:1000], y_te[:1000]
    return (x_tr, y_tr), (x_val, y_val), (x_te, y_te)


def build_model(cfg, x_train):
    model = keras.Sequential(name="dcnn")
    model.add(keras.Input(shape=x_train.shape[1:]))

    # Standardize: (x - mean) / std ต่อช่องสี โดยคำนวณจากชุดฝึกเท่านั้น
    # (กันข้อมูลรั่วจาก validation/test) และฝังไว้ในโมเดล
    norm = layers.Normalization(axis=-1)
    norm.adapt(x_train.astype("float32"))
    model.add(norm)

    for filters, n_conv in cfg["blocks"]:
        for _ in range(n_conv):
            model.add(layers.Conv2D(filters, 3, padding="same"))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(cfg["dense"], activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(CLASSES), activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


class TestAtEpochs(keras.callbacks.Callback):
    """วัด accuracy บนชุดทดสอบ ณ epoch ที่กำหนด (ใช้รายงานผลเท่านั้น)"""

    def __init__(self, x_te, y_te, checkpoints):
        super().__init__()
        self.x_te, self.y_te, self.checkpoints = x_te, y_te, set(checkpoints)
        self.test_acc = {}

    def on_epoch_end(self, epoch, logs=None):
        n = epoch + 1
        if n in self.checkpoints:
            _, acc = self.model.evaluate(self.x_te, self.y_te, verbose=0)
            self.test_acc[n] = float(acc)
            print(f"   >> epoch {n}: test accuracy = {acc * 100:.2f}%")


def plot_history(hist, name, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, title in ((ax[0], "accuracy", "Accuracy"), (ax[1], "loss", "Loss")):
        a.plot(hist[key], label="train")
        a.plot(hist["val_" + key], label="validation")
        a.set_xlabel("Epoch"); a.set_ylabel(title)
        a.set_title(f"{name}: {title}"); a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="ทดสอบเร็ว")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs_cifar10"))
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    checkpoints = [1, 2, 3] if args.quick else [5, 10, 20, 30]
    max_epoch = max(checkpoints)
    os.makedirs(args.out, exist_ok=True)
    set_seed()

    print("[1] โหลดและแบ่งชุดข้อมูล CIFAR-10")
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = load_cifar10(args.quick)
    print(f"    train={len(x_tr)} val={len(x_val)} test={len(x_te)} "
          f"shape={x_tr.shape[1:]}")

    rows, histories, models = [], {}, {}
    for name, cfg in CONFIGS.items():
        print(f"\n[2] เทรน {name}")
        set_seed()
        model = build_model(cfg, x_tr)
        cb = TestAtEpochs(x_te, y_te, checkpoints)
        t0 = time.time()
        h = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                      epochs=max_epoch, batch_size=args.batch,
                      callbacks=[cb], verbose=2)
        secs = time.time() - t0
        hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
        histories[name], models[name] = hist, model
        plot_history(hist, name, os.path.join(args.out, f"history_{name}.png"))
        for ep in checkpoints:
            rows.append({
                "config": name,
                "conv_layers": sum(n for _, n in cfg["blocks"]),
                "dense_neurons": cfg["dense"],
                "params": model.count_params(),
                "epochs": ep,
                "train_acc": round(hist["accuracy"][ep - 1], 4),
                "val_acc": round(hist["val_accuracy"][ep - 1], 4),
                "test_acc": round(cb.test_acc[ep], 4),
                "train_seconds_total": round(secs, 1),
            })

    # ---- ตารางผลลัพธ์ ----
    with open(os.path.join(args.out, "results.csv"), "w", newline="",
              encoding="utf-8-sig") as f:  # utf-8-sig ให้ Excel เปิดได้
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- กราฟเทียบโครงสร้าง (ที่ epoch สูงสุด) ----
    final = [r for r in rows if r["epochs"] == max_epoch]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([r["config"] for r in final], [r["test_acc"] * 100 for r in final])
    ax.bar_label(bars, fmt="%.2f%%")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Test accuracy by DCNN configuration ({max_epoch} epochs)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "accuracy_by_config.png"), dpi=150)
    plt.close(fig)

    # ---- กราฟเทียบจำนวน epoch ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        ax.plot([r["epochs"] for r in rs], [r["test_acc"] * 100 for r in rs],
                marker="o", label=name)
    ax.set_xlabel("Epochs"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test accuracy vs number of epochs")
    ax.set_xticks(checkpoints); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "accuracy_by_epoch.png"), dpi=150)
    plt.close(fig)

    # ---- ตัวอย่างการทำนาย ด้วยโมเดลที่ดีที่สุด ----
    best = max(final, key=lambda r: r["test_acc"])["config"]
    probs = models[best].predict(x_te, verbose=0)
    pred = probs.argmax(axis=1)
    pick = np.random.RandomState(SEED).choice(len(x_te), 12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(9, 7.5))
    for a, i in zip(axes.ravel(), pick):
        a.imshow(x_te[i]); a.axis("off")
        ok = pred[i] == y_te[i]
        a.set_title(f"true: {CLASSES[y_te[i]]}\npred: {CLASSES[pred[i]]}",
                    color="green" if ok else "red", fontsize=9)
    fig.suptitle(f"Predictions ({best})"); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "predictions.png"), dpi=150); plt.close(fig)

    # ---- สรุปผลเป็นข้อความ ----
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_te, pred):
        cm[t, p] += 1
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best configuration: {best} "
                f"(test accuracy {max(r['test_acc'] for r in final) * 100:.2f}% "
                f"at {max_epoch} epochs)\n\n")
        f.write("config | epochs | train | val | test\n")
        for r in rows:
            f.write(f"{r['config']} | {r['epochs']} | {r['train_acc']:.4f} | "
                    f"{r['val_acc']:.4f} | {r['test_acc']:.4f}\n")
        f.write("\nPer-class accuracy (best model):\n")
        for k, c in enumerate(CLASSES):
            f.write(f"  {c:<11}{cm[k, k] / max(cm[k].sum(), 1) * 100:6.2f}%\n")

    print(f"\nเสร็จสิ้น -> {args.out}")
    print(open(os.path.join(args.out, "summary.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
