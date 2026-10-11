"""
ML-LAB-08: DCNN บน CIFAR-10 (PyTorch) -- ใช้การ์ดจอ Intel Arc / NVIDIA อัตโนมัติ
=============================================================================
เลือกอุปกรณ์อัตโนมัติตามลำดับ: Intel GPU (xpu) -> NVIDIA GPU (cuda) -> CPU
รัน:   python compare_gpu.py            # เต็มรูปแบบ
       python compare_gpu.py --quick    # ทดสอบเร็ว
       python compare_gpu.py --device cpu   # บังคับใช้ CPU (ถ้าจำเป็น)

ผลลัพธ์ในโฟลเดอร์ outputs_cifar10_gpu/ :
  results.csv, accuracy_by_config.png, accuracy_by_epoch.png,
  history_<config>.png, predictions.png, summary.txt

การเทียบจำนวน epoch: เทรนแต่ละโครงสร้างครั้งเดียวถึง epoch สูงสุด (learning rate คงที่,
ไม่มี early stopping) แล้ววัด accuracy บนชุดทดสอบ ณ epoch ที่กำหนด
ชุดทดสอบใช้รายงานผลเท่านั้น ไม่ถูกใช้ปรับโมเดล
"""
import argparse
import csv
import os
import random
import time

import matplotlib

matplotlib.use("Agg")  # ไม่ต้องมีหน้าจอ
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SEED = 42
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# blocks = [(จำนวน filter, จำนวนชั้น Conv ต่อ block)], dense = จำนวน neurons ก่อนชั้นสุดท้าย
CONFIGS = {
    "A_1block_32":        {"blocks": [(32, 1)],                       "dense": 64},
    "B_2block_32-64":     {"blocks": [(32, 1), (64, 1)],              "dense": 128},
    "C_3block_32-64-128": {"blocks": [(32, 1), (64, 1), (128, 1)],    "dense": 256},
    "D_VGG_3block_x2":    {"blocks": [(32, 2), (64, 2), (128, 2)],    "dense": 256},
}


# ---------------------------------------------------------------- อุปกรณ์
def pick_device(name):
    """เลือก GPU ที่ใช้ได้ ถ้าไม่มีจึงถอยไป CPU และแจ้งให้ทราบชัดเจน"""
    if name != "auto":
        return torch.device(name)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")        # Intel Arc
    if torch.cuda.is_available():
        return torch.device("cuda")       # NVIDIA
    print("!! ไม่พบ GPU ที่ PyTorch ใช้ได้ -> ใช้ CPU (ตรวจไดรเวอร์/การติดตั้ง torch รุ่น XPU)")
    return torch.device("cpu")


def device_name(dev):
    if dev.type == "xpu":
        return torch.xpu.get_device_name(0)
    if dev.type == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------- ข้อมูล
def load_cifar10(root, quick):
    """โหลด CIFAR-10 (ดาวน์โหลดอัตโนมัติครั้งแรก) แบ่ง train 40,000 / val 10,000 / test 10,000"""
    from torchvision.datasets import CIFAR10
    tr = CIFAR10(root, train=True, download=True)
    te = CIFAR10(root, train=False, download=True)
    x_tr, y_tr = tr.data, np.array(tr.targets)       # uint8, (N,32,32,3)
    x_te, y_te = te.data, np.array(te.targets)

    idx = np.random.RandomState(SEED).permutation(len(x_tr))
    x_tr, y_tr = x_tr[idx], y_tr[idx]
    x_val, y_val = x_tr[:10000], y_tr[:10000]
    x_tr, y_tr = x_tr[10000:], y_tr[10000:]

    if quick:
        x_tr, y_tr = x_tr[:4000], y_tr[:4000]
        x_val, y_val = x_val[:1000], y_val[:1000]
        x_te, y_te = x_te[:1000], y_te[:1000]
    return (x_tr, y_tr), (x_val, y_val), (x_te, y_te)


def standardize(x_tr, x_val, x_te, dev):
    """Standardize: (x - mean) / std ต่อช่องสี โดยใช้ค่าจากชุดฝึกเท่านั้น
    แล้วย้ายข้อมูลทั้งหมดขึ้น GPU ครั้งเดียว (ชุดข้อมูลเล็กพอ ทำให้เทรนเร็ว)"""
    def to_t(a):
        return torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255)
    a, b, c = to_t(x_tr), to_t(x_val), to_t(x_te)
    mean = a.mean(dim=(0, 2, 3), keepdim=True)
    std = a.std(dim=(0, 2, 3), keepdim=True)
    return [((t - mean) / std).to(dev) for t in (a, b, c)], (mean, std)


# ---------------------------------------------------------------- โมเดล
def build_model(cfg):
    layers, in_ch, size = [], 3, 32
    for filters, n_conv in cfg["blocks"]:
        for _ in range(n_conv):
            layers += [nn.Conv2d(in_ch, filters, 3, padding=1),
                       nn.BatchNorm2d(filters), nn.ReLU()]
            in_ch = filters
        layers += [nn.MaxPool2d(2), nn.Dropout(0.25)]
        size //= 2
    layers += [nn.Flatten(),
               nn.Linear(in_ch * size * size, cfg["dense"]), nn.ReLU(),
               nn.Dropout(0.5), nn.Linear(cfg["dense"], len(CLASSES))]
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, x, y, batch=1000):
    """คืนค่า (loss, accuracy) -- ปิด dropout/batchnorm ด้วย eval()"""
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, len(x), batch):
        out = model(x[i:i + batch])
        loss_sum += nn.functional.cross_entropy(out, y[i:i + batch], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + batch]).sum().item()
    return loss_sum / len(x), correct / len(x)


def train_one(cfg, data, dev, max_epoch, checkpoints, batch):
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = data
    model = build_model(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}
    test_acc = {}
    n = len(x_tr)

    for ep in range(1, max_epoch + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        loss_sum, correct = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = x_tr[idx], y_tr[idx]
            # augmentation แบบเบา: พลิกภาพซ้าย-ขวาแบบสุ่มครึ่งหนึ่งของ batch
            flip = torch.rand(len(xb), device=dev) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)

            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = nn.functional.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()

        v_loss, v_acc = evaluate(model, x_val, y_val)
        hist["loss"].append(loss_sum / n)
        hist["accuracy"].append(correct / n)
        hist["val_loss"].append(v_loss)
        hist["val_accuracy"].append(v_acc)
        msg = (f"  epoch {ep:>2}/{max_epoch}  loss {hist['loss'][-1]:.4f}  "
               f"acc {hist['accuracy'][-1]:.4f}  val_loss {v_loss:.4f}  val_acc {v_acc:.4f}")
        if ep in checkpoints:
            _, test_acc[ep] = evaluate(model, x_te, y_te)
            msg += f"  >> TEST acc {test_acc[ep] * 100:.2f}%"
        print(msg, flush=True)
    return model, hist, test_acc


# ---------------------------------------------------------------- กราฟ
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
    ap.add_argument("--device", default="auto", help="auto | xpu | cuda | cpu")
    ap.add_argument("--batch", type=int, default=128)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "outputs_cifar10_gpu"))
    ap.add_argument("--data", default=os.path.join(here, "data"))
    args = ap.parse_args()

    dev = pick_device(args.device)
    print(f"อุปกรณ์ที่ใช้: {dev.type.upper()} ({device_name(dev)})")
    print(f"PyTorch {torch.__version__}")

    checkpoints = [1, 2, 3] if args.quick else [5, 10, 20, 30]
    max_epoch = max(checkpoints)
    os.makedirs(args.out, exist_ok=True)
    set_seed()

    print("[1] โหลดและแบ่งชุดข้อมูล CIFAR-10")
    raw = load_cifar10(args.data, args.quick)
    (x_tr, x_val, x_te), _ = standardize(raw[0][0], raw[1][0], raw[2][0], dev)
    to_y = lambda a: torch.from_numpy(a).long().to(dev)
    data = ((x_tr, to_y(raw[0][1])), (x_val, to_y(raw[1][1])), (x_te, to_y(raw[2][1])))
    print(f"    train={len(x_tr)} val={len(x_val)} test={len(x_te)}")

    rows, models = [], {}
    for name, cfg in CONFIGS.items():
        print(f"\n[2] เทรน {name}")
        set_seed()
        t0 = time.time()
        model, hist, test_acc = train_one(cfg, data, dev, max_epoch, checkpoints, args.batch)
        secs = time.time() - t0
        models[name] = model
        params = sum(p.numel() for p in model.parameters())
        print(f"  ใช้เวลา {secs:.0f} วินาที, พารามิเตอร์ {params:,}")
        plot_history(hist, name, os.path.join(args.out, f"history_{name}.png"))
        for ep in checkpoints:
            rows.append({
                "config": name,
                "conv_layers": sum(n for _, n in cfg["blocks"]),
                "dense_neurons": cfg["dense"],
                "params": params,
                "epochs": ep,
                "train_acc": round(hist["accuracy"][ep - 1], 4),
                "val_acc": round(hist["val_accuracy"][ep - 1], 4),
                "test_acc": round(test_acc[ep], 4),
                "train_seconds_total": round(secs, 1),
                "device": dev.type,
            })

    with open(os.path.join(args.out, "results.csv"), "w", newline="",
              encoding="utf-8-sig") as f:   # utf-8-sig ให้ Excel เปิดได้
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    final = [r for r in rows if r["epochs"] == max_epoch]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([r["config"] for r in final], [r["test_acc"] * 100 for r in final])
    ax.bar_label(bars, fmt="%.2f%%")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Test accuracy by DCNN configuration ({max_epoch} epochs)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "accuracy_by_config.png"), dpi=150)
    plt.close(fig)

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

    # ตัวอย่างการทำนายด้วยโมเดลที่ดีที่สุด
    best = max(final, key=lambda r: r["test_acc"])["config"]
    bm = models[best].eval()
    with torch.no_grad():
        pred = torch.cat([bm(data[2][0][i:i + 1000]).argmax(1)
                          for i in range(0, len(data[2][0]), 1000)]).cpu().numpy()
    y_np = raw[2][1]
    pick = np.random.RandomState(SEED).choice(len(y_np), 12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(9, 7.5))
    for a, i in zip(axes.ravel(), pick):
        a.imshow(raw[2][0][i]); a.axis("off")
        a.set_title(f"true: {CLASSES[y_np[i]]}\npred: {CLASSES[pred[i]]}",
                    color="green" if pred[i] == y_np[i] else "red", fontsize=9)
    fig.suptitle(f"Predictions ({best})"); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "predictions.png"), dpi=150); plt.close(fig)

    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_np, pred):
        cm[t, p] += 1
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Device: {dev.type} ({device_name(dev)})\n")
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
