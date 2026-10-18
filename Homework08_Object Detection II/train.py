from ultralytics import YOLO
from pathlib import Path
import argparse
import torch

ROOT = Path(__file__).resolve().parent


def pick_device():
    """เลือกอุปกรณ์ประมวลผลอัตโนมัติ: Intel Arc (XPU) -> NVIDIA (CUDA) -> CPU"""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu:0"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n on BDD100K")
    parser.add_argument("--epochs", type=int, default=10, help="จำนวนรอบการเทรน")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="สัดส่วนข้อมูลเทรนที่ใช้ (0-1) เช่น 0.05 = 5%% ใช้ทดสอบความเร็ว")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--name", default="bdd100k_new",
                        help="ชื่อโฟลเดอร์ผลลัพธ์ (ไม่เขียนทับของเดิม ถ้าซ้ำจะต่อท้ายเลขให้)")
    args = parser.parse_args()

    device = pick_device()
    print(f"Using device: {device}")

    # Load pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # Train model
    # exist_ok=False: ถ้าชื่อซ้ำจะสร้างโฟลเดอร์ใหม่ (name2, name3, ...)
    # เพื่อไม่ให้เขียนทับโมเดลเดิมใน runs/detect/runs/bdd100k
    model.train(
        data=str(ROOT / "dataset" / "data.yaml"),
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        amp=True,
        cache=False,
        project=str(ROOT / "runs" / "detect" / "runs"),
        name=args.name,
        exist_ok=False,
        save=True,
        plots=True
    )

    # Output directory
    save_dir = Path(model.trainer.save_dir)

    print("Training completed!")
    print(f"Best Model: {save_dir / 'weights/best.pt'}")
    print(f"Last Model: {save_dir / 'weights/last.pt'}")
    print(f"Training CSV: {save_dir / 'results.csv'}")
    print("ประเมินผลโมเดลใหม่ด้วย:")
    print(f'  python evaluate.py --weights "{save_dir / "weights/best.pt"}"')


if __name__ == "__main__":
    main()
