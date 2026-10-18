
from pathlib import Path
import argparse
from ultralytics import YOLO
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
    device = pick_device()
    print(f'Using device: {device}')

    # Load trained model
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None,
                        help='พาธไฟล์ best.pt (ค่าเริ่มต้น: runs/detect/runs/bdd100k/weights/best.pt)')
    args = parser.parse_args()
    weights = (
        Path(args.weights) if args.weights
        else ROOT / 'runs' / 'detect' / 'runs' / 'bdd100k' / 'weights' / 'best.pt'
    )

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    model = YOLO(str(weights))

    # Evaluate on test dataset
    metrics = model.val(
        data=str(ROOT / 'dataset' / 'data.yaml'),
        split='test',
        imgsz=480,
        batch=8,
        device=device,
        workers=0,
        plots=True,
        project=str(ROOT / 'runs'),
        name='evaluation',
        exist_ok=True,
    )

    # Display evaluation results
    print('\n===== Evaluation Results =====')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall: {metrics.box.mr:.4f}')
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')


if __name__ == '__main__':
    main()
