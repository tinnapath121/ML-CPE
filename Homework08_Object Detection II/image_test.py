
from pathlib import Path
import argparse
import random
import cv2
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

    # Model path
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None,
                        help='พาธไฟล์ best.pt (ค่าเริ่มต้น: runs/detect/runs/bdd100k/weights/best.pt)')
    args = parser.parse_args()
    weights = (
        Path(args.weights) if args.weights
        else ROOT / 'runs' / 'detect' / 'runs' / 'bdd100k' / 'weights' / 'best.pt'
    )

    # Test dataset
    image_dir = ROOT / 'dataset' / 'test' / 'images'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    if not image_dir.exists():
        raise FileNotFoundError(f'Dataset not found: {image_dir}')

    # Load trained model
    model = YOLO(str(weights))

    # Get test images
    extensions = {'.jpg', '.jpeg', '.png'}
    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    ]

    if not images:
        raise RuntimeError('No test images found')

    # Randomly select 5 images
    selected_images = random.sample(
        images, min(5, len(images))
    )

    # Object detection
    for image in selected_images:
        results = model.predict(
            source=str(image),
            conf=0.25,
            imgsz=480,
            device=device,
            save=True,
            project=str(ROOT / 'runs'),
            name='image_test',
            exist_ok=True
        )

        for result in results:
            print(f'\nImage: {image.name}')

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                print(
                    f'{model.names[class_id]}: '
                    f'{confidence:.2f}'
                )

            # Display detection result
            annotated = result.plot()
            cv2.imshow('YOLO Image Detection', annotated)

            # Press any key for next image
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print('\nImage testing completed!')


if __name__ == '__main__':
    main()
