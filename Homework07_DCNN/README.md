# ML-08: Deep Convolutional Neural Network (DCNN)

ใบงานที่ 8 วิชา Machine Learning (04-624-201) มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี
(โจทย์: `ML-LAB-08.docx`)

---

## บทสรุปสำหรับผู้ตรวจ

- **งานในใบงาน:** ใช้ DCNN จำแนกภาพ CIFAR-10 เปรียบเทียบ 4 โครงสร้าง (จำนวนชั้น Conv และ neurons) และจำนวน epoch (5, 10, 20, 30)
- **ผลที่ดีที่สุด:** โครงสร้าง D_VGG_3block_x2 ได้ test accuracy **85.24%** ที่ 30 epoch
- **การใช้การ์ดจอ:** เทรนบน Intel Arc B580 (XPU) ด้วย PyTorch ทั้งหมดใช้เวลาไม่ถึง 4 นาที
- **ตัวชี้วัด:** ปัญหา: โครงสร้างตื้นสุด (A) ได้ 62.82% (สุ่มเดาได้ 10%) → วัตถุประสงค์: เปรียบเทียบโครงสร้างและจำนวน epoch ตามใบงาน → ผลสัมฤทธิ์: โครงสร้างที่ดีที่สุดสูงกว่า A อยู่ 22.42 จุดเปอร์เซ็นต์
- **รายละเอียดการติดตั้งและรัน:** ดู `mini-proj/README.md`

หน้า 1 จาก 3

---

## 1. โครงสร้างโฟลเดอร์

```text
ML-08-DCNN/
├── ML-LAB-08.docx              # ใบงาน
├── README.md                   # ไฟล์นี้ (ภาพรวมและผลการทดลอง)
└── mini-proj/
    ├── README.md               # คู่มือติดตั้ง/รัน/ปัญหาที่พบบ่อย
    ├── compare_gpu.py          # สคริปต์หลักของใบงาน (PyTorch, ใช้การ์ดจอ)
    ├── compare.py              # เวอร์ชัน TensorFlow ลอจิกเดียวกัน
    ├── data/                   # CIFAR-10 (ดาวน์โหลดอัตโนมัติ, ไม่อัปขึ้น GitHub)
    ├── outputs_cifar10_gpu/    # ผลลัพธ์ของการรันจริง
    │   ├── results.csv
    │   ├── summary.txt
    │   ├── accuracy_by_config.png
    │   ├── accuracy_by_epoch.png
    │   ├── history_<config>.png (4 ไฟล์)
    │   └── predictions.png
    └── main.py, data_loader.py, preprocessing.py, split_data.py,
        vgg_model.py, evaluate.py, test_vgg.py   # โปรเจกต์ Cat vs Dog (ดูภาคผนวก)
```

## 2. วิธีรันโดยสรุป

```bat
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install matplotlib numpy
cd mini-proj
python compare_gpu.py --quick
python compare_gpu.py
```

ชุดข้อมูล: CIFAR-10 (ไม่ได้อัปขึ้น GitHub เพราะไฟล์ใหญ่เกิน 100 MB ผู้ที่รัน `compare_gpu.py` จะได้ข้อมูลดาวน์โหลดเองอัตโนมัติ) แบ่ง train 40,000 / validation 10,000 / test 10,000 (seed 42) ทำ standardize ด้วย mean/std ของชุด train
รายละเอียดเพิ่มเติมอยู่ใน `mini-proj/README.md`

หน้า 2 จาก 3

---

## 3. ผลการทดลอง

ผลจากการรันจริงบน Intel Arc B580 (XPU) ที่มา: `mini-proj/outputs_cifar10_gpu/results.csv` (test accuracy %)

| โครงสร้าง | ชั้น Conv | Neurons | พารามิเตอร์ | epoch 5 | epoch 10 | epoch 20 | epoch 30 | เวลาเทรน 30 epoch |
|---|---|---|---|---|---|---|---|---|
| A_1block_32 | 1 | 64 | 525,962 | 55.48 | 59.68 | 61.76 | 62.82 | 32.6 วินาที |
| B_2block_32-64 | 2 | 128 | 545,290 | 62.79 | 67.06 | 70.26 | 72.05 | 40.8 วินาที |
| C_3block_32-64-128 | 3 | 256 | 620,810 | 69.73 | 75.28 | 77.61 | 80.56 | 48.0 วินาที |
| D_VGG_3block_x2 | 6 | 256 | 815,018 | 74.99 | 80.74 | 84.68 | **85.24** | 82.6 วินาที |

**ข้อสังเกต**
- ยิ่งมีชั้น Conv มาก accuracy ยิ่งสูงที่ epoch เดียวกัน (A < B < C < D) และความลึกสำคัญกว่าจำนวนพารามิเตอร์ (A มีพารามิเตอร์ใกล้เคียง B แต่ต่ำกว่า 9 จุดที่ epoch 30)
- เพิ่ม epoch แล้วดีขึ้นทุกโครงสร้าง แต่ผลตอบแทนลดลง เช่น D เพิ่มจาก 20 เป็น 30 epoch ได้เพียง 0.56 จุด
- โครงสร้าง D เริ่มมีช่องว่างระหว่าง train (87.27%) กับ validation (85.61%) ที่ epoch 30 ซึ่งเป็นสัญญาณเริ่มต้นของ overfitting
- คลาสที่ทำนายยากที่สุดคือ cat (67.2%) และ bird (73.1%) ส่วนง่ายที่สุดคือ truck (94.9%)
- ข้อจำกัด: เทรนรอบเดียว (seed 42) ผลอาจต่างเล็กน้อยเมื่อเปลี่ยน seed หรืออุปกรณ์

## ภาคผนวก
> หมายเหตุ: README เดิมอ้างโฟลเดอร์ `classification/` แต่โฟลเดอร์จริงชื่อ `mini-proj/` และโฟลเดอร์ข้อมูล `PetImages` ไม่ได้อยู่ในที่เก็บนี้

Build a DCNN based on the VGG architecture using Python for image classification. The project covers image loading, preprocessing, dataset splitting, DCNN (VGG) model training, evaluation, and Cat and Dog classification

- Data: Kaggle Dogs vs. Cats https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
- Files: `main.py` (training pipeline), `data_loader.py` (load images, skip corrupted files), `preprocessing.py` (resize, BGR to RGB), `split_data.py` (train/validation/test split), `vgg_model.py` (build, train, save, predict), `evaluate.py` (accuracy, report, confusion matrix, history plots), `test_vgg.py` (test with random images)
- Summary: a VGG-based DCNN for Cat and Dog image classification with a modular Python pipeline.

หน้า 3 จาก 3
