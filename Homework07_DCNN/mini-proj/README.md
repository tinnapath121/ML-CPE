# ใบงานที่ 8: Deep Convolutional Neural Network (DCNN) บนชุดข้อมูล CIFAR-10

รายวิชา Machine Learning (04-624-201) ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

---

## บทสรุปสำหรับผู้ตรวจ

- **โจทย์:** สร้าง DCNN จำแนกภาพ เปรียบเทียบประสิทธิภาพตามจำนวน epoch และตามโครงสร้างโมเดล (จำนวนชั้น Convolution และจำนวน neurons) ตามใบงาน `ML-LAB-08.docx`
- **ชุดข้อมูล:** CIFAR-10 (ภาพสี 32×32 พิกเซล 10 คลาส)
- **วิธีทำ:** เทรน 4 โครงสร้างด้วยชุดข้อมูลและ seed เดียวกัน วัด accuracy บนชุดทดสอบที่ epoch 5, 10, 20 และ 30
- **การใช้การ์ดจอ:** สคริปต์หลัก `compare_gpu.py` (PyTorch) เลือกอุปกรณ์อัตโนมัติ Intel GPU (XPU) → NVIDIA (CUDA) → CPU และพิมพ์ชื่ออุปกรณ์ตอนเริ่มรัน
- **ผลลัพธ์:** ตัวเลขทั้งหมดได้จากการรันจริงและบันทึกใน `outputs_cifar10_gpu/` (ดูหัวข้อที่ 6)

หน้า 1 จาก 4

---

## 1. โครงสร้างไฟล์

```text
mini-proj/
│
├── compare_gpu.py        # สคริปต์หลักของใบงานที่ 8 (PyTorch, ใช้การ์ดจอ) ← ใช้ไฟล์นี้
├── compare.py            # เวอร์ชัน TensorFlow ลอจิกเดียวกัน (ใช้ CPU บน Windows)
├── data/                 # สร้างอัตโนมัติ เก็บ cifar-10-python.tar.gz และข้อมูลที่แตกแล้ว
├── outputs_cifar10_gpu/  # ผลลัพธ์ของ compare_gpu.py (สร้างอัตโนมัติ)
│   ├── results.csv               # ตาราง accuracy ทุกโครงสร้าง x ทุก epoch
│   ├── accuracy_by_config.png    # กราฟแท่งเทียบโครงสร้าง
│   ├── accuracy_by_epoch.png     # กราฟเส้นเทียบจำนวน epoch
│   ├── history_<config>.png      # กราฟ train/validation accuracy และ loss
│   ├── predictions.png           # ตัวอย่างการทำนายของโมเดลที่ดีที่สุด
│   └── summary.txt               # สรุปผลและ accuracy รายคลาส
│
├── main.py, data_loader.py, preprocessing.py, split_data.py,
│   vgg_model.py, evaluate.py, test_vgg.py   # โปรเจกต์เดิม Cat vs Dog (VGG)
└── README.md             # ไฟล์นี้
```

> `compare.py` และ `compare_gpu.py` เป็นอิสระต่อกันและไม่แก้ไขโค้ดโปรเจกต์ Cat vs Dog เดิม

หน้า 2 จาก 4

---

## 2. การติดตั้ง (ทำครั้งเดียว)

1. ติดตั้งไดรเวอร์การ์ดจอรุ่นล่าสุด (Intel Arc ใช้ไดรเวอร์กราฟิก Intel)
2. ติดตั้งไลบรารีใน Command Prompt:

```bat
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install matplotlib numpy
```

3. ตรวจว่า PyTorch เห็นการ์ดจอ (ต้องพิมพ์ `True`):

```bat
python -c "import torch; print(torch.xpu.is_available())"
```

ถ้าเป็นการ์ด NVIDIA ให้ติดตั้ง PyTorch รุ่น CUDA ตามคำแนะนำที่ https://pytorch.org แทนคำสั่งข้างต้น

## 3. ชุดข้อมูล

- ดาวน์โหลดอัตโนมัติครั้งแรกที่รัน (ประมาณ 170 MB) ไปยัง `data/`
- ดาวน์โหลดเอง: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  แล้ววางไฟล์ **โดยไม่แตกไฟล์และไม่เปลี่ยนชื่อ** ที่ `mini-proj/data/cifar-10-python.tar.gz`
- การแบ่งข้อมูล (seed คงที่ = 42): train 40,000 / validation 10,000 / test 10,000
  - ชุด test คือชุดทดสอบทางการของ CIFAR-10 ใช้รายงานผลเท่านั้น ไม่ถูกใช้ปรับโมเดล

## 4. วิธีรัน

```bat
cd mini-proj
python compare_gpu.py --quick     :: ทดสอบเร็ว (ข้อมูลย่อย, 3 epoch) รันก่อนเสมอ
python compare_gpu.py             :: รันจริง (30 epoch x 4 โครงสร้าง)
python compare_gpu.py --device cpu   :: บังคับใช้ CPU (กรณีเปรียบเทียบความเร็ว)
```

บรรทัดแรกของผลลัพธ์ต้องระบุอุปกรณ์ เช่น `อุปกรณ์ที่ใช้: XPU (ชื่อการ์ดจอ)` หากแสดง CPU
แปลว่า PyTorch ยังไม่เห็นการ์ดจอ ให้ตรวจไดรเวอร์และขั้นตอนที่ 2

หน้า 3 จาก 4

---

## 5. รายละเอียดวิธีการ

| หัวข้อ | รายละเอียด |
|---|---|
| Standardize | (x − mean) / std ต่อช่องสี คำนวณจากชุด train เท่านั้น (ป้องกันข้อมูลรั่วจาก validation/test) |
| Optimizer | Adam, learning rate 1e-3 คงที่, batch 128 |
| Augmentation | พลิกภาพซ้าย-ขวาแบบสุ่ม |
| Regularization | BatchNorm, Dropout 0.25 หลังแต่ละ block และ 0.5 ก่อนชั้นสุดท้าย |
| การเทียบ epoch | เทรนแต่ละโครงสร้าง **ครั้งเดียว** ถึง epoch สูงสุด แล้ววัด test accuracy ที่ epoch 5, 10, 20, 30 (ไม่มี early stopping) ผลเทียบเท่าการเทรนแยกครั้งด้วยจำนวน epoch นั้น |

**โครงสร้างที่เปรียบเทียบ**

| โครงสร้าง | ชั้น Conv | Filters ต่อ block | Neurons (Dense) |
|---|---|---|---|
| A_1block_32 | 1 | 32 | 64 |
| B_2block_32-64 | 2 | 32, 64 | 128 |
| C_3block_32-64-128 | 3 | 32, 64, 128 | 256 |
| D_VGG_3block_x2 | 6 | (32,32), (64,64), (128,128) | 256 |

แก้ไขโครงสร้างหรือจำนวน epoch ได้ที่ตัวแปร `CONFIGS` และ `checkpoints` ใน `compare_gpu.py`

## 6. ผลการทดลอง

ผลจากการรันจริงด้วย `python compare_gpu.py` บน **Intel Arc B580 (XPU)** ที่มา: `outputs_cifar10_gpu/results.csv`
(ค่าเป็น test accuracy %)

| โครงสร้าง | epoch 5 | epoch 10 | epoch 20 | epoch 30 | เวลาเทรนรวม 30 epoch |
|---|---|---|---|---|---|
| A_1block_32 | 55.48 | 59.68 | 61.76 | 62.82 | 32.6 วินาที |
| B_2block_32-64 | 62.79 | 67.06 | 70.26 | 72.05 | 40.8 วินาที |
| C_3block_32-64-128 | 69.73 | 75.28 | 77.61 | 80.56 | 48.0 วินาที |
| D_VGG_3block_x2 | 74.99 | 80.74 | 84.68 | **85.24** | 82.6 วินาที |

- อุปกรณ์ที่ใช้เทรน: Intel Arc B580 (XPU)
- โมเดลที่ดีที่สุด: D_VGG_3block_x2 test accuracy **85.24%** ที่ 30 epoch
- ภาพประกอบ: `accuracy_by_config.png`, `accuracy_by_epoch.png`, `history_*.png`, `predictions.png`

**ข้อสังเกตจากผล**
- **จำนวนชั้น Conv มีผลมากที่สุด:** ที่ epoch เดียวกัน accuracy เรียงตามความลึก A < B < C < D ชัดเจนทุกจุด
  โครงสร้าง A มีพารามิเตอร์ 525,962 ตัว ไม่น้อยกว่า B (545,290) มาก แต่ accuracy ต่ำกว่า แปลว่าความลึกสำคัญกว่าจำนวนพารามิเตอร์
- **จำนวน epoch:** accuracy เพิ่มขึ้นตามจำนวน epoch ในทุกโครงสร้าง แต่ผลตอบแทนลดลง
  เช่น D เพิ่มจาก 84.68% (epoch 20) เป็น 85.24% (epoch 30) เพียง 0.56 จุด
- **Overfitting:** โครงสร้าง D มี train 87.27% สูงกว่า validation 85.61% เล็กน้อยที่ epoch 30 และกราฟ validation เริ่มนิ่งถึงลดลงเล็กน้อยปลายการเทรน
  โครงสร้าง A, B, C ยังมี train ต่ำกว่า validation ซึ่งน่าจะเป็นเพราะ Dropout และการพลิกภาพทำงานเฉพาะตอนเทรน และโมเดลยังไม่ถึงจุดอิ่มตัว (เป็นข้อสันนิษฐาน ไม่ได้ทดสอบแยก)
- **รายคลาส (โมเดล D):** ทำได้ดีที่สุดคือ truck 94.9%, ship 90.7%, horse 90.6% และแย่ที่สุดคือ cat 67.2%, bird 73.1% (รายละเอียดใน `summary.txt`)
- **เวลา:** D ใช้เวลาเทรนมากกว่า A ประมาณ 2.5 เท่า (82.6 เทียบกับ 32.6 วินาที) แลกกับ accuracy ที่สูงกว่า 22.42 จุด

**ข้อจำกัด:** เทรนและวัดผลรอบเดียว (seed 42) ผลอาจต่างกันเล็กน้อยเมื่อเปลี่ยน seed หรืออุปกรณ์

## 7. ปัญหาที่พบบ่อย

| อาการ | สาเหตุและวิธีแก้ |
|---|---|
| ขึ้น "ไม่พบ GPU ... ใช้ CPU" | ติดตั้ง PyTorch ผิดรุ่น หรือไดรเวอร์เก่า ติดตั้งใหม่ด้วยคำสั่งรุ่น `xpu` |
| ดาวน์โหลด CIFAR-10 ไม่สำเร็จ | ดาวน์โหลดไฟล์เองตามหัวข้อ 3 แล้ววางใน `data/` |
| หน่วยความจำการ์ดจอไม่พอ | ลด batch เช่น `python compare_gpu.py --batch 64` |
| เส้น validation แกว่งมาก | ดูกราฟ `history_*.png` เลือก epoch ที่ validation ดีที่สุดเมื่อรายงานผล |

หน้า 4 จาก 4
