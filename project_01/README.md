# SVM Applied Project: Face Mask Detection (Mask vs No Mask)

ระบบตรวจจับการสวมหน้ากากอนามัยจากภาพใบหน้า ด้วยอัลกอริทึม **Support Vector Machine (SVM)** ร่วมกับ **Principal Component Analysis (PCA)** — ต่อยอดจากระบบ Cat vs Dog เดิม (`Homework04_SVM`) โดยเปลี่ยนโดเมนข้อมูลให้เป็นงานที่มีการใช้งานจริง พร้อมทดสอบการใช้งานจริงกับภาพถ่าย/วิดีโอที่โมเดลไม่เคยเห็นมาก่อน (ไม่ใช่แค่ test set ของ dataset เดิม)

## ทำไมต้องประยุกต์ใช้กับโดเมนนี้ (Why this application)

* **Problem**: การตรวจสอบว่าพนักงาน/ผู้เข้าใช้บริการสวมหน้ากากอนามัยถูกต้องหรือไม่ ปกติต้องใช้คนเฝ้าตรวจด้วยสายตาที่ทางเข้า ซึ่งไม่สม่ำเสมอและใช้แรงงานคนตลอดเวลา
* **Goal**: สร้างระบบจำแนกภาพอัตโนมัติ (mask / no mask) ที่สามารถต่อยอดเป็นระบบเตือน ณ จุดตรวจ (เช่น กล้องหน้าประตูทางเข้าโรงงาน/ออฟฟิศ/โรงพยาบาล) แทนการเฝ้าดูด้วยคน
* **Result**: โมเดล SVM (RBF Kernel) ทำ **Test Accuracy = 84.62%** บนชุดทดสอบ 819 ภาพที่โมเดลไม่เคยเห็นมาก่อน และเมื่อนำไปทดสอบกับภาพถ่ายภายนอก dataset จริง 4 ภาพ **ทายถูกทั้ง 4/4 ภาพ** — แต่เมื่อทดสอบแบบ real-time ผ่านกล้องเว็บแคม/มือถือกับใบหน้าจริง กลับพบข้อจำกัดใหม่ (ดูหัวข้อ "การทดสอบใช้งานจริง" ด้านล่าง)

## Structure

```text
mask_project/
├── dataset/                       # ภาพแบ่งตามคลาส (ไม่ได้แนบมาด้วย ดูหัวข้อ Dataset)
│   ├── with_mask/                   # 2,162 ภาพ
│   └── without_mask/                # 1,930 ภาพ
├── demo_photos/                   # ภาพทดสอบ "ใช้งานจริง" — ไม่ได้อยู่ใน dataset ฝึก/test เลย
│   ├── test_unseen_*.jpg              # ภาพต้นฉบับ
│   └── pred_test_unseen_*_v2.jpg      # ผลทำนาย (bounding box + label)
├── outputs/                       # โมเดลและผลลัพธ์ที่บันทึกไว้
│   ├── svm_model.pkl                 # โมเดล SVM ที่ผ่านการเทรน
│   ├── scaler.pkl                    # Pipeline: StandardScaler + PCA
│   ├── classes.json                  # ["with_mask", "without_mask"]
│   ├── confusion_matrix.png
│   └── prediction_sample.png
├── data_load.py                   # โหลดภาพจากโฟลเดอร์ + gán label ตามชื่อคลาส (sorted)
├── preprocess.py                  # Grayscale + Resize + Flatten + Normalize
├── split_data.py                  # แบ่ง Train/Test (Stratified 80/20)
├── svm_model.py                   # StandardScaler + PCA(150) + SVC(kernel=rbf, C=10)
├── evaluate.py                    # Accuracy, Classification Report, Confusion Matrix
├── test_svm.py                    # ทดสอบโมเดล: ถ้ามีรูปใน test_photos/ ใช้รูปนั้นก่อน ไม่มีค่อยสุ่มจาก Test Set
├── test_photos/                   # *** วางรูปที่อยากทดสอบไว้ที่นี่ (ว่าง = ใช้ Test Set แทน) ***
├── predict_image.py               # *** ใช้งานจริง: รับภาพถ่ายอะไรก็ได้ + Haar Cascade auto-crop ***
├── webcam_demo.py                 # *** ใช้งานจริงแบบ real-time ผ่านกล้อง/เว็บแคม ***
├── main.py                        # รัน Pipeline ทั้งหมดตั้งแต่เริ่มจนจบ
├── flowchart.dot / flowchart.png  # ผังงานของระบบ
└── README.md
```

โครงสร้างและโค้ดนี้ปรับมาจาก `Homework04_SVM` (Cat vs Dog) โดยตรง — สถาปัตยกรรม pipeline เหมือนเดิมทุกไฟล์ เปลี่ยนแค่ dataset และหัวข้อ ทำให้มั่นใจได้ว่าวิธีการเดิมที่เข้าใจแล้วสามารถ "ประยุกต์ใช้" ข้ามโดเมนได้จริง

## Dataset

* **Dataset**: Face-Mask-Detection dataset (chandrikadeb7)
* **Link**: https://github.com/chandrikadeb7/Face-Mask-Detection
* ใช้ภาพทั้งหมดในชุดข้อมูล: `with_mask` 2,162 ภาพ + `without_mask` 1,930 ภาพ = **4,092 ภาพ**
* dataset แนบมาเป็นไฟล์เดียว `dataset.zip` (17MB) — แตกไฟล์ด้วย:
  ```bash
  python extract_dataset.py
  ```
  จะได้ `dataset/with_mask/` (2,165 ภาพ) และ `dataset/without_mask/` (1,930 ภาพ) กลับมาครบ (ใช้เฉพาะตอนจะเทรนใหม่เท่านั้น)
* `dataset.zip` เป็น**สำเนาที่ย่อขนาดแล้ว** (ย่อด้านยาวสุดเหลือ 160px, JPEG q85) จากต้นฉบับ 163MB ให้เหลือ 17MB เพื่อให้เก็บไว้กับโปรเจกต์ได้ — ไม่กระทบผลลัพธ์เพราะ pipeline ย่อทุกภาพเหลือ 100×100 grayscale อยู่แล้ว (ตรวจสอบแล้ว: เทรนใหม่จากไฟล์นี้ได้ test set 819 ภาพเท่าเดิม, accuracy 84.0% เทียบกับ 84.62% ของต้นฉบับ) ถ้าต้องการภาพความละเอียดเต็ม ดาวน์โหลดจากลิงก์ด้านบน
* **ถ้าแค่จะใช้งาน/สาธิต ไม่ต้องแตก dataset เลย** — โมเดลที่เทรนเสร็จแล้วอยู่ใน `outputs/svm_model.pkl` + `outputs/scaler.pkl` พร้อมใช้งานทันที
* ไฟล์ `dataset_part*.zip` และโฟลเดอร์ `dataset/` ถูกใส่ไว้ใน `.gitignore` แล้ว จะไม่ถูก push ขึ้น GitHub (ใหญ่เกินไป)

## Setup

1. สร้าง virtual environment แล้วติดตั้ง dependencies:
   ```bash
   pip install numpy opencv-python scikit-learn matplotlib joblib
   ```
   > ใช้ `opencv-python` (ไม่ใช่ `opencv-python-headless`) เพราะ `webcam_demo.py` ต้องใช้ `cv2.imshow` แสดงหน้าต่างวิดีโอ และห้ามลงทั้งสองแพ็กเกจพร้อมกัน (ทำให้ `cv2.CascadeClassifier` หายไปได้)
2. วางรูปภาพไว้ที่ `dataset/with_mask/` และ `dataset/without_mask/` (ถ้าต้องการเทรนใหม่ — ข้ามได้ถ้าจะใช้โมเดลที่เทรนไว้แล้วใน `outputs/`)

## How to Run

```bash
python main.py                       # รัน Pipeline ทั้งหมด: โหลด -> Preprocess -> แบ่งข้อมูล -> เทรน SVM -> ประเมินผล
python test_svm.py                   # ทดสอบโมเดล (ดูหัวข้อ test_photos/ ด้านล่าง)

# --- ใช้งานจริง (Applied usage) ---
python predict_image.py photo.jpg    # ใส่รูปอะไรก็ได้ (ไม่ต้อง crop เอง) ระบบตรวจจับใบหน้าอัตโนมัติแล้วทาย
python webcam_demo.py                # เปิดกล้อง ตรวจจับ+ทายแบบ real-time (กด q เพื่อออก) — ต้องรันบนเครื่องที่มีกล้องจริง ไม่ใช่ cloud sandbox
```

### โฟลเดอร์ `test_photos/` — วางรูปทดสอบเองได้ทันที

`test_svm.py` เช็คโฟลเดอร์นี้ก่อนเสมอ:

* **มีรูปอยู่** -> ใช้รูปในโฟลเดอร์นี้ทดสอบก่อน (ผ่าน Haar Cascade auto-crop เหมือน `predict_image.py` จริง)
* **ว่างเปล่า** -> สุ่มภาพจาก test set ของ dataset (X_test.npy) มาทดสอบแทนเหมือนเดิม

ถ้าตั้งชื่อไฟล์ขึ้นต้นด้วยชื่อคลาส เช่น `with_mask_1.jpg` หรือ `without_mask_test.png` สคริปต์จะรู้ ground truth จากชื่อไฟล์เองและขึ้นสีเขียว/แดงบอกถูก-ผิดให้อัตโนมัติ ถ้าตั้งชื่ออย่างอื่นจะโชว์แค่ผลทำนายเฉย ๆ (ไม่ฟันธงถูก/ผิดเพราะไม่รู้คำตอบจริง)

## Flowchart (ผังงาน)

ดูขั้นตอนการทำงานแบบผังงาน (มาตรฐาน Start/End, Input/Output, Process, Decision) ได้ที่ [`flowchart.png`](flowchart.png)

## Pipeline และ Model Configuration

1. **Preprocessing** (`preprocess.py`) — แปลง BGR → Grayscale, Resize เป็น 100×100 พิกเซล, แผ่เป็นเวกเตอร์ 10,000 features, Normalize ช่วง [0,1]
2. **Feature Scaling & PCA** (`svm_model.py`) — `StandardScaler` + `PCA(n_components=150, whiten=True)` ลดมิติจาก 10,000 → 150
3. **SVM Classification** (`svm_model.py`) — `SVC(kernel="rbf", C=10, gamma="scale", probability=True)`
4. **Applied inference** (`predict_image.py`, `webcam_demo.py`) — เพิ่ม Haar Cascade Face Detector (`cv2.data.haarcascades`) หน้า pipeline เดิม เพื่อตัดใบหน้าจากภาพถ่ายทั่วไปโดยอัตโนมัติ (ขยายกรอบที่ตรวจจับได้ ~20-25% ให้ครอบคลุมถึงคาง/ปาก ก่อนส่งเข้าโมเดล)

## ผลลัพธ์จริง (Real Results)

| Metric | Value |
| --- | --- |
| Total images | 4,092 (2,162 with_mask + 1,930 without_mask) |
| Train / Test split | 3,273 / 819 (Stratified, 80/20) |
| **Test Accuracy** | **84.62%** |
| Confusion Matrix | with_mask: 372/433 ถูก · without_mask: 321/386 ถูก |
| Precision (with_mask / without_mask) | 0.85 / 0.84 |
| Recall (with_mask / without_mask) | 0.86 / 0.83 |
| F1-score (with_mask / without_mask) | 0.86 / 0.84 |

## การทดสอบใช้งานจริง (Generalization test — สำคัญมาก)

### รอบที่ 1: จำนวนข้อมูลเทรนกระทบการ generalize

ตอนแรกเทรนด้วยข้อมูลแค่ 500 ภาพ/คลาส ได้ test accuracy 81.5% บน held-out set ของ dataset เดียวกัน — **แต่พอเอาไปทดสอบกับภาพถ่ายใหม่จริง 4 ภาพที่ไม่ได้มาจากส่วนที่ใช้เทรน/ทดสอบเลย โมเดลทายผิดไปถึง 3 ใน 4 ภาพ** แม้ภาพจะเห็นใบหน้าชัดเจนก็ตาม — สัญญาณของ **overfitting กับข้อมูลจำนวนน้อย**

หลังจากเทรนใหม่ด้วยข้อมูลเต็มชุด (4,092 ภาพ) แล้วทดสอบกับภาพชุดเดิมทั้ง 4 ภาพอีกครั้ง **ผลลัพธ์ถูกต้องครบ 4/4 ภาพ** (ดูภาพจริงใน `demo_photos/`):

| ภาพ | Ground Truth | ทำนาย (500/class) | ทำนาย (Full dataset) |
| --- | --- | --- | --- |
| test_unseen_with_mask.jpg | with_mask | ❌ without_mask (57%) | ✅ with_mask (96%) |
| test_unseen_without_mask.jpg | without_mask | ❌ with_mask (76%) | ✅ without_mask (55%) |
| test_unseen_with_mask2.jpg | with_mask | ✅ with_mask (70%) | ✅ with_mask (96%) |
| test_unseen_without_mask2.jpg | without_mask | ❌ with_mask (76%) | ✅ without_mask (90%) |

**บทเรียน**: accuracy บน held-out test set ของ dataset เดียวกัน (81.5% vs 84.6%) ไม่ต่างกันมาก แต่ **ความสามารถในการ generalize ไปยังภาพใหม่จริงต่างกันชัดเจน** (1/4 ถูก vs 4/4 ถูก) — ควรทดสอบระบบ ML กับข้อมูลนอกชุด test เสมอ ก่อนบอกว่า "ใช้งานจริงได้"

### รอบที่ 2: real-time ผ่านกล้อง/มือถือ พบข้อจำกัดใหม่ (domain shift)

ทดสอบเพิ่มด้วยภาพถ่ายเซลฟีจริงของผู้พัฒนาเอง 4 ภาพ (ไม่มี/มีแมส × ไม่ใส่/ใส่แว่น) ผ่าน `predict_image.py`:

| ภาพ | Ground Truth | Haar Cascade ตรวจจับใบหน้า | ทำนาย | ผล |
| --- | --- | --- | --- | --- |
| ไม่ใส่แมส | without_mask | เจอ | with_mask (55%) | ❌ |
| ใส่แมส | with_mask | **ไม่เจอ** (fallback: classify ทั้งภาพ) | with_mask (56%) | ✅ |
| ไม่ใส่แมส + แว่น | without_mask | เจอ | without_mask (65%) | ✅ |
| ใส่แมส + แว่น | with_mask | **ไม่เจอ** (fallback: classify ทั้งภาพ) | without_mask (74%) | ❌ |

ผลลัพธ์ **2/4 ถูก** ต่ำกว่าชุด `demo_photos/` เดิมมาก (4/4) ทั้งที่เป็นโมเดลเดียวกัน — สาเหตุคือ **domain shift**: ภาพจากกล้องเว็บแคม/มือถือ (สี, แสง, การบีบอัดจาก Camo virtual camera) ต่างจากสไตล์ภาพใน dataset ต้นทางที่ใช้เทรน โมเดลแบบ raw-pixel + PCA + SVM ไวต่อความต่างของ "อุปกรณ์ที่มา" ของภาพมาก (คนละปัญหากับ overfitting ในรอบที่ 1) และ **Haar Cascade หาใบหน้าไม่เจอเลยเมื่อใส่แมส** ในทั้ง 2 ภาพที่ทดสอบ ต้องอาศัย fallback "classify ทั้งภาพ" ที่เขียนไว้ใน `predict_image.py`

**สรุปข้อจำกัดที่ค้นพบจากการทดสอบจริง 2 รอบ**:
1. ข้อมูลเทรนน้อยเกินไป → overfit → generalize แย่ (แก้แล้วด้วยข้อมูลเต็มชุด)
2. Haar Cascade ตรวจจับใบหน้าพลาด/ไม่เจอเลยเมื่อมีการบดบัง (ใส่แมส) — เป็นข้อจำกัดของตัว face detector เอง
3. Domain shift ระหว่างภาพเทรนกับภาพจากอุปกรณ์จับภาพจริง (กล้องเว็บแคม/มือถือ) — เป็นข้อจำกัดของวิธี raw-pixel ที่ไม่ทนต่อความต่างของอุปกรณ์เท่า deep learning

## จุดที่ต่อยอดได้ต่อ (Future work)

* เปลี่ยนจาก Haar Cascade เป็น face detector ที่ทนต่อการบดบัง/แมสมากกว่า (เช่น MTCNN, RetinaFace, DNN-based face detector)
* เก็บข้อมูลเทรนเพิ่มจากกล้องเว็บแคม/มือถือโดยตรง ไม่ใช่แค่ภาพจาก dataset สาธารณะ เพื่อลด domain shift
* เพิ่มคลาสที่ 3 เช่น "สวมหน้ากากไม่ถูกวิธี" (mask worn incorrectly) เพื่อความสมบูรณ์ของระบบตรวจสอบ compliance
