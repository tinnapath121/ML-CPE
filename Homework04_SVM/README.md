```python
readme_content = """# SVM Image Recognition (Cat vs. Dog Classification)

ระบบจำแนกและประมวลผลภาพสัตว์ (Cat vs. Dog) ด้วยอัลกอริทึม **Support Vector Machine (SVM)** ร่วมกับ **Principal Component Analysis (PCA)** พัฒนาด้วยภาษา Python และไลบรารีทางด้าน Machine Learning / Computer Vision

---

## 📌 คุณสมบัติเด่น (Features)

- **Automatic Class Detection**: ตรวจจับโฟลเดอร์คลาสของรูปภาพในโครงสร้างข้อมูลโดยอัตโนมัติ
- **Image Preprocessing & Normalization**: ปรับขนาดภาพเป็น $100 \\times 100$ พิกเซล, แปลงเป็น Grayscale และทำ Min-Max Normalization (ช่วง $0 - 1$)
- **Dimensionality Reduction**: ลดมิติข้อมูลรูปภาพ ($10,000$ คุณลักษณะ) ด้วย **PCA (150 Components)** ร่วมกับ **StandardScaler** เพิ่มประสิทธิภาพและลดเวลาในการฝึกโมเดล
- **SVM Classifier**: ใช้ **Support Vector Machine (RBF Kernel)** ในการจำแนกคลาส พร้อมคำนวณค่าความมั่นใจ (Confidence Score / Probability %)
- **Model Evaluation & Visualization**: 
  - คำนวณ Accuracy, Classification Report และ Confusion Matrix
  - สร้างภาพผลลัพธ์การทำนายพร้อมเปรียบเทียบค่าจริง (True) และค่าทำนาย (Pred) แสดงผลสีเขียว/แดงตามความถูกต้อง

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)


```

.
├── animals/                     # โฟลเดอร์เก็บข้อมูลภาพแบ่งตามคลาส
│   ├── cat/                     # รูปภาพแมว (.jpg, .png ฯลฯ)
│   └── dog/                     # รูปภาพสุนัข (.jpg, .png ฯลฯ)
├── outputs/                     # โฟลเดอร์เก็บผลลัพธ์และโมเดลที่บันทึกไว้
│   ├── svm_model.pkl            # โมเดล SVM ที่ผ่านการเทรน
│   ├── scaler.pkl               # Pipeline สำหรับ Scaler + PCA
│   ├── confusion_matrix.png     # กราฟ Confusion Matrix
│   └── prediction_sample.png    # สรุปผลการสุ่มทำนายภาพ 4 ภาพ
├── data_load.py                 # โหลดและอ่านไฟล์รูปภาพจากโฟลเดอร์
├── preprocess.py               # ปรับขนาด แปลงสี และแปลงภาพเป็น Vector คุณลักษณะ
├── split_data.py                # แบ่งข้อมูลเป็น Train Set และ Test Set (Stratified)
├── svm_model.py                 # สร้างและเทรนโมเดล SVM (StandardScaler + PCA + SVC)
├── evaluate.py                  # ประเมินผลโมเดล (Accuracy, Report, Confusion Matrix)
├── test_svm.py                  # สุ่มดึงภาพจาก Test Set มาทำนายและแสดงผลภาพตาราง 2x2
├── main.py                      # สคริปต์หลักรัน Pipeline ทั้งหมดตั้งแต่เริ่มจนจบ
└── README.md                    # เอกสารอธิบายโปรเจกต์

```

---

## 🛠️ การติดตั้งและเตรียมสภาพแวดล้อม (Installation & Setup)

### 1. ความต้องการของระบบ (Requirements)
โปรเจกต์นี้รองรับ **Python 3.8+** และใช้ไลบรารีหลักดังนี้:

```bash
pip install numpy opencv-python scikit-learn matplotlib joblib

```

### 2. การเตรียมข้อมูลภาพ (Dataset Setup)

จัดเก็บรูปภาพแยกโฟลเดอร์ตามชื่อคลาสภายในโฟลเดอร์ `animals/` ดังนี้:

```text
animals/
├── cat/
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   └── ...
└── dog/
    ├── dog.1.jpg
    ├── dog.2.jpg
    └── ...

```

---

## 🚀 วิธีการใช้งาน (Usage)

### 1. ฝึกสอนโมเดลและประเมินผล (Train & Evaluate)

รันไฟล์ `main.py` เพื่อรัน Pipeline ทั้งหมด (โหลดภาพ -> Preprocess -> แบ่งข้อมูล -> เทรน SVM -> บันทึกโมเดล -> ประเมินผล):

```bash
python main.py

```

**ขั้นตอนที่โปรแกรมจะทำงาน:**

1. โหลดภาพจาก `animals/` และแปลงเป็นรูปแบบ Numpy Array
2. แปลงภาพเป็น feature matrix ขนาด $(N, 10000)$ และแบ่ง Train/Test (80/20)
3. ปรับสเกลข้อมูล ทำ PCA Reduction เหลือ 150 components และเทรน SVM (RBF Kernel)
4. บันทึกโมเดลไว้ที่ `outputs/svm_model.pkl` และ `outputs/scaler.pkl`
5. คำนวณค่า Performance และเซฟไฟล์ `outputs/confusion_matrix.png`

---

### 2. ทดสอบสุ่มทำนายภาพและแสดงผลแบบ Visual (Test & Visualize)

รันไฟล์ `test_svm.py` เพื่อสุ่มดึงภาพทดสอบมา 4 รูป ทำนายผล และบันทึกเป็นรูปภาพตารางแสดงผลลัพธ์:

```bash
python test_svm.py

```

* ผลลัพธ์จะถูกบันทึกเป็นไฟล์ `outputs/prediction_sample.png`
* แสดงชื่อคลาสที่ทาย (`Pred`), เปอร์เซ็นต์ความมั่นใจ (`%`), คลาสจริง (`True`) พร้อมการไฮไลต์สีเขียวเมื่อทายถูก หรือสีแดงเมื่อทายผิด

---

## 📊 รายละเอียด Pipeline และอัลกอริทึม

1. **Preprocessing (`preprocess.py`)**
* แปลงภาพ BGR เป็น Grayscale เพื่อลดขนาดมิติสีจาก 3 แชนเนลเหลือ 1 แชนเนล
* Resize ภาพเป็นขนาด $100 \times 100$ พิกเซล ด้วย `cv2.INTER_AREA`
* แผ่อาร์เรย์เป็น 1D Vector ความยาว $10,000$ คุณลักษณะ และหารด้วย $255.0$ เพื่อแปลงช่วงสเกลพิกเซลเป็น $[0.0, 1.0]$


2. **Feature Scaling & PCA (`svm_model.py`)**
* **`StandardScaler`**: ปรับมาตรฐานข้อมูลให้มีเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1
* **`PCA(n_components=150)`**: สกัดเฉพาะคุณลักษณะสำคัญที่สุด 150 องค์ประกอบ ช่วยแก้ปัญหา *Curse of Dimensionality* และเพิ่มความเร็วในการเทรนโมเดลอย่างมาก


3. **SVM Classification (`svm_model.py`)**
* **`SVC(kernel='rbf', C=10, probability=True)`**: ใช้ RBF Kernel ในการสร้าง Hyperplane แบ่งคลาสแบบ Non-linear พร้อมเปิดคำนวณ Probability



---

## 📈 ผลลัพธ์ไฟล์ในโฟลเดอร์ `outputs/`

| ชื่อไฟล์ | คำอธิบาย |
| --- | --- |
| `svm_model.pkl` | โมเดล SVM (SVC) ที่ฝึกสอนเสร็จเรียบร้อย |
| `scaler.pkl` | Pipeline รวม `StandardScaler` และ `PCA` สำหรับแปลงข้อมูลนำเข้า |
| `classes.json` | รายชื่อคลาสทั้งหมด (เช่น `["cat", "dog"]`) |
| `confusion_matrix.png` | กราฟ Confusion Matrix แสดงประสิทธิภาพการทายผล |
| `prediction_sample.png` | ภาพตารางสรุปผลการสุ่มทำนายภาพพร้อม % ความมั่นใจ |
| """ |  |

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

```

Your Markdown file is ready
[file-tag: code-generated-file-2ac1337e-76b6-450d-865b-a963f6b41994]

เนื้อหาภายใน **`README.md`** ที่สร้างขึ้นสำหรับโปรเจกต์ของคุณ:

```markdown
# SVM Image Recognition (Cat vs. Dog Classification)

ระบบจำแนกและประมวลผลภาพสัตว์ (Cat vs. Dog) ด้วยอัลกอริทึม **Support Vector Machine (SVM)** ร่วมกับ **Principal Component Analysis (PCA)** พัฒนาด้วยภาษา Python และไลบรารีทางด้าน Machine Learning / Computer Vision

---

## 📌 คุณสมบัติเด่น (Features)

- **Automatic Class Detection**: ตรวจจับโฟลเดอร์คลาสของรูปภาพในโครงสร้างข้อมูลโดยอัตโนมัติ
- **Image Preprocessing & Normalization**: ปรับขนาดภาพเป็น 100x100 พิกเซล, แปลงเป็น Grayscale และทำ Min-Max Normalization (ช่วง 0 - 1)
- **Dimensionality Reduction**: ลดมิติข้อมูลรูปภาพ (10,000 คุณลักษณะ) ด้วย **PCA (150 Components)** ร่วมกับ **StandardScaler** เพิ่มประสิทธิภาพและลดเวลาในการฝึกโมเดล
- **SVM Classifier**: ใช้ **Support Vector Machine (RBF Kernel)** ในการจำแนกคลาส พร้อมคำนวณค่าความมั่นใจ (Confidence Score / Probability %)
- **Model Evaluation & Visualization**: 
  - คำนวณ Accuracy, Classification Report และ Confusion Matrix
  - สร้างภาพผลลัพธ์การทำนายพร้อมเปรียบเทียบค่าจริง (True) และค่าทำนาย (Pred) แสดงผลสีเขียว/แดงตามความถูกต้อง

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
.
├── animals/                     # โฟลเดอร์เก็บข้อมูลภาพแบ่งตามคลาส
│   ├── cat/                     # รูปภาพแมว (.jpg, .png ฯลฯ)
│   └── dog/                     # รูปภาพสุนัข (.jpg, .png ฯลฯ)
├── outputs/                     # โฟลเดอร์เก็บผลลัพธ์และโมเดลที่บันทึกไว้
│   ├── svm_model.pkl            # โมเดล SVM ที่ผ่านการเทรน
│   ├── scaler.pkl               # Pipeline สำหรับ Scaler + PCA
│   ├── confusion_matrix.png     # กราฟ Confusion Matrix
│   └── prediction_sample.png    # สรุปผลการสุ่มทำนายภาพ 4 ภาพ
├── data_load.py                 # โหลดและอ่านไฟล์รูปภาพจากโฟลเดอร์
├── preprocess.py               # ปรับขนาด แปลงสี และแปลงภาพเป็น Vector คุณลักษณะ
├── split_data.py                # แบ่งข้อมูลเป็น Train Set และ Test Set (Stratified)
├── svm_model.py                 # สร้างและเทรนโมเดล SVM (StandardScaler + PCA + SVC)
├── evaluate.py                  # ประเมินผลโมเดล (Accuracy, Report, Confusion Matrix)
├── test_svm.py                  # สุ่มดึงภาพจาก Test Set มาทำนายและแสดงผลภาพตาราง 2x2
├── main.py                      # สคริปต์หลักรัน Pipeline ทั้งหมดตั้งแต่เริ่มจนจบ
└── README.md                    # เอกสารอธิบายโปรเจกต์

```

---

## 🛠️ การติดตั้งและเตรียมสภาพแวดล้อม (Installation & Setup)

### 1. ความต้องการของระบบ (Requirements)

โปรเจกต์นี้รองรับ **Python 3.8+** และใช้ไลบรารีหลักดังนี้:

```bash
pip install numpy opencv-python scikit-learn matplotlib joblib

```

### 2. การเตรียมข้อมูลภาพ (Dataset Setup)

จัดเก็บรูปภาพแยกโฟลเดอร์ตามชื่อคลาสภายในโฟลเดอร์ `animals/` ดังนี้:

```text
animals/
├── cat/
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   └── ...
└── dog/
    ├── dog.1.jpg
    ├── dog.2.jpg
    └── ...

```

---

## 🚀 วิธีการใช้งาน (Usage)

### 1. ฝึกสอนโมเดลและประเมินผล (Train & Evaluate)

รันไฟล์ `main.py` เพื่อรัน Pipeline ทั้งหมด (โหลดภาพ -> Preprocess -> แบ่งข้อมูล -> เทรน SVM -> บันทึกโมเดล -> ประเมินผล):

```bash
python main.py

```

**ขั้นตอนที่โปรแกรมจะทำงาน:**

1. โหลดภาพจาก `animals/` และแปลงเป็นรูปแบบ Numpy Array
2. แปลงภาพเป็น feature matrix ขนาด $(N, 10000)$ และแบ่ง Train/Test (80/20)
3. ปรับสเกลข้อมูล ทำ PCA Reduction เหลือ 150 components และเทรน SVM (RBF Kernel)
4. บันทึกโมเดลไว้ที่ `outputs/svm_model.pkl` และ `outputs/scaler.pkl`
5. คำนวณค่า Performance และเซฟไฟล์ `outputs/confusion_matrix.png`

---

### 2. ทดสอบสุ่มทำนายภาพและแสดงผลแบบ Visual (Test & Visualize)

รันไฟล์ `test_svm.py` เพื่อสุ่มดึงภาพทดสอบมา 4 รูป ทำนายผล และบันทึกเป็นรูปภาพตารางแสดงผลลัพธ์:

```bash
python test_svm.py

```

* ผลลัพธ์จะถูกบันทึกเป็นไฟล์ `outputs/prediction_sample.png`
* แสดงชื่อคลาสที่ทาย (`Pred`), เปอร์เซ็นต์ความมั่นใจ (`%`), คลาสจริง (`True`) พร้อมการไฮไลต์สีเขียวเมื่อทายถูก หรือสีแดงเมื่อทายผิด

---

## 📊 รายละเอียด Pipeline และอัลกอริทึม

1. **Preprocessing (`preprocess.py`)**
* แปลงภาพ BGR เป็น Grayscale เพื่อลดขนาดมิติสีจาก 3 แชนเนลเหลือ 1 แชนเนล
* Resize ภาพเป็นขนาด 100x100 พิกเซล ด้วย `cv2.INTER_AREA`
* แผ่อาร์เรย์เป็น 1D Vector ความยาว 10,000 คุณลักษณะ และหารด้วย 255.0 เพื่อแปลงช่วงสเกลพิกเซลเป็น [0.0, 1.0]


2. **Feature Scaling & PCA (`svm_model.py`)**
* **`StandardScaler`**: ปรับมาตรฐานข้อมูลให้มีเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1
* **`PCA(n_components=150)`**: สกัดเฉพาะคุณลักษณะสำคัญที่สุด 150 องค์ประกอบ ช่วยแก้ปัญหา *Curse of Dimensionality* และเพิ่มความเร็วในการเทรนโมเดลอย่างมาก


3. **SVM Classification (`svm_model.py`)**
* **`SVC(kernel='rbf', C=10, probability=True)`**: ใช้ RBF Kernel ในการสร้าง Hyperplane แบ่งคลาสแบบ Non-linear พร้อมเปิดคำนวณ Probability



---

## 📈 ผลลัพธ์ไฟล์ในโฟลเดอร์ `outputs/`

| ชื่อไฟล์ | คำอธิบาย |
| --- | --- |
| `svm_model.pkl` | โมเดล SVM (SVC) ที่ฝึกสอนเสร็จเรียบร้อย |
| `scaler.pkl` | Pipeline รวม `StandardScaler` และ `PCA` สำหรับแปลงข้อมูลนำเข้า |
| `classes.json` | รายชื่อคลาสทั้งหมด (เช่น `["cat", "dog"]`) |
| `confusion_matrix.png` | กราฟ Confusion Matrix แสดงประสิทธิภาพการทายผล |
| `prediction_sample.png` | ภาพตารางสรุปผลการสุ่มทำนายภาพพร้อม % ความมั่นใจ |

```

```