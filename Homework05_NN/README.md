```python
readme_content = """# 🧠 Machine Learning Lab 06: Neural Network Image Classification

ระบบจำแนกภาพด้วย **Neural Network (Multi-Layer Perceptron - MLP)** สำหรับรายวิชา **04-624-201 Machine Learning** ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี (RMUTT)

โครงการนี้รองรับการโหลดข้อมูลภาพ การเตรียมข้อมูล (Preprocessing) การแบ่งข้อมูลเป็น Train/Validation/Test การเทรนโมเดลโครงข่ายประสาทเทียมพร้อมติดตามประสิทธิภาพ การเปรียบเทียบสถาปัตยกรรม (Configurations & Epochs) และการแสดงผลลัพธ์ผ่านกราฟประเมินผล

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
ML-06-NN/
├── mini-proj/
│   ├── data_loader.py      # โหลดรูปภาพจากโฟลเดอร์ตามชื่อ Class อัตโนมัติ
│   ├── preprocessing.py    # ปรับขนาดภาพ (Resize) และแปลงโครงสร้าง Feature
│   ├── split_data.py        # แบ่งชุดข้อมูลเป็น Train / Validation / Test
│   ├── nn_model.py          # สร้างและเทรนโครงข่ายประสาทเทียม (Keras/TensorFlow)
│   ├── evaluate.py         # คำนวณ Accuracy, Classification Report, Confusion Matrix
│   ├── main.py             # สคริปต์หลักรัน Pipeline การเทรนและการเปรียบเทียบผล
│   ├── test_nn.py          # ทดสอบสุ่มทำนายภาพ 4 รูปภาพหลังเทรนเสร็จ
│   ├── animals/            # โฟลเดอร์เก็บ Dataset (แบ่งตามโฟลเดอร์ย่อยคลาส)
│   │   ├── Cat/
│   │   └── Dog/
│   └── outputs/            # โฟลเดอร์เก็บผลลัพธ์ (สร้างอัตโนมัติเมื่อรันโค้ด)
│       ├── nn_model.keras
│       ├── training_history.png
│       ├── confusion_matrix.png
│       ├── prediction_sample.png
│       └── experiment_results.csv
└── README.md

```

---

## 🛠️ ความต้องการของระบบ (Prerequisites & Installation)

โปรเจกต์นี้ทำงานบน **Python 3.8+** โดยต้องติดตั้งไลบรารีพื้นฐานดังนี้:

```bash
pip install tensorflow opencv-python matplotlib scikit-learn numpy pandas

```

---

## 📂 การเตรียม Dataset (Dataset Setup)

จัดวางรูปภาพในโฟลเดอร์ตามชื่อคลาส (Class Name) ภายในโฟลเดอร์ `animals` หรือชื่อโฟลเดอร์ที่กำหนดไว้ใน `DATA_PATH`:

```text
mini-proj/animals/
├── Cat/
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   └── ...
└── Dog/
    ├── dog.1.jpg
    ├── dog.2.jpg
    └── ...

```

*รองรับไฟล์นามสกุล `.jpg`, `.jpeg`, `.png`, และ `.bmp*`

---

## 🚀 วิธีการใช้งาน (How to Run)

### 1. เทรนโมเดลและประเมินผล (Main Pipeline)

รันสคริปต์หลักเพื่อประมวลผล เทรนโมเดล และสร้างรายงานการทดลอง:

```bash
python main.py

```

**สิ่งที่ระบบจะดำเนินการ:**

1. โหลดข้อมูลภาพและปรับขนาดเป็น $100 \times 100$ พิกเซล
2. แบ่งข้อมูลอัตราส่วน **Train 70% / Validation 10% / Test 20%**
3. ทดลองเปรียบเทียบโครงสร้าง Neural Network ต่างๆ (1, 2, 3 Hidden Layers) และจำนวน Epochs (10, 20, 30)
4. บันทึกผลลัพธ์การทดลองลงใน `outputs/`

### 2. ทดสอบการทำนายผลด้วยภาพสุ่ม (Test Model)

หลังรัน `main.py` เรียบร้อยแล้ว สั่งรันสคริปต์นี้เพื่อสุ่มรูปภาพ 4 รูปขึ้นมาทำนายผล:

```bash
python test_nn.py

```

---

## 📊 ผลลัพธ์และไฟล์ Output (Output Artifacts)

เมื่อรันโปรแกรมสำเร็จ ระบบจะสร้างไฟล์ผลลัพธ์ไว้ในโฟลเดอร์ `outputs/`:

* **`nn_model.keras`**: ไฟล์โมเดลที่ผ่านการเทรนเรียบร้อยแล้ว
* **`training_history.png`**: กราฟเปรียบเทียบค่า **Accuracy** และ **Loss** ระหว่าง Train และ Validation
* **`confusion_matrix.png`**: แผนภาพ Confusion Matrix แสดงความถูกต้องในการจำแนกแต่ละคลาส
* **`prediction_sample.png`**: ภาพตัวอย่างการทำนายผลแบบ Grid 2x2 พร้อมแสดงค่าความมั่นใจ (Confidence Score %)
* **`experiment_results.csv`**: ตารางสรุปการเปรียบเทียบประสิทธิภาพของแต่ละ Configuration และ Epochs

---

## 🏗️ สถาปัตยกรรมโมเดล (Model Architecture)

โมเดลที่ใช้คือ **Multi-Layer Perceptron (MLP)** สร้างด้วย TensorFlow / Keras ประกอบด้วย:

* **Rescaling Layer**: ปรับ Scale พิกเซลภาพจาก $[0, 255]$ เป็น $[0.0, 1.0]$ อัตโนมัติ
* **Flatten Layer**: แปลงภาพ 2D ให้เป็น Vector 1D
* **Hidden Dense Layers**: ประกอบด้วยโครงสร้างแบบปรับเปลี่ยนได้ (เช่น 256, 128, 64 neurons) พร้อมใช้งาน **BatchNormalization** และ **Dropout** เพื่อลดปัญหา Overfitting
* **Output Layer**: ใช้ **Sigmoid** สำหรับ Binary Classification (2 คลาส) หรือ **Softmax** สำหรับ Multi-class Classification (>2 คลาส)
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

print("README.md generated successfully.")

```

```text?code_stdout&code_event_index=1
README.md generated successfully.


```

ไฟล์ Markdown (README.md) สำหรับโปรเจกต์ของคุณเสร็จเรียบร้อยแล้วครับ


---

### เนื้อหาภายใน `README.md`

```markdown
# 🧠 Machine Learning Lab 06: Neural Network Image Classification

ระบบจำแนกภาพด้วย **Neural Network (Multi-Layer Perceptron - MLP)** สำหรับรายวิชา **04-624-201 Machine Learning** ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี (RMUTT)

โครงการนี้รองรับการโหลดข้อมูลภาพ การเตรียมข้อมูล (Preprocessing) การแบ่งข้อมูลเป็น Train/Validation/Test การเทรนโมเดลโครงข่ายประสาทเทียมพร้อมติดตามประสิทธิภาพ การเปรียบเทียบสถาปัตยกรรม (Configurations & Epochs) และการแสดงผลลัพธ์ผ่านกราฟประเมินผล

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
ML-06-NN/
├── mini-proj/
│   ├── data_loader.py      # โหลดรูปภาพจากโฟลเดอร์ตามชื่อ Class อัตโนมัติ
│   ├── preprocessing.py    # ปรับขนาดภาพ (Resize) และแปลงโครงสร้าง Feature
│   ├── split_data.py        # แบ่งชุดข้อมูลเป็น Train / Validation / Test
│   ├── nn_model.py          # สร้างและเทรนโครงข่ายประสาทเทียม (Keras/TensorFlow)
│   ├── evaluate.py         # คำนวณ Accuracy, Classification Report, Confusion Matrix
│   ├── main.py             # สคริปต์หลักรัน Pipeline การเทรนและการเปรียบเทียบผล
│   ├── test_nn.py          # ทดสอบสุ่มทำนายภาพ 4 รูปภาพหลังเทรนเสร็จ
│   ├── animals/            # โฟลเดอร์เก็บ Dataset (แบ่งตามโฟลเดอร์ย่อยคลาส)
│   │   ├── Cat/
│   │   └── Dog/
│   └── outputs/            # โฟลเดอร์เก็บผลลัพธ์ (สร้างอัตโนมัติเมื่อรันโค้ด)
│       ├── nn_model.keras
│       ├── training_history.png
│       ├── confusion_matrix.png
│       ├── prediction_sample.png
│       └── experiment_results.csv
└── README.md

```

---

## 🛠️ ความต้องการของระบบ (Prerequisites & Installation)

โปรเจกต์นี้ทำงานบน **Python 3.8+** โดยต้องติดตั้งไลบรารีพื้นฐานดังนี้:

```bash
pip install tensorflow opencv-python matplotlib scikit-learn numpy pandas

```

---

## 📂 การเตรียม Dataset (Dataset Setup)

จัดวางรูปภาพในโฟลเดอร์ตามชื่อคลาส (Class Name) ภายในโฟลเดอร์ `animals` หรือชื่อโฟลเดอร์ที่กำหนดไว้ใน `DATA_PATH`:

```text
mini-proj/animals/
├── Cat/
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   └── ...
└── Dog/
    ├── dog.1.jpg
    ├── dog.2.jpg
    └── ...

```

*รองรับไฟล์นามสกุล `.jpg`, `.jpeg`, `.png`, และ `.bmp*`

---

## 🚀 วิธีการใช้งาน (How to Run)

### 1. เทรนโมเดลและประเมินผล (Main Pipeline)

รันสคริปต์หลักเพื่อประมวลผล เทรนโมเดล และสร้างรายงานการทดลอง:

```bash
python main.py

```

**สิ่งที่ระบบจะดำเนินการ:**

1. โหลดข้อมูลภาพและปรับขนาดเป็น 100x100 พิกเซล
2. แบ่งข้อมูลอัตราส่วน **Train 70% / Validation 10% / Test 20%**
3. ทดลองเปรียบเทียบโครงสร้าง Neural Network ต่างๆ (1, 2, 3 Hidden Layers) และจำนวน Epochs (10, 20, 30)
4. บันทึกผลลัพธ์การทดลองลงใน `outputs/`

### 2. ทดสอบการทำนายผลด้วยภาพสุ่ม (Test Model)

หลังรัน `main.py` เรียบร้อยแล้ว สั่งรันสคริปต์นี้เพื่อสุ่มรูปภาพ 4 รูปขึ้นมาทำนายผล:

```bash
python test_nn.py

```

---

## 📊 ผลลัพธ์และไฟล์ Output (Output Artifacts)

เมื่อรันโปรแกรมสำเร็จ ระบบจะสร้างไฟล์ผลลัพธ์ไว้ในโฟลเดอร์ `outputs/`:

* **`nn_model.keras`**: ไฟล์โมเดลที่ผ่านการเทรนเรียบร้อยแล้ว
* **`training_history.png`**: กราฟเปรียบเทียบค่า **Accuracy** และ **Loss** ระหว่าง Train และ Validation
* **`confusion_matrix.png`**: แผนภาพ Confusion Matrix แสดงความถูกต้องในการจำแนกแต่ละคลาส
* **`prediction_sample.png`**: ภาพตัวอย่างการทำนายผลแบบ Grid 2x2 พร้อมแสดงค่าความมั่นใจ (Confidence Score %)
* **`experiment_results.csv`**: ตารางสรุปการเปรียบเทียบประสิทธิภาพของแต่ละ Configuration และ Epochs

---

## 🏗️ สถาปัตยกรรมโมเดล (Model Architecture)

โมเดลที่ใช้คือ **Multi-Layer Perceptron (MLP)** สร้างด้วย TensorFlow / Keras ประกอบด้วย:

* **Rescaling Layer**: ปรับ Scale พิกเซลภาพจาก [0, 255] เป็น [0.0, 1.0] อัตโนมัติ
* **Flatten Layer**: แปลงภาพ 2D ให้เป็น Vector 1D
* **Hidden Dense Layers**: ประกอบด้วยโครงสร้างแบบปรับเปลี่ยนได้ (เช่น 256, 128, 64 neurons) พร้อมใช้งาน **BatchNormalization** และ **Dropout** เพื่อลดปัญหา Overfitting
* **Output Layer**: ใช้ **Sigmoid** สำหรับ Binary Classification (2 คลาส) หรือ **Softmax** สำหรับ Multi-class Classification (>2 คลาส)

```

```