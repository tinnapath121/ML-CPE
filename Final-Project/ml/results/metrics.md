# ผลการทดลอง (2026-11-03 21:10)

## อัตราตรวจจับใบหน้า (SSD MobileNet + เติมขอบ 20%) — project_01 ใช้ Haar Cascade แล้วหาหน้าไม่เจอเมื่อใส่แมส

|                          |   detect_rate |   count |
|:-------------------------|--------------:|--------:|
| ('ds1', 'with_mask')     |         0.828 | 900.000 |
| ('ds1', 'without_mask')  |         0.873 | 900.000 |
| ('ds2', 'with_mask')     |         0.983 | 644.000 |
| ('ds2', 'without_mask')  |         1.000 |  12.000 |
| ('real', 'with_mask')    |         1.000 |   2.000 |
| ('real', 'without_mask') |         1.000 |   2.000 |


## A) 5-fold CV บน dataset หลัก (chandrikadeb7)

| Model                                     |   Accuracy |   Macro F1 |
|:------------------------------------------|-----------:|-----------:|
| SVM บน pixel ดิบ (project_01 เดิม, 80/20) |      0.846 |      0.850 |
| Logistic Regression                       |      0.939 |      0.939 |
| SVM (RBF)                                 |      0.975 |      0.975 |
| Random Forest                             |      0.939 |      0.939 |
| MLP (deploy บนเว็บ)                       |      0.955 |      0.955 |



## B) เทรน ds1 → ทดสอบ ds2 (prajnasb, คนละแหล่ง)

| Model               |   Accuracy |   Macro F1 |   n |   with_mask n |
|:--------------------|-----------:|-----------:|----:|--------------:|
| Logistic Regression |      0.831 |      0.531 | 645 |           633 |
| SVM (RBF)           |      0.958 |      0.714 | 645 |           633 |
| Random Forest       |      0.794 |      0.512 | 645 |           633 |
| MLP (deploy บนเว็บ) |      0.878 |      0.576 | 645 |           633 |


หมายเหตุ: รูปไม่สวมหน้ากากของ ds2 ส่วนใหญ่ซ้ำกับ ds1 จึงถูกตัดทิ้ง ชุดนี้จึงวัดได้เป็นหลักว่า 'จับแมสแบบใหม่ที่ไม่เคยเห็นได้ไหม' (Accuracy ≈ Recall ของ with_mask) ส่วน Macro F1 ใช้ตีความไม่ได้


## C) รูปจริงจากมือถือผู้พัฒนา

| Model               |   Accuracy |   Macro F1 |   n |   with_mask n |
|:--------------------|-----------:|-----------:|----:|--------------:|
| Logistic Regression |      1.000 |      1.000 |   4 |             2 |
| SVM (RBF)           |      1.000 |      1.000 |   4 |             2 |
| Random Forest       |      0.750 |      0.733 |   4 |             2 |
| MLP (deploy บนเว็บ) |      1.000 |      1.000 |   4 |             2 |



### รายรูป (MLP)

| รูป                                      | จริง         | ทาย          |
|:-----------------------------------------|:-------------|:-------------|
| real/with_mask/ketar_mask.jpg            | with_mask    | with_mask    |
| real/with_mask/ketar_mask_glasses.jpg    | with_mask    | with_mask    |
| real/without_mask/ketar_bare.jpg         | without_mask | without_mask |
| real/without_mask/ketar_bare_glasses.jpg | without_mask | without_mask |



## เกณฑ์จดจำใบหน้า (คู่รูปคนเดียวกัน หน้าตรง vs ใส่แมส จาก ds2)

- จำนวนคู่ 154 · ระยะเฉลี่ย คนเดียวกัน 0.517 / ต่างคน 0.754

- เกณฑ์ที่เลือก **0.55** → FAR 0.7% · FRR 26.6%

- หมายเหตุ: แมสใน ds2 เป็นแมสสังเคราะห์ที่วางทับรูปเดิม ผลจริงอาจต่ำกว่านี้ ต้องทดสอบกับคนจริงด้วย
