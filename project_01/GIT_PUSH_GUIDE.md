# วิธี Push โปรเจกต์นี้ขึ้น GitHub (ML-CPE)

เป้าหมาย: เอาโฟลเดอร์โปรเจกต์นี้ขึ้น repo `https://github.com/tinnapath121/ML-CPE` ให้อาจารย์เปิดดู source code ตอนสอบได้

---

## สิ่งที่ต้องมี

| ต้องมี | เช็คด้วยคำสั่ง | ถ้าไม่มี |
| --- | --- | --- |
| Git ติดตั้งแล้ว | `git --version` | โหลดจาก https://git-scm.com/download/win |
| ตั้งชื่อ/อีเมลแล้ว | `git config --global user.name` | `git config --global user.name "Ketar"` และ `git config --global user.email "ketar.tnp@gmail.com"` |
| สิทธิ์ push เข้า repo | ครั้งแรกจะเด้งหน้าต่างให้ login GitHub | login ผ่านเบราว์เซอร์ที่เด้งขึ้นมา (Git Credential Manager) |

> ถ้าเคย push `Homework04_SVM` / `Homework05_NN` ขึ้น repo นี้ได้แล้ว แปลว่า 3 ข้อนี้ผ่านหมดแล้ว ข้ามไปขั้นตอนถัดไปได้เลย

---

## ขั้นตอน (กรณีมี repo อยู่แล้วในเครื่อง)

```bash
# 1. เข้าไปที่โฟลเดอร์ repo ที่ clone ไว้
cd path\to\ML-CPE

# 2. ดึงของใหม่จาก GitHub ก่อน กันชนกัน
git pull

# 3. ก็อปโฟลเดอร์โปรเจกต์นี้เข้าไปใน repo (ทำผ่าน File Explorer ก็ได้)
#    แนะนำตั้งชื่อโฟลเดอร์ให้สื่อ เช่น  Project_SVM_MaskDetection

# 4. เช็คว่า git เห็นอะไรบ้าง (ต้องไม่เห็น dataset/ กับ *.npy เพราะมี .gitignore แล้ว)
git status

# 5. เพิ่มเข้า staging
git add Project_SVM_MaskDetection

# 6. commit
git commit -m "Add applied SVM project: face mask detection (mask vs no mask)"

# 7. push ขึ้น GitHub
git push
```

## ขั้นตอน (กรณียังไม่มี repo ในเครื่อง)

```bash
cd C:\Users\ketar\OneDrive\Desktop\AI_Workspace
git clone https://github.com/tinnapath121/ML-CPE.git
cd ML-CPE
# แล้วทำตามข้อ 3-7 ด้านบน
```

---

## ข้อควรระวัง

1. **อย่า push โฟลเดอร์ `dataset/`** — ใหญ่ประมาณ 170MB GitHub จะช้ามากหรือปฏิเสธ (ไฟล์เดี่ยวเกิน 100MB จะถูกบล็อก) — ไฟล์ `.gitignore` ที่แนบมาในโปรเจกต์กันให้แล้ว
2. **ไฟล์ `outputs/*.npy` ก็ไม่ต้อง push** (รวมกันหลายร้อย MB สร้างใหม่ได้ด้วย `python main.py`) — `.gitignore` กันให้แล้วเช่นกัน
3. **ไฟล์ที่ต้อง push แน่ ๆ**: โค้ด `.py` ทุกไฟล์, `README.md`, `flowchart.png`, `demo_photos/`, และ `outputs/svm_model.pkl` + `outputs/scaler.pkl` + `outputs/classes.json` (รวมกันประมาณ 9MB — ผ่านสบาย และทำให้คนอื่น clone ไปรันได้เลยโดยไม่ต้องเทรนใหม่)
4. ถ้า `git push` ขึ้น error เรื่อง authentication ให้ลองใหม่อีกครั้ง หน้าต่าง login GitHub มักเด้งขึ้นรอบสอง หรือใช้ GitHub Desktop แทนก็ได้ (ง่ายกว่าถ้าเร่งเวลา)

---

## เช็คก่อนสอบ

เปิด `https://github.com/tinnapath121/ML-CPE` ในเบราว์เซอร์แล้วดูว่า:

- [ ] เห็นโฟลเดอร์โปรเจกต์ใหม่
- [ ] กดเข้าไปแล้วเห็น `README.md` แสดงผลสวยงาม (มี Structure / Dataset / Setup / How to Run)
- [ ] กดดู `svm_model.py`, `predict_image.py` แล้วเห็นโค้ดครบ
- [ ] `flowchart.png` เปิดดูได้
