"""
train_mask.py — เทรนโมเดลตรวจการสวมหน้ากาก + หาเกณฑ์จดจำใบหน้า แล้ว export ไปใช้บนเว็บ

อินพุต: data/features.jsonl  (สร้างจาก ml/extract/run_extract.py)
    แต่ละบรรทัด = 1 รูป: path, src (ds1/ds2/real), label (with_mask/without_mask),
    found (ตัวตรวจจับเจอหน้าหรือไม่), d (ลายใบหน้า 128 ค่า จาก face-api ตัวเดียวกับบนเว็บ)

ผลลัพธ์:
    web/model/mask_model.json   MLP สำหรับรันในเบราว์เซอร์ (web/js/model.js)
    web/model/config.json       เกณฑ์ระยะจดจำใบหน้า + เกณฑ์แมส
    ml/results/*.md / *.png     ตารางและกราฟสำหรับสไลด์

การประเมินผล 3 แบบ:
    A) 5-fold CV ภายใน dataset หลัก (ds1)           — เทียบกับ SVM เดิมของ project_01 (84.62%)
    B) เทรน ds1 -> ทดสอบ ds2 (คนละแหล่งข้อมูล)       — วัดการ generalize ข้าม dataset
    C) รูปจริงจากมือถือของผู้พัฒนา (src = real)       — ปัญหา domain shift ที่ project_01 เจอ
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ["with_mask", "without_mask"]
SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def models() -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=3, gamma="scale")),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "MLP (deploy บนเว็บ)": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=2000, random_state=SEED)),
    }


def scores(y, p) -> dict:
    return {"Accuracy": accuracy_score(y, p), "Macro F1": f1_score(y, p, average="macro")}


def export_mlp(pipe, path: str) -> None:
    sc, mlp = pipe.named_steps["standardscaler"], pipe.named_steps["mlpclassifier"]
    classes = [str(c) for c in mlp.classes_]
    W, b = [w.tolist() for w in mlp.coefs_], [x.tolist() for x in mlp.intercepts_]
    if len(classes) == 2 and len(b[-1]) == 1:
        # sklearn ใช้ output เดียว (sigmoid) กับ 2 คลาส -> แปลงเป็น 2 output ให้ softmax ใน model.js ใช้ได้
        # softmax([0, z]) = [1 - sigmoid(z), sigmoid(z)] จึงตรงกับ predict_proba ทุกค่า
        W[-1] = [[0.0, row[0]] for row in W[-1]]
        b[-1] = [0.0, b[-1][0]]
    json.dump({"type": "mlp", "classes": classes, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
               "activation": mlp.activation, "weights": W, "biases": b,
               "trained_at": datetime.now().isoformat(timespec="seconds")}, open(path, "w"))


def plot_cm(y, p, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, p, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASSES); ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def face_threshold(df: pd.DataFrame, data_root: str, out_png: str):
    """หาเกณฑ์ระยะจดจำใบหน้า จากคู่รูป 'คนเดียวกัน ไม่ใส่แมส vs ใส่แมส' ใน ds2 (ชื่อไฟล์ N.jpg กับ N-with-mask.jpg)
    genuine = คนเดียวกัน, impostor = ต่างคน  -> เลือกเกณฑ์ที่ false accept <= 1%"""
    by_md5 = {}
    for r in df.itertuples():
        p = os.path.join(data_root, r.path)
        if os.path.exists(p):
            by_md5[hashlib.md5(open(p, "rb").read()).hexdigest()] = np.array(r.d)
    base = os.path.join(data_root, "ds2")
    if not os.path.isdir(base):
        return None
    pairs = []
    for f in sorted(os.listdir(os.path.join(base, "without_mask"))):
        g = os.path.join(base, "with_mask", f.replace(".jpg", "-with-mask.jpg"))
        a = os.path.join(base, "without_mask", f)
        if not os.path.exists(g):
            continue
        ha = hashlib.md5(open(a, "rb").read()).hexdigest(); hg = hashlib.md5(open(g, "rb").read()).hexdigest()
        if ha in by_md5 and hg in by_md5:
            pairs.append((by_md5[ha], by_md5[hg]))
    if len(pairs) < 20:
        return None
    A = np.stack([p[0] for p in pairs]); G = np.stack([p[1] for p in pairs])
    genuine = np.linalg.norm(A - G, axis=1)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pairs))
    impostor = np.linalg.norm(A - G[idx], axis=1)[idx != np.arange(len(pairs))]
    ths = np.linspace(0.2, 1.0, 161)
    far = np.array([(impostor < t).mean() for t in ths]); frr = np.array([(genuine >= t).mean() for t in ths])
    ok = ths[far <= 0.01]
    th = float(ok.max()) if len(ok) else float(ths[np.argmin(np.abs(far - frr))])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(genuine, bins=40, alpha=.7, label="same person (bare vs masked)")
    ax.hist(impostor, bins=40, alpha=.7, label="different people")
    ax.axvline(th, color="k", ls="--", label=f"threshold {th:.2f}")
    ax.set_xlabel("Face descriptor distance (Euclidean)"); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    i = int(np.argmin(np.abs(ths - th)))
    return {"threshold": round(th, 2), "pairs": len(pairs), "FAR": float(far[i]), "FRR": float(frr[i]),
            "genuine_mean": float(genuine.mean()), "impostor_mean": float(impostor.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ROOT, "data", "features.jsonl"))
    ap.add_argument("--images", default=os.path.join(ROOT, "ml", "extract"), help="โฟลเดอร์ที่มี ds1/ ds2/ real/")
    ap.add_argument("--web", default=os.path.join(ROOT, "web", "model"))
    ap.add_argument("--results", default=os.path.join(ROOT, "ml", "results"))
    a = ap.parse_args()
    os.makedirs(a.web, exist_ok=True); os.makedirs(a.results, exist_ok=True)

    df = pd.DataFrame([json.loads(l) for l in open(a.features, encoding="utf-8")])
    X = np.stack(df.d.values); y = df.label.values
    rep = [f"# ผลการทดลอง ({datetime.now():%Y-%m-%d %H:%M})\n"]

    # อัตราที่ตัวตรวจจับใบหน้าเจอหน้า แยกตามคลาส (project_01 ใช้ Haar แล้วหาไม่เจอเมื่อใส่แมส)
    det = df.groupby(["src", "label"]).found.agg(["mean", "count"]).rename(columns={"mean": "detect_rate"})
    rep += ["## อัตราตรวจจับใบหน้า (SSD MobileNet + เติมขอบ 20%) — project_01 ใช้ Haar Cascade แล้วหาหน้าไม่เจอเมื่อใส่แมส\n", det.to_markdown(floatfmt=".3f"), "\n"]

    # เทรน/ประเมินเฉพาะรูปที่ตรวจเจอหน้า เพราะบนเว็บก็จำแนกเฉพาะตอนเจอหน้าเท่านั้น
    df = df[df.found == 1].reset_index(drop=True)
    X = np.stack(df.d.values); y = df.label.values
    ds1, ds2, real = (df.src == "ds1").values, (df.src == "ds2").values, (df.src == "real").values

    # A) 5-fold CV ใน ds1
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, preds = [], {}
    for name, m in models().items():
        p = cross_val_predict(m, X[ds1], y[ds1], cv=cv)
        preds[name] = p; rows.append({"Model": name, **scores(y[ds1], p)})
    rows.insert(0, {"Model": "SVM บน pixel ดิบ (project_01 เดิม, 80/20)", "Accuracy": 0.8462, "Macro F1": 0.85})
    tA = pd.DataFrame(rows)
    rep += ["## A) 5-fold CV บน dataset หลัก (chandrikadeb7)\n", tA.to_markdown(index=False, floatfmt=".3f"), "\n"]
    plot_cm(y[ds1], preds["MLP (deploy บนเว็บ)"], os.path.join(a.results, "cm_cv.png"), "MLP — 5-fold CV")

    # B) ข้าม dataset และ C) รูปจริง
    for tag, mask in (("B) เทรน ds1 → ทดสอบ ds2 (prajnasb, คนละแหล่ง)", ds2), ("C) รูปจริงจากมือถือผู้พัฒนา", real)):
        if mask.sum() == 0:
            continue
        rows, per = [], None
        for name, m in models().items():
            m.fit(X[ds1], y[ds1]); p = m.predict(X[mask])
            rows.append({"Model": name, **scores(y[mask], p), "n": int(mask.sum()),
                         "with_mask n": int((y[mask] == "with_mask").sum())})
            if name.startswith("MLP") and tag.startswith("C"):
                per = pd.DataFrame({"รูป": df.path[mask].values, "จริง": y[mask], "ทาย": p})
        rep += [f"\n## {tag}\n", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"), "\n"]
        if tag.startswith("B"):
            rep += ["หมายเหตุ: รูปไม่สวมหน้ากากของ ds2 ส่วนใหญ่ซ้ำกับ ds1 จึงถูกตัดทิ้ง ชุดนี้จึงวัดได้เป็นหลักว่า"
                    " 'จับแมสแบบใหม่ที่ไม่เคยเห็นได้ไหม' (Accuracy ≈ Recall ของ with_mask) ส่วน Macro F1 ใช้ตีความไม่ได้\n"]
        if per is not None:
            rep += ["\n### รายรูป (MLP)\n", per.to_markdown(index=False), "\n"]

    # เทรนตัวจริงด้วยข้อมูลสาธารณะทั้งหมด (ไม่รวมรูปจริง เพื่อเก็บไว้เป็นชุดทดสอบ)
    final = models()["MLP (deploy บนเว็บ)"].fit(X[~real], y[~real])
    export_mlp(final, os.path.join(a.web, "mask_model.json"))

    ft = face_threshold(df, a.images, os.path.join(a.results, "face_distance.png"))
    cfg = {"match_threshold": ft["threshold"] if ft else 0.5, "mask_threshold": 0.5}
    json.dump(cfg, open(os.path.join(a.web, "config.json"), "w"))
    if ft:
        rep += ["\n## เกณฑ์จดจำใบหน้า (คู่รูปคนเดียวกัน หน้าตรง vs ใส่แมส จาก ds2)\n",
                f"- จำนวนคู่ {ft['pairs']} · ระยะเฉลี่ย คนเดียวกัน {ft['genuine_mean']:.3f} / ต่างคน {ft['impostor_mean']:.3f}\n",
                f"- เกณฑ์ที่เลือก **{ft['threshold']}** → FAR {ft['FAR']*100:.1f}% · FRR {ft['FRR']*100:.1f}%\n",
                "- หมายเหตุ: แมสใน ds2 เป็นแมสสังเคราะห์ที่วางทับรูปเดิม ผลจริงอาจต่ำกว่านี้ ต้องทดสอบกับคนจริงด้วย\n"]
    open(os.path.join(a.results, "metrics.md"), "w", encoding="utf-8").write("\n".join(rep))
    print("\n".join(rep)); print("config:", cfg)


if __name__ == "__main__":
    main()
