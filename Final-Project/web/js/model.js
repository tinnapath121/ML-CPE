// model.js — รันโมเดล MLP ที่เทรนด้วย scikit-learn (export จาก ml/train.py) ในเบราว์เซอร์
// คำนวณ forward pass เอง: standardize -> (Wx+b -> ReLU) ... -> softmax

export async function loadModel(url = "model/model.json") {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`ไม่พบไฟล์โมเดล (${res.status}) — รัน ml/train.py ก่อน`);
  const m = await res.json();
  return { ...m, predict: (x) => predict(m, x) };
}

const act = {
  relu: (v) => Math.max(0, v),
  tanh: Math.tanh,
  logistic: (v) => 1 / (1 + Math.exp(-v)),
  identity: (v) => v,
};

/** @returns {{label:string, prob:number, probs:Object<string,number>}} */
export function predict(m, x) {
  // 1) standardize ด้วยค่าเฉลี่ย/ส่วนเบี่ยงเบนจากชุดเทรน
  let h = x.map((v, i) => (v - m.mean[i]) / (m.scale[i] || 1));
  const L = m.weights.length;
  for (let l = 0; l < L; l++) {
    const W = m.weights[l], b = m.biases[l];
    const out = new Array(b.length);
    for (let j = 0; j < b.length; j++) {
      let s = b[j];
      for (let i = 0; i < h.length; i++) s += h[i] * W[i][j];
      out[j] = l < L - 1 ? act[m.activation](s) : s; // ชั้นสุดท้ายยังไม่ใส่ activation
    }
    h = out;
  }
  // 2) softmax (sklearn ใช้ softmax กับปัญหาหลายคลาส)
  const mx = Math.max(...h);
  const e = h.map((v) => Math.exp(v - mx));
  const sum = e.reduce((a, b) => a + b, 0);
  const probs = Object.fromEntries(m.classes.map((c, i) => [c, e[i] / sum]));
  let best = 0;
  for (let i = 1; i < e.length; i++) if (e[i] > e[best]) best = i;
  return { label: m.classes[best], prob: e[best] / sum, probs };
}
