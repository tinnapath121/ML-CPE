// face.js — โหลด face-api (รันในเบราว์เซอร์ทั้งหมด ไม่ส่งภาพออกนอกเครื่อง) และดึง "ลายใบหน้า" 128 มิติ
// ตัวเลข 128 ค่านี้ (face descriptor) ใช้ทำ 2 อย่าง:
//   1) เป็น feature ให้โมเดลตรวจแมสของเรา (เทรนใน ml/train_mask.py)
//   2) เทียบระยะห่างกับลายใบหน้าที่ลงทะเบียนไว้ เพื่อระบุว่าเป็นใคร
// การตั้งค่าตัวตรวจจับต้องตรงกับตอนดึง feature จาก dataset (ml/extract) ไม่อย่างนั้นโมเดลจะเพี้ยน

const VER = "1.7.15";
const CDN = `https://cdn.jsdelivr.net/npm/@vladmandic/face-api@${VER}`;
let faceapi = null;

// ตั้งค่าตัวตรวจจับ — ต้องตรงกับ ml/extract/extract.html
// SSD MobileNet + ขอบสีเทา 20% ตรวจเจอหน้าที่ใส่แมสได้ดีกว่า Haar (project_01) และ Tiny detector มาก
export const DETECTOR = { minConfidence: 0.15, pad: 0.2 };
let padCanvas = null;

export async function loadFaceApi(onStatus = () => {}) {
  onStatus("กำลังโหลดไลบรารีตรวจจับใบหน้า…");
  faceapi = await import(`${CDN}/dist/face-api.esm.js`);
  try {
    await faceapi.tf.setBackend("webgl");
  } catch {
    // มือถือที่ไม่มี WebGL -> ใช้ WebAssembly แทน
    faceapi.tf.setWasmPaths("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/");
    await faceapi.tf.setBackend("wasm");
  }
  await faceapi.tf.ready();
  onStatus("กำลังโหลดโมเดลใบหน้า (ครั้งแรกประมาณ 12 MB)…");
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri(`${CDN}/model`),
    faceapi.nets.faceLandmark68Net.loadFromUri(`${CDN}/model`),
    faceapi.nets.faceRecognitionNet.loadFromUri(`${CDN}/model`),
  ]);
  return faceapi.tf.getBackend();
}

/** ตรวจจับใบหน้า แล้วคืน { box (พิกัดบนภาพกล้อง), descriptor:number[128], score } หรือ null */
export async function analyzeFace(input) {
  const w = input.videoWidth || input.width, h = input.videoHeight || input.height;
  const p = Math.round(Math.max(w, h) * DETECTOR.pad);
  padCanvas ??= document.createElement("canvas");
  padCanvas.width = w + 2 * p; padCanvas.height = h + 2 * p;
  const x = padCanvas.getContext("2d");
  x.fillStyle = "#808080"; x.fillRect(0, 0, padCanvas.width, padCanvas.height); x.drawImage(input, p, p, w, h);
  const det = await faceapi
    .detectSingleFace(padCanvas, new faceapi.SsdMobilenetv1Options({ minConfidence: DETECTOR.minConfidence }))
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (!det) return null;
  const b = det.detection.box;
  return { box: { x: b.x - p, y: b.y - p, width: b.width, height: b.height }, score: det.detection.score,
           descriptor: Array.from(det.descriptor), cues: coverCues(x, det.landmarks.positions, b) };
}

export function distance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

export async function startCamera(video, facingMode = "user") {
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
  });
  await video.play();
}

export function stopCamera(video) {
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

// ---------- ตรวจ "สิ่งที่ปิดหน้า" ด้วยสีจากจุดสำคัญ 68 จุด (กันของที่ไม่ใช่หน้ากาก) ----------
// mouthCheek: สีบริเวณปาก vs สีแก้มส่วนล่าง  -> หน้ากากจริงปิดทั้งปากและแก้ม สีจึงใกล้กัน (วัดได้ 2–12)
//             แผ่นมาสก์หน้า/ปากโผล่ สีปากต่างจากแก้มชัด (วัดได้ 61)
// cheekNeck / cheekForehead: ถ้าสิ่งที่ปิดหน้าสีเดียวกับบริเวณใต้คาง แต่ต่างจากหน้าผากมาก
//             = เอาเสื้อดึงขึ้นมาปิด (วัดได้ 23 / 219)
// skinTone + cheekForehead ต่ำ: สิ่งที่ปิดเป็นสีผิวเดียวกับหน้าผาก = เอามือปิด (วัดได้ 22)
function coverCues(ctx, P, B) {
  const mean = (cx, cy, w, h) => {
    w = Math.max(2, Math.round(w)); h = Math.max(2, Math.round(h));
    const d = ctx.getImageData(Math.round(cx - w / 2), Math.round(cy - h / 2), w, h).data;
    let r = 0, g = 0, bl = 0; const n = d.length / 4;
    for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; bl += d[i + 2]; }
    return [r / n, g / n, bl / n];
  };
  const dist = (a, c) => Math.hypot(a[0] - c[0], a[1] - c[1], a[2] - c[2]);
  const lerp = (a, c, t) => ({ x: a.x + (c.x - a.x) * t, y: a.y + (c.y - a.y) * t });
  const fw = B.width, fh = B.height;
  const M = mean((P[48].x + P[54].x) / 2, (P[51].y + P[57].y) / 2,
                 Math.abs(P[54].x - P[48].x) * 0.6, Math.max(4, Math.abs(P[57].y - P[51].y) * 0.6));
  const l = lerp(P[4], P[48], 0.5), r = lerp(P[12], P[54], 0.5), s = fw * 0.08;
  const C1 = mean(l.x, l.y, s, s), C2 = mean(r.x, r.y, s, s), C = C1.map((v, i) => (v + C2[i]) / 2);
  const N = mean(P[8].x, P[8].y + fh * 0.2, fw * 0.3, fh * 0.12);
  const F = mean((P[19].x + P[24].x) / 2, Math.min(P[19].y, P[24].y) - fh * 0.08, fw * 0.25, fh * 0.06);
  const skinTone = C[0] - C[1] > 12 && C[0] - C[2] > 20;   // โทนผิว: แดง > เขียว > น้ำเงิน
  return { mouthCheek: dist(M, C), cheekNeck: dist(C, N), cheekForehead: dist(C, F), skinTone };
}

/** true = สิ่งที่ปิดหน้าดูเป็นหน้ากากจริง (ปากถูกปิด และไม่ใช่เสื้อที่ดึงขึ้นมา) */
export function looksLikeMask(cues, cfg = {}) {
  if (!cues) return true;
  const mouthMax = cfg.mouth_cheek_max ?? 25, neckMin = cfg.cheek_neck_min ?? 30, foreMax = cfg.cheek_forehead_max ?? 60;
  if (cues.mouthCheek > mouthMax) return false;                                  // ปากโผล่ (เช่น แผ่นมาสก์หน้า)
  if (cues.cheekNeck < neckMin && cues.cheekForehead > foreMax) return false;    // ต่อเนื่องกับเสื้อ
  if (cues.skinTone && cues.cheekForehead < (cfg.skin_forehead_max ?? 40)) return false; // ที่ปิดเป็นสีผิวเหมือนหน้าผาก = มือ/ผิว
  return true;
}
