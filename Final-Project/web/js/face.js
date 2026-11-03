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
           descriptor: Array.from(det.descriptor) };
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
