// app.js — ควบคุมหน้าจอทั้งหมด
// สแกน: กล้อง -> ตรวจจับใบหน้า -> ลายใบหน้า 128 ค่า -> (1) โมเดลตรวจแมสของเรา (2) 1-NN เทียบผู้ลงทะเบียน
// ผ่านเมื่อ: จำหน้าได้ (ระยะ < เกณฑ์) และ สวมหน้ากาก
import { loadFaceApi, analyzeFace, distance, startCamera } from "./face.js?v=11";
import { loadModel } from "./model.js?v=11";
import { people, log, identify } from "./store.js?v=11";

// ถ้าหา element ไม่เจอ (เช่น เบราว์เซอร์ยังจำ HTML เวอร์ชันเก่า) คืนตัวหลอกแทน null เพื่อไม่ให้ทั้งแอปพัง
const dummy = new Proxy(function () {}, { get: (t, k) => (k === "classList" ? { toggle() {}, add() {}, remove() {} } : k === "querySelector" ? () => dummy : k === "dataset" ? {} : ""), set: () => true });
const $ = (id) => document.getElementById(id) || (console.warn("missing element #" + id), dummy);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");

let maskModel = null, cfg = { match_threshold: 0.5, mask_threshold: 0.5 };
let view = "scan", busy = false, lastFace = null;
const recent = [];                 // ผลสแกนล่าสุดหลายเฟรม ใช้โหวตให้ผลนิ่ง
const COOLDOWN_MS = 30000;          // คนเดิมลงเวลาซ้ำไม่ได้ภายใน 30 วินาที
const lastCheckin = new Map();
const SCAN_EVERY_MS = 250;

// ---------- เริ่มต้น ----------
(async () => {
  try {
    const [backend, m, c] = await Promise.all([
      loadFaceApi((s) => ($("overlay").textContent = s)),
      loadModel("model/mask_model.json?v=4"),
      fetch("model/config.json", { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]);
    maskModel = m; if (c) cfg = { ...cfg, ...c };
    $("overlay").textContent = "พร้อม — กดเปิดกล้อง";
    $("modelInfo").textContent = `โมเดลตรวจแมส: MLP · เทรนเมื่อ ${m.trained_at} · เกณฑ์จำหน้า ${cfg.match_threshold} · backend ${backend}`;
  } catch (e) {
    $("overlay").textContent = "โหลดไม่สำเร็จ: " + e.message;
  }
})();

let camMode = "user";
$("btnCamFront").onclick = () => openCam(camMode);
$("btnCamSwitch").onclick = () => openCam(camMode === "user" ? "environment" : "user");
async function openCam(mode) {
  camMode = mode;
  try {
    await startCamera(video, mode);
    $("btnCamFront").hidden = true;   // กล้องเปิดแล้ว เหลือแค่ปุ่มสลับกล้อง ไม่บังภาพ
    video.classList.toggle("mirror", mode === "user");
    canvas.classList.toggle("mirror", mode === "user");
    loop();
  } catch (e) { $("overlay").textContent = "เปิดกล้องไม่ได้: " + e.message; }
}

// ---------- แท็บ ----------
// แต่ละแท็บมี URL ของตัวเอง (#scan, #register, ...) ปุ่มย้อนกลับของมือถือจึงย้อนระหว่างแท็บ ไม่หลุดออกจากเว็บ
const VIEWS = ["scan", "register", "log", "people", "collect"];
document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => show(b.dataset.view)));
window.addEventListener("popstate", () => show(viewFromHash(), false));
function viewFromHash() { const v = location.hash.slice(1); return VIEWS.includes(v) ? v : "scan"; }
history.replaceState(null, "", "#" + viewFromHash());
function show(v, push = true) {
  if (push && v !== view) history.pushState(null, "", "#" + v);
  view = v;
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  document.querySelectorAll("section.view").forEach((s) => (s.hidden = s.dataset.view !== v));
  const cam = v === "scan" || v === "register" || v === "collect";
  $("cameraBox").hidden = !cam;
  document.querySelector("main").classList.toggle("with-cam", cam);   // จอใหญ่: จัดกล้องซ้าย แผงขวา
  if (v === "collect") renderCollect();
  if (v === "log") renderLog();
  if (v === "people") renderPeople();
}

// ---------- วนประมวลผล ----------
let timer = 0;
function loop() {
  clearTimeout(timer);
  const tick = async () => {
    if (!busy && maskModel && video.readyState >= 2 && !$("cameraBox").hidden) {
      busy = true;
      try {
        const f = await analyzeFace(video);
        lastFace = f;
        draw(f);
        if (view === "scan") await onScanFrame(f);
        else if (view === "collect") onCollectFrame(f);
        else $("overlay").textContent = f ? `เจอใบหน้า · แมส ${(maskProb(f) * 100).toFixed(0)}%` : "ไม่พบใบหน้า";
      } finally { busy = false; }
    }
    timer = setTimeout(tick, SCAN_EVERY_MS);
  };
  tick();
}

function maskProb(f) { return maskModel.predict(f.descriptor).probs.with_mask ?? 0; }

function draw(f) {
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!f) return;
  const ok = maskProb(f) >= cfg.mask_threshold;
  ctx.lineWidth = 4; ctx.strokeStyle = ok ? "#22c55e" : "#ef4444";
  ctx.strokeRect(f.box.x, f.box.y, f.box.width, f.box.height);
}

// ---------- ตรรกะสแกนเข้างาน ----------
async function onScanFrame(f) {
  if (!f) { recent.length = 0; $("overlay").textContent = "ไม่พบใบหน้า — ขยับเข้ามาใกล้กล้อง"; return; }
  const pm = maskProb(f);
  let match;
  try { match = await identify(f.descriptor, cfg.match_threshold); }
  catch (e) { $("overlay").textContent = "เชื่อมต่อฐานข้อมูลไม่ได้: " + e.message; return; }
  const known = !!(match && match.person);
  recent.push({ pm, id: known ? match.person.id : null, match, box: f.box });
  if (recent.length > 4) recent.shift();
  $("overlay").textContent = "กำลังตรวจ…";
  $("sMask").textContent = `${(pm * 100).toFixed(0)}%`;
  $("sWho").textContent = known ? match.person.name : "ไม่รู้จัก";
  $("sDist").textContent = match ? match.dist.toFixed(2) : "–";
  if (recent.length < 4) return;

  // โหวต 4 เฟรมล่าสุด: ต้องได้คนเดียวกันทุกเฟรม และใช้ค่าเฉลี่ยความน่าจะเป็นแมส
  const ids = new Set(recent.map((r) => r.id));
  const avgMask = recent.reduce((s, r) => s + r.pm, 0) / recent.length;
  const masked = avgMask >= cfg.mask_threshold;
  if (ids.size !== 1) return;
  const id = [...ids][0];

  if (!id) {
    setResult("warn", "ไม่พบข้อมูลในระบบ", "ยังไม่ได้ลงทะเบียน กรุณาลงทะเบียนก่อนใช้งาน",
      { label: "ไปลงทะเบียน", fn: () => show("register") });
    return;
  }
  const p = recent.at(-1).match.person, dist = recent.at(-1).match.dist;
  if (!masked) {
    setResult("deny", `${p.name} — กรุณาสวมหน้ากาก`, "ต้องสวมหน้ากากอนามัยก่อนเข้าพื้นที่โรงงาน");
    logOnce(p, "denied_no_mask", avgMask, dist, recent.at(-1).box);
    return;
  }
  const t = new Date();
  setResult("ok", `ลงเวลาเข้างานสำเร็จ ✓`, `${p.name} (${p.id}) · ${t.toLocaleTimeString("th-TH")}`);
  logOnce(p, "checked_in", avgMask, dist, recent.at(-1).box);
}

// ตัดรูปหน้าย่อ 96x96 JPEG (~4-8 KB) จากกล้อง เก็บคู่กับประวัติ (ผู้ลงทะเบียนยินยอมแล้วตอนลงทะเบียน)
function faceThumb(box) {
  if (!box || !video.videoWidth) return null;
  const pad = box.width * 0.25, x = Math.max(0, box.x - pad), y = Math.max(0, box.y - pad);
  const w = Math.min(video.videoWidth - x, box.width + pad * 2), hh = Math.min(video.videoHeight - y, box.height + pad * 2);
  const c = document.createElement("canvas"); c.width = c.height = 96;
  c.getContext("2d").drawImage(video, x, y, w, hh, 0, 0, 96, 96);
  const url = c.toDataURL("image/jpeg", 0.7);
  return url.length <= 20000 ? url : null;
}
function logOnce(p, result, pm, dist, box) {
  const key = p.id + result, now = Date.now();
  if (now - (lastCheckin.get(key) || 0) < COOLDOWN_MS) return;
  lastCheckin.set(key, now);
  log.add({ id: p.id, result, mask: +pm.toFixed(3), dist: +dist.toFixed(3), photo: faceThumb(box) })
    .catch((e) => ($("modelInfo").textContent = "บันทึกประวัติไม่สำเร็จ: " + e.message));
  if (navigator.vibrate) navigator.vibrate(result === "checked_in" ? 80 : [200, 100, 200]);
}

function setResult(kind, title, sub, action) {
  const el = $("result");
  el.className = "result " + kind;
  el.innerHTML = "";
  el.append(Object.assign(document.createElement("div"), { className: "result-title", textContent: title }));
  el.append(Object.assign(document.createElement("div"), { className: "result-sub", textContent: sub }));
  if (action) {
    const b = Object.assign(document.createElement("button"), { className: "primary", textContent: action.label });
    b.style.marginTop = "8px"; b.onclick = action.fn; el.append(b);
  }
}

// ---------- ลงทะเบียน ----------
const reg = { plain: [], masked: [] };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function capture(kind) {
  // เก็บ 3 เฟรมห่างกันเล็กน้อย เพื่อให้ได้ลายใบหน้าหลายมุมเล็กน้อย จำได้แม่นขึ้น
  const out = [];
  for (let i = 0; i < 12 && out.length < 3; i++) {
    await sleep(350);
    const f = lastFace;
    if (!f) continue;
    const pm = maskProb(f);
    if (kind === "plain" && pm >= cfg.mask_threshold) return { error: "ระบบเห็นว่ายังสวมหน้ากากอยู่ — ถอดหน้ากากแล้วถ่ายใหม่" };
    if (kind === "masked" && pm < cfg.mask_threshold) return { error: "ระบบยังไม่เห็นหน้ากาก — สวมหน้ากากให้ปิดจมูกและปากแล้วถ่ายใหม่" };
    if (!out.length || distance(out.at(-1), f.descriptor) > 0.02) out.push(f.descriptor);
  }
  if (!out.length) return { error: "ไม่พบใบหน้า — มองตรงเข้ากล้องให้เห็นทั้งหน้า แล้วลองใหม่" };
  return { list: out };
}

$("btnShot1").onclick = async () => {
  if (!validForm()) return;
  $("regStatus").textContent = "กำลังถ่าย… มองตรงเข้ากล้อง ไม่สวมหน้ากาก";
  const r = await capture("plain");
  if (r.error) { $("regStatus").textContent = r.error; return; }
  let dup = null;
  try { dup = await identify(r.list[0], cfg.match_threshold); }
  catch (e) { $("regStatus").textContent = "เชื่อมต่อฐานข้อมูลไม่ได้: " + e.message; return; }
  if (dup && dup.person && dup.person.id !== $("rId").value.trim()) {
    $("regStatus").textContent = `ใบหน้านี้ลงทะเบียนไว้แล้วเป็น ${dup.person.name} (${dup.person.id})`; return;
  }
  reg.plain = r.list; mark(1); $("btnShot2").disabled = false;
  $("regStatus").textContent = "ขั้นที่ 1 เรียบร้อย — สวมหน้ากากแล้วกดถ่ายขั้นที่ 2";
};
$("btnShot2").onclick = async () => {
  $("regStatus").textContent = "กำลังถ่าย… สวมหน้ากากแล้วมองตรงเข้ากล้อง";
  const r = await capture("masked");
  if (r.error) { $("regStatus").textContent = r.error; return; }
  reg.masked = r.list; mark(2); $("btnSave").disabled = false;
  $("regStatus").textContent = "ครบทั้ง 2 ขั้น — กดบันทึก";
};
$("btnSave").onclick = async () => {
  if (!validForm() || !reg.plain.length || !reg.masked.length) return;
  const p = {
    id: $("rId").value.trim(), name: $("rName").value.trim(), dept: $("rDept").value.trim(),
    consent_at: new Date().toISOString(),
    templates: [...reg.plain.map((d) => ({ kind: "plain", d })), ...reg.masked.map((d) => ({ kind: "masked", d }))],
  };
  $("btnSave").disabled = true; $("regStatus").textContent = "กำลังบันทึก…";
  try { await people.add(p); }
  catch (e) { $("regStatus").textContent = "บันทึกไม่สำเร็จ: " + e.message; $("btnSave").disabled = false; return; }
  resetReg(true);
  // ลงทะเบียนเสร็จ -> ไปหน้าสแกนทันที พร้อมข้อความยืนยัน
  show("scan");
  recent.length = 0;
  setResult("ok", `ลงทะเบียน ${p.name} สำเร็จ ✓`, "ยืนหน้ากล้องพร้อมสวมหน้ากากเพื่อลงเวลาเข้างาน");
};
$("btnRegReset").onclick = () => resetReg(true);

function validForm() {
  if (!$("rId").value.trim() || !$("rName").value.trim()) { $("regStatus").textContent = "กรอกรหัสพนักงานและชื่อก่อน"; return false; }
  if (!$("rConsent").checked) { $("regStatus").textContent = "ต้องกดยินยอมการเก็บลายใบหน้าก่อน"; return false; }
  return true;
}
function mark(n) { $("step" + n).classList.add("done"); $("st" + n).textContent = "✓ เรียบร้อย"; }
function resetReg(clearForm) {
  reg.plain = []; reg.masked = [];
  for (const n of [1, 2]) { $("step" + n).classList.remove("done"); $("st" + n).textContent = "ยังไม่ได้ถ่าย"; }
  $("btnShot2").disabled = true; $("btnSave").disabled = true;
  if (clearForm) { ["rId", "rName", "rDept"].forEach((i) => ($(i).value = "")); $("rConsent").checked = false; $("regStatus").textContent = ""; }
}

// ---------- ประวัติ / พนักงาน ----------
const RESULT_TH = { checked_in: "เข้างาน ✓", denied_no_mask: "ไม่สวมหน้ากาก ✗" };
const fmt = (t) => (t ? new Date(t).toLocaleString("th-TH", { dateStyle: "short", timeStyle: "short" }) : "–");
// data-label = หัวคอลัมน์ ใช้แสดงเป็นการ์ดบนจอมือถือ (ดู table.list ใน style.css)
const cell = (tr, t) => {
  const td = Object.assign(document.createElement("td"), { textContent: t });
  const th = tr.closest("table")?.querySelectorAll("thead th")[tr.children.length];
  if (th) td.dataset.label = th.textContent;
  tr.append(td); return td;
};
async function renderLog() {
  const tb = $("logTable").querySelector("tbody"); tb.innerHTML = "<tr><td colspan=4>กำลังโหลด…</td></tr>";
  try {
    const rows = await log.all(200); tb.innerHTML = "";
    if (!rows.length) { const td = cell(tb.insertRow(), "ยังไม่มีประวัติเข้างาน"); td.colSpan = 4; td.className = "empty"; }
    for (const r of rows) { const tr = tb.insertRow(); [fmt(r.created_at), r.id, r.name, RESULT_TH[r.result] || r.result].forEach((t) => cell(tr, t)); }
  } catch (e) { tb.innerHTML = ""; cell(tb.insertRow(), "โหลดไม่สำเร็จ: " + e.message); }
}
async function renderPeople() {
  const tb = $("peopleTable").querySelector("tbody"); tb.innerHTML = "<tr><td colspan=6>กำลังโหลด…</td></tr>";
  let list = [];
  try { list = await people.all(); } catch (e) { tb.innerHTML = ""; cell(tb.insertRow(), "โหลดไม่สำเร็จ: " + e.message); return; }
  $("peopleCount").textContent = `รายชื่อผู้ลงทะเบียนทั้งหมด ${list.length} คน`;
  $("peopleEmpty").hidden = list.length > 0;
  tb.innerHTML = "";
  for (const p of list) {
    const tr = tb.insertRow();
    [p.id, p.name, p.dept || "–", fmt(p.registered_at), fmt(p.last_checkin)].forEach((t) => cell(tr, t));
    const del = Object.assign(document.createElement("button"), { textContent: "ลบ", className: "danger" });
    del.onclick = async () => {
      const pin = $("adminPin").value.trim();
      if (!pin) { $("peopleStatus").textContent = "ใส่ PIN ผู้ดูแลก่อนลบ"; return; }
      if (!del.dataset.armed) { del.dataset.armed = 1; del.textContent = "ยืนยันลบ"; return; }
      try { await people.remove(p.id, pin); $("peopleStatus").textContent = `ลบ ${p.name} แล้ว`; renderPeople(); }
      catch (e) { $("peopleStatus").textContent = e.message; }
    };
    const td = document.createElement("td"); td.className = "cell-action"; td.append(del); tr.append(td);
  }
}
$("btnGoScan").onclick = () => show("scan");
$("btnGoReg").onclick = () => show("register");
$("btnPeopleRefresh").onclick = () => renderPeople();
const download = (name, text, type) => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["\ufeff" + text], { type })); a.download = name; a.click();
};
$("btnCsv").onclick = async () => {
  const b = $("btnCsv"); b.disabled = true; b.textContent = "กำลังสร้างรายงาน…";
  try {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(await log.toExcel());
    a.download = `รายงานเข้างาน_${new Date().toLocaleDateString("en-CA", { timeZone: "Asia/Bangkok" })}.xlsx`; a.click();
  } catch (e) { alert("สร้างรายงานไม่สำเร็จ: " + e.message); }
  b.disabled = false; b.textContent = "ดาวน์โหลดรายงาน";
};
$("btnLogClear").onclick = async () => {
  const pin = $("logPin").value.trim();
  if (!pin) { $("logStatus").textContent = "ใส่ PIN ผู้ดูแลก่อนล้างประวัติ"; return; }
  try { await log.clear(pin); $("logStatus").textContent = "ล้างประวัติแล้ว"; renderLog(); }
  catch (e) { $("logStatus").textContent = e.message; }
};

// ---------- เก็บข้อมูลเทรนเพิ่ม ----------
const KEY_COLLECT = "mc-collect-v1";
const LABEL_TH = { with_mask: "สวมหน้ากาก", without_mask: "ไม่สวมหน้ากาก", hand_cover: "มือ/ของอื่นปิดปาก" };
let collectRows = (() => { try { return JSON.parse(localStorage.getItem(KEY_COLLECT)) || []; } catch { return []; } })();
let rec = null;   // { session, label, person, until }
const saveCollect = () => { try { localStorage.setItem(KEY_COLLECT, JSON.stringify(collectRows)); } catch { /* เต็ม */ } };

$("btnRec").onclick = () => {
  if (rec) return;
  let n = 3; $("btnRec").disabled = true;
  const t = setInterval(() => {
    $("overlay").textContent = `เตรียมท่า… ${n}`;
    if (n-- <= 0) {
      clearInterval(t);
      const person = $("cPerson").value.trim() || "p?", label = $("cLabel").value;
      rec = { person, label, session: `${person}_${label}_${Date.now()}`, until: Date.now() + Number($("cSecs").value) * 1000 };
    }
  }, 1000);
};
function onCollectFrame(f) {
  if (!rec) { $("overlay").textContent = f ? `เจอใบหน้า · แมส ${(maskProb(f) * 100).toFixed(0)}%` : "ไม่พบใบหน้า"; return; }
  if (Date.now() > rec.until) {
    rec = null; $("btnRec").disabled = false; saveCollect(); renderCollect();
    $("overlay").textContent = "อัดเสร็จ ✓"; return;
  }
  if (!f) { $("overlay").textContent = "● ไม่พบใบหน้า"; return; }
  collectRows.push({ path: rec.session, src: "phone", person: rec.person, session: rec.session, label: rec.label,
                     found: 1, score: +f.score.toFixed(3), d: f.descriptor.map((v) => +v.toFixed(5)) });
  $("overlay").textContent = `● ${LABEL_TH[rec.label]} — ${collectRows.filter((r) => r.session === rec.session).length} เฟรม`;
}
function renderCollect() {
  const tb = $("collectTable").querySelector("tbody"); tb.innerHTML = "";
  for (const k of Object.keys(LABEL_TH)) {
    const tr = document.createElement("tr");
    [LABEL_TH[k], collectRows.filter((r) => r.label === k).length]
      .forEach((t) => tr.append(Object.assign(document.createElement("td"), { textContent: t })));
    tb.append(tr);
  }
}
$("btnCollectDl").onclick = () =>
  download(`phone_${$("cPerson").value.trim() || "data"}_${Date.now()}.jsonl`, collectRows.map((r) => JSON.stringify(r)).join("\n"), "application/json");
$("btnCollectClear").onclick = (e) => {
  const b = e.currentTarget;
  if (b.dataset.armed) { collectRows = []; saveCollect(); renderCollect(); delete b.dataset.armed; b.textContent = "ล้างข้อมูลที่เก็บ"; }
  else { b.dataset.armed = 1; b.textContent = "กดอีกครั้งเพื่อยืนยัน"; }
};

show(viewFromHash(), false);

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
