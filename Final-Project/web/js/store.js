// store.js — ฐานข้อมูลกลางบน Supabase (ทุกเครื่องเห็นรายชื่อและประวัติเดียวกัน)
// เรียกผ่านฟังก์ชัน RPC เท่านั้น: ตารางจริงอยู่ใน schema ที่ไม่เปิดผ่าน API
// -> เบราว์เซอร์ส่งลายใบหน้า 128 ค่าไปให้ "ฐานข้อมูลเทียบเอง" และได้กลับมาแค่ ชื่อ/รหัส/ระยะห่าง
//    ไม่มีใครดึงลายใบหน้าของคนอื่นออกมาได้ (ดู supabase/schema.sql)
// คีย์ด้านล่างเป็น publishable key ที่ออกแบบให้ใส่ในหน้าเว็บได้ สิทธิ์ถูกจำกัดด้วยฟังก์ชันฝั่งฐานข้อมูล

const URL = "https://vwhuxqxugjrvwqaroiap.supabase.co";
const KEY = "sb_publishable_5i3UpYDL-ykRMzZOWWEx5Q_fHcSkXAS";

async function rpc(fn, args = {}) {
  const res = await fetch(`${URL}/rest/v1/rpc/${fn}`, {
    method: "POST",
    headers: { apikey: KEY, Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(data?.message || `เชื่อมต่อฐานข้อมูลไม่สำเร็จ (${res.status})`);
  return data;
}

const round = (d) => d.map((v) => +v.toFixed(6));

export const people = {
  all: () => rpc("list_employees"),
  add: (p) => rpc("register_employee", {
    p_id: p.id, p_name: p.name, p_dept: p.dept, p_consent: true,
    p_templates: p.templates.map((t) => ({ kind: t.kind, d: round(t.d) })),
  }),
  remove: (id, pin) => rpc("delete_employee", { p_id: id, p_pin: pin }),
};

export const log = {
  all: (limit = 200) => rpc("list_checkins", { p_limit: limit }),
  add: (e) => rpc("log_checkin", { p_id: e.id, p_result: e.result, p_mask: e.mask, p_distance: e.dist, p_photo: e.photo || null }),
  clear: (pin) => rpc("clear_checkins", { p_pin: pin }),
  // รายงานประวัติเข้างาน: ไฟล์ HTML เปิดได้ทุกเครื่อง มีรูปหน้าย่อ และกด "พิมพ์ / บันทึกเป็น PDF" ได้
  async toReport() {
    const [rows, staff] = await Promise.all([log.all(1000), people.all().catch(() => [])]);
    const dept = Object.fromEntries((staff || []).map((p) => [p.id, p.dept || ""]));
    const RESULT = { checked_in: "เข้างาน", denied_no_mask: "ไม่ให้เข้า (ไม่สวมหน้ากาก)" };
    const tz = { timeZone: "Asia/Bangkok" };
    const date = (t) => new Date(t).toLocaleDateString("th-TH", { ...tz, dateStyle: "medium" });
    const time = (t) => new Date(t).toLocaleTimeString("th-TH", { ...tz, hour12: false });
    // escape ทุกค่าที่มาจากผู้ใช้ กัน XSS ในไฟล์รายงาน
    const h = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    const img = (p) => (p && p.startsWith("data:image/jpeg;base64,") ? `<img src="${p}" alt="">` : `<span class="nop">ไม่มีรูป</span>`);
    const list = [...rows].reverse();
    const ok = list.filter((r) => r.result === "checked_in").length;
    const body = list.map((r, i) => `<tr class="${r.result === "checked_in" ? "" : "deny"}">
<td>${i + 1}</td><td>${img(r.photo)}</td><td>${h(date(r.created_at))}<br><b>${h(time(r.created_at))}</b></td>
<td>${h(r.id)}</td><td>${h(r.name)}</td><td>${h(dept[r.id] || "–")}</td>
<td class="res">${h(RESULT[r.result] || r.result)}</td>
<td>${r.mask_prob == null ? "–" : (r.mask_prob * 100).toFixed(1) + "%"}</td></tr>`).join("\n");
    const now = new Date().toLocaleString("th-TH", { ...tz, dateStyle: "long", timeStyle: "short" });
    return `<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>รายงานประวัติเข้างาน</title><style>
body{font-family:Sarabun,Tahoma,sans-serif;margin:24px;color:#111;background:#fff}
h1{font-size:22px;margin:0 0 4px}.meta{color:#555;margin-bottom:16px}
.sum{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}.sum div{border:1px solid #ddd;border-radius:8px;padding:8px 14px}
.sum b{display:block;font-size:20px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:middle}
th{background:#f1f5f9}img{width:56px;height:56px;object-fit:cover;border-radius:6px;display:block}
.deny .res{color:#b91c1c;font-weight:700}.res{color:#15803d;font-weight:700}.nop{color:#999;font-size:12px}
button{margin-bottom:16px;padding:8px 16px;font:inherit;cursor:pointer}
@media print{button{display:none}body{margin:0}tr{break-inside:avoid}}
</style></head><body>
<h1>รายงานประวัติเข้างาน — Mask Check-in</h1>
<div class="meta">ออกรายงานเมื่อ ${h(now)} · ข้อมูลส่วนบุคคลตาม พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล ใช้ภายในหน่วยงานเท่านั้น</div>
<div class="sum"><div>ทั้งหมด<b>${list.length}</b></div><div>เข้างาน<b>${ok}</b></div><div>ไม่ให้เข้า<b>${list.length - ok}</b></div></div>
<button onclick="print()">พิมพ์ / บันทึกเป็น PDF</button>
<table><thead><tr><th>#</th><th>รูป</th><th>วันที่ / เวลา</th><th>รหัส</th><th>ชื่อ-นามสกุล</th><th>แผนก</th><th>ผล</th><th>สวมหน้ากาก</th></tr></thead>
<tbody>${body || '<tr><td colspan="8">ยังไม่มีประวัติ</td></tr>'}</tbody></table></body></html>`;
  },
};

/** 1-NN ในฐานข้อมูล: คืน { person: {id,name,dept} | null (ไม่รู้จัก), dist } หรือ null ถ้ายังไม่มีใครลงทะเบียน */
export async function identify(descriptor, threshold) {
  const rows = await rpc("identify", { p_desc: round(descriptor), p_threshold: threshold });
  if (!rows || !rows.length) return null;
  const r = rows[0];
  return { person: r.id ? { id: r.id, name: r.name, dept: r.dept } : null, dist: r.distance, kind: r.kind };
}
