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
  add: (e) => rpc("log_checkin", { p_id: e.id, p_result: e.result, p_mask: e.mask, p_distance: e.dist }),
  clear: (pin) => rpc("clear_checkins", { p_pin: pin }),
  // ส่งออกประวัติเป็น CSV ที่เปิดใน Excel / Google Sheets แล้วอ่านง่าย
  // - หัวคอลัมน์ภาษาไทย, แยกวันที่กับเวลา (เวลาไทย), เรียงจากเก่าไปใหม่, มีคอลัมน์ลำดับ
  // - ผลลัพธ์เป็นคำไทย, ค่าความมั่นใจเป็น %, ระยะห่าง 2 ตำแหน่ง
  // - ใส่แผนก โดยจับคู่กับรายชื่อผู้ลงทะเบียน
  async toCSV() {
    const [rows, staff] = await Promise.all([log.all(1000), people.all().catch(() => [])]);
    const dept = Object.fromEntries((staff || []).map((p) => [p.id, p.dept || ""]));
    const RESULT = { checked_in: "เข้างาน", denied_no_mask: "ไม่ให้เข้า (ไม่สวมหน้ากาก)" };
    const tz = { timeZone: "Asia/Bangkok" };
    const date = (t) => new Date(t).toLocaleDateString("en-CA", tz);                     // 2026-11-05 (Excel เรียงได้)
    const time = (t) => new Date(t).toLocaleTimeString("en-GB", { ...tz, hour12: false }); // 08:15:30
    // กัน CSV/Formula injection (OWASP): ค่าที่ขึ้นต้นด้วย = + - @ จะถูก Excel มองเป็นสูตร
    const safe = (v) => { const s = String(v ?? ""); return /^[=+\-@\t\r]/.test(s) ? "'" + s : s; };
    const esc = (v) => `"${safe(v).replace(/"/g, '""')}"`;
    // รหัสที่เป็นตัวเลขล้วน เช่น 00123 ให้ Excel แสดงเป็นข้อความ ไม่ตัดเลข 0 หน้า
    const idCell = (id) => (/^\d+$/.test(id) ? `="${id}"` : esc(id));
    const head = ["ลำดับ", "วันที่", "เวลา", "รหัสพนักงาน", "ชื่อ-นามสกุล", "แผนก", "ผลการตรวจ", "ความมั่นใจว่าสวมหน้ากาก (%)", "ระยะห่างใบหน้า"];
    const body = [...rows].reverse().map((r, i) => [
      esc(i + 1), esc(date(r.created_at)), esc(time(r.created_at)), idCell(r.id), esc(r.name), esc(dept[r.id] ?? ""),
      esc(RESULT[r.result] || r.result),
      esc(r.mask_prob == null ? "" : (r.mask_prob * 100).toFixed(1)),
      esc(r.distance == null ? "" : Number(r.distance).toFixed(2)),
    ].join(","));
    return [head.map(esc).join(","), ...body].join("\r\n"); // CRLF ตามมาตรฐาน CSV (RFC 4180)
  },
};

/** 1-NN ในฐานข้อมูล: คืน { person: {id,name,dept} | null (ไม่รู้จัก), dist } หรือ null ถ้ายังไม่มีใครลงทะเบียน */
export async function identify(descriptor, threshold) {
  const rows = await rpc("identify", { p_desc: round(descriptor), p_threshold: threshold });
  if (!rows || !rows.length) return null;
  const r = rows[0];
  return { person: r.id ? { id: r.id, name: r.name, dept: r.dept } : null, dist: r.distance, kind: r.kind };
}
