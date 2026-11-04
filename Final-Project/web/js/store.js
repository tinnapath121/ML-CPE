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
  // รายงานประวัติเข้างานเป็นไฟล์ Excel (.xlsx) พร้อมรูปหน้าย่อฝังในเซลล์ (ใช้ ExcelJS โหลดเมื่อกดเท่านั้น)
  async toExcel() {
    const [rows, staff] = await Promise.all([log.all(1000), people.all().catch(() => [])]);
    if (!window.ExcelJS) await new Promise((ok, bad) => {
      const sc = document.createElement("script");
      sc.src = "https://cdnjs.cloudflare.com/ajax/libs/exceljs/4.4.0/exceljs.min.js";
      sc.onload = ok; sc.onerror = () => bad(new Error("โหลดตัวสร้าง Excel ไม่สำเร็จ"));
      document.head.append(sc);
    });
    const dept = Object.fromEntries((staff || []).map((p) => [p.id, p.dept || ""]));
    const RESULT = { checked_in: "เข้างาน", denied_no_mask: "ไม่ให้เข้า (ไม่สวมหน้ากาก)" };
    const tz = { timeZone: "Asia/Bangkok" };
    // กัน Formula injection ใน Excel (OWASP)
    const safe = (v) => { const t = String(v ?? ""); return /^[=+\-@]/.test(t) ? "'" + t : t; };
    const wb = new ExcelJS.Workbook();
    const ws = wb.addWorksheet("ประวัติเข้างาน", { views: [{ state: "frozen", ySplit: 1 }] });
    ws.columns = [
      { header: "ลำดับ", width: 7 }, { header: "รูป", width: 11 }, { header: "วันที่", width: 13 },
      { header: "เวลา", width: 10 }, { header: "รหัสพนักงาน", width: 14 }, { header: "ชื่อ-นามสกุล", width: 26 },
      { header: "แผนก", width: 14 }, { header: "ผลการตรวจ", width: 26 }, { header: "สวมหน้ากาก (%)", width: 15 },
    ];
    const head = ws.getRow(1);
    head.font = { bold: true, color: { argb: "FFFFFFFF" } }; head.height = 24;
    head.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1E3A5F" } };
    head.alignment = { vertical: "middle", horizontal: "center" };
    [...rows].reverse().forEach((r, i) => {
      const row = ws.addRow([i + 1, "",
        new Date(r.created_at).toLocaleDateString("en-CA", tz),
        new Date(r.created_at).toLocaleTimeString("en-GB", { ...tz, hour12: false }),
        safe(r.id), safe(r.name), safe(dept[r.id]), RESULT[r.result] || r.result,
        r.mask_prob == null ? null : +(r.mask_prob * 100).toFixed(1)]);
      row.height = 60; row.alignment = { vertical: "middle" };
      row.getCell(5).numFmt = "@";
      row.getCell(8).font = { bold: true, color: { argb: r.result === "checked_in" ? "FF15803D" : "FFB91C1C" } };
      if (r.photo && r.photo.startsWith("data:image/jpeg;base64,")) {
        const id = wb.addImage({ base64: r.photo, extension: "jpeg" });
        ws.addImage(id, { tl: { col: 1.1, row: row.number - 1 + 0.05 }, ext: { width: 72, height: 72 } });
      } else row.getCell(2).value = "ไม่มีรูป";
    });
    return new Blob([await wb.xlsx.writeBuffer()], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  },
};

/** 1-NN ในฐานข้อมูล: คืน { person: {id,name,dept} | null (ไม่รู้จัก), dist } หรือ null ถ้ายังไม่มีใครลงทะเบียน */
export async function identify(descriptor, threshold) {
  const rows = await rpc("identify", { p_desc: round(descriptor), p_threshold: threshold });
  if (!rows || !rows.length) return null;
  const r = rows[0];
  return { person: r.id ? { id: r.id, name: r.name, dept: r.dept } : null, dist: r.distance, kind: r.kind };
}
