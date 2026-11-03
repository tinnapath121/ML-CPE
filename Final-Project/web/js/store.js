// store.js — เก็บข้อมูลผู้ลงทะเบียนและประวัติเข้างาน "ในเครื่องนี้เท่านั้น" (localStorage)
// เก็บเฉพาะลายใบหน้า (ตัวเลข 128 ค่า) ไม่เก็บรูปถ่าย เพื่อลดข้อมูลส่วนบุคคลให้น้อยที่สุด (PDPA data minimization)
// ระบบจริงในโรงงานควรย้ายไปเก็บบนเซิร์ฟเวอร์ของหน่วยงานที่เข้ารหัสและควบคุมสิทธิ์

const KEY_PEOPLE = "mc-people-v1";
const KEY_LOG = "mc-log-v1";

function read(key) { try { return JSON.parse(localStorage.getItem(key)) || []; } catch { return []; } }
function write(key, v) {
  try { localStorage.setItem(key, JSON.stringify(v)); return true; } catch { return false; }
}

export const people = {
  all: () => read(KEY_PEOPLE),
  add(p) { const list = read(KEY_PEOPLE).filter((x) => x.id !== p.id); list.push(p); return write(KEY_PEOPLE, list); },
  remove(id) { write(KEY_PEOPLE, read(KEY_PEOPLE).filter((x) => x.id !== id)); },
  clear() { write(KEY_PEOPLE, []); },
};

export const log = {
  all: () => read(KEY_LOG),
  add(entry) { const l = read(KEY_LOG); l.push(entry); write(KEY_LOG, l.slice(-2000)); },
  clear() { write(KEY_LOG, []); },
  toCSV() {
    const rows = read(KEY_LOG);
    const esc = (s) => `"${String(s ?? "").replace(/"/g, '""')}"`;
    return ["time,employee_id,name,result,mask_prob,distance",
      ...rows.map((r) => [r.time, r.id, r.name, r.result, r.mask, r.dist].map(esc).join(","))].join("\n");
  },
};

/** หาคนที่ลายใบหน้าใกล้ที่สุด (1-Nearest Neighbor ข้ามทุก template ของทุกคน) */
export function identify(descriptor, distanceFn) {
  let best = null;
  for (const p of read(KEY_PEOPLE)) {
    for (const t of p.templates) {
      const d = distanceFn(descriptor, t.d);
      if (!best || d < best.dist) best = { person: p, dist: d, kind: t.kind };
    }
  }
  return best;
}
