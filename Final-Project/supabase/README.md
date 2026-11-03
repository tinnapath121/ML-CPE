# ฐานข้อมูลกลาง (Supabase)

- โปรเจกต์: `mask-checkin` (ภูมิภาค ap-southeast-1 สิงคโปร์) · URL อยู่ใน `web/js/store.js`
- ตารางอยู่ใน schema `app` ซึ่ง**ไม่เปิดผ่าน API** (เรียก `/rest/v1/templates` ตรงๆ ได้ 404)
- เว็บเรียกได้เฉพาะฟังก์ชัน RPC:

| ฟังก์ชัน | หน้าที่ | คืนค่า |
|---|---|---|
| `list_employees()` | รายชื่อผู้ลงทะเบียน | รหัส ชื่อ แผนก วันที่ลงทะเบียน เข้างานล่าสุด (ไม่มีลายใบหน้า) |
| `register_employee(...)` | ลงทะเบียน (ต้อง `p_consent = true`) | รหัส |
| `identify(p_desc, p_threshold)` | 1-NN ด้วย pgvector (`<->` = Euclidean) | ชื่อเฉพาะเมื่อระยะ < เกณฑ์ (สูงสุด 0.6) |
| `log_checkin(...)`, `list_checkins(...)` | บันทึก/ดูประวัติ | |
| `delete_employee(id, pin)`, `clear_checkins(pin)` | ลบ ต้องใช้ PIN ผู้ดูแล (เก็บเป็น bcrypt hash) | |

PIN ผู้ดูแลไม่ได้เก็บใน repo · เปลี่ยน PIN ได้ใน SQL Editor:
```sql
update app.settings set value = extensions.crypt('PINใหม่', extensions.gen_salt('bf')) where key = 'admin_pin';
```
