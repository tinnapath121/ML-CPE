-- Mask Check-in: ฐานข้อมูลกลาง (รันใน Supabase SQL Editor ของโปรเจกต์ใหม่ได้ทันที)
-- ตารางทั้งหมดอยู่ใน schema "app" ซึ่งไม่ถูกเปิดผ่าน API -> เบราว์เซอร์อ่านลายใบหน้าตรงๆ ไม่ได้
-- เข้าถึงได้เฉพาะผ่านฟังก์ชัน RPC ซึ่งคืนแค่ ชื่อ/รหัส/แผนก/ระยะห่าง
create extension if not exists vector with schema extensions;
create extension if not exists pgcrypto with schema extensions;
create schema if not exists app;
revoke all on schema app from public, anon, authenticated;

create table app.employees (
  id text primary key check (char_length(id) between 1 and 32),
  name text not null check (char_length(name) between 1 and 100),
  dept text check (char_length(dept) <= 60),
  consent_at timestamptz not null,
  created_at timestamptz not null default now()
);
create table app.templates (
  id bigserial primary key,
  employee_id text not null references app.employees(id) on delete cascade,
  kind text not null check (kind in ('plain', 'masked')),
  embedding extensions.vector(128) not null
);
create index on app.templates (employee_id);
create table app.checkins (
  id bigserial primary key,
  employee_id text not null references app.employees(id) on delete cascade,
  result text not null check (result in ('checked_in', 'denied_no_mask')),
  mask_prob real, distance real,
  photo text check (photo is null or (photo like 'data:image/jpeg;base64,%' and length(photo) <= 20000)), -- รูปหน้าย่อ ~96px
  created_at timestamptz not null default now()
);
create index on app.checkins (created_at desc);
create table app.settings (key text primary key, value text not null);
alter table app.employees enable row level security;
alter table app.templates enable row level security;
alter table app.checkins enable row level security;
alter table app.settings enable row level security;

create or replace function public.list_employees()
returns table (id text, name text, dept text, registered_at timestamptz, last_checkin timestamptz)
language sql stable security definer set search_path = '' as $$
  select e.id, e.name, e.dept, e.created_at,
         (select max(c.created_at) from app.checkins c where c.employee_id = e.id and c.result = 'checked_in')
  from app.employees e order by e.id;
$$;

create or replace function public.register_employee(p_id text, p_name text, p_dept text, p_consent boolean, p_templates jsonb)
returns text language plpgsql security definer set search_path = '' as $$
declare t jsonb;
begin
  if p_consent is not true then raise exception 'ต้องได้รับความยินยอมก่อนลงทะเบียน'; end if;
  if jsonb_typeof(p_templates) <> 'array' or jsonb_array_length(p_templates) not between 1 and 8 then
    raise exception 'ข้อมูลลายใบหน้าไม่ถูกต้อง'; end if;
  if exists (select 1 from app.employees where id = trim(p_id)) then raise exception 'รหัสพนักงานนี้ลงทะเบียนแล้ว'; end if;
  insert into app.employees (id, name, dept, consent_at) values (trim(p_id), trim(p_name), nullif(trim(p_dept), ''), now());
  for t in select * from jsonb_array_elements(p_templates) loop
    if jsonb_array_length(t->'d') <> 128 then raise exception 'ลายใบหน้าต้องมี 128 ค่า'; end if;
    insert into app.templates (employee_id, kind, embedding) values (trim(p_id), t->>'kind', (t->'d')::text::extensions.vector);
  end loop;
  return trim(p_id);
end $$;

-- 1-Nearest Neighbor ในฐานข้อมูล (pgvector: <-> = ระยะ Euclidean) คืนชื่อเฉพาะเมื่อระยะต่ำกว่าเกณฑ์
create or replace function public.identify(p_desc jsonb, p_threshold real default 0.55)
returns table (id text, name text, dept text, distance real, kind text)
language plpgsql stable security definer set search_path = '' as $$
-- 1-NN + margin test (เข้มขึ้น):
--  * ใกล้สุดกับ template "ใส่แมส" ใช้เกณฑ์ 0.42 (หน้าใส่แมสเหลือข้อมูลครึ่งบน คนต่างกันดูคล้ายกันง่าย)
--  * ใกล้สุดกับ template หน้าเปล่า ใช้เกณฑ์ตามที่ส่งมา (สูงสุด 0.55)
--  * ต้องใกล้กว่าคนอันดับ 2 อย่างน้อย 0.08 กันจำสลับ
declare v extensions.vector; d1 real; d2 real; e1 text; k1 text; thr real; margin constant real := 0.08;
begin
  if jsonb_array_length(p_desc) <> 128 then raise exception 'ลายใบหน้าต้องมี 128 ค่า'; end if;
  v := p_desc::text::extensions.vector;
  select x.employee_id, x.dist, x.kind into e1, d1, k1 from (
    select distinct on (t.employee_id) t.employee_id, t.kind, (t.embedding operator(extensions.<->) v)::real dist
    from app.templates t order by t.employee_id, dist) x order by x.dist limit 1;
  if e1 is null then return; end if;
  thr := case when k1 = 'masked' then least(p_threshold, 0.42) else least(p_threshold, 0.55) end;
  select min(t.embedding operator(extensions.<->) v)::real into d2 from app.templates t where t.employee_id <> e1;
  return query
    select case when ok then e.id end, case when ok then e.name end, case when ok then e.dept end, d1, k1
    from app.employees e, (select d1 < thr and (d2 is null or d2 - d1 >= margin) as ok) c
    where e.id = e1;
end $$;

create or replace function public.log_checkin(p_id text, p_result text, p_mask real, p_distance real, p_photo text)
returns void language sql security definer set search_path = '' as $$
  insert into app.checkins (employee_id, result, mask_prob, distance, photo) values (p_id, p_result, p_mask, p_distance, p_photo);
$$;

create or replace function public.list_checkins(p_limit int default 200)
returns table (created_at timestamptz, id text, name text, result text, mask_prob real, distance real, photo text)
language sql stable security definer set search_path = '' as $$
  select c.created_at, c.employee_id, e.name, c.result, c.mask_prob, c.distance, c.photo
  from app.checkins c join app.employees e on e.id = c.employee_id
  order by c.created_at desc limit least(greatest(p_limit, 1), 1000);
$$;

create or replace function public.delete_employee(p_id text, p_pin text)
returns void language plpgsql security definer set search_path = '' as $$
begin
  if not exists (select 1 from app.settings where key = 'admin_pin' and value = extensions.crypt(p_pin, value)) then
    raise exception 'PIN ผู้ดูแลไม่ถูกต้อง'; end if;
  delete from app.employees where id = p_id;
end $$;

create or replace function public.clear_checkins(p_pin text)
returns void language plpgsql security definer set search_path = '' as $$
begin
  if not exists (select 1 from app.settings where key = 'admin_pin' and value = extensions.crypt(p_pin, value)) then
    raise exception 'PIN ผู้ดูแลไม่ถูกต้อง'; end if;
  delete from app.checkins where true;
end $$;

revoke all on function public.list_employees(), public.register_employee(text,text,text,boolean,jsonb),
  public.identify(jsonb,real), public.log_checkin(text,text,real,real,text), public.list_checkins(int),
  public.delete_employee(text,text), public.clear_checkins(text) from public;
grant execute on function public.list_employees(), public.register_employee(text,text,text,boolean,jsonb),
  public.identify(jsonb,real), public.log_checkin(text,text,real,real,text), public.list_checkins(int),
  public.delete_employee(text,text), public.clear_checkins(text) to anon, authenticated;

-- ตั้ง PIN ผู้ดูแล (เปลี่ยน 'xxxxxx' เป็น PIN จริง ห้าม commit PIN จริงลง repo)
-- insert into app.settings(key, value) values ('admin_pin', extensions.crypt('xxxxxx', extensions.gen_salt('bf')));

-- ===== อัปเดต v14: ป้องกันรูปหน้า + จำกัดความยาวข้อมูล (รันต่อจากด้านบน) =====
-- ประวัติสาธารณะไม่ส่งรูปออกไป (PDPA: รูปหน้าเห็นได้เฉพาะผู้ดูแล)
drop function if exists public.list_checkins(int);
create function public.list_checkins(p_limit int default 200)
returns table (created_at timestamptz, id text, name text, result text, mask_prob real, distance real)
language sql stable security definer set search_path = '' as $$
  select c.created_at, c.employee_id, e.name, c.result, c.mask_prob, c.distance
  from app.checkins c join app.employees e on e.id = c.employee_id
  order by c.created_at desc limit least(greatest(p_limit, 1), 1000);
$$;
-- รายงานพร้อมรูป ต้องใช้ PIN ผู้ดูแล
create or replace function public.report_checkins(p_pin text, p_limit int default 1000)
returns table (created_at timestamptz, id text, name text, dept text, result text, mask_prob real, distance real, photo text)
language plpgsql stable security definer set search_path = '' as $$
begin
  if not exists (select 1 from app.settings where key = 'admin_pin' and value = extensions.crypt(p_pin, value)) then
    raise exception 'PIN ผู้ดูแลไม่ถูกต้อง'; end if;
  return query select c.created_at, c.employee_id, e.name, e.dept, c.result, c.mask_prob, c.distance, c.photo
    from app.checkins c join app.employees e on e.id = c.employee_id
    order by c.created_at desc limit least(greatest(p_limit, 1), 1000);
end $$;
drop function if exists public.log_checkin(text,text,real,real);
-- ลงทะเบียน: จำกัดความยาว รหัส 30 / ชื่อ 100 / แผนก 60 ตัวอักษร และตรวจชนิด template
create or replace function public.register_employee(p_id text, p_name text, p_dept text, p_consent boolean, p_templates jsonb)
returns text language plpgsql security definer set search_path = '' as $$
declare t jsonb;
begin
  if p_consent is not true then raise exception 'ต้องได้รับความยินยอมก่อนลงทะเบียน'; end if;
  if length(trim(coalesce(p_id,''))) not between 1 and 30 or length(trim(coalesce(p_name,''))) not between 1 and 100
     or length(coalesce(p_dept,'')) > 60 then raise exception 'รหัส/ชื่อ/แผนก ยาวเกินกำหนด'; end if;
  if jsonb_typeof(p_templates) <> 'array' or jsonb_array_length(p_templates) not between 1 and 8 then
    raise exception 'ข้อมูลลายใบหน้าไม่ถูกต้อง'; end if;
  if exists (select 1 from app.employees where id = trim(p_id)) then raise exception 'รหัสพนักงานนี้ลงทะเบียนแล้ว'; end if;
  insert into app.employees (id, name, dept, consent_at) values (trim(p_id), trim(p_name), nullif(trim(p_dept), ''), now());
  for t in select * from jsonb_array_elements(p_templates) loop
    if t->>'kind' not in ('plain','masked') or jsonb_array_length(t->'d') <> 128 then raise exception 'ลายใบหน้าไม่ถูกต้อง'; end if;
    insert into app.templates (employee_id, kind, embedding) values (trim(p_id), t->>'kind', (t->'d')::text::extensions.vector);
  end loop;
  return trim(p_id);
end $$;
revoke all on function public.list_checkins(int), public.report_checkins(text,int) from public;
grant execute on function public.list_checkins(int), public.report_checkins(text,int) to anon, authenticated;

-- ===== อัปเดต v16 (v19: เกณฑ์ใส่แมสผ่อนเป็น 0.51): จำหน้าเข้มขึ้น (คนใส่แว่น+แมสคล้ายกัน) =====
create or replace function public.identify(p_desc jsonb, p_threshold real default 0.55)
returns table (id text, name text, dept text, distance real, kind text)
language plpgsql stable security definer set search_path = '' as $$
-- คะแนนแต่ละคน = เฉลี่ยระยะของ template ใกล้สุด 2 อัน · เกณฑ์ใส่แมส 0.51 / หน้าเปล่า 0.50 · ต้องห่างคนอันดับ 2 ≥ 0.10
declare v extensions.vector; d1 real; d2 real; e1 text; k1 text; thr real; margin constant real := 0.10;
begin
  if jsonb_array_length(p_desc) <> 128 then raise exception 'ลายใบหน้าต้องมี 128 ค่า'; end if;
  v := p_desc::text::extensions.vector;
  with d as (select t.employee_id e, t.kind k, (t.embedding operator(extensions.<->) v)::real dist,
                    row_number() over (partition by t.employee_id order by t.embedding operator(extensions.<->) v) rn
             from app.templates t),
       s as (select e, avg(dist)::real sc, (array_agg(k order by dist))[1] k from d where rn <= 2 group by e)
  select x.e, x.sc, x.k into e1, d1, k1 from s x order by x.sc limit 1;
  if e1 is null then return; end if;
  with d as (select t.employee_id e, (t.embedding operator(extensions.<->) v)::real dist,
                    row_number() over (partition by t.employee_id order by t.embedding operator(extensions.<->) v) rn
             from app.templates t where t.employee_id <> e1)
  select min(sc) into d2 from (select avg(dist)::real sc from d where rn <= 2 group by e) z;
  thr := case when k1 = 'masked' then least(p_threshold, 0.51) else least(p_threshold, 0.50) end;
  return query
    select case when ok then e.id end, case when ok then e.name end, case when ok then e.dept end, d1, k1
    from app.employees e, (select d1 < thr and (d2 is null or d2 - d1 >= margin) as ok) c
    where e.id = e1;
end $$;
