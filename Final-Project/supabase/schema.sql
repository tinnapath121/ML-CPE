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
declare v extensions.vector;
begin
  if jsonb_array_length(p_desc) <> 128 then raise exception 'ลายใบหน้าต้องมี 128 ค่า'; end if;
  v := p_desc::text::extensions.vector;
  return query
    select case when d.dist < least(p_threshold, 0.6) then e.id end,
           case when d.dist < least(p_threshold, 0.6) then e.name end,
           case when d.dist < least(p_threshold, 0.6) then e.dept end,
           d.dist::real, d.kind
    from (select t.employee_id, t.kind, (t.embedding operator(extensions.<->) v) as dist
          from app.templates t order by dist limit 1) d
    join app.employees e on e.id = d.employee_id;
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
