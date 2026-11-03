// sw.js — เก็บไฟล์ไว้ในเครื่อง ให้แอปเปิดได้แม้เน็ตในห้องไม่ดี (หลังเปิดครั้งแรกแล้ว)
// ไฟล์ของเรา: network-first · ไลบรารี/โมเดลจาก CDN (ระบุเวอร์ชันตายตัว): cache-first
const CACHE = "mask-checkin-v8";
const LOCAL = ["./", "index.html", "css/style.css", "manifest.webmanifest",
  "js/app.js", "js/face.js", "js/model.js", "js/store.js", "model/mask_model.json", "model/config.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => Promise.allSettled(LOCAL.map((u) => c.add(u)))));
  self.skipWaiting();
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  const url = new URL(e.request.url);
  const save = (res) => { if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); } return res; };
  if (url.hostname === "cdn.jsdelivr.net") {
    e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request).then(save)));
  } else if (url.origin === location.origin) {
    // cache: no-cache = ถามเซิร์ฟเวอร์ทุกครั้งว่ามีเวอร์ชันใหม่ไหม กันไฟล์เก่า/ใหม่ปนกันหลังอัปเดตเว็บ
    e.respondWith(fetch(e.request, { cache: "no-cache" }).then(save).catch(() => caches.match(e.request, { ignoreSearch: true })));
  }
});
