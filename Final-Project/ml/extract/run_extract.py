"""run_extract.py — ดึงลายใบหน้า 128 ค่าจาก dataset ด้วย face-api ตัวเดียวกับบนเว็บ (รันใน Chromium แบบ headless)
ติดตั้ง:  pip install playwright && playwright install chromium
          npm i @vladmandic/face-api@1.7.15 @tensorflow/tfjs-backend-wasm@4.22.0
          คัดลอก face-api/dist/face-api.esm.js, face-api/model/, tfjs-backend-wasm/dist/*.wasm มาไว้ในโฟลเดอร์นี้
          วางรูปไว้ที่ ds1/{with_mask,without_mask}, ds2/..., real/...
รัน:      python ml/extract/run_extract.py 1 data/features.jsonl ds1,ds2,real
"""
import asyncio, json, os, sys, time, hashlib
from playwright.async_api import async_playwright
import http.server, threading, functools
ROOT=os.path.dirname(os.path.abspath(__file__))  # ml/extract/ ต้องมี face-api.esm.js, model/, *.wasm, ds1/ ds2/ real/
class Q(http.server.SimpleHTTPRequestHandler):
  def log_message(self,*a): pass
srv=http.server.ThreadingHTTPServer(("127.0.0.1",8765),functools.partial(Q,directory=ROOT))
threading.Thread(target=srv.serve_forever,daemon=True).start()
limit=int(sys.argv[1]) if len(sys.argv)>1 else None
out=sys.argv[2] if len(sys.argv)>2 else os.path.join(ROOT,"..","..","data","features.jsonl")
files=[]; seen=set()
for src in (sys.argv[3].split(",") if len(sys.argv)>3 else ["ds1","ds2"]):
  for lab in ["with_mask","without_mask"]:
    d=os.path.join(ROOT,src,lab)
    for f in sorted(os.listdir(d)):
      if not f.lower().endswith((".jpg",".jpeg",".png")): continue
      h=hashlib.md5(open(os.path.join(d,f),"rb").read()).hexdigest()
      if h in seen: continue
      seen.add(h); files.append((src,lab,f))
print("unique files",len(files),flush=True)
import random; random.seed(7)
keep=[]
for lab in ["with_mask","without_mask"]:
  c=[x for x in files if x[0]=="ds1" and x[1]==lab]; keep+=random.sample(c,min(900,len(c)))
keep+=[x for x in files if x[0]!="ds1"]
files=keep if limit else files
done=set()
if os.path.exists(out):
  for l in open(out): done.add(json.loads(l)["path"])
async def main():
  async with async_playwright() as p:
    b=await p.chromium.launch()
    pg=await b.new_page(); await pg.add_init_script("window.BACKEND='wasm';"); await pg.goto("http://127.0.0.1:8765/extract.html")
    print("backend",await pg.evaluate("window.ready"),flush=True)
    t=time.time(); n=0
    with open(out,"a") as fo:
      for src,lab,f in files:
        path=f"{src}/{lab}/{f}"
        if path in done: continue
        try: r=await pg.evaluate("p=>extract(p)",path)
        except Exception as e: print("ERR",path,str(e)[:80]); continue
        fo.write(json.dumps({"path":path,"src":src,"label":lab,**r})+"\n"); n+=1
        if n%200==0: print(n,f"{(time.time()-t)/n:.3f}s/img",flush=True)
    print("done",n,f"{(time.time()-t)/max(n,1):.3f}s/img")
    await b.close()
asyncio.run(main())
