"""run_synth.py — สร้างตัวอย่าง "มือปิดปาก" สังเคราะห์ (วาดมือสีผิวทับปากบนรูป ds1/without_mask) แล้วดึงลายใบหน้า
แนวคิดจาก NatOcc/RandOcc (Voo et al., CVPRW 2022)
รัน: python ml/extract/run_synth.py 700 data/synth/synth_hand.jsonl [keep]   (keep = บันทึกรูปตัวอย่างไว้ดู)
"""
import asyncio,os,sys,json,random,base64,http.server,threading,functools
from playwright.async_api import async_playwright
ROOT=os.path.dirname(os.path.abspath(__file__))  # ml/extract (ต้องมี ds1/ และไฟล์ face-api เหมือน run_extract.py)
class Q(http.server.SimpleHTTPRequestHandler):
  def log_message(self,*a): pass
srv=http.server.ThreadingHTTPServer(("127.0.0.1",8771),functools.partial(Q,directory=ROOT));threading.Thread(target=srv.serve_forever,daemon=True).start()
n=int(sys.argv[1]); out=sys.argv[2]; keep=len(sys.argv)>3
random.seed(11)
src=["ds1/without_mask/"+f for f in sorted(os.listdir(ROOT+"/ds1/without_mask"))]
src=random.sample(src,min(n,len(src)))
async def main():
  async with async_playwright() as p:
    b=await p.chromium.launch(); pg=await b.new_page()
    if keep: await pg.add_init_script("window.KEEP=1")
    await pg.goto("http://127.0.0.1:8771/synth.html"); print(await pg.evaluate("window.ready"),flush=True)
    ok=0
    with open(out,"w") as fo:
      for i,s in enumerate(src):
        try: r=await pg.evaluate("([s,k])=>synth(s,k)",[s,i*7+3])
        except Exception as e: print("ERR",s,str(e)[:100]); continue
        if not r: continue
        if keep and r.get("png"):
          os.makedirs(os.path.join(ROOT,"synth_preview"),exist_ok=True)
          open(os.path.join(ROOT,"synth_preview",f"{i}.jpg"),"wb").write(base64.b64decode(r["png"].split(",")[1]))
        if r["found"]:
          ok+=1; fo.write(json.dumps({"path":"synth/"+s,"src":"synth","person":None,"session":"synth","label":"hand_cover","found":1,"score":0,"d":r["d"]})+"\n")
        if (i+1)%50==0: print(i+1,"ok",ok,flush=True)
    print("done",ok,"of",len(src))
    await b.close()
asyncio.run(main())
