"""Compare local FP4-Local server profiles with sequential and overlapping requests.
Uses private ports, never stops a pre-existing server. Not a quality benchmark.
"""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,threading,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',action='append',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--port',type=int,default=8102);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 import socket
 def get(path):
  with urllib.request.urlopen(base+path,timeout=3) as r:return json.load(r)
 report={}
 prompt='Explain how a CPU and a GPU cooperate to run a large language model on a desktop PC. Cover memory bandwidth, expert caching, and batch scheduling in detail.'
 for index,name in enumerate(a.config):
  port=a.port+index;base=f'http://127.0.0.1:{port}'
  with socket.socket() as sock:sock.bind(('127.0.0.1',port))
  cfg=Path(name).resolve();label=cfg.stem;run={'config':json.loads(cfg.read_text(encoding='utf8'))};report[label]=run
  log=(a.output/(label+'.log')).open('w',encoding='utf8');env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
  p=subprocess.Popen([sys.executable,'-m','serve.server','--engine','strata','--config',str(cfg),'--host','127.0.0.1','--port',str(port)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
  stop=threading.Event();samples=[];worker=None
  def monitor():
   while not stop.wait(.4):
    try:
     m=get('/metrics');samples.append({'slots':m['live'].get('slots'),'running':m['live'].get('running'),'hardware':m.get('hardware',{})})
    except Exception:pass
  def request(text):
   payload={'model':'Qwen3.8-Flash-Next-FP4-Local','messages':[{'role':'user','content':text}],'reasoning_effort':'none','temperature':0,'max_tokens':256,'strata_checkpoint':False}
   req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
   t=time.monotonic()
   with urllib.request.urlopen(req,timeout=180) as r:reply=json.load(r)
   return {'wall_s':time.monotonic()-t,'response':reply}
  try:
   deadline=time.monotonic()+600
   while time.monotonic()<deadline:
    if p.poll() is not None:raise RuntimeError('server exited')
    try:
     if get('/props').get('total_slots')==2:break
    except Exception:pass
    time.sleep(2)
   else:raise RuntimeError('server did not become ready')
   run['engine']=get('/metrics')['engine']
   assert run['engine'].get('weight_source')=='safetensors','Not a real safetensors engine'
   run['exe_sha256']=hashlib.sha256(Path(run['config']['exe']).read_bytes()).hexdigest()
   worker=threading.Thread(target=monitor,daemon=True);worker.start()
   run['solo']=[request(prompt),request(prompt)]
   run['repeat_equal']=run['solo'][0]['response']['choices']==run['solo'][1]['response']['choices']
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    jobs=[pool.submit(request,x) for x in [prompt,'Write a numbered list of 30 practical tips for learning Python, with one explanatory sentence for each.']]
    run['concurrent']=[x.result() for x in jobs]
   run['metrics']=get('/metrics')['requests'];run['peak_running']=max((s.get('running') or 0 for s in samples),default=0)
   run['pass']=run['repeat_equal'] and run['peak_running']>=2 and all(x['response'].get('choices') for x in run['solo']+run['concurrent'])
   print(label,json.dumps({'pass':run['pass'],'repeat_equal':run['repeat_equal'],'peak_running':run['peak_running'],'solo_tok_s':[x['response']['usage']['completion_tokens']/x['wall_s'] for x in run['solo']],'concurrent_wall_s':[x['wall_s'] for x in run['concurrent']]}),flush=True)
  finally:
   stop.set()
   if worker:worker.join(timeout=5)
   run['samples']=samples
   subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);log.close()
   (a.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
