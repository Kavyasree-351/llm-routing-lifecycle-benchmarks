"""Run bounded checks; no training sweeps, downloads or final evaluation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
COMMANDS=[['-m', 'unittest', 'discover', '-s', '.', '-p', 'test_study.py', '-v']]
def main():
    p=argparse.ArgumentParser();p.add_argument('--record',action='store_true');a=p.parse_args();records=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for args in COMMANDS:
        print('Checking: '+' '.join(args),flush=True);start=time.perf_counter()
        done=subprocess.run([sys.executable,'-B']+args,cwd=ROOT,env=env)
        records.append(dict(command=args,exit_code=done.returncode,seconds=time.perf_counter()-start))
        if done.returncode:break
    success=len(records)==len(COMMANDS) and all(r['exit_code']==0 for r in records)
    if a.record:(ROOT/'PUBLICATION_TESTS.json').write_text(json.dumps(dict(successful=success,checks=records,final_queries_evaluated=False),indent=2)+'\n')
    return 0 if success else 1
if __name__=='__main__':raise SystemExit(main())
