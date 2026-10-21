"""P3.4 fast, read-only CSV integrity audit. Python standard library only."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

FILES = {
 'DeepChange': ('deepchange_unified_v1.csv','deepchange_p3_provisional.csv'),
 'Occluded-PoseTrack-ReID': ('occluded_posetrack_unified_v1.csv','occluded_posetrack_p3_provisional.csv'),
 'CUHK03_NP': ('cuhk03_np_unified_v1.csv','cuhk03_np_p3_provisional.csv'),
}

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
 return h.hexdigest()

def validate(original, provisional, label):
 issues=[]; roles_by_pid={}; seen_uid=set(); seen_path=set()
 stats={r:{'images':0,'identities':set(),'views':set(),'videos':set(),'cameras':set(),'tracklets':set()} for r in ('train','validation')}
 id_features=defaultdict(lambda:{'views':set(),'cameras':set(),'videos':set(),'tracklets':set()})
 with original.open('r',encoding='utf-8-sig',newline='') as af, provisional.open('r',encoding='utf-8-sig',newline='') as bf:
  a=csv.DictReader(af); b=csv.DictReader(bf)
  if not a.fieldnames or not b.fieldnames: raise ValueError('Missing CSV header')
  if b.fieldnames != a.fieldnames+['p3_role']: issues.append('Columns/order changed or p3_role not last')
  expected=set(a.fieldnames)
  for required in ('sample_uid','relative_path','identity_key','p3_role','source_split'):
   if required not in set(b.fieldnames): raise ValueError(f'Missing {required}')
  count=0
  from itertools import zip_longest
  for x,y in zip_longest(a,b):
   count+=1
   if x is None or y is None:
    issues.append('Original/provisional row counts differ'); break
   if None in x or None in y: issues.append(f'Malformed row {count}'); break
   if {k:y.get(k) for k in a.fieldnames} != x:
    issues.append(f'Original metadata changed or reordered at row {count}'); break
   uid=y['sample_uid']; path=y['relative_path']; pid=y['identity_key']; role=y['p3_role']
   if not uid or uid in seen_uid: issues.append(f'Missing/duplicate sample UID at row {count}'); break
   if not path or path in seen_path: issues.append(f'Missing/duplicate path at row {count}'); break
   if not pid: issues.append(f'Missing identity at row {count}'); break
   if y['source_split']!='train': issues.append(f'Unexpected source_split at row {count}'); break
   if role not in stats: issues.append(f'Invalid p3_role at row {count}'); break
   if pid in roles_by_pid and roles_by_pid[pid]!=role:
    issues.append(f'Identity appears in both roles: {pid}'); break
   roles_by_pid[pid]=role; seen_uid.add(uid); seen_path.add(path)
   s=stats[role]; s['images']+=1; s['identities'].add(pid)
   for col,key in [('raw_camera_or_view','views'),('camera_context_id','cameras'),('video_id','videos'),('tracklet_id','tracklets')]:
    v=y.get(col,'')
    if v: s[key].add(v); id_features[pid][key].add(v)
   if count%20000==0: print(f'  {label}: checked {count:,} rows',flush=True)
 for role,s in stats.items():
  if not s['images']: issues.append(f'{role} is empty')
  if label=='CUHK03_NP':
   missing=sum(id_features[p]['views']!={'1','2'} for p in s['identities'])
   if missing: issues.append(f'{role}: {missing} identities do not have both views')
   s['ids_without_both_views']=missing
  if label=='DeepChange':
   s['ids_with_multiple_cameras']=sum(len(id_features[p]['cameras'])>1 for p in s['identities'])
   s['ids_with_multiple_tracklets']=sum(len(id_features[p]['tracklets'])>1 for p in s['identities'])
  if label=='Occluded-PoseTrack-ReID':
   s['ids_with_multiple_videos']=sum(len(id_features[p]['videos'])>1 for p in s['identities'])
  for k in ('identities','views','videos','cameras','tracklets'):
   s[k]=len(s[k])
 return {'status':'PASS' if not issues else 'FAIL','rows_checked':len(seen_uid),
         'original_sha256':digest(original),'provisional_sha256':digest(provisional),
         'splits':stats,'issues':issues}

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--input-dir',type=Path,default=Path('.'))
 p.add_argument('--out-dir',type=Path,default=Path('./phase3'))
 args=p.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
 report={'scope':'P3.4 metadata split integrity only, not pixel hashing/official eval verification/DeepChange exclusions','datasets':{}}
 for label,(original,split) in FILES.items():
  print(f'Checking {label}...',flush=True)
  try: result=validate(args.input_dir/original,args.out_dir/split,label)
  except Exception as ex: result={'status':'FAIL','issues':[str(ex)]}
  report['datasets'][label]=result
  print(f'  {result["status"]}: {result.get("splits",{})}',flush=True)
  if result['issues']: print('  Issues:',result['issues'],flush=True)
 report['overall']='PASS' if all(r['status']=='PASS' for r in report['datasets'].values()) else 'FAIL'
 target=args.out_dir/'p3_4_validation_report.json'
 target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
 print(f'\nOverall: {report["overall"]} (metadata only)\nReport: {target}',flush=True)
 return 0 if report['overall']=='PASS' else 1

if __name__=='__main__': raise SystemExit(main())
