#!/usr/bin/env python3
"""Read-only final verification of Phase 3 internal retrieval CSVs.

Checks metadata assignments and retrieval semantics, NOT underlying image pixels
or external official evaluation sets. Python standard library only.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

CONFIG = {
    'deepchange_internal_cross_tracklet_v1': ('deepchange_p3_provisional.csv', 'deepchange'),
    'posetrack_internal_same_or_cross_video_v1': ('occluded_posetrack_p3_provisional.csv', 'pose'),
    'cuhk03_internal_c1_to_c2_v1': ('cuhk03_np_p3_provisional.csv', 'c1_to_c2'),
    'cuhk03_internal_c2_to_c1_v1': ('cuhk03_np_p3_provisional.csv', 'c2_to_c1'),
}
REQUIRED = {'sample_uid', 'identity_key', 'relative_path', 'p3_role',
            'source_split', 'raw_camera_or_view', 'video_id', 'tracklet_id'}

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f'{path.name}: repeated CSV column names')
        if not REQUIRED.issubset(fields):
            raise ValueError(f'{path.name}: missing fields {sorted(REQUIRED - set(fields))}')
        rows = list(rd)
    if len(rows) != len({r['sample_uid'] for r in rows}):
        raise ValueError(f'{path.name}: duplicate sample_uid')
    return rows, fields

def digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def check(protocol, original_rows, original_fields, path, kind):
    rows, fields = read(path)
    if fields != original_fields + ['validation_protocol', 'retrieval_role']:
        raise ValueError('Unexpected column schema or order')
    expected = {r['sample_uid']:r for r in original_rows if r['p3_role'] == 'validation'}
    if not expected:
        raise ValueError('Empty validation set')
    got = {r['sample_uid']:r for r in rows}
    if set(got) != set(expected):
        raise ValueError(f'Row mismatch: missing={len(set(expected)-set(got))}, extra={len(set(got)-set(expected))}')
    train_ids = {r['identity_key'] for r in original_rows if r['p3_role'] == 'train'}
    val_ids = {r['identity_key'] for r in original_rows if r['p3_role'] == 'validation'}
    if train_ids & val_ids:
        raise ValueError('Training and validation share identities')
    if any(r['source_split'] != 'train' for r in original_rows):
        raise ValueError('Unexpected original source_split')
    if any(r['p3_role'] not in {'train','validation'} for r in original_rows):
        raise ValueError('Unexpected p3_role in source')
    if any(r['validation_protocol'] != protocol for r in rows):
        raise ValueError('Inconsistent validation_protocol field')
    if any(r['retrieval_role'] not in {'query','gallery'} for r in rows):
        raise ValueError('Unexpected retrieval_role')
    for uid, row in got.items():
        if any(row[c] != expected[uid][c] for c in original_fields):
            raise ValueError(f'Original metadata modified at {uid}')
    if len({r['relative_path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate relative paths in validation retrieval')
    queries = [r for r in rows if r['retrieval_role']=='query']
    gallery = [r for r in rows if r['retrieval_role']=='gallery']
    if not queries or not gallery:
        raise ValueError('Empty query or gallery')
    byid = defaultdict(list)
    for r in gallery: byid[r['identity_key']].append(r)
    no_positive=[]; positives=[]; cross_camera=0; cross_video=0
    for q in queries:
        pool = byid[q['identity_key']]
        if kind=='deepchange':
            if not q['tracklet_id']:
                raise ValueError('DeepChange query missing tracklet')
            matches=[g for g in pool if g['tracklet_id'] != q['tracklet_id']]
            cross_camera += any(g['raw_camera_or_view']!=q['raw_camera_or_view'] for g in matches)
        elif kind=='pose':
            matches=pool
            cross_video += any(g['video_id'] != q['video_id'] for g in matches)
        else:
            matches=[g for g in pool if g['raw_camera_or_view'] != q['raw_camera_or_view']]
        positives.append(len(matches))
        if not matches: no_positive.append(q['sample_uid'])
    if no_positive:
        raise ValueError(f'{len(no_positive)} queries without valid positives; first: {no_positive[:3]}')
    if kind=='c1_to_c2' and (any(q['raw_camera_or_view']!='1' for q in queries) or any(g['raw_camera_or_view']!='2' for g in gallery)):
        raise ValueError('CUHK03 view direction c1 -> c2 failed')
    if kind=='c2_to_c1' and (any(q['raw_camera_or_view']!='2' for q in queries) or any(g['raw_camera_or_view']!='1' for g in gallery)):
        raise ValueError('CUHK03 view direction c2 -> c1 failed')
    if kind=='pose' and len(queries) != len(val_ids):
        raise ValueError('PoseTrack: expected one query per validation identity')
    if kind=='pose' and len({r['identity_key'] for r in queries}) != len(queries):
        raise ValueError('PoseTrack: repeated query identity')
    if kind=='deepchange':
        for q in queries:
            if not any(g['tracklet_id'] != q['tracklet_id'] for g in byid[q['identity_key']]):
                raise ValueError('DeepChange: no other-tracklet positive')
    summary={'status':'PASS', 'validation_rows':len(rows),'query_images':len(queries),
             'gallery_images':len(gallery),'query_ids':len({q['identity_key'] for q in queries}),
             'gallery_ids':len(byid), 'zero_valid_positive_queries':0,
             'min_valid_positives':min(positives),'max_valid_positives':max(positives)}
    if kind=='deepchange':summary['queries_with_cross_camera_positive']=cross_camera
    if kind=='pose':summary['queries_with_cross_video_positive']=cross_video
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,default=Path('phase3'))
    p.add_argument('--retrieval-dir',type=Path,default=Path('phase3/retrieval'))
    a=p.parse_args()
    results={}
    cache={}
    print('P3 FINAL RETRIEVAL VERIFICATION',flush=True)
    for protocol,(source,kind) in CONFIG.items():
        if source not in cache:cache[source]=read(a.input_dir/source)
        print(f'Checking {protocol}...',flush=True)
        results[protocol]=check(protocol,*cache[source],a.retrieval_dir/f'{protocol}.csv',kind)
        print('  PASS:',results[protocol],flush=True)
    a.retrieval_dir.mkdir(parents=True, exist_ok=True)
    report={'overall':'PASS', 'scope':'Metadata and internal retrieval protocol only',
            'not_checked':['Raw image pixels and physical file presence',
                           'Official evaluation image/content overlap',
                           'Cross-dataset acquisition leakage'], 'protocols':results,
            'metric_rules':{
                'deepchange':'Positive iff same identity and different tracklet. Same identity and same tracklet gallery items are junk; filter them before ranking.',
                'posetrack':'Positive iff same identity and other image; same-video allowed. Optional cross-video reporting on eligible queries only.',
                'cuhk03':'Positive iff same identity and opposite pair-local view; rank directions separately.'}}
    out=a.retrieval_dir/'p3_final_verification_report.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'\nOVERALL: PASS (internal metadata and retrieval semantics only)\nReport: {out}',flush=True)

if __name__=='__main__':
    try:main()
    except (ValueError, OSError) as e:
        print(f'FAIL: {e}',flush=True)
        raise SystemExit(1)
