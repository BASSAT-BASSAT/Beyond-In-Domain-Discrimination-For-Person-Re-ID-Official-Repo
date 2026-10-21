#!/usr/bin/env python3
"""P3 source-only identity-disjoint split planner. Default = report only.

Usage:
 python p3_split_sources.py --input-dir /path/to/csvs --out-dir /path/to/phase3
 python p3_split_sources.py --input-dir /path/to/csvs --out-dir /path/to/phase3 --apply

No original input files are modified. This does NOT check official evaluation images or P4 leakage.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

FILES = {
    'DeepChange': 'deepchange_unified_v1.csv',
    'Occluded-PoseTrack-ReID': 'occluded_posetrack_unified_v1.csv',
    'CUHK03_NP': 'cuhk03_np_unified_v1.csv',
}
REQUIRED = {'sample_uid', 'dataset', 'candidate_pool', 'relative_path',
            'source_split', 'raw_pid', 'identity_namespace', 'identity_key',
            'raw_camera_or_view', 'camera_semantics', 'video_id', 'tracklet_id',
            'source_metadata_json'}

def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_summary(path, expected):
    ids = Counter()
    uid, paths = set(), set()
    features = defaultdict(lambda: {'view': set(), 'video': set(), 'camera': set()})
    tracks = {}
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f'{path}: missing required columns {REQUIRED - set(reader.fieldnames or [])}')
        fields = reader.fieldnames
        for line, r in enumerate(reader, start=2):
            dataset = r['dataset']
            if dataset != expected or r['source_split'] != 'train':
                raise ValueError(f'{path}:{line}: unexpected dataset or source split')
            k = r['identity_key']
            if not k or k != r['identity_namespace'] + '::' + r['raw_pid']:
                raise ValueError(f'{path}:{line}: invalid identity key')
            if not r['sample_uid'] or r['sample_uid'] in uid:
                raise ValueError(f'{path}:{line}: empty/duplicate sample_uid')
            if not r['relative_path'] or r['relative_path'] in paths:
                raise ValueError(f'{path}:{line}: empty/duplicate path')
            if Path(r['relative_path']).is_absolute() or '..' in Path(r['relative_path']).parts:
                raise ValueError(f'{path}:{line}: unsafe relative_path')
            try:
                json.loads(r['source_metadata_json'])
            except ValueError as e:
                raise ValueError(f'{path}:{line}: invalid metadata JSON') from e
            uid.add(r['sample_uid']); paths.add(r['relative_path']); ids[k] += 1
            if r['raw_camera_or_view']:
                features[k]['view'].add(r['raw_camera_or_view'])
                features[k]['camera'].add(r['camera_context_id'])
            if r['video_id']:
                features[k]['video'].add(r['video_id'])
            if expected == 'DeepChange' and r['tracklet_id']:
                tr = r['tracklet_id']
                old = tracks.setdefault(tr, (k, r['raw_camera_or_view']))
                if old != (k, r['raw_camera_or_view']):
                    raise ValueError(f'{path}:{line}: tracklet crosses identity/camera')
            if expected == 'Occluded-PoseTrack-ReID' and r['raw_camera_or_view'] != r['video_id']:
                raise ValueError(f'{path}:{line}: logical video proxy mismatch')
            if expected == 'CUHK03_NP' and ('/detected/' not in '/' + r['relative_path']):
                raise ValueError(f'{path}:{line}: not detected variant')
    return fields, ids, features, len(uid)

def make_roles(ids, seed, fraction):
    keys = sorted(ids, key=lambda k: hashlib.sha256(f'{seed}|{k}'.encode()).digest())
    n_val = round(len(keys) * fraction)
    val = set(keys[:n_val])
    return {k: ('validation' if k in val else 'train') for k in keys}

def write_csv(path, out, fields, roles):
    with path.open(newline='', encoding='utf-8-sig') as inp, out.open('w', newline='', encoding='utf-8') as dest:
        rd = csv.DictReader(inp)
        wr = csv.DictWriter(dest, fieldnames=fields + ['p3_role'])
        wr.writeheader()
        for r in rd:
            r['p3_role'] = roles[r['identity_key']]
            wr.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-fraction', type=float, default=0.20)
    ap.add_argument('--apply', action='store_true', help='Write provisional split manifests; default reports only')
    args = ap.parse_args()
    if not 0 < args.val_fraction < 1:
        ap.error('val-fraction must be between 0 and 1')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {'status': 'PROVISIONAL — DeepChange exclusion provenance and official evaluation checks pending',
              'seed': args.seed, 'val_fraction': args.val_fraction,
              'note': 'P3 is metadata-only; P4 cross-dataset leakage and official evaluation checks still required',
              'datasets': {}}
    for name, filename in FILES.items():
        path = args.input_dir / filename
        fields, ids, features, nrows = load_summary(path, name)
        roles = make_roles(ids, args.seed, args.val_fraction)
        train = {k for k, v in roles.items() if v == 'train'}
        val = {k for k, v in roles.items() if v == 'validation'}
        assert not train & val
        assert len(train) + len(val) == len(ids)
        images = {role: sum(ids[k] for k in ids if roles[k] == role) for role in ('train', 'validation')}
        assert sum(images.values()) == nrows
        views = {role: sorted(set().union(*(features[k]['view'] for k in ids if roles[k] == role))) for role in ('train','validation')}
        videos = {role: len(set().union(*(features[k]['video'] for k in ids if roles[k] == role))) for role in ('train','validation')}
        entry = {'input': str(path), 'input_sha256': sha256_file(path), 'rows': nrows,
                 'identities': len(ids), 'train_identities': len(train), 'validation_identities': len(val),
                 'train_images': images['train'], 'validation_images': images['validation'],
                 'observed_views_or_camera_codes': views, 'observed_video_counts': videos,
                 'identity_overlap': 0, 'row_conservation': True}
        if name == 'CUHK03_NP':
            entry['all_identities_have_both_views'] = all(features[k]['view'] == {'1','2'} for k in ids)
            if not entry['all_identities_have_both_views']:
                raise ValueError('CUHK03 identity without both views')
        if args.apply:
            out = args.out_dir / filename.replace('_unified_v1.csv', '_p3_provisional.csv')
            write_csv(path, out, fields, roles)
            entry['output'] = str(out)
            entry['output_sha256'] = sha256_file(out)
        report['datasets'][name] = entry
        print(f'{name}: {nrows:,} images, {len(train)} train IDs / {len(val)} val IDs, '
              f'{images["train"]:,} train images / {images["validation"]:,} val images')
    out_report = args.out_dir / 'p3_split_preview.json'
    out_report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Report: {out_report}\nMode: {"PROVISIONAL WRITES" if args.apply else "PREVIEW ONLY"}')

if __name__ == '__main__':
    main()
