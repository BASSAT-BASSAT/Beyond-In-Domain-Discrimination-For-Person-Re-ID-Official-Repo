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
import random
from collections import Counter, defaultdict
from pathlib import Path

FILES = {
    'DeepChange': 'deepchange_unified_v1.csv',
    'Occluded-PoseTrack-ReID': 'occluded_posetrack_unified_v1.csv',
    'CUHK03_NP': 'cuhk03_np_unified_v1.csv',
    'ORBench_RGB': 'orbench_rgb_unified_v1.csv',
    'PRID2011': 'prid2011_unified_v1.csv',
    'RPIfield': 'rpifield_unified_v1.csv',
}
# Rows eligible for identity-disjoint train/val (per dataset audit convention).
SPLITTABLE_SOURCE_SPLIT = {
    'CUHK03_NP': frozenset({'train'}),
    'DeepChange': frozenset({'train'}),
    'Occluded-PoseTrack-ReID': frozenset({'train'}),
    'ORBench_RGB': frozenset({'train'}),
    'PRID2011': frozenset({'unassigned'}),
    'RPIfield': frozenset({'unassigned'}),
}
# ``provisional``: identity-disjoint val_fraction (Member 1 default).
# ``official_member3``: ORBench all train; RPIfield camera-group holdout; PRID identity-disjoint.
SPLIT_POLICY = {
    'CUHK03_NP': 'identity_disjoint',
    'DeepChange': 'identity_disjoint',
    'Occluded-PoseTrack-ReID': 'identity_disjoint',
    'ORBench_RGB': 'identity_disjoint',
    'PRID2011': 'identity_disjoint',
    'RPIfield': 'camera_scene_disjoint',
}
SPLIT_POLICY_PROVISIONAL = {k: 'identity_disjoint' for k in FILES}
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
    allowed_splits = SPLITTABLE_SOURCE_SPLIT.get(expected, frozenset({'train'}))
    ids = Counter()
    id_cam_counts: dict[str, Counter] = defaultdict(Counter)
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
            if dataset != expected or r['source_split'] not in allowed_splits:
                raise ValueError(
                    f'{path}:{line}: unexpected dataset or source_split '
                    f'(got {r["source_split"]!r}, allowed {sorted(allowed_splits)})'
                )
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
            cam = (r.get('camera_context_id') or r.get('raw_camera_or_view') or '0').strip() or '0'
            id_cam_counts[k][cam] += 1
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
    id_to_cam = {k: counts.most_common(1)[0][0] for k, counts in id_cam_counts.items()}
    return fields, ids, features, len(uid), id_to_cam

def assign_pid_groups(groups: dict[str, list[str]], *, train_fraction: float, seed: int):
    """Assign whole camera/scene groups to train/val (~train_fraction identities in train)."""
    all_pids = sorted({p for ps in groups.values() for p in ps})
    if len(all_pids) <= 1:
        return all_pids, [], {'strategy': 'camera_scene', 'n_groups': len(groups), 'skipped': 'too_few_ids'}

    rng = random.Random(seed)
    total = len(all_pids)
    target_val = max(1, int(round((1.0 - train_fraction) * total)))
    target_val = min(target_val, total - 1)

    keys = list(groups.keys())
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]))

    val_set: set[str] = set()
    for key in keys:
        if len(val_set) >= target_val:
            break
        val_set |= set(groups[key])
    train_set = set(all_pids) - val_set
    if not train_set and len(keys) > 1:
        largest = max(keys, key=lambda k: len(groups[k]) if set(groups[k]) <= val_set else 0)
        val_set -= set(groups[largest])
        train_set = set(all_pids) - val_set
    train = sorted(train_set)
    val = sorted(val_set)
    return train, val, {
        'strategy': 'camera_scene',
        'n_groups': len(groups),
        'n_train_ids': len(train),
        'n_val_ids': len(val),
        'train_fraction_target': train_fraction,
        'train_fraction_actual': len(train) / total if total else None,
    }

def make_roles_camera_scene(ids, id_to_cam: dict[str, str], seed: int, train_fraction: float = 0.8):
    by_cam: dict[str, list[str]] = defaultdict(list)
    for k in ids:
        by_cam[id_to_cam.get(k, 'unknown')].append(k)
    groups = {f'cam:{c}': sorted(v) for c, v in by_cam.items() if v}
    train, val, meta = assign_pid_groups(groups, train_fraction=train_fraction, seed=seed)
    roles = {k: 'train' for k in train}
    roles.update({k: 'validation' for k in val})
    return roles, meta

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
    ap.add_argument(
        '--policy-mode',
        choices=('provisional', 'official_member3'),
        default='provisional',
        help='provisional = identity 80/20 for all; official_member3 = ORBench+PRID identity 80/20, RPIfield camera holdout',
    )
    ap.add_argument(
        '--train-fraction',
        type=float,
        default=0.8,
        help='Target train identity fraction for camera_scene_disjoint (RPIfield)',
    )
    args = ap.parse_args()
    if not 0 < args.val_fraction < 1:
        ap.error('val-fraction must be between 0 and 1')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policies = SPLIT_POLICY if args.policy_mode == 'official_member3' else SPLIT_POLICY_PROVISIONAL
    report = {'status': 'PROVISIONAL — DeepChange exclusion provenance and official evaluation checks pending',
              'seed': args.seed, 'val_fraction': args.val_fraction,
              'policy_mode': args.policy_mode,
              'train_fraction_camera': args.train_fraction,
              'note': 'P3 is metadata-only; P4 cross-dataset leakage and official evaluation checks still required',
              'datasets': {}}
    for name, filename in FILES.items():
        path = args.input_dir / filename
        if not path.is_file():
            continue
        fields, ids, features, nrows, id_to_cam = load_summary(path, name)
        policy = policies.get(name, 'identity_disjoint')
        split_meta = {}
        if policy == 'official_train_only':
            roles = {k: 'train' for k in ids}
            split_meta = {
                'strategy': 'official_train_only',
                'note': (
                    'All unified train candidates are p3_role=train. Official in-training val uses '
                    'test_gallery_and_queries.json (reidfm pool val_sets / ORBenchM query+gallery), '
                    'not a random holdout from train IDs.'
                ),
            }
        elif policy == 'camera_scene_disjoint':
            roles, split_meta = make_roles_camera_scene(
                ids, id_to_cam, args.seed, train_fraction=args.train_fraction,
            )
        else:
            roles = make_roles(ids, args.seed, args.val_fraction)
            split_meta = {'strategy': 'identity_disjoint', 'val_fraction': args.val_fraction}
        train = {k for k, v in roles.items() if v == 'train'}
        val = {k for k, v in roles.items() if v == 'validation'}
        assert not train & val
        assert len(train) + len(val) == len(ids)
        images = {role: sum(ids[k] for k in ids if roles[k] == role) for role in ('train', 'validation')}
        assert sum(images.values()) == nrows
        views = {role: sorted(set().union(*(features[k]['view'] for k in ids if roles[k] == role))) for role in ('train','validation')}
        videos = {role: len(set().union(*(features[k]['video'] for k in ids if roles[k] == role))) for role in ('train','validation')}
        entry = {'input': str(path), 'input_sha256': sha256_file(path), 'rows': nrows,
                 'split_policy': policy, 'split_meta': split_meta,
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
        print(f'{name} [{policy}]: {nrows:,} images, {len(train)} train IDs / {len(val)} val IDs, '
              f'{images["train"]:,} train images / {images["validation"]:,} val images')
    out_report = args.out_dir / 'p3_split_preview.json'
    out_report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Report: {out_report}\nMode: {"PROVISIONAL WRITES" if args.apply else "PREVIEW ONLY"}')

if __name__ == '__main__':
    main()
