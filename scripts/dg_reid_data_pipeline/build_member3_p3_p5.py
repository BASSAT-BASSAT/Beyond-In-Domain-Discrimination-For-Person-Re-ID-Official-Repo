#!/usr/bin/env python3
"""Member 3 provisional P3/P5 handoff: exactly six outputs, one command.

Inputs under the project repository:
  unified/iust_unified_v1.csv
  unified/last_unified_v1.csv
  unified/lpw_unified_v1.csv
  manifests/LaST/last_manifest.csv   (official LaST validation from audited manifest)
  unified/final_dataset/Member {1,2}/identity_class_mapping.json

P3: identity-disjoint ~80/20 internal validation for IUST and LPW;
    preserve LaST's official train and val/query + val/gallery incl. distractors.
P5: exactly six *provisional* handoff files under unified/final_dataset/Member 3.

Run from any directory: python scripts/dg_reid_data_pipeline/build_member3_p3_p5.py
Stdlib-only, Python 3.10+. Never edits the four input CSVs or source images.
Does NOT certify P4. Input full image bytes are not checked here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

UNIFIED_FIELDS = [
    'sample_uid','dataset','candidate_pool','dataset_root_id','relative_path',
    'source_split','raw_pid','identity_namespace','identity_key',
    'raw_camera_or_view','camera_semantics','camera_context_id','video_id',
    'tracklet_id','frame_id','clothes_id','source_row_number',
    'source_image_index','source_metadata_json',
]
TRAIN_FIELDS = [
    'sample_uid','dataset','dataset_root_id','relative_path','identity_key',
    'class_id','raw_camera_or_view','camera_semantics','camera_context_id',
    'video_id','tracklet_id','frame_id','clothes_id','candidate_pool',
    'source_row_number',
]
VAL_FIELDS = [
    'sample_uid','dataset','dataset_root_id','relative_path','identity_key',
    'validation_protocol_id','retrieval_role','raw_camera_or_view',
    'camera_semantics','camera_context_id','video_id','tracklet_id',
    'frame_id','clothes_id','candidate_pool','source_row_number',
]
OUTPUTS = (
    'train_manifest.csv','validation_manifest.csv',
    'identity_class_mapping.json','source_registry.json',
    'dataset_protocol.json','checksums.sha256',
)
SOURCES = {
    'iust': dict(file='iust_unified_v1.csv', name='IUST_PersonReID', rows=72393, ids=1193,
                 camera_semantics='physical_camera_code',
                 sha='ae577916bf2fd79cdb275b6c36bf2b223243ae4228af16499b7f28a2008a3503',
                 env='DGREID_ROOT_IUST', protocol='iust_internal_cross_camera_identity_val_v1'),
    'last': dict(file='last_unified_v1.csv', name='LaST', rows=71248, ids=5000,
                 camera_semantics='not_available',
                 sha='10f69e3cc005db46f111f49ae9f1872b01b0cc74713c744e61c08dc545f1d256',
                 env='DGREID_ROOT_LAST', protocol='last_official_val_query_gallery_v1'),
    'lpw': dict(file='lpw_unified_v1.csv', name='LPW', rows=418447, ids=1975,
                camera_semantics='scene_local_view',
                sha='0f9c8d5808b3b935750dae0136ac88b4670e514c323d75cd28f0cd78969a0354',
                env='DGREID_ROOT_LPW', protocol='lpw_internal_cross_view_identity_val_v1'),
}
LAST_EXPECTED = dict(query=100, gallery=21279, distractors=20000, val_ids=56,
                     labeled_gallery=1279, train=71248)

class AuditError(RuntimeError):
    pass

def check(condition, message):
    if not condition:
        raise AuditError(message)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(2 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def uid(dataset: str, relative_path: str) -> str:
    return hashlib.sha256((dataset+'\x00'+relative_path).encode('utf-8')).hexdigest()

def json_write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')

def csv_iter(path, *, unified=False):
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        check(reader.fieldnames is not None, f'{path}: no CSV header')
        if unified:
            check(reader.fieldnames == UNIFIED_FIELDS,
                  f'{path.name}: frozen Unified v1 header/schema mismatch:\n{reader.fieldnames}')
        for lineno, row in enumerate(reader, start=2):
            check(None not in row and all(x is not None for x in row.values()),
                  f'{path.name}:{lineno}: malformed CSV row')
            yield lineno, row

def checked_relative(path, origin):
    value = str(path).strip().replace('\\','/')
    parts = PurePosixPath(value).parts
    check(value and not value.startswith('/') and not re.match(r'^[a-zA-Z]:', value)
          and ':' not in value and '..' not in parts,
          f'{origin}: unsafe or absolute relative path: {value!r}')
    return value

def check_unified(row, spec, lineno):
    ds, name = spec['name'], spec['file']
    check(row['dataset'] == ds, f'{name}:{lineno}: dataset {row["dataset"]!r} != {ds!r}')
    check(row['source_split'] == 'train',
          f'{name}:{lineno}: source_split={row["source_split"]!r}, expected train only')
    check(row['camera_semantics'] == spec['camera_semantics'],
          f'{name}:{lineno}: camera semantics mismatch')
    check(row['identity_key'] == row['identity_namespace']+'::'+row['raw_pid']
          and row['raw_pid'] and row['dataset_root_id'],
          f'{name}:{lineno}: invalid identity namespace, PID or root ID')
    rel = checked_relative(row['relative_path'], f'{name}:{lineno}')
    check(row['sample_uid'] == uid(ds, rel), f'{name}:{lineno}: wrong sample_uid')
    if ds != 'LaST':
        check(bool(row['camera_context_id']),
              f'{name}:{lineno}: missing camera/view context required for internal retrieval')

def read_prior_mappings(repo):
    seen_keys, classes, provenance = set(), set(), []
    for member in ('Member 1','Member 2'):
        path = repo/'unified'/'final_dataset'/member/'identity_class_mapping.json'
        check(path.is_file(), f'Missing {path}; cannot invent global classifier offset')
        doc = json.loads(path.read_text(encoding='utf-8'))
        mapping = doc.get('identity_to_class', doc)
        check(isinstance(mapping, dict) and mapping, f'{path}: missing identity_to_class mapping')
        ids = [int(x) for x in mapping.values()]
        check(len(ids) == len(set(ids)) and not (classes & set(ids)),
              f'{path}: overlapping/duplicate class IDs')
        check(not (seen_keys & set(mapping)), f'{path}: duplicate identity keys across members')
        classes.update(ids); seen_keys.update(mapping)
        provenance.append(dict(member=member, sha256=sha256_file(path), classes=len(ids)))
    check(classes == set(range(len(classes))),
          'Member 1 + Member 2 class IDs must be unique/contiguous starting at 0')
    return len(classes), seen_keys, provenance

def read_train_sources(repo, allow_unpinned):
    stats, identities, all_keys, input_digests = {}, {}, set(), {}
    for short, spec in SOURCES.items():
        path = repo/'unified'/spec['file']
        check(path.is_file(), f'Input not found: {path}')
        digest = sha256_file(path)
        if not allow_unpinned:
            check(digest == spec['sha'],
                  f'{path.name}: frozen P2 SHA mismatch. Actual={digest}, expected={spec["sha"]}. '
                  'Do not silently use changed inputs.')
        print(f'[SCAN] {spec["name"]}: {path.name}', flush=True)
        per_id, root_ids, seen_uids, pids = defaultdict(Counter), set(), set(), set()
        count = 0
        for line, row in csv_iter(path, unified=True):
            check_unified(row, spec, line)
            check(row['sample_uid'] not in seen_uids, f'{spec["file"]}:{line}: repeated sample_uid')
            seen_uids.add(row['sample_uid'])
            count += 1
            root_ids.add(row['dataset_root_id'])
            pids.add(row['raw_pid'])
            per_id[row['identity_key']][row['camera_context_id']] += 1
        check(count == spec['rows'] and len(per_id) == spec['ids'],
              f'{spec["file"]}: got {count:,} rows/{len(per_id):,} IDs; '
              f'expected {spec["rows"]:,}/{spec["ids"]:,}')
        check(len(root_ids) == 1, f'{spec["file"]}: multiple dataset_root_id values: {root_ids}')
        check(not (set(per_id) & all_keys), 'Duplicate identity keys between candidate sources')
        all_keys.update(per_id)
        stats[short] = dict(dataset=spec['name'], source_rows=count, candidate_ids=len(per_id),
                            root_id=next(iter(root_ids)), raw_pids=pids)
        identities[short] = per_id
        input_digests[spec['file']] = digest
        print(f'   {count:,} images, {len(per_id):,} IDs, root={stats[short]["root_id"]}',flush=True)
    return stats, identities, input_digests, all_keys

def choose_internal_val(per_id, seed, fraction, dataset):
    ids = sorted(per_id)
    eligible = [k for k in ids if len(per_id[k]) >= 2]
    desired = max(1,round(len(ids)*fraction))
    check(len(eligible) >= desired,
          f'{dataset}: only {len(eligible)} multi-context IDs for desired {desired} val IDs')
    scores = lambda k: (hashlib.sha256(f'{seed}\x00{k}'.encode()).digest(), k)
    selected = set(sorted(eligible,key=scores)[:desired])
    return selected

def choose_query_uids(path, selected, per_id):
    preferred = {k:min((num, cam) for cam, num in per_id[k].items())[1] for k in selected}
    candidates={}
    for _, row in csv_iter(path, unified=True):
        key=row['identity_key']
        if key in selected and row['camera_context_id']==preferred[key]:
            if key not in candidates or row['sample_uid'] < candidates[key]:
                candidates[key]=row['sample_uid']
    check(len(candidates)==len(selected), f'{path.name}: missing chosen query rows')
    return set(candidates.values())

def first_present(row, candidates):
    for name in candidates:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    return ''

PATH_COLUMNS = ('relative_path','image_path','img_path','path','file_path','filepath',
                'image_relpath','relative_image_path','image_file','file_name','filename','image')
SPLIT_COLUMNS = ('split','source_split','dataset_split','set','partition','subset')
ROLE_COLUMNS = ('retrieval_role','role','query_gallery','query_or_gallery','query_gallery_role',
                'set_type','image_type','subset','split')
PID_COLUMNS = ('folder_pid','folder_person_id','folder_id','person_id','identity_id',
               'raw_pid','pid','identity','label')


def canonical_last_val_path(original, split_value='', role_value='', folder=''):
    """Derive a portable official val/query|gallery path, never invent a role.
    Supports manifest paths: val/query/ID/name.jpg, last/val/..., absolute
    original paths containing /val/query|gallery/..., or filename + explicit
    split/role/folder fields (only when unambiguous).
    """
    raw = original.strip().replace('\\','/')
    pieces=[x for x in raw.split('/') if x and x!='.']
    lower=[x.lower() for x in pieces]
    hits=[i for i,x in enumerate(lower) if x in ('val','validation')]
    if hits:
        idx = hits[-1]
        check(idx+1<len(pieces) and lower[idx+1] in ('query','gallery'),
              f'LaST ambiguous validation path {original!r}; expected val/query or val/gallery')
        role=lower[idx+1]
        relative='val/'+'/'.join(pieces[idx+1:])
        if role_value:
            check(role_value.lower() in (role, 'val_'+role, 'validation_'+role),
                  f'LaST conflict between role field={role_value!r} and path={original!r}')
        return role, checked_relative(relative, 'LaST official validation')
    # Non-val rows may be train/test; caller skips them before demanding a path.
    check(split_value.lower() in ('val','validation') and role_value.lower() in ('query','gallery'),
          f'LaST row has no identifiable official val/query|gallery in path: {original!r}')
    if len(pieces)>1 and lower[0] in ('query','gallery'):
        check(lower[0]==role_value.lower(),'LaST role/path disagreement')
        relative='val/'+'/'.join(pieces)
    elif len(pieces)>1 and pieces[-2].isdigit():
        relative='val/'+role_value.lower()+'/'+('/'.join(pieces[-2:]))
    else:
        check(folder and folder.isdigit(),
              f'LaST filename-only path needs explicit folder_pid: {original!r}')
        relative='val/'+role_value.lower()+'/'+folder+'/'+pieces[-1]
    return role_value.lower(), checked_relative(relative, 'LaST official validation')


def iter_last_official_val(last_path, last_root_id):
    """Read LaST original audit manifest, yield normalized official val rows.
    If column names/layout are incompatible, fail with a useful error instead of
    manufacturing labels or paths. Test rows never enter output.
    """
    with last_path.open('r',newline='',encoding='utf-8-sig') as stream:
        rdr=csv.DictReader(stream)
        check(rdr.fieldnames is not None, f'{last_path}: empty/invalid CSV')
        fields=set(rdr.fieldnames)
        check(fields.intersection(PATH_COLUMNS),
              f'LaST manifest has no recognized path column; found columns={rdr.fieldnames}. '
              'Need actual LaST manifest schema before generating official val.')
        check(fields.intersection(SPLIT_COLUMNS) or fields.intersection(ROLE_COLUMNS),
              f'LaST manifest needs a split/role field or val paths; columns={rdr.fieldnames}')
        for line,row in enumerate(rdr,2):
            check(None not in row and all(v is not None for v in row.values()),
                  f'{last_path.name}:{line}: malformed row')
            original=first_present(row,PATH_COLUMNS)
            split=first_present(row,SPLIT_COLUMNS).lower().strip()
            role=first_present(row,ROLE_COLUMNS).lower().strip()
            if role in ('train','test','val','validation'):
                role=''  # split field alone is not a retrieval role
            norm='/'+original.replace('\\','/').lower()+'/'
            has_val_path= '/val/' in norm or '/validation/' in norm
            # Some manifests encode a single split field as 'val_query'.
            if split in ('val_query','validation_query'):
                split,role='val','query'
            elif split in ('val_gallery','validation_gallery'):
                split,role='val','gallery'
            if not (split in ('val','validation') or has_val_path):
                continue
            check(original, f'{last_path.name}:{line}: missing path')
            folder=first_present(row,('folder_pid','folder_person_id','folder_id'))
            role2,rel=canonical_last_val_path(original,split,role,folder)
            filename=PurePosixPath(rel).name
            stem=Path(filename).stem
            # LaST native filename: PID_IMAGEID_VIDEOID_FRAMEID_BBOXID_CLOTHESID.
            tokens=[token for token in stem.split('_') if token]
            check(len(tokens)==6 and all(t.isdigit() for t in tokens),
                  f'{last_path.name}:{line}: cannot parse LaST filename {filename!r}')
            name_pid,image_index,video_id,frame_id,bbox_id,clothes_id=tokens
            path_parts=PurePosixPath(rel).parts
            inferred_folder=path_parts[2] if len(path_parts)>=4 and path_parts[2].isdigit() else ''
            if folder and inferred_folder:
                check(int(folder)==int(inferred_folder),
                      f'{last_path.name}:{line}: folder PID metadata != relative path')
            pid=folder or inferred_folder or first_present(row,PID_COLUMNS) or name_pid
            check(pid.isdigit(),f'{last_path.name}:{line}: invalid PID {pid!r}')
            if int(pid)!=0:
                check(int(name_pid)==int(pid),
                      f'{last_path.name}:{line}: official val folder/filename PID mismatch {pid}/{name_pid}')
            else:
                check(role2=='gallery',f'{last_path.name}:{line}: distractor cannot be query')
            # Manifest columns, if present, must not disagree with the verified filename.
            for candidates, expected, what in (
                (('video_id',),video_id,'video_id'),
                (('frame_id',),frame_id,'frame_id'),
                (('clothes_id',),clothes_id,'clothes_id'),
            ):
                raw=first_present(row,candidates)
                if raw:
                    check(raw.isdigit() and int(raw)==int(expected),
                          f'{last_path.name}:{line}: {what} metadata conflicts with filename')
            sample_uid=uid('LaST',rel)
            is_distractor=int(pid)==0
            # Per-image distinct keys avoid incorrectly treating 20k unknown distractors
            # as a single ground-truth person. Not classifier/train IDs.
            identity_key=(f'LaST::official_val::distractor::{sample_uid}' if is_distractor
                          else f'LaST::official_val::{int(pid)}')
            yield {
                'sample_uid':sample_uid,'dataset':'LaST','dataset_root_id':last_root_id,
                'relative_path':rel,'identity_key':identity_key,
                'validation_protocol_id':'last_official_val_query_gallery_v1',
                'retrieval_role':role2,
                'raw_camera_or_view':'','camera_semantics':'not_available',
                'camera_context_id':'','video_id':video_id,'tracklet_id':'',
                'frame_id':frame_id,'clothes_id':clothes_id,
                'candidate_pool':'last_official_validation_from_audited_manifest',
                'source_row_number':str(line),
                '_raw_pid':str(int(pid)),'_distractor':is_distractor,
            }

def check_destination(path):
    if path.exists():
        check(path.is_dir(), f'Output destination exists but is not a directory: {path}')
        existing={p.name for p in path.iterdir()}
        check(existing.issubset({'.gitkeep'}),
              f'Output directory already contains files {sorted(existing)}. '
              'Refusing to overwrite or mix versions.')

def build(repo, seed, val_fraction, allow_unpinned):
    check(0<val_fraction<1,'--val-fraction must be between 0 and 1')
    out=repo/'unified'/'final_dataset'/'Member 3'
    check_destination(out)
    base,prior_keys,prior_provenance=read_prior_mappings(repo)
    stats, per_id, digests, source_keys=read_train_sources(repo,allow_unpinned)
    check(not prior_keys.intersection(source_keys),
          'Member 3 identity key overlaps existing Member 1/2 classifier keys')
    last_manifest=repo/'manifests'/'LaST'/'last_manifest.csv'
    check(last_manifest.is_file(),f'Missing LaST audit manifest: {last_manifest}')
    digests['manifests/LaST/last_manifest.csv']=sha256_file(last_manifest)
    selections={
        'iust':choose_internal_val(per_id['iust'],seed,val_fraction,'IUST'),
        'last':set(),
        'lpw':choose_internal_val(per_id['lpw'],seed,val_fraction,'LPW'),
    }
    query_uids={
        short:choose_query_uids(repo/'unified'/SOURCES[short]['file'],selections[short],per_id[short])
        for short in ('iust','lpw')
    }
    print('[P3] Internal validation IDs:', {s:len(selections[s]) for s in ('iust','lpw')},flush=True)
    class_keys=sorted(set().union(*(set(per_id[s])-selections[s] for s in SOURCES)))
    mapping={k:base+i for i,k in enumerate(class_keys)}
    print(f'[P5] Member 3: {len(mapping):,} classes, range {base}..{base+len(mapping)-1}',flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.member3_stage_',dir=out.parent))
    try:
        tc,vc,qc,gc=Counter(),Counter(),Counter(),Counter()
        query_ids=set(); gallery_ids=set(); distractors=0; val_uids=set()
        last_raw_val_ids=set()
        with (stage/'train_manifest.csv').open('w',newline='',encoding='utf-8') as trainfile, \
             (stage/'validation_manifest.csv').open('w',newline='',encoding='utf-8') as valfile:
            trainw=csv.DictWriter(trainfile,fieldnames=TRAIN_FIELDS,lineterminator='\n')
            valw=csv.DictWriter(valfile,fieldnames=VAL_FIELDS,lineterminator='\n')
            trainw.writeheader();valw.writeheader()
            for short,spec in SOURCES.items():
                for line,row in csv_iter(repo/'unified'/spec['file'],unified=True):
                    check_unified(row,spec,line)
                    ds,key=spec['name'],row['identity_key']
                    if key in selections[short]:
                        r={col:row.get(col,'') for col in VAL_FIELDS}
                        r['validation_protocol_id']=spec['protocol']
                        r['retrieval_role']='query' if row['sample_uid'] in query_uids[short] else 'gallery'
                        valw.writerow(r)
                        vc[ds]+=1
                        if r['retrieval_role']=='query': qc[ds]+=1
                        else:gc[ds]+=1
                    else:
                        r={col:row.get(col,'') for col in TRAIN_FIELDS}
                        r['class_id']=mapping[key]
                        trainw.writerow(r);tc[ds]+=1
            print('[LaST] Reading official validation from last_manifest.csv...',flush=True)
            for row in iter_last_official_val(last_manifest,stats['last']['root_id']):
                check(row['sample_uid'] not in val_uids,
                      f'LaST duplicate official val path/UID {row["relative_path"]}')
                val_uids.add(row['sample_uid'])
                if row['_distractor']:
                    distractors+=1
                else:
                    last_raw_val_ids.add(int(row['_raw_pid']))
                    (query_ids if row['retrieval_role']=='query' else gallery_ids).add(row['identity_key'])
                valw.writerow({col:row[col] for col in VAL_FIELDS})
                vc['LaST']+=1
                if row['retrieval_role']=='query':qc['LaST']+=1
                else:gc['LaST']+=1
        for short,spec in SOURCES.items():
            ds=spec['name']
            if short!='last':
                check(tc[ds]+vc[ds]==spec['rows'],f'{ds}: train/val row loss')
                check(qc[ds]==len(selections[short]),f'{ds}: one query/held-out ID contract failed')
                check(gc[ds]>0 and all(len(per_id[short][k])>=2 for k in selections[short]),
                      f'{ds}: invalid query/gallery')
            else:
                check(tc[ds]==LAST_EXPECTED['train'],'LaST train count mismatch')
                check(qc[ds]==LAST_EXPECTED['query'] and gc[ds]==LAST_EXPECTED['gallery'],
                      f'LaST official val count mismatch: query={qc[ds]},gallery={gc[ds]} '
                      f'(expected {LAST_EXPECTED["query"]}/{LAST_EXPECTED["gallery"]})')
                check(distractors==LAST_EXPECTED['distractors'],
                      f'LaST expected {LAST_EXPECTED["distractors"]} gallery distractors, got {distractors}')
                check(len(query_ids)==LAST_EXPECTED['val_ids'] and query_ids<=gallery_ids,
                      f'LaST query identities missing in labeled gallery: query={len(query_ids)}, '
                      f'gallery={len(gallery_ids)},missing={len(query_ids-gallery_ids)}')
                check(gc[ds]-distractors==LAST_EXPECTED['labeled_gallery'],
                      'LaST labeled gallery count mismatch')
                # Verify native PID separation, not just split-scoped identity_key.
                source_pids={int(pid) for pid in stats['last']['raw_pids'] if pid.isdigit()}
                check(not (source_pids & last_raw_val_ids),
                      f'LaST native PID overlap between train and official val: {source_pids&last_raw_val_ids}')
        summary={}
        registry=[]
        for short,spec in SOURCES.items():
            ds=spec['name']
            registry.append(dict(dataset=ds,dataset_root_id=stats[short]['root_id'],
                    relative_base_path='',root_env_var=spec['env'],
                    train_rows=tc[ds],validation_unique_rows=vc[ds]))
            summary[ds]=dict(source_candidate_images=spec['rows'],source_candidate_ids=spec['ids'],
                    train_images=tc[ds],validation_images=vc[ds],
                    train_ids=len(per_id[short])-len(selections[short]),
                    val_ids=(LAST_EXPECTED['val_ids'] if short=='last' else len(selections[short])),
                    query_images=qc[ds],gallery_images=gc[ds],
                    p3_policy=('official_train_and_official_val_query_gallery' if short=='last'
                               else 'deterministic_identity_disjoint_internal_val'),
                    p3_validation_protocol_id=spec['protocol'],
                    camera_semantics=spec['camera_semantics'])
        summary['LaST'].update(distractor_gallery_images=distractors,
                                labeled_gallery_images=gc['LaST']-distractors,
                                official_val_identity_match_checked=True)
        json_write(stage/'identity_class_mapping.json',dict(
            schema_version='v1',scope='Member 3 provisional',
            continues_after_member='Member 2',class_id_offset=base,
            num_training_identities=len(mapping),
            class_id_range_inclusive=[base,base+len(mapping)-1],
            identity_to_class=mapping))
        json_write(stage/'source_registry.json',dict(schema_version='v1',sources=registry))
        json_write(stage/'dataset_protocol.json',dict(
            schema_version='v1',protocol_id='dg_reid_member3_p3_p5_v2',
            status='PROVISIONAL_P4_PENDING',ready_for_finetuning=False,
            p4_approval_evidence=None,p4_status='NOT_CLEARED',
            validation_roles_are_official={'LaST':True,'IUST_PersonReID':False,'LPW':False},
            source_official_test_used=False,
            actual_image_bytes_verified_in_this_run=False,
            seed=seed,identity_holdout_fraction_target=val_fraction,
            selected_datasets=[v['name'] for v in SOURCES.values()],
            global_classifier_offset=base,prior_member_mappings=prior_provenance,
            num_training_classes=len(mapping),
            train_images=sum(tc.values()),validation_images=sum(vc.values()),
            input_sha256=digests,per_dataset=summary,
            last_validation_distractor_policy=(
                'All 20000 unlabeled gallery images retained. Each receives unique '
                'nonmatching validation-only identity_key; never a supervised class.'),
            caveats=[
                'P4 source-source/source-target leakage has NOT been cleared.',
                'Source imagery and root image paths were NOT accessed or re-decoded in this script; audit reports are historical.',
                'IUST/LPW query/gallery are custom internal retrieval protocols; evaluator must apply appropriate camera/view matching rules.',
                'LaST val protocol is official but dominated by same-video positives; do not interpret as cross-video DG proof.',
                'Final unseen domains (Market1501/MARS) are never used for model selection.',
                'Full CSV manifests are local-only. Generate 10-row previews separately for git.',
            ],
        ))
        with (stage/'checksums.sha256').open('w',encoding='utf-8',newline='\n') as checksum:
            for name in OUTPUTS[:-1]:
                checksum.write(f'{sha256_file(stage/name)}  {name}\n')
        for line in (stage/'checksums.sha256').read_text().splitlines():
            actual, filename=line.split('  ',1)
            check(sha256_file(stage/filename)==actual, f'Generated checksum mismatch: {filename}')
        check_destination(out)
        out.mkdir(parents=True,exist_ok=True)
        for name in OUTPUTS:
            os.replace(stage/name,out/name)
        (out/'.gitkeep').unlink(missing_ok=True)
    finally:
        shutil.rmtree(stage,ignore_errors=True)
    print('\n[SUCCESS] Exactly 6 files created:')
    for name in OUTPUTS: print('  ',out/name)
    for dataset,s in summary.items():
        print(f'{dataset}: train {s["train_images"]:,} | val {s["validation_images"]:,} '
              f'| query {s["query_images"]:,} | gallery {s["gallery_images"]:,}')
    print(f'TOTAL: train {sum(tc.values()):,}, val {sum(vc.values()):,}, classes {len(mapping):,}')
    print('STATUS: PROVISIONAL — P4 pending. Do NOT push full CSVs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--val-fraction',type=float,default=0.20)
    parser.add_argument('--allow-unpinned-inputs-for-test',action='store_true',
                        help='Synthetic testing only; never use for actual frozen input.')
    args=parser.parse_args()
    try:
        build(args.repo.resolve(),args.seed,args.val_fraction,args.allow_unpinned_inputs_for_test)
    except (AuditError, OSError, ValueError, KeyError, csv.Error, UnicodeError) as error:
        parser.exit(1,f'\n[FAILED] {error}\nNo P3/P5 approval. Fix inputs and rerun.\n')

if __name__=='__main__':
    main()
