"""Dataset-specific cardiac sampling and local indexed-mask provenance.

Reference: MemSAM commit 36f1b988f14b518a6346d2edb1267340387e646c.
No annotation is assigned to a frame by its position in a Python dictionary.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json
import numpy as np
from .dataset import _read_records, _video_array, _load_masks, select_frames

def cardiac_windows(total, length, annotated, *, profile, training=False):
    """Unique chronological samples; short videos are never duplicated/padded."""
    annotations=sorted(set(int(i) for i in annotated))
    if total<1 or length<1 or any(i<0 or i>=total for i in annotations):
        raise ValueError('Invalid cardiac frame extent or annotations')
    if profile=='echonet_interval':
        if len(annotations)!=2: raise ValueError('EchoNet interval protocol requires exactly two indexed masks')
        first,last=annotations[0],annotations[-1]
        local=select_frames(last-first+1,length,[0,last-first])
        return [[first+i for i in local]]
    if profile not in ('camus_sequence','cardiac_full_video'):
        raise ValueError('Unknown cardiac sampling profile '+str(profile))
    # Dense CAMUS training uses a single bounded sequence, not one additional
    # training forward for every excess annotated frame. Evaluation covers GT.
    required=annotations if len(annotations)<=length else []
    windows=[select_frames(total,length,required)]
    if training: return windows
    uncovered=sorted(set(annotations)-set(windows[0]))
    while uncovered:
        window=select_frames(total,length,uncovered[:length]); windows.append(window)
        uncovered=sorted(set(uncovered)-set(window))
    return windows

def index_existing(data_root,inventory,output,splits=('train','val'),datasets=('CAMUS','EchoNet')):
    """Read local exported annotations; preserve inventory patient partitions.

    An indexed dictionary proves storage-frame correspondence, not original
    acquisition time, human annotation provenance or ED/ES identity.
    """
    root=Path(data_root).resolve(); out=Path(output).resolve(); out.mkdir(parents=True,exist_ok=True)
    wanted={'public/video_seg/'+name for name in datasets}
    rows=_read_records(Path(inventory)); mapping={}; accepted=[]; issues=[]; sources={}
    def resolved(relative):
        path=(root/str(relative).replace('\\','/')).resolve()
        if not path.is_relative_to(root): raise ValueError('Source path escapes data root')
        return path
    for row in rows:
        if row.get('split') not in splits: continue
        if row.get('dataset_id') not in wanted:
            accepted.append(row); continue
        staged={}; details=[]
        try:
            videos=row['sample_paths']; annotations=row.get('annotation_paths',[])
            if len(videos)!=len(annotations): raise ValueError('Video/annotation source counts disagree')
            for i,(video,annotation) in enumerate(zip(videos,annotations)):
                vp,ap=resolved(video),resolved(annotation); array=_video_array(vp,False); total=len(array)
                # error forbids center/first reference fallback. Indexed legacy
                # fnum_mask objects are read only from the selected local root.
                masks,provenance=_load_masks(ap,total,row,i,'error')
                basis=provenance['mask_frame_mapping']
                if basis not in ('explicit_fnum_mask','explicit_frame_indices','dense_same_length'):
                    raise ValueError('No indexed cardiac supervision')
                keys=sorted(masks)
                if not keys: raise ValueError('No indexed masks')
                if row['dataset_id'].endswith('/EchoNet') and len(keys)!=2:
                    raise ValueError('EchoNet protocol expects two annotated frames')
                values=set(); distinct=set(); empty=0
                for mask in masks.values():
                    if tuple(mask.shape)!=tuple(array.shape[-2:]): raise ValueError('Mask/video spatial shape mismatch')
                    unique=np.unique(mask)
                    if any(not np.isfinite(v) or float(v)!=int(v) for v in unique): raise ValueError('Mask contains noninteger values')
                    values.update(int(v) for v in unique)
                    distinct.add(hashlib.sha256(np.ascontiguousarray(mask).tobytes()).digest())
                    empty+=not np.any(mask)
                allowed={0,1,2,3,100,200,255} if row['dataset_id'].endswith('/CAMUS') else {0,1,255}
                if not values.issubset(allowed): raise ValueError('Unexpected cardiac label encoding: '+str(sorted(values)))
                entry={'frame_indices':list(range(total)),'mask_frame_indices':keys,'annotation_valid':[True]*len(keys),
                       'index_semantics':'stored_array_ordinal_not_verified_acquisition_frame','mapping_basis':basis,
                       'annotation_sha256':hashlib.sha256(ap.read_bytes()).hexdigest(),
                       'dense_indexed':keys==list(range(total)),
                       'temporal_direction':'as_stored; original acquisition direction unverified'}
                staged[video]=entry
                details.append({'sample_path':video,'frames':total,'annotated_frames':len(keys),
                    'dense_indexed':entry['dense_indexed'],'distinct_masks':len(distinct),'empty_masks':int(empty),
                    'raw_values':sorted(values),'mapping_basis':basis})
            indexed=dict(row,cardiac_annotation_index='verified_from_local_archive')
            if row['dataset_id'].endswith('/CAMUS'):
                # CAMUS raw class 2 is myocardium, not the union enclosed by the
                # epicardial contour. Preserve legacy ID 6 and use reserved 24.
                indexed['segmentation_targets']=['1:LV endo','2:lv_myocardium','3:LA']
            accepted.append(indexed)
            mapping.update(staged)
            name=row['dataset_id']; info=sources.setdefault(name,{'entities':0,'sequences':0,'annotated_frames':0,'dense_sequences':0,'sparse_sequences':0,'raw_values':set()})
            info['entities']+=1
            for d in details:
                info['sequences']+=1; info['annotated_frames']+=d['annotated_frames']; info['dense_sequences']+=int(d['dense_indexed']); info['sparse_sequences']+=int(not d['dense_indexed']); info['raw_values'].update(d['raw_values'])
        except (OSError,ValueError,KeyError,TypeError) as exc:
            # Exclude the entire entity, including both CAMUS views, consistently.
            issues.append({'entity':row.get('entity_uid'),'dataset':row.get('dataset_id'),'split':row.get('split'),'reason':str(exc)})
    for info in sources.values(): info['raw_values']=sorted(info['raw_values'])
    paths={'frame_map':out/'frame_map.json','inventory':out/'inventory_trainval.jsonl','audit':out/'cardiac_audit.json'}
    paths['frame_map'].write_text(json.dumps(mapping,ensure_ascii=False),encoding='utf-8')
    paths['inventory'].write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted),encoding='utf-8')
    report={'data_root':str(root),'inventory_sha256':hashlib.sha256(Path(inventory).read_bytes()).hexdigest(),
            'splits':list(splits),'test_opened':'test' in splits,'source_arrays_modified':False,
            'pixel_arrays_used_for_annotation_validation_only':True,'prediction_or_training_performed':False,
            'sources':sources,'excluded_entities':issues,'frame_map':str(paths['frame_map']),'inventory':str(paths['inventory']),
            'interpretation':'dense_indexed means per-stored-frame masks exist; it does not certify human annotation or raw acquisition chronology'}
    paths['audit'].write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
