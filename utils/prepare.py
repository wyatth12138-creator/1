"""Prepare manifest-only study inputs and audit group/label isolation.

This command reads metadata only. It never opens a media or annotation file.
The optional frame map for segmentation must come from independently verified
source metadata; a classification frame is labeled a local stored-array ordinal.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from tum_us.data.dataset import _read_records
from tum_us.data.registry import classification_label
from tum_us.generalist.config import load_config

def _key(row):
    result=row.get('group_id') or row.get('entity_uid') or row.get('case_id')
    if not result: raise ValueError('Missing group_id/entity_uid/case_id')
    return str(result)

def prepare(config, inventory, output, frame_map=None, include_test=False):
    cfg=load_config(overrides=config)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    manifest_dir=output/'manifests'; manifest_dir.mkdir(exist_ok=True)
    supplied = json.loads(Path(frame_map).read_text(encoding='utf-8-sig')) if frame_map else {}
    rows=_read_records(Path(inventory)); by_endpoint=defaultdict(lambda:defaultdict(list))
    entities=defaultdict(set); labels=defaultdict(set); bad_label=[]
    allowed={str(d):name for name,spec in cfg['endpoints'].items() for d in spec.get('dataset_ids',[])}
    if len(allowed)!=sum(len(s.get('dataset_ids',[])) for s in cfg['endpoints'].values()): raise ValueError('A dataset_id belongs to multiple endpoints')
    for row in rows:
        name=allowed.get(str(row.get('dataset_id')))
        if name is None: continue
        spec=cfg['endpoints'][name]
        if row.get('task')!=spec['task']: raise ValueError(f'{name}: task disagrees with inventory')
        split={'validation':'dev','val':'dev'}.get(row.get('split'),row.get('split'))
        if split not in ('train','dev','test'): raise ValueError(f'{name}: invalid split')
        key=_key(row); entities[key].add(split)
        if spec['kind']=='classification':
            value,valid,space=classification_label(row)
            if valid: labels[(key,space)].add(value)
            else: bad_label.append({'endpoint':name,'entity':key,'split':split,'status':row.get('classification_label_status','unavailable')})
        if split != 'test' or include_test: by_endpoint[name][split].append(row)
    overlaps={k:sorted(v) for k,v in entities.items() if len(v)>1}
    conflicts=[{'entity':k[0],'label_space_id':k[1],'values':sorted(v)} for k,v in labels.items() if len(v)>1]
    bad_entities={item['entity'] for item in bad_label} | {item['entity'] for item in conflicts} | set(overlaps)
    report={'inventory':str(Path(inventory).resolve()),'image_decode_performed':False,'test_exported':bool(include_test),
            'group_definition':'group_id, then entity_uid, then case_id; source identifiers must be validated independently',
            'identity_limitation':'Namespaced identifiers cannot establish or exclude cross-source patient overlap; independent identity linkage is required.',
            'exclude_invalid_groups':bool(cfg.get('prepare',{}).get('exclude_invalid_groups',True)),
            'cross_split_groups':overlaps,'label_conflicts':conflicts,'invalid_classification_labels':bad_label,'endpoints':{}}
    for name,spec in cfg['endpoints'].items():
        state={}; maps={}; final_maps={}; missing=[]; invalid=[]
        for split in ('train','dev') + (('test',) if include_test else ()):
            original=by_endpoint[name][split]
            excluded=sorted({_key(r) for r in original} & bad_entities)
            records=[r for r in original if _key(r) not in bad_entities] if report['exclude_invalid_groups'] else original
            out=manifest_dir/f'{name}_{split}.jsonl'
            with out.open('w',encoding='utf-8',newline='\n') as stream:
                for row in records: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            spec['train_manifest' if split=='train' else 'dev_manifest' if split=='dev' else 'final_manifest']=str(out)
            split_missing=[]
            for row in records:
                if spec['task'].startswith('image'): continue
                for i,source in enumerate(row.get('sample_paths',[])):
                    if spec['kind']=='classification':
                        lengths=row.get('temporal_lengths') or []
                        n=lengths[i] if i < len(lengths) else None
                        if not isinstance(n,int) or n<1: split_missing.append(source); continue
                        entry={'frame_indices':list(range(n)),'index_semantics':'stored_array_ordinal_not_acquisition_frame'}
                    else:
                        entry=supplied.get(source)
                        if not isinstance(entry,dict) or not all(k in entry for k in ('frame_indices','mask_frame_indices','annotation_valid')):
                            split_missing.append(source); continue
                        frames=entry['frame_indices']; annotated=entry['mask_frame_indices']; valid=entry['annotation_valid']
                        lengths=row.get('temporal_lengths') or []
                        n=lengths[i] if i < len(lengths) else None
                        if not isinstance(n,int) or n<1 or not isinstance(frames,list) or len(frames)!=n or not frames or len(set(frames))!=len(frames) or any(not isinstance(x,int) or x<0 for x in frames) or any(a>=b for a,b in zip(frames,frames[1:])) or not isinstance(annotated,list) or not isinstance(valid,list) or len(annotated)!=len(valid) or len(set(annotated))!=len(annotated) or not set(annotated).issubset(frames) or any(not isinstance(v,bool) for v in valid):
                            invalid.append(source); split_missing.append(source); continue
                    selected_map = final_maps if split == 'test' else maps
                    if source in selected_map and selected_map[source]!=entry: raise ValueError(f'Conflicting frame map for {source}')
                    selected_map[source]=entry
            affected=sorted({_key(r) for r in records} & bad_entities)
            state[split]={'records_before':len(original),'entities_before':len({_key(r) for r in original}),'excluded_entities':excluded if report['exclude_invalid_groups'] else [],'records':len(records),'entities':len({_key(r) for r in records}),'missing_frame_map_count':len(split_missing),'missing_frame_map_examples':split_missing[:20], 'invalid_or_cross_split_entities':affected, 'ready':bool(records) and not split_missing and not affected}
            missing.extend(split_missing)
        if maps:
            map_path=manifest_dir/f'{name}_frame_map.json'; map_path.write_text(json.dumps(maps,ensure_ascii=False),encoding='utf-8')
            spec['frame_map_path']=str(map_path)
        if final_maps:
            final_map_path=manifest_dir/f'{name}_final_frame_map.json'; final_map_path.write_text(json.dumps(final_maps,ensure_ascii=False),encoding='utf-8')
            spec['final_frame_map_path']=str(final_map_path)
        spec['readiness']={k:v['ready'] for k,v in state.items()}
        report['endpoints'][name]={'splits':state,'invalid_frame_map_examples':invalid[:20], 'role':spec.get('role'), 'frame_index_semantics':'stored_array_ordinal_not_acquisition_frame' if spec['task']=='ceus_cls' else 'explicit_verified_mapping_required' if spec['task']=='video_seg' else 'static'}
    cfg_path=output/'prepared_config.json'; cfg_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'readiness_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return cfg,report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--config'); parser.add_argument('--frame-map'); parser.add_argument('--include-test',action='store_true')
    parser.add_argument('--data-root',help='Override the data root when preparing a newly exported standalone dataset')
    parser.add_argument('--index-cardiac',action='store_true',help='Recover local CAMUS/EchoNet indexed masks before preparation')
    args=parser.parse_args()
    config=load_config(args.config,{'data_root':args.data_root} if args.data_root else None)
    if args.index_cardiac:
        if args.frame_map: parser.error('Choose --index-cardiac or an explicit --frame-map')
        from tum_us.data.cardiac import index_existing
        indexed=index_existing(config['data_root'],args.inventory,Path(args.output)/'cardiac_index',
                               splits=('train','val','test') if args.include_test else ('train','val'))
        args.inventory=indexed['inventory']; args.frame_map=indexed['frame_map']
    _,report=prepare(config,args.inventory,args.output,args.frame_map,args.include_test)
    print(json.dumps({'output':str(Path(args.output).resolve()),'endpoints':{k:v['splits'] for k,v in report['endpoints'].items()}},ensure_ascii=False))

if __name__=='__main__': main()
