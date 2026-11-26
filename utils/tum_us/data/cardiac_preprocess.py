"""Raw cardiac dataset export adapted from MemSAM's MIT-licensed protocols.

Attribution and source revision: vendor/MemSAM_LICENSE, SOURCES.md.
Images use linear interpolation, masks nearest-neighbor. Arrays contain explicit
storage/source indices; no EF label is fabricated from a segmentation mask.
"""
from __future__ import annotations
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def _write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def _resize(array,size,mask=False):
    import cv2
    method=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR
    return np.stack([cv2.resize(frame,(size,size),interpolation=method) for frame in array])

def _uint8_video(array):
    array=np.asarray(array)
    if not np.isfinite(array).all() or array.min()<0: raise ValueError('Invalid raw image intensity')
    if array.dtype.kind=='f' and array.max()<=1: array=array*255
    if array.max()>255: raise ValueError('Raw cardiac image outside [0,255]; specify and document intensity conversion first')
    return np.rint(array).astype(np.uint8)

def _paths(output,split,name):
    root=Path(output); vp=root/'videos'/split/(name+'.npy'); ap=root/'annotations'/split/(name+'.npz')
    if vp.exists() or ap.exists(): raise FileExistsError('Refusing to overwrite exported sequence '+name)
    return vp,ap

def _save_case(output,split,name,video,masks,raw_indices,metadata):
    vp,ap=_paths(output,split,name)
    if video.ndim!=4 or video.shape[1]!=3: raise ValueError('Expected exported video [T,3,H,W]')
    indices=sorted(masks)
    if not indices or any(i<0 or i>=len(video) for i in indices): raise ValueError('Invalid stored mask frame index')
    if len(raw_indices)!=len(video): raise ValueError('Raw/storage index lengths differ')
    vp.parent.mkdir(parents=True,exist_ok=True); ap.parent.mkdir(parents=True,exist_ok=True)
    # Safe numeric archives are understood by TUM-US without object unpickling.
    # Original MemSAM writes fnum_mask dictionaries; index_cardiac accepts both.
    np.save(vp,video.transpose(1,0,2,3),allow_pickle=False)
    numeric={k:np.array(metadata[k],dtype=np.float64) for k in ('ef','edv','esv') if metadata.get(k) is not None}
    np.savez(ap,mask=np.stack([masks[i] for i in indices]),frame_indices=np.array(indices,dtype=np.int64),
             raw_source_frame_indices=np.asarray(raw_indices,dtype=np.int64),**numeric)
    return {'video':vp.relative_to(output).as_posix(),'annotation':ap.relative_to(output).as_posix(),
            'frames':len(video),'mask_frames':indices,'raw_source_frame_indices':list(map(int,raw_indices)),
            'annotation_sha256':hashlib.sha256(ap.read_bytes()).hexdigest(),'metadata':metadata}

def _finish(output,dataset,rows,errors,options):
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    frame_map={r['video']:{'frame_indices':list(range(r['frames'])),'mask_frame_indices':r['mask_frames'],
        'annotation_valid':[True]*len(r['mask_frames']),'raw_source_frame_indices':r['raw_source_frame_indices'],
        'mapping_basis':'raw_export_explicit_indices','index_semantics':'stored_array_ordinal_with_raw_index_sidecar'} for r in rows}
    _write_json(root/'frame_map.json',frame_map)
    for split in options['splits']:
        (root/(dataset.lower()+'_'+split+'_filenames.txt')).write_text(''.join(r['name']+'\n' for r in rows if r['split']==split),encoding='utf-8')
    targets=['1:LV endo','2:myocardium','3:LA'] if dataset=='CAMUS' else ['1:LV endo']
    inventory=[]
    for r in rows:
        inventory.append({'dataset_id':'public/video_seg/'+dataset,'dataset_name':dataset,'task':'video_seg','privacy':'public',
            'split':r['split'],'entity_uid':dataset+'::'+r['entity'],'record_id':dataset+'::'+r['name'],
            'case_id':r['entity'],'sample_paths':[r['video']],'annotation_paths':[r['annotation']],
            'temporal_lengths':[r['frames']],'anatomy':'heart','segmentation_targets':targets,
            'segmentation_raw_values':{str(i):i for i in range(4 if dataset=='CAMUS' else 2)},
            'annotation_status':'dense_indexed' if len(r['mask_frames'])==r['frames'] else 'partial'})
    (root/'inventory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inventory),encoding='utf-8')
    _write_json(root/'preprocess_report.json',{'dataset':dataset,'reference':'dengxl0520/MemSAM@36f1b988f14b518a6346d2edb1267340387e646c',
        'options':options,'sequences':rows,'errors':errors,'test_converted':'test' in options['splits'],
        'array_layout':'C,T,H,W; RGB','image_interpolation':'linear','mask_interpolation':'nearest',
        'input_files_modified':False,'missing_frame_masks_created':False})
    if not rows: raise ValueError('No valid sequences exported; inspect '+str(root/'preprocess_report.json'))
    return {'output':str(root.resolve()),'sequences':len(rows),'errors':errors,'report':str(root/'preprocess_report.json')}

def _camus_config(path):
    result={}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if ':' in line:
            key,value=line.split(':',1); result[key.strip()]=value.strip()
    return result

def _read_nifti(path):
    import SimpleITK as sitk
    image=sitk.ReadImage(str(path)); array=sitk.GetArrayFromImage(image)
    if array.ndim==2: array=array[None]
    if array.ndim!=3: raise ValueError('CAMUS image must be [T,H,W] or [H,W]')
    return array,{'spacing':list(image.GetSpacing()),'origin':list(image.GetOrigin()),'direction':list(image.GetDirection())}

def preprocess_camus(input_dir,output_dir,*,size=256,split_file=None,splits=('train','val'),temporal_order='source',mask_mode='auto'):
    root=Path(input_dir); output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    if root.resolve()==output.resolve() or size<1: raise ValueError('Use a distinct output directory and positive resize size')
    if not root.is_dir(): raise NotADirectoryError(root)
    if temporal_order not in ('source','ed-to-es') or mask_mode not in ('auto','dense','ed-es'): raise ValueError('Invalid CAMUS export mode')
    patients=sorted(p.name for p in root.glob('patient*') if p.is_dir())
    if not patients: raise ValueError('No patient directories in CAMUS input')
    if split_file:
        groups=json.loads(Path(split_file).read_text(encoding='utf-8-sig'))
        groups={k:list(groups.get(k,groups.get('dev',[]) if k=='val' else [])) for k in ('train','val','test')}
    else:
        a,b=int(.7*len(patients)),int(.1*len(patients))
        groups={'train':patients[:a],'val':patients[a:a+b],'test':patients[a+b:]}
    membership={}
    for split,names in groups.items():
        for name in names:
            if name in membership: raise ValueError('Patient occurs in multiple splits: '+name)
            membership[name]=split
    if set(patients)-set(membership): raise ValueError('CAMUS split file omits patients')
    rows=[]; errors=[]
    for patient in patients:
        split=membership[patient]
        if split not in splits: continue
        staged=[]
        try:
            # Validate both views and both output names before writing either.
            for view in ('2CH','4CH'):
                folder=root/patient; prefix=patient+'_'+view
                sequence=next((folder/(prefix+suffix) for suffix in ('_half_sequence.nii.gz','_sequence.nii.gz') if (folder/(prefix+suffix)).is_file()),None)
                if sequence is None: raise FileNotFoundError('Missing CAMUS sequence '+prefix)
                raw,geometry=_read_nifti(sequence); cfg=_camus_config(folder/('Info_'+view+'.cfg')); t,h,w=raw.shape
                if int(cfg.get('NbFrame',t))!=t: raise ValueError('CAMUS NbFrame does not match selected sequence')
                ed,es=int(cfg['ED'])-1,int(cfg['ES'])-1
                if not 0<=ed<t or not 0<=es<t or ed==es: raise ValueError('CAMUS ED/ES must be distinct one-based source indices')
                dense_path=sequence.with_name(sequence.name.replace('.nii.gz','_gt.nii.gz'))
                masks={}; mode='dense' if mask_mode=='dense' or (mask_mode=='auto' and dense_path.is_file()) else 'ed-es'
                if mode=='dense':
                    labels,gt_geometry=_read_nifti(dense_path)
                    if labels.shape!=raw.shape or gt_geometry!=geometry: raise ValueError('CAMUS sequence/GT geometry mismatch')
                    masks={i:labels[i] for i in range(t)}
                else:
                    for phase,index in (('ED',ed),('ES',es)):
                        label,_=_read_nifti(folder/(prefix+'_'+phase+'_gt.nii.gz'))
                        if label.shape!=(1,h,w): raise ValueError('CAMUS phase-mask spatial mismatch')
                        masks[index]=label[0]
                if any(not set(np.unique(m)).issubset({0,1,2,3}) for m in masks.values()): raise ValueError('Unexpected CAMUS raw segmentation class')
                indices=list(range(t))
                reverse=temporal_order=='ed-to-es' and ed>es
                if reverse:
                    raw=raw[::-1]; masks={t-1-i:m for i,m in masks.items()}; indices=indices[::-1]
                resized=_resize(_uint8_video(raw),size); video=np.repeat(resized[:,None],3,axis=1)
                masks={i:_resize(np.asarray(m)[None],size,True)[0] for i,m in masks.items()}
                spacing=list(geometry['spacing']); spacing[0]*=w/size; spacing[1]*=h/size
                meta={'source_sequence':str(sequence.resolve()),'source_geometry':geometry,'export_spacing':spacing,
                    'source_ed_index':ed,'source_es_index':es,'stored_ed_index':t-1-ed if reverse else ed,
                    'stored_es_index':t-1-es if reverse else es,'source_index_base':0,'annotation_mode':mode,
                    'temporal_order':temporal_order,'reversed':reverse,'source_config':cfg,
                    'ef':float(cfg['EF']) if 'EF' in cfg else None,'ef_status':'source_cfg' if 'EF' in cfg else 'not_exported'}
                _paths(output,split,prefix); staged.append((prefix,video,masks,indices,meta))
            for name,video,masks,indices,meta in staged:
                rows.append(dict(_save_case(output,split,name,video,masks,indices,meta),name=name,entity=patient,split=split))
        except (OSError,ValueError,KeyError,RuntimeError) as exc:
            errors.append({'entity':patient,'split':split,'reason':str(exc)})
    return _finish(output,'CAMUS',rows,errors,{'size':size,'splits':list(splits),'split_file':str(split_file) if split_file else None,
            'default_split':'ordered patient IDs 70/10/20, following upstream active split() rather than unused random_split()',
            'temporal_order':temporal_order,'mask_mode':mask_mode})

def trace_to_mask(traces,shape):
    """MemSAM/EchoNet contour order: omit first long-axis row, reverse side 2."""
    from skimage.draw import polygon
    points=np.asarray(traces,dtype=float)
    if points.ndim!=2 or points.shape[1]!=4 or len(points)<3 or not np.isfinite(points).all(): raise ValueError('Invalid EchoNet trace rows')
    x=np.concatenate((points[1:,0],points[1:,2][::-1])); y=np.concatenate((points[1:,1],points[1:,3][::-1]))
    r,c=polygon(np.rint(y).astype(int),np.rint(x).astype(int),shape)
    mask=np.zeros(shape,dtype=np.uint8); mask[r,c]=1
    return mask

def _load_avi(path):
    import cv2
    cap=cv2.VideoCapture(str(path)); frames=[]
    try:
        expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened(): raise ValueError('Cannot open AVI '+str(path))
        while True:
            ok,frame=cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if not frames or (expected>0 and len(frames)!=expected): raise ValueError('Incomplete AVI decode')
        return np.stack(frames)
    finally: cap.release()

def preprocess_echonet(input_dir,output_dir,*,size=112,splits=('train','val'),expected_trace_rows=21):
    root=Path(input_dir); output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    if root.resolve()==output.resolve() or size<1: raise ValueError('Use a distinct output directory and positive resize size')
    with (root/'FileList.csv').open(newline='',encoding='utf-8-sig') as stream: files=list(csv.DictReader(stream))
    traces=defaultdict(lambda:defaultdict(list))
    with (root/'VolumeTracings.csv').open(newline='',encoding='utf-8-sig') as stream:
        for r in csv.DictReader(stream):
            value=float(r['Frame'])
            if not np.isfinite(value) or value!=int(value): raise ValueError('Noninteger EchoNet frame index')
            traces[Path(r['FileName']).stem][int(value)].append([float(r[k]) for k in ('X1','Y1','X2','Y2')])
    rows=[]; errors=[]; seen=set()
    for row in files:
        name=Path(row['FileName']).stem; split=row['Split'].lower()
        if name in seen: raise ValueError('Duplicate EchoNet FileList record '+name)
        seen.add(name)
        if split not in ('train','val','test'): raise ValueError('Unknown official EchoNet split')
        if split not in splits: continue
        try:
            indexed=traces[name]
            if len(indexed)!=2 or any(len(t)!=expected_trace_rows for t in indexed.values()):
                raise ValueError('Expected two indexed frames, each with '+str(expected_trace_rows)+' tracing rows')
            filename=row['FileName'] if Path(row['FileName']).suffix else row['FileName']+'.avi'
            path=root/'Videos'/filename; video=_load_avi(path); t,h,w,_=video.shape
            if any(i<0 or i>=t for i in indexed): raise ValueError('EchoNet traced frame outside decoded video')
            masks={i:trace_to_mask(trace,(h,w)) for i,trace in indexed.items()}
            if any(not m.any() for m in masks.values()): raise ValueError('Empty EchoNet trace polygon')
            if size:
                video=_resize(video,size); masks={i:_resize(m[None],size,True)[0] for i,m in masks.items()}
            meta={'source_video':str(path.resolve()),'source_index_base':0,'temporal_order':'source','reversed':False,
                'traced_frames':sorted(indexed),'ed_es_identity':'not_inferred_from_dictionary_order',
                'spacing_status':'unknown; no physical millimeter spacing inferred',
                **{k.lower():float(row[k]) if row.get(k) not in (None,'') else None for k in ('EF','EDV','ESV')}}
            rows.append(dict(_save_case(output,split,name,video.transpose(0,3,1,2),masks,list(range(t)),meta),name=name,entity=name,split=split))
        except (OSError,ValueError,KeyError,RuntimeError) as exc:
            errors.append({'entity':name,'split':split,'reason':str(exc)})
    return _finish(output,'EchoNet',rows,errors,{'size':size,'splits':list(splits),'expected_trace_rows':expected_trace_rows,
            'partition':'official FileList.csv; no random repartition','unannotated_frames':'kept without masks'})
