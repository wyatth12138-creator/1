from __future__ import annotations
import hashlib, json
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tum_us.data.dataset import (_read_records, _video_array, _bmode_source_view, _ceus_media, _media_tensor, _load_masks, _target_definitions, _mask_channels, select_frames)
from tum_us.data.registry import TASK_TO_ID, classification_label
from tum_us.data.schema import Sample, collate_samples

@dataclass
class Entity:
    key: str
    batches: list
    label: int | None
    endpoint: str

class EndpointDataset:
    def __init__(self, spec, split, config, labels_allowed=True):
        if split not in ('train','dev','val','test') or (split == 'test' and (not config.get('allow_final_evaluation') or not spec.get('final_manifest'))):
            raise ValueError('Only explicit train/dev manifests are permitted')
        self.spec, self.config, self.labels_allowed = dict(spec), config, labels_allowed
        megabytes=float(spec.get('annotation_cache_mb',config.get('video',{}).get('annotation_cache_mb',0)))
        if not np.isfinite(megabytes) or megabytes<0: raise ValueError('annotation_cache_mb must be finite and nonnegative')
        self._annotation_cache=OrderedDict(); self._cache_limit=int(megabytes*1024*1024)
        self.cache_info={'hits':0,'misses':0,'bytes':0}
        self.split = split
        self.endpoint = str(spec.get('name', spec.get('endpoint', '')))
        self.root = Path(config.get('data_root', config.get('data', {}).get('data_root', '.'))).resolve()
        path = Path(spec['train_manifest' if split == 'train' else 'final_manifest' if split == 'test' else 'dev_manifest'])
        self.manifest_fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        mapping_path = spec.get('final_frame_map_path', spec.get('frame_map_path')) if split == 'test' else spec.get('frame_map_path')
        self.frame_map = json.loads(Path(mapping_path).read_text(encoding='utf-8-sig')) if mapping_path else {}
        if mapping_path:
            self.manifest_fingerprint = hashlib.sha256((self.manifest_fingerprint + hashlib.sha256(Path(mapping_path).read_bytes()).hexdigest()).encode()).hexdigest()
        self.size = config.get('model', {}).get('image_size', 256)
        self.size = (self.size, self.size) if isinstance(self.size, int) else tuple(self.size)
        mask_size = config.get('evaluation', {}).get('mask_size') if split != 'train' else None
        mask_size = self.size if mask_size is None else mask_size
        self.mask_size = (mask_size, mask_size) if isinstance(mask_size, int) else tuple(mask_size)
        if len(self.mask_size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in self.mask_size):
            raise ValueError('evaluation.mask_size must be a positive integer or pair')
        self.clip = int(config.get('video', {}).get('train_frames' if split == 'train' else 'eval_frames', config.get('data', {}).get('clip_length', 8)))
        if self.clip < 1: raise ValueError('clip_length must be positive')
        groups = defaultdict(list)
        for row in _read_records(path):
            if row.get('split') and row['split'] not in ({'train'} if split == 'train' else {'test'} if split == 'test' else {'dev','val','validation'}): continue
            if row.get('task') != spec['task']: continue
            if spec.get('dataset_names') and row.get('dataset_name') not in spec['dataset_names']: continue
            if spec.get('dataset_ids') and row.get('dataset_id') not in spec['dataset_ids']: continue
            key = str(row.get('group_id') or row.get('entity_uid') or row.get('case_id') or '')
            if not key: raise ValueError('Missing entity/group identifier')
            groups[key].append(row)
        keys = sorted(groups)
        budget = spec.get('label_budget') if split == 'train' else None
        if budget is not None:
            if int(budget) < 1: raise ValueError('label_budget must be positive')
            seed = spec.get('label_subset_seed', config.get('data', {}).get('label_subset_seed', 6000))
            if spec.get('subset_keys') is not None:
                selected = set(map(str, spec['subset_keys']))
                if not selected.issubset(groups): raise ValueError('Unknown subset group')
                keys = sorted(selected)
                if len(keys) != int(budget): raise ValueError('Subset size differs from budget')
            elif spec['kind'] == 'classification':
                if not labels_allowed: raise ValueError('Unlabeled classification requires explicit subset_keys')
                strata = defaultdict(list)
                for key in keys:
                    labels = {classification_label(r)[0] for r in groups[key] if classification_label(r)[1]}
                    if len(labels) != 1: raise ValueError('Classification group requires one consistent label')
                    strata[next(iter(labels))].append(key)
                ranked = {k: sorted(v,key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).digest()) for k,v in strata.items()}
                n = min(int(budget),len(keys)); remaining = n
                selected=[]
                for i, label in enumerate(sorted(ranked)):
                    count = remaining if i == len(ranked)-1 else round(n*len(ranked[label])/len(keys))
                    count = min(count,len(ranked[label]),remaining)
                    selected.extend(ranked[label][:count]); remaining -= count
                if remaining:
                    candidates = sorted(set(keys)-set(selected),key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).digest())
                    selected.extend(candidates[:remaining])
                keys=sorted(selected)
            else:
                keys = sorted(keys, key=lambda k: hashlib.sha256(f'{seed}:{k}'.encode()).digest())[:int(budget)]
        self.groups = [(k, groups[k]) for k in keys]
        if not self.groups: raise ValueError('No records selected')
        self.normalizer = float(spec.get('normalizer', 1.0))
        self.mean_rgb = list(spec.get('mean_rgb', [0.5,0.5,0.5]))

    def __len__(self): return len(self.groups)
    def _resolve(self, path):
        p = (self.root / str(path).replace('\\','/')).resolve()
        if p != self.root and self.root not in p.parents: raise ValueError('Path escapes data root')
        return p
    def _load_annotation(self,path,total,row,sample_index,policy):
        stat=path.stat()
        fields={k:row.get(k) for k in ('mask_frame_indices','annotated_frame_indices','frame_index_base','frame_indices')}
        key=(str(path),stat.st_mtime_ns,stat.st_size,total,sample_index,policy,json.dumps(fields,sort_keys=True))
        if key in self._annotation_cache:
            self.cache_info['hits']+=1
            value,size=self._annotation_cache.pop(key); self._annotation_cache[key]=(value,size)
            return value
        self.cache_info['misses']+=1
        value=_load_masks(path,total,row,sample_index,policy)
        size=sum(mask.nbytes for mask in value[0].values())
        if self._cache_limit and size<=self._cache_limit:
            for mask in value[0].values(): mask.setflags(write=False)
            while self._annotation_cache and self.cache_info['bytes']+size>self._cache_limit:
                _,(_,old)=self._annotation_cache.popitem(last=False); self.cache_info['bytes']-=old
            self._annotation_cache[key]=(value,size); self.cache_info['bytes']+=size
        return value
    def __getitem__(self, index):
        key, rows = self.groups[index]
        batches, labels = [], set()
        task = self.spec['task']; seg = self.spec['kind'] == 'segmentation'; ceus = task.startswith('ceus')
        targets = self.spec.get('target_ids') or [0]
        for row in rows:
            if not seg and self.labels_allowed:
                label, valid, _ = classification_label(row)
                if valid: labels.add(label)
            for si, source in enumerate(row['sample_paths']):
                path = self._resolve(source)
                if task.startswith('image'):
                    with Image.open(path) as image: raw = np.asarray(image.convert('RGB')).copy()
                    total=1; video=None
                else:
                    video = _video_array(path, ceus); total=len(video)
                    mapping = self.frame_map.get(source) or self.frame_map.get(source.replace('\\', '/'))
                    if mapping is None: raise ValueError(f'Missing explicit native frame map for {source}')
                    native = [int(v) for v in mapping['frame_indices']]
                    if len(native) != total or len(set(native)) != total or any(v < 0 for v in native) or any(a >= b for a,b in zip(native,native[1:])): raise ValueError('Invalid native frame map')
                    if ceus: side, view_order_source = _bmode_source_view(video,row)
                masks={}
                if seg and self.labels_allowed:
                    annotations = row.get('annotation_paths') or []
                    if si >= len(annotations): raise ValueError('Missing paired annotation')
                    if video is None:
                        masks, _ = self._load_annotation(self._resolve(annotations[si]), 1, row, si, 'error')
                    else:
                        mask_indices = mapping.get('mask_frame_indices')
                        if mask_indices is None: raise ValueError('Explicit mask_frame_indices required')
                        native_mask_indices = [int(v) for v in mask_indices]
                        if len(set(native_mask_indices)) != len(native_mask_indices) or not set(native_mask_indices).issubset(native): raise ValueError('Invalid annotation frame map')
                        mask_row = dict(row, mask_frame_indices=native_mask_indices, frame_index_base=0)
                        raw_masks, _ = self._load_annotation(self._resolve(annotations[si]), max(native)+1, mask_row, si, 'error')
                        if set(raw_masks) != set(native_mask_indices): raise ValueError('Annotation archive and frame map disagree')
                        annotation_valid = mapping.get('annotation_valid')
                        if annotation_valid is None: raise ValueError('Explicit annotation_valid required')
                        if len(annotation_valid) != len(native_mask_indices): raise ValueError('annotation_valid length mismatch')
                        native_to_local = {n:i for i,n in enumerate(native)}
                        masks = {native_to_local[n]:raw_masks[n] for n,v in zip(native_mask_indices, annotation_valid) if v}
                # Each window samples the full source extent; excess supervised frames get extra windows.
                if video is None:
                    windows = [[0]]
                    native = [0]
                else:
                    profile=self.spec.get('sampling_profile')
                    if profile:
                        from tum_us.data.cardiac import cardiac_windows
                        annotations=sorted(masks) if self.labels_allowed else [native.index(k) for k in mapping.get('mask_frame_indices',[]) if k in native]
                        windows=cardiac_windows(total,self.clip,annotations,profile=profile,training=self.split=='train')
                    else:
                        windows = [select_frames(total, self.clip, sorted(masks))]
                        uncovered = sorted(set(masks) - set(windows[0]))
                        while uncovered:
                            required = uncovered[:self.clip]
                            window = select_frames(total, self.clip, required)
                            windows.append(window)
                            uncovered = sorted(set(uncovered) - set(window))
                supervised_seen = set()
                seg_view = 0
                if ceus and masks:
                    first = next(iter(masks.values()))
                    if first.shape[1] == video.shape[2]:
                        half = first.shape[1] // 2
                        left = sum(np.count_nonzero(m[:, :half]) for m in masks.values())
                        right = sum(np.count_nonzero(m[:, -half:]) for m in masks.values())
                        mask_side = int(right > left) if left or right else side
                        seg_view = int(mask_side != side)
                    else:
                        seg_view = int(row.get('seg_view_id', 0))
                for window_number, indices in enumerate(windows):
                    if video is None: media=_media_tensor(raw.transpose(2,0,1)[None,None],self.size)
                    elif ceus: media=_ceus_media(video,indices,side,self.size)
                    else: media=_media_tensor(np.asarray(video[indices])[:,None],self.size)
                    mask_tensor=torch.zeros((len(indices),len(targets),*self.mask_size)); valid_mask=torch.zeros((len(indices),len(targets)),dtype=torch.bool)
                    if masks:
                        _, local, source_targets, raw_values = _target_definitions(row)
                        for j, frame in enumerate(indices):
                            if frame not in masks or frame in supervised_seen: continue
                            supervised_seen.add(frame)
                            m=masks[frame]
                            if ceus and m.shape[1] == video.shape[2]:
                                half=m.shape[1]//2; mask_side = side if seg_view == 0 else 1-side; m=m[:,:half] if mask_side == 0 else m[:,-half:]
                            channels=_mask_channels(m,local,raw_values,self.mask_size)
                            for q,target in enumerate(targets):
                                if target in source_targets:
                                    mask_tensor[j,q]=channels[source_targets.index(target)]; valid_mask[j,q]=True
                    label = next(iter(labels)) if len(labels)==1 and not seg else -100
                    sample=Sample(media,torch.ones(len(indices),dtype=torch.bool),torch.ones(media.shape[1],dtype=torch.bool),mask_tensor,valid_mask,torch.tensor(targets),torch.tensor(label),torch.tensor(label != -100),torch.tensor(self.spec.get('label_space_id',0)),torch.tensor(TASK_TO_ID[task]),torch.tensor(self.spec.get('anatomy_id',0)),torch.tensor(seg_view),torch.tensor([native[i] for i in indices]),{'entity_uid':key,'record_id':row.get('record_id'),'sample_index':si,'source_frame_indices':[native[i] for i in indices], 'local_frame_indices':indices, 'window_index':window_number, 'memory_reset':True})
                    sample.metadata['input_size'] = list(self.size)
                    sample.metadata['mask_size'] = list(self.mask_size)
                    if self.spec.get('sampling_profile'):
                        sample.metadata['sampling_profile']=self.spec['sampling_profile']
                        sample.metadata['supervision_protocol']='indexed_actual_masks_only'
                        sample.metadata['annotation_mapping_basis']=mapping.get('mapping_basis','explicit_user_mapping')
                    if ceus:
                        sample.metadata['view_order_source'] = view_order_source
                        sample.metadata['source_bmode_view_index'] = side
                    batches.append(collate_samples([sample]))
        if len(labels)>1: raise ValueError('Conflicting entity labels')
        return Entity(key,batches,next(iter(labels)) if labels else None,self.endpoint)
    def compute_normalization(self):
        # Explicit call on the training dataset only; source RGB first and second moments.
        if self.split != 'train': raise ValueError('Normalization requires training split')
        pixel_sum = None
        pixel_squares = None
        count = 0
        for i in range(len(self)):
            for batch in self[i].batches:
                x = batch.media.double().reshape(-1, 3, *self.size)
                pixel_sum = x.sum(0) if pixel_sum is None else pixel_sum + x.sum(0)
                pixel_squares = (x*x).sum(0) if pixel_squares is None else pixel_squares + (x*x).sum(0)
                count += len(x)
        template = pixel_sum / count
        self.mean_rgb = template.mean(dim=(-2,-1)).tolist()
        self.normalizer = max(float((pixel_squares/count-template.square()).mean()), 1e-8)
        return self.normalizer,self.mean_rgb

from .metrics import evaluate_entities, supervised_entity_loss





