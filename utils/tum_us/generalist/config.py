"""Configuration for the task-utility study; paths are explicit runtime inputs."""
import copy
import json
from pathlib import Path
from tum_us.paths import ROOT

def _endpoint(task, dataset_id, anatomy_id, targets, role, label_space=0):
    return {'kind':'segmentation' if task.endswith('seg') else 'classification', 'task':task,
            'dataset_ids':[dataset_id], 'anatomy_id':anatomy_id, 'target_ids':targets,
            'label_space_id':label_space, 'role':role, 'label_budget':None}

def default_config():
    endpoints = {
      'bus_cls':_endpoint('image_cls','public/image_cls/BUS-BRA',2,[1],'train'),
      'appendix_cls':_endpoint('image_cls','public/image_cls/Appendix',1,[19],'train',1),
      'bus_seg':_endpoint('image_seg','public/image_seg/BUSIS',2,[1],'train'),
      'fetal_seg':_endpoint('image_seg','public/image_seg/Fetal_HC',3,[4],'train'),
      'camus':_endpoint('video_seg','public/video_seg/CAMUS',4,[5],'train'),
      'ceus_breast':_endpoint('ceus_cls','private/ceus_cls/Breast',2,[1],'train'),
      'echonet':_endpoint('video_seg','public/video_seg/EchoNet',4,[5],'heldout_source'),
      'ceus_liver':_endpoint('ceus_cls','private/ceus_cls/Liver',6,[14],'heldout_source'),
      'ceus_thyroid':_endpoint('ceus_cls','private/ceus_cls/Thyroid',9,[2],'heldout_source')}
    endpoints['camus']['sampling_profile']='camus_sequence'
    endpoints['echonet']['sampling_profile']='echonet_interval'
    train=['bus_cls','appendix_cls','bus_seg','fetal_seg','camus','ceus_breast']
    return {'data_root':'data', 'output_dir':str(ROOT/'run'),
      'model':{'backbone':'small','image_size':256,'memory_capacity':4,'policy':'fifo','checkpoint':'pre_checkpoints/sam2.1_hiera_small.pt','freeze_trunk':True,'joint_ceus_memory':False},
      'training':{'seed':3407,'seeds':[3407,3408,3409],'lr':0.0003,'weight_decay':0.01,'reader_updates':1200,'effective_batch':16,'device':'auto','amp':True},
      'video':{'train_frames':16,'eval_frames':32,'annotation_cache_mb':8}, 'prepare':{'exclude_invalid_groups':True},
      'utility':{'capacities':[2,4,8],'horizons':[1,2,4],'max_states':2,'behavior_policy':'fifo','continuation_policy':'fifo','recollection_rounds':1,'run_expand':True,'reuse_prefix':True,'target_objective':'task','tie_continuation_to_behavior':True,'variants':['fixed_state','task_agnostic','seg_specialist','cls_specialist']},
      'controller':{'lr':0.0003,'updates':1200,'ranking_weight':1.0,'replay_fraction':0.5},
      'evaluation':{'mask_size':512,'policies':['none','fifo','random','quality','novelty','tum','full'],'capacities':[2,4,6,8],'bootstrap':1000,'seed':3407,'report_case_cap':100},
      'endpoints':endpoints,'endpoint_sets':{'readers':train,'collect':train,'evaluate':train+['echonet','ceus_liver','ceus_thyroid'],'expand_old':train,'expand_new':['ceus_liver']}}

def _merge(dst,src):
    for key,value in src.items():
        if isinstance(value,dict) and isinstance(dst.get(key),dict): _merge(dst[key],value)
        else: dst[key]=copy.deepcopy(value)
    return dst

def load_config(path=None, overrides=None):
    cfg=default_config()
    updates = []
    if path is not None:
        with Path(path).open(encoding='utf-8-sig') as stream:
            updates.append(json.load(stream))
    if overrides:
        updates.append(overrides)
    for update in updates:
        if update.get('replace_endpoints') is True and 'endpoints' in update:
            cfg['endpoints'] = {}
        _merge(cfg, update)
    for section,key in (('training','reader_updates'),('training','effective_batch'),('controller','updates')):
        value=cfg[section][key]
        if not isinstance(value,int) or isinstance(value,bool) or value < 1:
            raise ValueError(f'{section}.{key} must be a positive integer')
    image_size = cfg['model']['image_size']
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 32 or image_size % 32:
        raise ValueError('model.image_size must be a positive integer multiple of 32')
    capacities=cfg['utility']['capacities']
    if not capacities or any(not isinstance(v,int) or isinstance(v,bool) or v<1 for v in capacities):
        raise ValueError('utility.capacities must contain positive integers')
    if cfg['video']['train_frames'] <= max(capacities)+1:
        raise ValueError('video.train_frames must exceed max utility capacity plus one')
    checkpoint=cfg['model'].get('checkpoint')
    if checkpoint:
        from tum_us.paths import pretrained_path
        cfg['model']['checkpoint']=str(pretrained_path(checkpoint))
    return cfg
