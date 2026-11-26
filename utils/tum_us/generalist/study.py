"""Run all configured seeds and aggregate paired entity predictions."""
from copy import deepcopy
import json
from pathlib import Path
from .analysis import summarize_predictions, paired_comparison

def run_reader_baselines(cfg,out):
    """Joint, task-specialist and source-specialist readers from official init.

    Scale updates by source count so each source has approximately the joint
    reader's exposure; actual cyclic exposures are recorded, not assumed equal.
    Each reader has its own development selection, then a fixed FIFO evaluation.
    """
    import math
    if cfg.get('training',{}).get('multitask',{}).get('enabled',False):
        raise ValueError('Use mtl_study.py for multitask optimizer comparisons; specialist baselines require a single-family training config with multitask disabled')
    from .engine import run_readers, evaluate
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    names=list(cfg['endpoint_sets']['readers'])
    if not names: raise ValueError('Reader baselines require training endpoints')
    grouped={}
    for ep in names: grouped.setdefault(cfg['endpoints'][ep]['task'],[]).append(ep)
    arms={'joint':names,**{'task_'+k:v for k,v in grouped.items()},**{'source_'+n:[n] for n in names}}
    results={}; total=int(cfg['training'].get('reader_updates',1200)); batch=int(cfg['training'].get('effective_batch',16))
    for name,sources in arms.items():
        local=deepcopy(cfg); local['endpoint_sets']['readers']=sources
        updates=math.ceil(total*len(sources)/len(names)); local['training']['reader_updates']=updates
        def compatible(endpoint):
            destination=cfg['endpoints'][endpoint]
            for source in sources:
                origin=cfg['endpoints'][source]
                if origin['task']!=destination['task']: continue
                if origin['task'].endswith('cls'):
                    if origin.get('label_space_id',0)==destination.get('label_space_id',0): return True
                elif set(destination.get('target_ids',[0])).issubset(origin.get('target_ids',[0])): return True
            return False
        local['endpoint_sets']['evaluate']=[e for e in cfg['endpoint_sets'].get('evaluate',sources)
                                            if e in sources or compatible(e)]
        directory=out/name; local['output_dir']=str(directory)
        trained=run_readers(local,directory)
        local.setdefault('evaluation',{}).update(checkpoint=trained['checkpoint'],checkpoints={},policies=['none','fifo'])
        evaluated=evaluate(local,directory)
        counts={ep:sum(1 for i in range(updates*batch) if sources[i%len(sources)]==ep) for ep in sources}
        results[name]={'training':trained,'evaluation':evaluated,'reader_updates':updates,
                       'scheduled_entity_exposures':counts,'selection':'each reader independently selected on its development endpoints'}
    result={'arms':results,'initialization':'official SAM weights plus same configured seed; no joint-reader transfer',
            'budget':'approximately source-exposure matched; exact cyclic counts reported'}
    (out/'reader_baselines.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result

def run_study(cfg,out):
    from .engine import run
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    seeds=cfg.get('training',{}).get('seeds',[cfg['training'].get('seed',3407)])
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError('Study seeds must be nonempty and distinct')
    aggregate={}; completed=[]
    for seed in seeds:
        local=deepcopy(cfg); local['training']['seed']=int(seed); local.setdefault('evaluation',{})['seed']=int(seed)
        destination=out/('seed_'+str(seed)); local['output_dir']=str(destination)
        report=run(local,destination)
        if report.get('status') not in (None,'complete'):
            result={'status':'incomplete','completed_seeds':completed,'stopped_seed':seed,'detail':report}
            (out/'study_result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
            return result
        for policy in report['evaluation']['policies']:
            key=(policy['policy'],policy['capacity']); rows=aggregate.setdefault(key,[])
            for source in policy['endpoints'].values():
                rows.extend([dict(row,seed=int(seed)) for row in source['records']])
        completed.append(seed)
    boot=int(cfg.get('evaluation',{}).get('bootstrap',1000)); reports={}; comparisons={}
    for (policy,capacity),rows in aggregate.items():
        name=f'{policy}@{capacity}'; reports[name]=summarize_predictions(rows,bootstrap=boot)
        baseline=aggregate.get(('fifo',capacity))
        if baseline is not None and policy!='fifo': comparisons[name]=paired_comparison(rows,baseline,bootstrap=boot)
    result={'status':'complete','seeds':completed,'policies':reports,'versus_fifo':comparisons,
            'note':'Per-endpoint metrics; crossed seed/entity bootstrap, no independent-window assumption.'}
    (out/'study_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result
