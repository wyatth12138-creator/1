"""Four explicit reader arms, each passed through the whole utility pipeline."""
from copy import deepcopy
import json
from pathlib import Path
from .analysis import paired_comparison,reader_policy_interaction

ARMS={'native':(False,False),'query':(True,False),'object':(False,True),'combined':(True,True)}

def run_object_study(cfg,out):
    from .engine import run
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    seeds=cfg.get('training',{}).get('seeds',[cfg['training'].get('seed',3407)])
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError('Object-study seeds must be nonempty and distinct')
    selected=cfg.get('object_study',{}).get('arms',list(ARMS))
    if not selected or len(set(selected))!=len(selected) or any(arm not in ARMS for arm in selected): raise ValueError('Unknown or repeated object-study reader arm')
    results={}
    for seed in seeds:
        for arm in selected:
            local=deepcopy(cfg); query,objects=ARMS[arm]
            local['model'].update(query_propagation=query,object_memory=objects)
            local['training']['seed']=int(seed); local.setdefault('evaluation',{})['seed']=int(seed)
            local['training']['post_init_seed']=int(seed)+1021
            destination=out/f'seed_{seed}'/arm; local['output_dir']=str(destination)
            result=run(local,destination)
            results[f'{seed}/{arm}']={'status':result.get('status','complete'),
                'pipeline_result':str(destination/'pipeline_result.json'),
                'evaluation':str(destination/'evaluation/policy_reports.json'),
                'reader_checkpoint':result.get('readers',{}).get('checkpoint'),
                'controller_checkpoint':result.get('controller',{}).get('checkpoint')}
            summary={'results':results,'arms':{k:dict(query_propagation=v[0],object_memory=v[1]) for k,v in ARMS.items()},
                     'scope':'Different reader arms are trained independently; policies share a frozen reader only within each arm.'}
            if result.get('status') not in (None,'complete'):
                summary['status']='incomplete'; (out/'object_study.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8'); return summary
    comparisons={}
    if 'native' in selected:
        for arm in selected:
            if arm=='native': continue
            for policy in ('fifo','tum'):
                for capacity in cfg.get('evaluation',{}).get('capacities',[2,4,8]):
                    left=[]; right=[]
                    for seed in seeds:
                        for name,destination in ((arm,left),('native',right)):
                            report=json.loads(Path(results[f'{seed}/{name}']['evaluation']).read_text(encoding='utf-8'))
                            entry=next((r for r in report['policies'] if r['policy']==policy and r['capacity']==capacity),None)
                            if entry:
                                destination.extend(dict(row,seed=int(seed)) for source in entry['endpoints'].values() for row in source['records'])
                    if left and right:
                        comparisons[f'{arm}-native/{policy}@{capacity}']=paired_comparison(left,right,
                            bootstrap=int(cfg.get('evaluation',{}).get('bootstrap',1000)),seed=int(cfg['training'].get('seed',3407)))
    interactions={}
    if 'native' in selected:
        for arm in selected:
            if arm=='native': continue
            for capacity in cfg.get('evaluation',{}).get('capacities',[2,4,8]):
                conditions=[[],[],[],[]]
                for seed in seeds:
                    for name,policy,rows in zip((arm,arm,'native','native'),('tum','fifo','tum','fifo'),conditions):
                        report=json.loads(Path(results[f'{seed}/{name}']['evaluation']).read_text(encoding='utf-8'))
                        entry=next((r for r in report['policies'] if r['policy']==policy and r['capacity']==capacity),None)
                        if entry: rows.extend(dict(row,seed=int(seed)) for src in entry['endpoints'].values() for row in src['records'])
                if all(conditions): interactions[f'{arm}@{capacity}']=reader_policy_interaction(*conditions,
                    bootstrap=int(cfg.get('evaluation',{}).get('bootstrap',1000)),seed=int(cfg['training'].get('seed',3407)))
    summary.update(status='complete',paired_reader_comparisons=comparisons,reader_policy_interactions=interactions)
    (out/'object_study.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    return summary
