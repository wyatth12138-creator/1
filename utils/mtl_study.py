"""Prepare or execute eight source-balanced reader optimization comparisons."""
import argparse,copy,json
from pathlib import Path
from tum_us.generalist.config import load_config

ARMS={f'{method}_{selection}':(method,selection=='cover')
      for method in ('mean','pcgrad','consistent') for selection in ('all','cover')}
ARMS.update({f'head_replay_{selection}':('consistent',selection=='cover') for selection in ('all','cover')})

def prepare_study(cfg,out,arms=None,seeds=None):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    arms=list(ARMS) if arms is None else list(arms)
    seeds=cfg['training'].get('seeds',[3407]) if seeds is None else seeds
    if not arms or len(arms)!=len(set(arms)) or any(a not in ARMS for a in arms): raise ValueError('Invalid study arms')
    if not seeds or len(seeds)!=len(set(seeds)): raise ValueError('Invalid study seeds')
    manifest={'arms':{},'split':'dev','scope':'reader optimization; fixed FIFO evaluation, no trained-controller confounding',
              'budget':'same supervised update/exposure schedule; consistency uses extra labelled alignment forwards; acquisition forwards and wall time reported separately',
              'clinical_results':False,'test_used':False}
    for seed in seeds:
        for arm in arms:
            method,selection=ARMS[arm]; local=copy.deepcopy(cfg)
            local['model']['private_seg_calibration']=True
            local['training'].update(seed=int(seed),post_init_seed=int(seed)+1021)
            options=local['training'].setdefault('multitask',{})
            options.update(enabled=True,optimizer=method)
            options['private_update']='supervised_replay' if arm.startswith('head_replay_') else 'alignment'
            options.setdefault('selection',{}).update(enabled=selection)
            local.setdefault('evaluation',{}).update(split='dev',policies=['fifo'],checkpoints={},seed=int(seed))
            local['evaluation'].pop('checkpoint',None)
            local['endpoint_sets']['evaluate']=list(local['endpoint_sets']['readers'])
            dest=out/f'seed_{seed}'/arm; dest.mkdir(parents=True,exist_ok=True)
            local['output_dir']=str(dest)
            path=dest/'config.json'
            serialized=json.dumps(local,ensure_ascii=False,indent=2)
            if path.exists() and path.read_text(encoding='utf-8')!=serialized: raise ValueError('Existing study config differs; use a new output')
            path.write_text(serialized,encoding='utf-8')
            manifest['arms'][f'{seed}/{arm}']={'config':str(path),'output':str(dest)}
    (out/'study_plan.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest

def execute_study(plan,out):
    from tum_us.generalist import engine
    from tum_us.generalist.analysis import paired_comparison
    rows={}; completed={}
    for key,item in plan['arms'].items():
        cfg=load_config(item['config']); directory=Path(item['output'])
        trained=engine.run_readers(cfg,directory)
        cfg['evaluation']['checkpoint']=trained['checkpoint']; evaluated=engine.evaluate(cfg,directory)
        seed,arm=key.split('/'); rows.setdefault(arm,[])
        for policy in evaluated['policies']:
            if policy['capacity']!=cfg['evaluation']['capacities'][0]: continue
            for ep,source in policy['endpoints'].items():
                rows[arm].extend(dict(r,seed=int(seed),endpoint=ep) for r in source['records'])
        completed[key]={'training':trained,'evaluation':str(directory/'evaluation/policy_reports.json')}
    comparisons={}
    if 'mean_all' in rows:
        comparisons={arm:paired_comparison(r,rows['mean_all'],bootstrap=1000) for arm,r in rows.items() if arm!='mean_all'}
    result={'completed':completed,'versus_mean_all':comparisons,'evaluation_capacity':cfg['evaluation']['capacities'][0],
            'note':'Endpoint-specific intervals. No claim of universal improvement from pooled AUROC and Dice.'}
    (Path(out)/'study_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    p.add_argument('--arms',help=','.join(ARMS)); p.add_argument('--seeds',help='comma-separated integers')
    p.add_argument('--execute',action='store_true',help='Run real reader training and dev evaluation; without this only prepare configurations')
    a=p.parse_args(); plan=prepare_study(load_config(a.config),a.output,a.arms.split(',') if a.arms else None,[int(s) for s in a.seeds.split(',')] if a.seeds else None)
    if a.execute: execute_study(plan,a.output)
    print(json.dumps({'plan':str(Path(a.output)/'study_plan.json'),'arms':len(plan['arms']),'executed':a.execute}))
if __name__=='__main__': main()
