import copy, json, time, hashlib, random
from pathlib import Path
import torch
from . import engine, analysis
from .checkpoint import save_checkpoint, load_checkpoint, rng_state, restore_rng, file_hash, digest
from .selection import build_gain_archive, select
from .model import TransferModel

METHODS=('shared','gain_selected2','drop_a_selected2','random2','same_selected2_frozen_no_copy','private_delta_old_route_frozen')

def _json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8'); tmp.replace(path)

def _manifest(cfg,ep,split):
    spec=cfg['endpoints'][ep]; h=file_hash(spec[split+'_manifest'])
    if spec.get('frame_map_path'): h=hashlib.sha256((h+file_hash(spec['frame_map_path'])).encode()).hexdigest()
    return h

def _save(out,name,model,cfg,parents=None,extra=None):
    manifests={}
    for ep in model.export_spec().get('endpoints', {}):
        for split in ('train','dev'):
            manifests[ep+':'+split]=_manifest(cfg,ep,split)
    path=Path(out)/(name+'.pt')
    if path.exists():
        previous,payload=load_checkpoint(path,expected_config=cfg,expected_parents=parents or {})
        if payload['manifests'] != manifests or previous.export_spec() != model.export_spec(): raise ValueError('Existing stage topology or manifests differ')
        state=model.state_dict(); old=previous.state_dict()
        if state.keys()!=old.keys() or any(not torch.equal(state[k].cpu(),old[k].cpu()) for k in state): raise ValueError('Existing stage state differs')
        restore_rng(payload['rng']); return str(path)
    return save_checkpoint(path,model,cfg,stage=name,parents=parents,manifests=manifests,extra=extra)

def _run_joint(model,cfg,out,name,parents,endpoint,old=(),**options):
    path=Path(out)/(name+'_progress.pt'); resume=None
    run_hash=digest({'name':name,'endpoint':endpoint,'old':list(old),'options':{k:v for k,v in options.items() if k!='initial_rng'}})
    if path.exists():
        model,resume=load_checkpoint(path,expected_config=cfg,expected_parents=parents)
        actual={}
        for ep in model.export_spec()['endpoints']:
            for split in ('train','dev'):
                actual[ep+':'+split]=_manifest(cfg,ep,split)
        if actual != resume['manifests'] or resume['extra'].get('run_hash') != run_hash or resume['stage'] != name: raise ValueError('Resume manifest or run contract mismatch')
    else:
        actual={}
        for ep in model.export_spec()['endpoints']:
            for split in ('train','dev'):
                actual[ep+':'+split]=_manifest(cfg,ep,split)
    every=int(cfg['training'].get('checkpoint_every',100))
    def checkpoint(m,opt,scheduler,counts,cursor):
        if counts['optimizer_updates'] % every == 0:
            save_checkpoint(path,m,cfg,stage=name,parents=parents,manifests=actual,optimizer=opt,scheduler=scheduler,cursor=cursor,extra={'counts':dict(counts),'run_hash':run_hash,'amp':'bf16' if cfg['training'].get('amp') else 'disabled'})
    result,cost=engine.train_joint(model,cfg,endpoint,old,callback=checkpoint,resume=resume,**options); cost['resumed_progress']=resume is not None
    return result,cost

def _eval(model,cfg,endpoints,split='dev'):
    device=engine.device_for(cfg); model.to(device)
    if split == 'test' and not cfg.get('allow_final_evaluation'): raise ValueError('Final evaluation requires explicit authorization')
    return {e:engine._evaluate(model,engine._dataset(cfg,e,split),e,device) for e in endpoints}

def _done(out,name,cfg,parents=None):
    path=Path(out)/(name+'.pt')
    if not path.exists(): return None
    model,payload=load_checkpoint(path,expected_config=cfg,expected_parents=parents or {})
    actual={ep+':'+split:_manifest(cfg,ep,split) for ep in model.export_spec()['endpoints'] for split in ('train','dev')}
    if actual != payload['manifests']: raise ValueError('Completed stage manifest mismatch')
    restore_rng(payload['rng'])
    cost=copy.deepcopy(payload['extra']); cost['reused_completed_stage']=True
    return model,str(path),cost

def _begin(cfg,out):
    Path(out).mkdir(parents=True,exist_ok=True); engine._seed(cfg['training'].get('seed',6000))
    existing=_done(out,'A_parent',cfg)
    if existing:
        model,a,cost=existing
    else:
        progress=Path(out)/'A_progress.pt'; resume=None
        manifests={'A:'+split:_manifest(cfg,'A',split) for split in ('train','dev')}
        if progress.exists():
            model,resume=load_checkpoint(progress,expected_config=cfg,expected_parents={})
            if resume['stage']!='A_progress' or resume['manifests']!=manifests: raise ValueError('A resume dependencies differ')
        else:
            model=TransferModel(cfg['model']); model.add_endpoint('A',cfg['endpoints']['A'])
        def save_A(m,opt,scheduler,cursor,extra):
            save_checkpoint(progress,m,cfg,stage='A_progress',manifests=manifests,optimizer=opt,scheduler=scheduler,cursor=cursor,extra=extra)
        model,cost=engine.train_A(model,cfg,callback=save_A,resume=resume); a=_save(out,'A_parent',model,cfg,extra=cost)
    base_rng=rng_state(); branches={}
    for name,supervised in [('minus',False),('plus',True)]:
        stage='B'+name; existing=_done(out,stage,cfg,{'A':a})
        if existing:
            branches[name]=existing; continue
        m=engine.clone_model(model,engine.device_for(cfg)); restore_rng(base_rng); m.add_endpoint('B',cfg['endpoints']['B'])
        m,w=_run_joint(m,cfg,out,stage+'_warm',{'A':a},'B',cycles=cfg['training'].get('head_updates',100),phase='head',supervised=supervised,initial_rng=base_rng)
        wp=_save(out,stage+'_W',m,cfg,{'A':a},w)
        m,c=_run_joint(m,cfg,out,stage+'_joint',{'W':wp},'B',['A'],supervised=supervised)
        cost_branch={'warm':w,'joint':c}; path=_save(out,stage,m,cfg,{'A':a},cost_branch); branches[name]=(m,path,cost_branch)
    minus,mp,mc=branches['minus']; plus,p,pc=branches['plus']
    baseline=_eval(model,cfg,['A'])['A']; minus_eval=_eval(minus,cfg,['A'])['A']; plus_eval=_eval(plus,cfg,['A'])['A']; _json(Path(out)/'A_baseline_predictions.json',baseline); _json(Path(out)/'discovery_predictions.json',{'parent':baseline,'minus':minus_eval,'plus':plus_eval}); gain=build_gain_archive(minus_eval,plus_eval,baseline); _json(Path(out)/'gain.json',gain)
    return plus,p,gain,{'A':cost,'Bminus':mc,'Bplus':pc}
def train_conditions(parent, parent_path, cfg, out, endpoint, old_endpoints, gain, *, diagnostic=False, supervised=True, methods=METHODS):
    device=engine.device_for(cfg); base=engine.clone_model(parent,device); base.add_endpoint(endpoint,cfg['endpoints'][endpoint]); base,warm=_run_joint(base,cfg,out,endpoint+'_warm',{'parent':parent_path},endpoint,cycles=cfg['training'].get('head_updates',100),phase='head',supervised=supervised)
    w_rng=rng_state(); wpath=_save(out,endpoint+'_W',base,cfg,{'parent':parent_path},warm)
    # P reference is the live state at W: head warmup never changes spatial modules.
    reference=base.current_live_state(); q=engine.clone_model(base,device)
    q,qcost=_run_joint(q,cfg,out,endpoint+'_probe',{'W':wpath},endpoint,old_endpoints,cycles=cfg['training'].get('probe_cycles',60),full_cycles=cfg['training'].get('joint_cycles',600),initial_rng=w_rng)
    qp=_save(out,endpoint+'_probe',q,cfg,{'W':wpath},qcost)
    start=time.perf_counter(); scores=engine.restore_and_score(q,reference,cfg,old_endpoints,endpoint,gain); selection_seconds=time.perf_counter()-start
    _json(Path(out)/(endpoint+'_scores.json'),scores)
    selected=select(scores); costs={'warm':warm,'probe':qcost,'selection_seconds':selection_seconds}; outputs={}; matrix=[]
    for method in methods:
        m=engine.clone_model(base,device); restore_rng(w_rng)
        if method in ('gain_selected2','drop_a_selected2','random2'):
            blocks=select(scores,method); m.split_for_new_task(endpoint,blocks)
        elif method=='same_selected2_frozen_no_copy':
            blocks=selected; m.freeze_selected(blocks)
        elif method=='private_delta_old_route_frozen':
            blocks=[]; m.add_distributed_delta(endpoint,freeze_old=True)
        else: blocks=[]
        m,cost=_run_joint(m,cfg,out,endpoint+'_'+method,{'W':wpath,'probe':qp},endpoint,old_endpoints,supervised=supervised,old_replay=method!='private_delta_old_route_frozen',initial_rng=w_rng)
        path=_save(out,endpoint+'_'+method,m,cfg,{'W':wpath,'probe':qp},{'blocks':blocks,'cost':cost})
        t=time.perf_counter(); metrics=_eval(m,cfg,list(old_endpoints)+[endpoint]); cost['evaluation_seconds']=time.perf_counter()-t
        report=m.parameter_report(); metrics['A']['gain_groups']=analysis.group_metrics(metrics['A']['records'],gain); _json(Path(out)/(endpoint+'_'+method+'_metrics.json'),metrics)
        outputs[method]=(m,path); costs[method]=cost
        matrix.append({'endpoint':endpoint,'method':method,'blocks':blocks,'metrics':{e:v['metrics'] for e,v in metrics.items()},'parameters':report,'cost':cost})
        if method=='shared':
            t=time.perf_counter(); full=engine.restore_and_score(m,reference,cfg,old_endpoints,endpoint,gain); costs['full_restoration_seconds']=time.perf_counter()-t; _json(Path(out)/(endpoint+'_full_restoration.json'),full)
    _json(Path(out)/(endpoint+'_stage_matrix.json'),matrix); _json(Path(out)/(endpoint+'_costs.json'),costs)
    return outputs, matrix, costs

def run_prototype(cfg, output_dir):
    started=time.perf_counter(); before=dict(engine._COSTS); out=Path(output_dir); plus,p,gain,costs=_begin(cfg,out)
    outputs,matrix,c=train_conditions(plus,p,cfg,out,'C',['A','B'],gain)
    costs['C']=c; costs['actual_wall_seconds']=time.perf_counter()-started; costs['normalization']={k:engine._COSTS[k]-before[k] for k in before}; _json(out/'resource_ledger.json',costs)
    return {'outputs':{k:v[1] for k,v in outputs.items()},'stage_matrix':matrix,'costs':costs}

def _seed_cfg(cfg, seed):
    c=copy.deepcopy(cfg); c['training']['seed']=seed; return c

def run_discovery(cfg,output_dir):
    started=time.perf_counter(); before=dict(engine._COSTS); out=Path(output_dir); results={}
    for seed in (6101,6102,6103,6104):
        _,p,_,costs=_begin(_seed_cfg(cfg,seed),out/str(seed)); results[str(seed)]={'Bplus':p,'costs':costs}
    archives=[json.loads((out/str(seed)/'gain.json').read_text(encoding='utf-8')) for seed in (6101,6102,6103,6104)]
    maps=[{r['key']:r for r in a['records']} for a in archives]
    if any(m.keys()!=maps[0].keys() for m in maps[1:]): raise ValueError('Discovery A entities differ across seeds')
    rows=[]
    for k,r in maps[0].items():
        row=copy.deepcopy(r); row['gain']=sum(m[k]['gain'] for m in maps)/len(maps); row['difficulty']=sum(m[k]['difficulty'] for m in maps)/len(maps); rows.append(row)
    ranked=sorted(rows,key=lambda r:(r['gain'],r['key']))
    for i,r in enumerate(ranked): r['group']='low' if i<len(ranked)//4 else ('high' if i>=len(ranked)-len(ranked)//4 else 'middle')
    _json(out/'discovery_gain.json',{'records':rows,'seeds':[6101,6102,6103,6104]}); predictions={str(seed):json.loads((out/str(seed)/'discovery_predictions.json').read_text(encoding='utf-8')) for seed in (6101,6102,6103,6104)}; _json(out/'discovery_statistics.json',analysis.discovery_summary(archives,predictions)); _json(out/'discovery_results.json',results)
    _json(out/'resource_ledger.json',{'actual_wall_seconds':time.perf_counter()-started,'normalization':{k:engine._COSTS[k]-before[k] for k in before},'per_seed':{k:v['costs'] for k,v in results.items()}})
    return results

def _paired_inference(out,seeds):
    result={}
    for endpoint in ('A','C'):
        values={}
        for seed in seeds:
            directory=Path(out)/str(seed)
            a=json.loads((directory/'C_gain_selected2_metrics.json').read_text(encoding='utf-8'))[endpoint]['records']
            b=json.loads((directory/'C_drop_a_selected2_metrics.json').read_text(encoding='utf-8'))[endpoint]['records']
            left={r['key']:r['loss'] for r in a}; right={r['key']:r['loss'] for r in b}
            if left.keys()!=right.keys(): raise ValueError('Paired method entity keys differ')
            values[seed]={key:left[key]-right[key] for key in left}
        keys=sorted(values[seeds[0]])
        if any(set(values[s])!=set(keys) for s in seeds): raise ValueError('Confirmation entities differ across seeds')
        means={str(s):sum(values[s].values())/len(keys) for s in seeds}
        rng=random.Random(6031); samples=[]
        for _ in range(2000):
            ss=[rng.choice(seeds) for _ in seeds]; kk=[rng.choice(keys) for _ in keys]
            samples.append(sum(values[s][k] for s in ss for k in kk)/(len(ss)*len(kk)))
        samples.sort()
        result[endpoint]={'metric':'loss: Brier' if endpoint=='A' else 'loss: 1-Dice','contrast':'gain_selected2 minus drop_a_selected2','per_seed':means,'mean':sum(means.values())/len(means),'paired_seed_entity_bootstrap_95':[samples[49],samples[1949]],'bootstrap_replicates':2000}
    return result
def run_confirmation(cfg,output_dir):
    started=time.perf_counter(); before=dict(engine._COSTS); out=Path(output_dir); source=Path(cfg.get('discovery_archive',out.parent/'discovery'/'discovery_gain.json'))
    gain=json.loads(source.read_text(encoding='utf-8')); results={}
    for seed in (6201,6202,6203,6204,6205):
        c=_seed_cfg(cfg,seed); c['discovery_archive']=str(source); directory=out/str(seed); plus,p,_,cost=_begin(c,directory)
        outputs,matrix,cc=train_conditions(plus,p,c,directory,'C',['A','B'],gain); cost['C']=cc; cost['discovery_archive_sha256']=file_hash(source); _json(directory/'resource_ledger.json',cost)
        diagnostic=run_diagnostic(c,directory); results[str(seed)]={'outputs':{k:v[1] for k,v in outputs.items()},'stage_matrix':matrix,'diagnostic':diagnostic}
    effects={seed:json.loads((out/str(seed)/'diagnostic_effects.json').read_text(encoding='utf-8'))['entities'] for seed in (6201,6202,6203,6204,6205)}; _json(out/'diagnostic_statistics.json',analysis.diagnostic_summary(effects,gain))
    _json(out/'paired_inference.json',_paired_inference(out,[6201,6202,6203,6204,6205])); _json(out/'confirmation_results.json',results); _json(out/'resource_ledger.json',{'actual_wall_seconds':time.perf_counter()-started,'normalization':{k:engine._COSTS[k]-before[k] for k in before},'per_seed':{k:{'stage_matrix':v['stage_matrix'],'diagnostic':v['diagnostic']} for k,v in results.items()}}); return results

def run_diagnostic(cfg,output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    plus,pp,_,_= _begin(cfg,out)
    a_path=str(out/'A_parent.pt'); minus,mp,_=_done(out,'Bminus',cfg,{'A':a_path})
    source=Path(cfg.get('discovery_archive',out.parent/'discovery'/'discovery_gain.json')); gain=json.loads(source.read_text(encoding='utf-8')); _json(out/'gain.json',gain); rows=[]; branch_rng=rng_state(); parent_records={'minus':_eval(minus,cfg,['A'])['A']['records'],'plus':_eval(plus,cfg,['A'])['A']['records']}; by_arm={}
    for b,parent,path in [('minus',minus,mp),('plus',plus,pp)]:
        for supervised in (False,True):
            m=engine.clone_model(parent,engine.device_for(cfg)); restore_rng(branch_rng); m.add_endpoint('C',cfg['endpoints']['C']); m,w=_run_joint(m,cfg,out,f'diag_B{b}_C{int(supervised)}_warm',{'B':path},'C',cycles=cfg['training'].get('head_updates',100),phase='head',supervised=supervised,initial_rng=branch_rng)
            wp=_save(out,f'diag_B{b}_C{int(supervised)}_W',m,cfg,{'B':path},w)
            m,c=_run_joint(m,cfg,out,f'diag_B{b}_C{int(supervised)}_joint',{'W':wp},'C',['A'],supervised=supervised)
            cp=_save(out,f'diag_B{b}_C{int(supervised)}',m,cfg,{'W':wp},c)
            metrics=_eval(m,cfg,['A','C']+(['B'] if b=='plus' else [])); by_arm[(b,supervised)]={r['key']:r['loss'] for r in metrics['A']['records']}; rows.append({'B':b,'C_supervised':supervised,'checkpoint':cp,'metrics':{k:v['metrics'] for k,v in metrics.items()},'cost':{'warm':w,'joint':c}})
    x={b:{r['key']:r['loss'] for r in recs} for b,recs in parent_records.items()}; groups={r['key']:r['group'] for r in gain['records']}; effects=[]
    for key in sorted(by_arm[('plus',True)]):
        d0=by_arm[('minus',True)][key]-by_arm[('minus',False)][key]; d1=by_arm[('plus',True)][key]-by_arm[('plus',False)][key]
        effects.append({'key':key,'group':groups.get(key),'x0':x['minus'][key],'x1':x['plus'][key],'y00':by_arm[('minus',False)][key],'y01':by_arm[('minus',True)][key],'y10':by_arm[('plus',False)][key],'y11':by_arm[('plus',True)][key],'d0':d0,'d1':d1,'h':d1-d0,'absolute_harm':by_arm[('plus',True)][key]-x['plus'][key],'advantage_erosion':(x['minus'][key]-x['plus'][key])-(by_arm[('minus',True)][key]-by_arm[('plus',True)][key])})
    _json(out/'diagnostic_effects.json',{'discovery_sha256':file_hash(source),'entities':effects}); _json(out/'diagnostic_matrix.json',rows); return rows

def run_continual(cfg,output_dir):
    order=list(cfg.get('task_order',['A','B','C','D','E','F']))
    if len(order)<3 or order[:2]!=['A','B'] or len(order)!=len(set(order)) or set(order)-cfg['endpoints'].keys():
        raise ValueError('Task order must start A,B and contain unique registered endpoints')
    started=time.perf_counter(); before=dict(engine._COSTS); out=Path(output_dir); plus,p,gain,costs=_begin(cfg,out/'initial')
    if cfg.get('discovery_archive'):
        source=Path(cfg['discovery_archive']);gain=json.loads(source.read_text(encoding='utf-8'))
        costs['gain_source']={'kind':'independent_discovery','sha256':file_hash(source),'path':str(source)}
    else:costs['gain_source']={'kind':'same_seed_engineering_prototype'}
    first=order[2];outputs,matrix,cc=train_conditions(plus,p,cfg,out/first,first,['A','B'],gain)
    allrows=list(matrix);costs[first]=cc;costs['continual']={};costs['task_order']=order
    for index,endpoint in enumerate(order[3:],3):
        old=order[:index]
        next_outputs={}; costs['continual'][endpoint]={}
        for method,(parent,path) in outputs.items():
            # Each method continues from its own preceding checkpoint and applies the same rule anew.
            candidates,rows,stage_cost=train_conditions(parent,path,cfg,out/endpoint/method,endpoint,old,gain,methods=(method,)); costs['continual'][endpoint][method]=stage_cost
            next_outputs[method]=candidates[method]; allrows.extend(r for r in rows if r['method']==method)
        outputs=next_outputs
    costs['actual_wall_seconds']=time.perf_counter()-started; costs['normalization']={k:engine._COSTS[k]-before[k] for k in before}; _json(out/'resource_ledger.json',costs); _json(out/'continual_stage_matrix.json',allrows); return {'outputs':{k:v[1] for k,v in outputs.items()},'stage_matrix':allrows}

def evaluate_checkpoint(path,cfg,output_dir,endpoints=None):
    model,payload=load_checkpoint(path); saved=copy.deepcopy(payload['config']); current=copy.deepcopy(cfg)
    for key in ('allow_final_evaluation','evaluation_split','output_dir'):
        saved.pop(key,None); current.pop(key,None)
    saved.get('training',{}).pop('device',None); current.get('training',{}).pop('device',None)
    if saved != current: raise ValueError('Evaluation config differs from checkpoint training config')
    for key,expected in payload['manifests'].items():
        ep,split=key.split(':',1); actual=_manifest(cfg,ep,split)
        if actual != expected: raise ValueError('Manifest fingerprint mismatch: '+key)
    eps=endpoints or list(model.export_spec()['endpoints']); t=time.perf_counter(); results=_eval(model,cfg,eps,cfg.get('evaluation_split','dev'))
    report={'checkpoint':str(path),'checkpoint_sha256':file_hash(path),'results':results,'seconds':time.perf_counter()-t}; _json(Path(output_dir)/'evaluation.json',report); return report




















