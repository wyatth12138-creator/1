"""Paired, source-locked summaries for the transfer study."""
import hashlib, random
import numpy as np
from .selection import weighted_difference

def group_metrics(records, gain):
    groups={r['key']:r['group'] for r in gain['records']}
    out={}
    for group in ('high','low'):
        vals=[r['loss'] for r in records if groups.get(r['key'])==group]
        out[group]={'n':len(vals),'brier':sum(vals)/len(vals) if vals else None}
    return out

def _ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); ranks=[0.]*len(order); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        rank=(i+j-1)/2
        for k in order[i:j]: ranks[k]=rank
        i=j
    return np.array(ranks)

def discovery_summary(archives,predictions):
    maps=[{r['key']:r for r in a['records']} for a in archives]; keys=sorted(maps[0])
    if any(set(m)!=set(keys) for m in maps): raise ValueError('Discovery keys differ')
    a=[(maps[0][k]['gain']+maps[1][k]['gain'])/2 for k in keys]; b=[(maps[2][k]['gain']+maps[3][k]['gain'])/2 for k in keys]
    ar,br=_ranks(a),_ranks(b)
    corr=float(np.corrcoef(ar,br)[0,1]) if len(keys)>1 and ar.std()>0 and br.std()>0 else None
    changes=[]
    for seed,p in predictions.items():
        base={r['key']:r['loss'] for r in p['parent']['records']}
        row={'seed':seed}
        for arm in ('minus','plus'):
            rec=p[arm]['records']
            if {r['key'] for r in rec} != set(base): raise ValueError('Paired discovery records differ')
            row[arm+'_absolute_brier_change']=sum(r['loss']-base[r['key']] for r in rec)/len(rec)
        changes.append(row)
    return {'split_half_spearman':corr,'split_seeds':[[6101,6102],[6103,6104]],'absolute_changes':changes}

def crossfit_effects(effects,gain):
    meta={r['key']:r for r in gain['records']}; rows=[]
    for r in effects:
        key=r['key']; fold=int(hashlib.sha256(key.encode()).hexdigest(),16)%2
        for arm in (0,1):
            x=r['x'+str(arm)]; y=meta[key]['label']; d=r['d'+str(arm)]
            rows.append((key,arm,fold,np.array([1.,x,x*x,float(y)]),d))
    predictions={}
    for fold in (0,1):
        train=[r for r in rows if r[2]!=fold]; test=[r for r in rows if r[2]==fold]
        if not train or not test: continue
        beta=np.linalg.lstsq(np.stack([r[3] for r in train]),np.array([r[4] for r in train]),rcond=None)[0]
        for key,arm,_,features,d in test: predictions[(key,arm)]=d-float(features@beta)
    return {r['key']:predictions[(r['key'],1)]-predictions[(r['key'],0)] for r in effects if (r['key'],0) in predictions and (r['key'],1) in predictions}

def diagnostic_summary(per_seed,gain,replicates=1000):
    values={}; residuals={}; summaries={}
    for seed,rows in per_seed.items():
        values[seed]={r['key']:r['h'] for r in rows}; residuals[seed]=crossfit_effects(rows,gain)
        raw,support=weighted_difference([{'key':k,'recovery':v} for k,v in values[seed].items()],gain)
        resid,rsupport=weighted_difference([{'key':k,'recovery':v} for k,v in residuals[seed].items()],gain)
        summaries[str(seed)]={'high_minus_low_h':raw,'common_support':support,'crossfit_residual_high_minus_low':resid,'residual_common_support':rsupport}
    seeds=list(values); keys=sorted(values[seeds[0]])
    if any(set(values[s])!=set(keys) for s in seeds): raise ValueError('Diagnostic keys differ by seed')
    rng=random.Random(6032); boots=[]
    for _ in range(replicates):
        ss=[rng.choice(seeds) for _ in seeds]; kk=[rng.choice(keys) for _ in keys]; sample=[]
        for s in ss:
            v,support=weighted_difference([{'key':k,'recovery':values[s][k]} for k in kk],gain)
            if support: sample.append(v)
        if len(sample)==len(ss): boots.append(sum(sample)/len(sample))
    boots.sort(); interval=[boots[int(.025*(len(boots)-1))],boots[int(.975*(len(boots)-1))]] if boots else None
    supported=[r['high_minus_low_h'] for r in summaries.values() if r['common_support']]
    return {'per_seed':summaries,'mean_high_minus_low_h':sum(supported)/len(supported) if len(supported)==len(seeds) else None,'paired_seed_entity_bootstrap_95':interval,'valid_bootstrap_replicates':len(boots),'conditional_analysis':'two-fold entity-paired, source-blind shared least-squares response in x, x² and label; descriptive, not a total effect'}
