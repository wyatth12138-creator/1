import random

BLOCKS = tuple(range(3,16))

def build_gain_archive(minus, plus, baseline=None):
    left = {r['key']: r for r in minus['records']}; right = {r['key']: r for r in plus['records']}
    if left.keys() != right.keys(): raise ValueError('Paired predictions must have identical entity keys')
    baselines = {r['key']:r for r in baseline['records']} if baseline else left
    if baselines.keys()!=left.keys(): raise ValueError('Baseline entity keys differ')
    rows = []
    for key in sorted(left):
        m, p = left[key], right[key]
        if m['label'] != p['label']: raise ValueError('Paired labels differ')
        y = m['label']; g = (m['probability']-y)**2-(p['probability']-y)**2
        rows.append({'key': key, 'label': y, 'difficulty': baselines[key].get('loss'), 'gain': g})
    ordered = sorted(rows, key=lambda r: (r['gain'], r['key']))
    n = len(ordered)//4
    for i, row in enumerate(ordered): row['group'] = 'low' if i < n else ('high' if i >= len(ordered)-n else 'middle')
    return {'records': rows}

def weighted_difference(records, gains):
    # Stratify by class and baseline difficulty; only matched strata contribute.
    gm = {r['key']: r for r in gains['records']}; strata = {}
    difficulties = sorted(r['difficulty'] for r in gains['records'] if r['difficulty'] is not None)
    cuts = [difficulties[min(len(difficulties)-1,len(difficulties)*k//3)] for k in (1,2)] if difficulties else [0,0]
    for r in records:
        g = gm.get(r['key'])
        if g is None: continue
        if g['group'] not in ('high','low'): continue
        k = (g['label'], sum((g['difficulty'] or 0) > cut for cut in cuts))
        strata.setdefault(k, {'high': [], 'low': []})[g['group']].append(r['recovery'])
    supported = [v for v in strata.values() if v['high'] and v['low']]
    if not supported: return 0.0, False
    total = sum(min(len(v['high']),len(v['low'])) for v in supported)
    a = sum(min(len(v['high']),len(v['low'])) * (sum(v['high'])/len(v['high'])-sum(v['low'])/len(v['low'])) for v in supported)/total
    return a, True

def select(scores, method='gain_selected2', seed=6001):
    if method == 'random2': return sorted(random.Random(seed).sample(list(BLOCKS), 2))
    key = 'score_no_a' if method == 'drop_a_selected2' else 'score'
    return sorted([r['block'] for r in sorted(scores, key=lambda x: (-x[key], x['block']))[:2]])

def score_row(block, recovery, a, supported):
    values = list(recovery.values())
    if not values: raise ValueError('At least one old endpoint is required')
    base = sum(v/0.02 for v in values)/len(values) - max(max(-v/0.02, 0) for v in values)
    return {'block': block, 'old_recovery': recovery, 'a': a, 'common_support': supported, 'score': a/0.02 + base, 'score_no_a': base}


