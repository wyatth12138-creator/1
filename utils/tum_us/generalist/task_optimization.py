"""Conflict optimization for independently labelled task families.

PCGrad on shared parameters; ConsMTL-inspired mixed-derivative private-head
alignment in common PARAMETER coordinates, not unrelated sample features.
This is an unpaired adaptation, not a reproduction of original ConsMTL.
"""
from contextlib import contextmanager
import torch


def dot(a,b):
    terms=[(x.float()*y.float()).sum() for x,y in zip(a,b) if x is not None and y is not None]
    return sum(terms) if terms else torch.tensor(0.)


def combine_shared(a,b,method='mean'):
    if method not in ('mean','pcgrad','consistent'): raise ValueError('Unknown shared optimizer')
    # Only project tensors present in BOTH computation graphs. No new gradient
    # is introduced into a parameter unused by a supervised task.
    overlap=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    if not overlap: raise ValueError('No jointly used shared parameter tensors')
    ab=sum((x*y).sum() for x,y in overlap)
    aa=sum(x.square().sum() for x,y in overlap); bb=sum(y.square().sum() for x,y in overlap)
    conflict=bool(ab<0)
    ca=(ab/bb.clamp_min(1e-20)) if conflict and method!='mean' else ab*0
    cb=(ab/aa.clamp_min(1e-20)) if conflict and method!='mean' else ab*0
    result=[]
    for x,y in zip(a,b):
        if x is None: result.append(None if y is None else y*.5)
        elif y is None: result.append(x*.5)
        else: result.append(((x-ca*y)+(y-cb*x))*.5)
    return result,{'conflict':conflict,'cosine':float(ab/(aa*bb).sqrt().clamp_min(1e-20))}


def private_correction(loss,probe,private,direction,main,weight=1.,max_ratio=.1):
    """Return head-only correction; main shared gradients are never overwritten."""
    if weight<0 or not 0<=max_ratio<=1: raise ValueError('Invalid private alignment settings')
    g=torch.autograd.grad(loss,probe,create_graph=True,retain_graph=True,allow_unused=True)
    paired=[(x,d.detach()) for x,d in zip(g,direction) if x is not None and d is not None]
    if not paired: raise ValueError('Alignment probe not used by this task')
    numerator=sum((x*d).sum() for x,d in paired)
    gn=sum(x.square().sum() for x,d in paired).clamp_min(1e-24).sqrt()
    dn=sum(d.square().sum() for x,d in paired).clamp_min(1e-24).sqrt()
    objective=-float(weight)*numerator/(gn*dn)
    extra=torch.autograd.grad(objective,private,allow_unused=True)
    correction=[None if x is None else x.detach().float() for x in extra]
    norm=dot(correction,correction).sqrt(); base=dot(main,main).sqrt().to(norm.device)
    factor=(max_ratio*base/norm.clamp_min(1e-20)).clamp(max=1.)
    correction=[None if x is None else x*factor for x in correction]
    return correction,{'alignment_cosine':float((-objective/max(weight,1e-20)).detach()),
                       'private_correction_norm':float(norm*factor),'private_main_norm':float(base)}


@contextmanager
def higher_order_attention():
    # Fused CUDA SDPA kernels need not implement double backward. Do not silently
    # substitute a detached first-order proxy for the mixed derivative.
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH): yield


def rank_and_cover(keys,uncertainties,features,budget,pool_factor=4):
    """Own-task uncertainty then normalized k-center. NO conflict penalty/CTMI."""
    n=len(keys)
    if n!=len(set(keys)) or n!=len(uncertainties) or features.ndim!=2 or n!=len(features):
        raise ValueError('Selection inputs must have unique aligned entity keys')
    if not 0<budget<=n or pool_factor<1: raise ValueError('Invalid selection budget')
    u=torch.as_tensor(uncertainties,dtype=torch.float32)
    if not torch.isfinite(u).all() or not torch.isfinite(features).all(): raise ValueError('Nonfinite selection inputs')
    ranked=sorted(range(n),key=lambda i:(-float(u[i]),str(keys[i])))[:min(n,budget*pool_factor)]
    f=torch.nn.functional.normalize(features.float(),dim=-1)
    chosen=[ranked[0]]; remaining=set(ranked[1:]); distances=torch.full((n,),float('inf'),device=f.device)
    while len(chosen)<budget:
        distances=torch.minimum(distances,(f-f[chosen[-1]]).square().sum(-1))
        next_id=min(remaining,key=lambda i:(-float(distances[i]),str(keys[i])))
        chosen.append(next_id); remaining.remove(next_id)
    return chosen
