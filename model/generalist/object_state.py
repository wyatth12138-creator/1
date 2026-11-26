"""Small query/pixel bridge inspired by VidEoMT and Cutie's object abstractions.

This is a SAM2 decoder-side adaptation, not an implementation of either full
architecture. All history is supplied by the caller's bounded memory bank.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def predicted_object_tokens(feature,mask_logits):
    """Return [Q,2,C] foreground/background means, using predictions only."""
    if feature.ndim!=3 or mask_logits.ndim!=3:
        raise ValueError('Expected feature [C,H,W] and predicted masks [Q,H0,W0]')
    if not torch.isfinite(feature).all() or not torch.isfinite(mask_logits).all():
        raise ValueError('Object pooling requires finite predictions and features')
    probabilities=F.interpolate(mask_logits.float().sigmoid()[:,None],size=feature.shape[-2:],mode='bilinear',align_corners=False)[:,0]
    weights=torch.stack([probabilities,1-probabilities],dim=1)
    sums=torch.einsum('qkhw,chw->qkc',weights,feature.float())
    return sums/weights.sum((-2,-1),keepdim=False)[...,None].clamp_min(1e-6)


def view_pointers(slot,view):
    return slot.pointers[view] if slot.pointers.ndim==3 else slot.pointers


def view_objects(slot,view):
    value=getattr(slot,'object_tokens',None)
    if value is None: return None
    return value[view] if value.ndim==4 else value


class ObjectStateBridge(nn.Module):
    """Zero-residual initialization preserves the existing reader at step zero.

    Two flags define four reader arms. Unused modules are not instantiated.
    There is no hidden last-query cache, EMA bank or per-patient mutable state.
    """
    def __init__(self,query_propagation=False,object_memory=False,dim=256):
        super().__init__()
        self.query_propagation=bool(query_propagation); self.object_memory=bool(object_memory)
        self.dim=dim
        if self.query_propagation:
            self.query_norm=nn.LayerNorm(dim)
            self.pointer_proj=nn.Linear(dim,dim)
            self.query_gate=nn.Linear(dim*2+2,1)
            self.query_gain=nn.Parameter(torch.zeros(()))
        if self.object_memory:
            self.object_norm=nn.LayerNorm(dim)
            self.attention=nn.MultiheadAttention(dim,4,dropout=0.,batch_first=True)
            self.spatial_proj=nn.Linear(dim,dim,bias=False)
            self.class_proj=nn.Linear(dim,dim,bias=False)
            self.spatial_gain=nn.Parameter(torch.zeros(()))
            self.class_gain=nn.Parameter(torch.zeros(()))

    def propagate(self,query,slots,view,frame):
        if not self.query_propagation or not slots: return query
        last=max(slots,key=lambda s:s.frame_index)
        previous=view_pointers(last,view)
        if previous.shape!=query.shape: raise ValueError('Object query slots changed within the sequence')
        # Only model-predicted quality and age enter the gate, never GT overlap.
        age=query.new_full((len(query),1),math.log1p(max(frame-last.frame_index,0))/10)
        quality=last.quality.float().reshape(1,1).expand(len(query),1).to(query)
        historical=self.pointer_proj(previous.to(query))
        gate=torch.sigmoid(self.query_gate(torch.cat([self.query_norm(query),self.query_norm(historical),age,quality],-1)))
        return query+self.query_gain.tanh()*gate*historical

    def object_context(self,query,slots,view,frame):
        valid=[s for s in slots if view_objects(s,view) is not None]
        if not self.object_memory or not valid: return None
        tokens=torch.stack([view_objects(s,view) for s in valid]) # N,Q,2,C
        if tokens.shape[1:]!=(len(query),2,self.dim): raise ValueError('Object summary/query identities disagree')
        age=query.new_tensor([max(frame-s.frame_index,0) for s in valid])
        # Per-object attention over retained times; foreground and background
        # remain separate channels rather than collapsing to global image mean.
        normalized_age=age/(age.max().clamp_min(1))
        frequencies=torch.exp(torch.arange(self.dim//2,device=query.device).float()*(-9.21034/max(self.dim//2-1,1)))
        phase=normalized_age[:,None]*frequencies[None]
        position=torch.cat([phase.sin(),phase.cos()],-1).to(query)
        # Fuse the two independent foreground/background reads in the batch
        # dimension. They share weights but cannot attend across this dimension.
        values=tokens.permute(2,1,0,3).reshape(2*len(query),len(valid),self.dim).to(query)
        q=self.object_norm(query)[:,None].repeat(2,1,1)
        keys=self.object_norm(values)+position[None]
        contexts,_=self.attention(q,keys,values,need_weights=False)
        foreground,background=contexts[:,0].chunk(2,dim=0)
        return foreground-background

    def enhance(self,query,fused,slots,view,frame):
        query=self.propagate(query,slots,view,frame)
        contrast=self.object_context(query,slots,view,frame)
        if contrast is not None:
            evidence=self.spatial_proj(contrast).to(fused)
            similarity=torch.einsum('qchw,qc->qhw',F.normalize(fused.float(),dim=1),F.normalize(evidence.float(),dim=-1))
            fused=fused+self.spatial_gain.tanh()*similarity.sigmoid()[:,None]*evidence[:,:,None,None]
        return query,fused

    def class_token(self,current,object_tokens):
        if not self.object_memory: return current
        # Equal averaging across supplied valid views/targets. No annotation is
        # used to choose a view or suppress a predicted object.
        contrast=torch.stack([value[:,0]-value[:,1] for value in object_tokens]).mean((0,1))
        return current+self.class_gain.tanh()*self.class_proj(contrast.to(current))
