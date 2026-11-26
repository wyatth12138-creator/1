"""Native TUM readers and one learned, contextual memory controller."""
from __future__ import annotations

from dataclasses import asdict,replace
import torch

from ..model.memory import MemoryBank
from ..model.network import TUMModel
from .controller import ContextualController


class GeneralistModel(TUMModel):
    def __init__(self, model_config: dict, controller_config: dict | None = None):
        model_config = {"joint_ceus_memory": False, "policy": "fifo", **model_config}
        self_private = model_config.pop('private_seg_calibration', False)
        if type(self_private) is not bool:
            raise ValueError('private_seg_calibration must be boolean')
        object_options={key:model_config.pop(key,False) for key in ('query_propagation','object_memory')}
        self.support_rule = model_config.pop('support_rule', None)
        support_profiles = model_config.pop('support_profiles', ())
        self.input_conditioning = model_config.pop('input_conditioning', None)
        if self.support_rule not in (None, 'fixed', 'shared'):
            raise ValueError('Reader support rule must be fixed or shared')
        if self.input_conditioning is not None and self.support_rule is not None:
            raise ValueError('Choose either support profiles or input conditioning')
        if any(type(v) is not bool for v in object_options.values()):
            raise ValueError('Object reader flags must be boolean')
        super().__init__(model_config)
        if self.support_rule is not None:
            from .support_metric import SupportMetric
            self.support_profile_indices = {
                tuple(key): index for index, key in enumerate(support_profiles)
            }
            if not self.support_profile_indices or len(self.support_profile_indices) != len(support_profiles):
                raise ValueError('Expected unique support profiles for the reader')
            self.support_metric = SupportMetric(variant=self.support_rule)
            self.register_buffer('support_prototypes', torch.zeros(len(support_profiles), 2, 256))
        self.object_options=object_options
        self.private_seg_calibration = self_private
        if self_private:
            self.seg_log_scale = torch.nn.Parameter(torch.zeros(self.config.num_targets))
            self.seg_bias = torch.nn.Parameter(torch.zeros(self.config.num_targets))
        self.controller_config = {"semantic_dropout": 0.2, "unseen_semantics_to_unknown": True,
                                  **(controller_config or {})}
        options = {"num_tasks": self.config.num_tasks, "num_anatomies": self.config.num_anatomies,
                   "num_targets": self.config.num_targets, "num_label_spaces": self.config.num_label_spaces,
                   **self.controller_config}
        self.controller = ContextualController(**options)
        if any(object_options.values()):
            from .object_state import ObjectStateBridge
            self.object_bridge=ObjectStateBridge(**object_options)
        if self.input_conditioning is not None:
            from .input_conditioning import InputConditioning
            self.feature_conditioning = InputConditioning(self.input_conditioning)
        self._training_phase = "readers"
        self.configure_readers()

    @property
    def model_config(self):
        config = {**asdict(self.config), **self.object_options}
        if self.private_seg_calibration:
            config['private_seg_calibration'] = True
        if self.support_rule is not None:
            config['support_rule'] = self.support_rule
            config['support_profiles'] = [list(key) for key in self.support_profile_indices]
        if self.input_conditioning is not None:
            config['input_conditioning'] = self.input_conditioning
        return config

    def _encode(self, media):
        features = super()._encode(media)
        if self.input_conditioning is not None:
            features = self.feature_conditioning.transform_features(features)
        return features

    def _read(self,features,view,targets,task,anatomy,label_space,frame,slots,view_valid):
        result=super()._read(features,view,targets,task,anatomy,label_space,frame,slots,view_valid)
        from ..data.registry import TASK_TO_ID
        segmentation={TASK_TO_ID[x] for x in ('image_seg','video_seg','ceus_seg')}
        if self.private_seg_calibration and int(task) in segmentation:
            masks=result[0]*self.seg_log_scale[targets].clamp(-3,3).exp()[:,None,None]+self.seg_bias[targets,None,None]
            result=(masks,*result[1:])
        return result

    def multitask_parameter_groups(self):
        """Private means absent from the OTHER supervised family's computation graph."""
        groups={'shared':[], 'classification':[], 'segmentation':[]}
        for name,p in self.named_parameters():
            if not p.requires_grad: continue
            if name.startswith('controller.'):
                raise ValueError('Configure readers before partitioning parameters')
            if name in ('seg_log_scale','seg_bias'): group='segmentation'
            elif name.startswith('global_readout.') or name.startswith('object_bridge.class_proj.') or name=='object_bridge.class_gain': group='classification'
            else: group='shared'
            groups[group].append((name,p))
        return groups

    def _prepare_object_read(self,query,fused,slots,view,frame):
        if not hasattr(self,'object_bridge'): return query,fused
        return self.object_bridge.enhance(query,fused,slots,view,frame)

    def _read_views(self,features,targets,task,anatomy,label_space,frame,slots,view_valid):
        if self.support_rule is not None:
            from .support_metric import support_key
            key = support_key(task, anatomy, label_space, targets.tolist())
            index = self.support_profile_indices[key]
            prototypes = self.support_prototypes[index]
            scales = self.support_metric.channel_scales(prototypes, task=None)
            pyramid = features['backbone_fpn']
            top = pyramid[-1]
            weighted_top = top * scales.to(top.dtype)[None, :, None, None]
            features = {**features, 'backbone_fpn': [*pyramid[:-1], weighted_top]}
        results=super()._read_views(features,targets,task,anatomy,label_space,frame,slots,view_valid)
        if not self.object_options['object_memory']: return results
        from .object_state import predicted_object_tokens
        summaries=[predicted_object_tokens(features['backbone_fpn'][-1][v],r[0]) for v,r in results.items()]
        current=next(iter(results.values()))[3]
        current=self.object_bridge.class_token(current,summaries)
        logits=self.global_readout(current,slots,label_space,frame)
        return {v:(r[0],r[1],r[2],current,logits) for v,r in results.items()}

    def _write_views(self,features,results,frame,anchor):
        slot=super()._write_views(features,results,frame,anchor)
        if not self.object_options['object_memory']: return slot
        from .object_state import predicted_object_tokens
        values=[predicted_object_tokens(features['backbone_fpn'][-1][v],r[0]) for v,r in results.items()]
        value=values[0] if len(values)==1 else torch.stack(values)
        if self.config.detach_memory: value=value.detach()
        return replace(slot,object_tokens=value)

    def configure_readers(self):
        self.requires_grad_(True)
        self.controller.requires_grad_(False)
        if self.config.freeze_trunk:
            self.sam.image_encoder.trunk.requires_grad_(False)
        if self.support_rule is not None or self.input_conditioning is not None:
            self.sam.image_encoder.requires_grad_(False)
        self._training_phase = "readers"
        self.train(self.training)
        return self

    def configure_controller(self):
        self.requires_grad_(False)
        self.controller.requires_grad_(True)
        self._training_phase = "controller"
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        phase = getattr(self, "_training_phase", "readers")
        if phase == "controller":
            for name, module in self.named_children():
                if name != "controller": module.eval()
            self.controller.train(mode)
        else:
            self.controller.eval()
        if self.support_rule is not None or self.input_conditioning is not None:
            self.sam.image_encoder.eval()
        return self

    def memory_context(self, task, anatomy, label_space, targets, capacity):
        target = int(targets[0]) if len(targets) == 1 else -1
        return torch.tensor([int(task), int(anatomy), int(label_space), target, int(capacity)],
                            dtype=torch.float32, device=self.image_mean.device)

    def _new_bank(self, task, anatomy, label_space, targets, capacity=None, policy=None, rng=None):
        capacity = self.config.memory_capacity if capacity is None else int(capacity)
        policy = self.config.policy if policy is None else policy
        context = self.memory_context(task, anatomy, label_space, targets, capacity)
        return MemoryBank(capacity, policy, self.controller, context=context, rng=rng)

    def set_memory_policy(self, policy, capacity=None):
        capacity = self.config.memory_capacity if capacity is None else int(capacity)
        MemoryBank(capacity, policy)  # validate before mutating model configuration
        self.config.policy, self.config.memory_capacity = policy, capacity
        return self

    def forward(self, batch, endpoint=None):
        # Endpoint names are evaluation grouping metadata, never controller input.
        return super().forward(batch)

    def utility_examples(self, batch, **kwargs):
        from .rollout import collect_examples
        return collect_examples(self, batch, **kwargs)
