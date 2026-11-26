"""Frozen native SAM2 with endpoint-private readers and versioned spatial adapters."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields, replace
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..data.registry import TASK_TO_ID
from ..model.memory import MemoryBank
from ..model.network import TUMConfig, TUMModel
from .adapters import BASE_RANKS, BLOCK_DIMS, BLOCK_INDICES, DELTA_RANKS, ResidualAdapter, RoutedHiera
from .routes import TaskRoutes


class TransferModel(TUMModel):
    """Load official weights first; keep all original SAM parameters immutable.

    Endpoint names are explicit on every encode/read. No global route context or
    forward hooks are used, so interleaved forward/backward calls remain safe.
    """

    SPEC_VERSION = "tum_us.transfer.model.v1"

    def __init__(self, model_config: dict):
        config = {"backbone": "small", "image_size": 256, "memory_capacity": 4,
                  "policy": "fifo", **deepcopy(model_config)}
        if config["backbone"] != "small":
            raise ValueError("This equal-budget experiment requires SAM2.1 Small")
        if config["policy"] != "fifo" or config["memory_capacity"] != 4:
            raise ValueError("Transfer experiments require fixed FIFO capacity 4")
        valid = {field.name for field in fields(TUMConfig)}
        unsupported = set(config) - valid - {"gradient_checkpointing"}
        if unsupported:
            raise ValueError(f"Unknown model config fields: {sorted(unsupported)}")
        # TUMModel/build_sam2 performs the official strict load before wrapping.
        super().__init__({k: v for k, v in config.items() if k in valid})
        self.model_config = config
        self.requires_grad_(False)
        self.sam.image_encoder.trunk = RoutedHiera(
            self.sam.image_encoder.trunk, config.get("gradient_checkpointing", False))
        self.routes = TaskRoutes()
        self.adapter_pool = nn.ModuleDict({str(i): nn.ModuleDict({
            "v0": ResidualAdapter(BLOCK_DIMS[i], BASE_RANKS[i])}) for i in BLOCK_INDICES})
        self.endpoint_specs: dict[str, dict] = {}
        self.endpoint_keys: dict[str, str] = {}
        self.readers = nn.ModuleDict()
        self.queries = nn.ParameterDict()
        self.mim_head = nn.Linear(256, 16 * 16 * 3, bias=True)
        self._training_selection = None
        self.train(self.training)

    def add_endpoint(self, name: str, spec: dict):
        if name in self.endpoint_specs:
            if self.endpoint_specs[name] != spec:
                raise ValueError(f"Endpoint {name!r} already has a different specification")
            return self
        spec = deepcopy(spec)
        kind, task = spec.get("kind"), spec.get("task")
        if kind not in {"classification", "segmentation"} or task not in TASK_TO_ID:
            raise ValueError("Endpoint requires a valid kind and native task")
        if (kind == "classification") != task.endswith("cls"):
            raise ValueError("Endpoint kind and task disagree")
        targets = spec.get("target_ids", [0])
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("Endpoint requires distinct nonempty target_ids")
        if any(not isinstance(i, int) or i < 0 or i >= self.config.num_targets for i in targets):
            raise ValueError("Endpoint target ID is outside the frozen embedding table")
        anatomy = int(spec.get("anatomy_id", 0))
        if not 0 <= anatomy < self.config.num_anatomies:
            raise ValueError("Endpoint anatomy ID is outside the frozen embedding table")
        key = f"e{len(self.endpoint_specs)}"
        self.endpoint_keys[name] = key
        self.endpoint_specs[name] = spec
        self.routes.add_endpoint(name)
        if kind == "classification":
            reader = deepcopy(self.global_readout)
            # All binary endpoint readers start from the same untrained template.
            reader.heads = nn.ModuleList([deepcopy(self.global_readout.heads[0])])
            self.readers[key] = reader
        else:
            self.queries[key] = nn.Parameter(self._template_query(name).detach().clone())
        if self._training_selection is not None:
            self.configure_training(**self._training_selection)
        return self

    def _template_query(self, endpoint):
        spec = self.endpoint_specs[endpoint]
        device = self.image_mean.device
        targets = torch.tensor(spec.get("target_ids", [0]), device=device, dtype=torch.long)
        task = torch.tensor(TASK_TO_ID[spec["task"]], device=device)
        anatomy = torch.tensor(spec.get("anatomy_id", 0), device=device)
        return (self.target_queries(targets) + self.task_embeddings(self.segmentation_task_id(task))
                + self.anatomy_embeddings(anatomy))

    def train(self, mode=True):
        super().train(mode)
        self.sam.eval()
        for name in ("target_queries", "task_embeddings", "anatomy_embeddings", "global_readout", "controller"):
            getattr(self, name).eval()
        if hasattr(self, "readers"):
            for reader in self.readers.values():
                reader.train(mode and any(p.requires_grad for p in reader.parameters()))
        return self

    def configure_training(self, endpoint: str, phase="joint", freeze_spatial=False):
        self.routes.route(endpoint)
        if phase not in {"joint", "head"}:
            raise ValueError("Training phase must be joint or head")
        self.requires_grad_(False)
        if phase == "joint" and not freeze_spatial:
            for block, versions in self.routes.versions.items():
                for version, meta in versions.items():
                    if not meta["frozen"] and not meta["archived"]:
                        self.adapter_pool[str(block)][version].requires_grad_(True)
        key = self.endpoint_keys[endpoint]
        if self.endpoint_specs[endpoint]["kind"] == "classification":
            self.readers[key].requires_grad_(True)
        else:
            self.queries[key].requires_grad_(True)
        self.mim_head.requires_grad_(True)
        self._training_selection = {"endpoint": endpoint, "phase": phase, "freeze_spatial": bool(freeze_spatial)}
        self.train(self.training)
        return self

    def _encode(self, media, endpoint):
        route = self.routes.route(endpoint)
        encoder = self.sam.image_encoder
        features, pos = encoder.neck(encoder.trunk(
            (media - self.image_mean) / self.image_std, route, self.adapter_pool))
        if encoder.scalp:
            features, pos = features[:-encoder.scalp], pos[:-encoder.scalp]
        output = {"vision_features": features[-1], "vision_pos_enc": pos, "backbone_fpn": features}
        if self.sam.use_high_res_features_in_sam:
            output["backbone_fpn"][0] = self.sam.sam_mask_decoder.conv_s0(features[0])
            output["backbone_fpn"][1] = self.sam.sam_mask_decoder.conv_s1(features[1])
        return output

    def _endpoint_read(self, features, view, targets, endpoint, frame, slots, view_valid):
        q = len(targets)
        fused = self._endpoint_condition(features, view, targets, slots, frame,
                                         view_valid.nonzero().flatten().tolist())
        key = self.endpoint_keys[endpoint]
        query = self.queries[key] if key in self.queries else self._template_query(endpoint)
        dense = self.sam.sam_prompt_encoder.no_mask_embed.weight.reshape(1, 256, 1, 1)
        dense = dense.expand(q, 256, fused.shape[-2], fused.shape[-1])
        high = [x[view:view + 1].expand(q, -1, -1, -1) for x in features["backbone_fpn"][:-1]]
        low, quality, token, _ = self.sam.sam_mask_decoder(
            image_embeddings=fused, image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=query[:, None], dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=high)
        masks = F.interpolate(low.float(), (self.config.image_size,) * 2,
                              mode="bilinear", align_corners=False)[:, 0]
        pointers = self.sam.obj_ptr_proj(token[:, 0])
        reader = self.readers[key] if key in self.readers else self.global_readout
        global_token = reader.fuse_views(features["backbone_fpn"][-1], view_valid)
        # Every private reader has exactly one binary head. label_space is metadata.
        logits = reader(global_token, slots, 0, frame)
        return masks, pointers, quality.mean(), global_token, logits

    def _endpoint_condition(self, features, view, targets, slots, frame, valid_view_indices):
        # Native memory stacks valid views densely; resolve the physical view ID.
        # A single surviving view has a 4D slot and needs no remapping.
        memory_view = valid_view_indices.index(view)
        remapped = []
        for slot in slots:
            if slot.spatial.ndim == 5:
                remapped.append(replace(slot, spatial=slot.spatial[memory_view],
                                        pos=slot.pos[memory_view], pointers=slot.pointers[memory_view]))
            else:
                remapped.append(slot)
        return self._condition(features, view, targets, remapped, frame)

    def forward(self, batch, endpoint: str):
        self.routes.route(endpoint)
        spec = self.endpoint_specs[endpoint]
        b, t, vmax, channels, h, w = batch.media.shape
        if channels != 3 or (h, w) != (self.config.image_size,) * 2 or not 1 <= vmax <= 2:
            raise ValueError("Media must match image_size and contain one/two RGB views")
        qmax = batch.target_ids.shape[-1]
        masks_out, views_out, classes_out, frame_classes_out, memory_indices, memory_bytes = [], [], [], [], [], []
        task = TASK_TO_ID[spec["task"]]
        video = task in (2, 3, 4, 5)
        for case in range(b):
            valid_targets = batch.target_ids[case] >= 0
            targets = batch.target_ids[case, valid_targets]
            if targets.tolist() != spec.get("target_ids", [0]):
                raise ValueError(f"Batch target order differs from endpoint {endpoint}")
            valid_views = batch.view_valid[case].nonzero().flatten().tolist()
            if not valid_views:
                raise ValueError("Every case must contain a valid view")
            canonical_view = int(batch.seg_view_ids[case])
            if canonical_view not in valid_views:
                raise ValueError("Canonical segmentation view must be valid")
            bank = MemoryBank(4, "fifo", self.controller)
            case_masks, case_views, case_classes = [], [], []
            last_logits = None
            previous_frame = -1
            for ti in range(t):
                if not bool(batch.frame_valid[case, ti]):
                    case_masks.append(batch.media.new_zeros(qmax, h, w))
                    case_views.append(batch.media.new_zeros(vmax, qmax, h, w))
                    case_classes.append(batch.media.new_zeros(2))
                    continue
                frame = int(batch.frame_indices[case, ti])
                if video and frame <= previous_frame:
                    raise ValueError("Video frame indices must be strictly increasing")
                previous_frame = frame
                features = self._encode(batch.media[case, ti], endpoint)
                results = {view: self._endpoint_read(features, view, targets, endpoint, frame,
                           bank.slots if video else [], batch.view_valid[case]) for view in valid_views}
                per_view = batch.media.new_zeros(vmax, qmax, h, w)
                for view, result in results.items():
                    per_view[view] = result[0].new_zeros(qmax, h, w).index_copy(
                        0, valid_targets.nonzero().flatten(), result[0])
                last_logits = results[canonical_view][4][0]
                case_masks.append(per_view[canonical_view])
                case_views.append(per_view)
                case_classes.append(last_logits)
                if video:
                    bank.update(self._write_views(features, results, frame, not bank.slots),
                                self.active_memory_tasks(task))
            if last_logits is None:
                raise ValueError("Every case requires a valid frame")
            masks_out.append(torch.stack(case_masks))
            views_out.append(torch.stack(case_views))
            classes_out.append(last_logits)
            frame_classes_out.append(torch.stack(case_classes))
            memory_indices.append([s.frame_index for s in bank.slots])
            memory_bytes.append(bank.nbytes)
        return {"mask_logits": torch.stack(masks_out), "view_mask_logits": torch.stack(views_out),
                "class_logits": torch.stack(classes_out), "frame_class_logits": torch.stack(frame_classes_out),
                "memory_indices": memory_indices, "memory_bytes": memory_bytes}

    def forward_reconstruction(self, media, endpoint: str, generator=None, mean_rgb=None):
        """Mask 25% of 16x16 patches; reconstruct raw RGB through routed top FPN.

        A 512 image has 32x32 tokens and exactly 256 masked patches. Other
        multiples of 32 are supported for CPU validation at the same mask ratio.
        """
        self.routes.route(endpoint)
        if media.ndim != 4 or media.shape[1] != 3 or media.shape[-2:] != (self.config.image_size,) * 2:
            raise ValueError("Reconstruction media must be [N,3,image_size,image_size]")
        n, _, h, w = media.shape
        gh, gw = h // 16, w // 16
        mask = torch.zeros(n, gh * gw, device=media.device, dtype=torch.bool)
        random_device = generator.device if generator is not None else media.device
        for sample in range(n):
            selected = torch.randperm(gh * gw, device=random_device, generator=generator)[:gh * gw // 4]
            mask[sample, selected.to(media.device)] = True
        pixels = mask.reshape(n, gh, gw).repeat_interleave(16, 1).repeat_interleave(16, 2)[:, None]
        if mean_rgb is None:
            raise ValueError("MIM requires the training-source mean_rgb; no implicit ImageNet/source estimate")
        mean = torch.as_tensor(mean_rgb, device=media.device, dtype=media.dtype).reshape(1, 3, 1, 1)
        if not torch.isfinite(mean).all() or bool(((mean < 0) | (mean > 1)).any()):
            raise ValueError("Training RGB mean must be finite and within [0,1]")
        masked = torch.where(pixels, mean, media)
        top = self._encode(masked, endpoint)["backbone_fpn"][-1]
        if tuple(top.shape) != (n, 256, gh, gw):
            raise ValueError(f"MIM expected top FPN {(n, 256, gh, gw)}, got {tuple(top.shape)}")
        prediction = self.mim_head(top.permute(0, 2, 3, 1))
        prediction = prediction.reshape(n, gh, gw, 16, 16, 3).permute(0, 5, 1, 3, 2, 4).reshape(n, 3, h, w)
        return {"reconstruction": prediction, "mask": pixels, "masked_media": masked}

    def reconstruction_loss(self, batch, endpoint: str, generator=None, normalizer=1.0, mean_rgb=None):
        normalizer = float(normalizer)
        if not math.isfinite(normalizer) or normalizer < 0:
            raise ValueError("MIM normalizer must be finite and nonnegative")
        media = []
        for case in range(len(batch.media)):
            for time in batch.frame_valid[case].nonzero().flatten().tolist():
                media.append(batch.media[case, time, batch.view_valid[case].bool()])
        if not media or not sum(len(item) for item in media):
            raise ValueError("MIM requires at least one valid frame/view")
        media = torch.cat(media)
        output = self.forward_reconstruction(media, endpoint, generator, mean_rgb)
        squared = (output["reconstruction"].float() - media.float()).square()
        mask = output["mask"].expand_as(squared)
        return squared.masked_select(mask).mean() / max(normalizer, 1e-6)

    @staticmethod
    def _blocks(block_indices):
        indices = tuple(int(i) for i in block_indices)
        if len(set(indices)) != len(indices) or any(i not in BLOCK_INDICES for i in indices):
            raise ValueError("Selected blocks must be distinct indices from 3 through 15")
        return indices

    def split_for_new_task(self, endpoint: str, block_indices):
        self.routes.route(endpoint)
        indices = self._blocks(block_indices)
        new_route = list(tuple(self.routes.live[i]) for i in BLOCK_INDICES)
        for block in indices:
            chain = self.routes.live[block]
            parent = chain[-1]
            meta = self.routes.versions[block][parent]
            meta.update(frozen=True, archived=True)
            self.adapter_pool[str(block)][parent].requires_grad_(False)
            version = self.routes.add_version(block, meta["rank"], meta["kind"])
            self.adapter_pool[str(block)][version] = deepcopy(self.adapter_pool[str(block)][parent])
            self.routes.live[block] = (*chain[:-1], version)
            new_route[block - 3] = self.routes.live[block]
        self.routes.endpoints[endpoint] = tuple(new_route)
        self._reapply_training()
        return self

    def freeze_selected(self, block_indices):
        for block in self._blocks(block_indices):
            for version in self.routes.live[block]:
                self.routes.versions[block][version]["frozen"] = True
                self.adapter_pool[str(block)][version].requires_grad_(False)
        self._reapply_training()
        return self

    def add_distributed_delta(self, endpoint: str, freeze_old=True, freeze_selected=None):
        self.routes.route(endpoint)
        if freeze_old:
            for block, versions in self.routes.versions.items():
                for version, metadata in versions.items():
                    metadata["frozen"] = True
                    self.adapter_pool[str(block)][version].requires_grad_(False)
        elif freeze_selected is None:
            raise ValueError("Shared-update delta requires the matching selected block indices")
        else:
            self.freeze_selected(freeze_selected)
        route = []
        for block, rank in zip(BLOCK_INDICES, DELTA_RANKS):
            version = self.routes.add_version(block, rank, "delta")
            parent = self.adapter_pool[str(block)][self.routes.live[block][-1]].down.weight
            self.adapter_pool[str(block)][version] = ResidualAdapter(BLOCK_DIMS[block], rank).to(parent)
            self.routes.live[block] = (*self.routes.live[block], version)
            route.append(self.routes.live[block])
        self.routes.endpoints[endpoint] = tuple(route)
        self._reapply_training()
        return self

    def _reapply_training(self):
        if self._training_selection is not None:
            self.configure_training(**self._training_selection)

    def current_live_state(self):
        return {block: {key: value.detach().clone() for key, value in
                        self.adapter_pool[str(block)][self.routes.live[block][-1]].state_dict().items()}
                for block in BLOCK_INDICES}

    @contextmanager
    def restore_modules(self, reference: dict, block_indices, endpoint=None):
        """Temporarily restore current live modules; topology/heads are untouched.

        endpoint, if supplied, must route through the live module being tested;
        this guards against silently restoring an unrelated archived version.
        """
        indices = self._blocks(block_indices)
        if endpoint is not None:
            route = self.routes.route(endpoint)
            if any(self.routes.live[i][-1] not in route[i - 3] for i in indices):
                raise ValueError("Requested endpoint does not use the selected current live version")
        modules, saved = {}, {}
        # Validate before the first mutation; a failed restoration cannot leak state.
        for block in indices:
            module = self.adapter_pool[str(block)][self.routes.live[block][-1]]
            state = reference.get(block, reference.get(str(block)))
            if state is None or set(state) != set(module.state_dict()):
                raise ValueError(f"Missing/incompatible restoration state for block {block}")
            if any(state[key].shape != value.shape for key, value in module.state_dict().items()):
                raise ValueError(f"Incompatible restoration tensor shape at block {block}")
            modules[block] = (module, state)
            saved[block] = {k: v.detach().clone() for k, v in module.state_dict().items()}
        try:
            for module, state in modules.values():
                module.load_state_dict(state, strict=True)
            yield self
        finally:
            for block, (module, _) in modules.items():
                module.load_state_dict(saved[block], strict=True)

    def export_spec(self):
        return {"format": self.SPEC_VERSION, "model_config": deepcopy(self.model_config),
                "endpoints": deepcopy(self.endpoint_specs), "endpoint_keys": dict(self.endpoint_keys),
                "routes": self.routes.export(), "training_selection": deepcopy(self._training_selection)}

    @classmethod
    def from_spec(cls, spec):
        if spec.get("format") != cls.SPEC_VERSION:
            raise ValueError("Unsupported transfer model topology format")
        result = cls(spec["model_config"])
        # Checkpoint serialization may sort JSON keys; preserve original module IDs.
        for endpoint in sorted(spec["endpoints"], key=lambda n: int(spec["endpoint_keys"][n][1:])):
            result.add_endpoint(endpoint, spec["endpoints"][endpoint])
        routes = TaskRoutes.from_spec(spec["routes"])
        if set(routes.endpoints) != set(result.endpoint_specs):
            raise ValueError("Topology endpoint registry mismatch")
        result.adapter_pool = nn.ModuleDict({str(block): nn.ModuleDict({
            version: ResidualAdapter(BLOCK_DIMS[block], meta["rank"])
            for version, meta in versions.items()}) for block, versions in routes.versions.items()})
        result.routes = routes
        result._training_selection = spec.get("training_selection")
        if result._training_selection is not None:
            result._reapply_training()
        else:
            for block, versions in routes.versions.items():
                for version, meta in versions.items():
                    result.adapter_pool[str(block)][version].requires_grad_(not(meta["frozen"] or meta["archived"]))
        return result

    def parameter_report(self):
        count = lambda ps: sum(p.numel() for p in ps)
        by_kind = {"base": 0, "delta": 0, "archived": 0, "spatial_trainable": 0}
        for block, versions in self.routes.versions.items():
            for version, meta in versions.items():
                module = self.adapter_pool[str(block)][version]
                n = count(module.parameters())
                by_kind[meta["kind"]] += n
                by_kind["archived"] += n if meta["archived"] else 0
                by_kind["spatial_trainable"] += count(p for p in module.parameters() if p.requires_grad)
        return {"total": count(self.parameters()), "trainable": count(p for p in self.parameters() if p.requires_grad),
                "original_sam": count(self.sam.parameters()),
                "original_sam_trainable": count(p for p in self.sam.parameters() if p.requires_grad),
                "base_spatial_budget": 319488, "spatial_total": count(self.adapter_pool.parameters()),
                "spatial_extra": count(self.adapter_pool.parameters()) - 319488,
                "mim_head": count(self.mim_head.parameters()),
                "private_readers": count(self.readers.parameters()), "private_queries": count(self.queries.parameters()),
                "by_kind": by_kind, "per_endpoint": {name: {
                    "reader": count(self.readers[key].parameters()) if key in self.readers else 0,
                    "query": self.queries[key].numel() if key in self.queries else 0}
                    for name, key in self.endpoint_keys.items()}}
