"""Build reader support prototypes from training cases."""

from collections import defaultdict
import random

import torch
from torch.nn import functional as F

from tum_us.data.registry import TASK_TO_ID, classification_label
from .support_metric import mask_prototypes, support_key


@torch.no_grad()
def _encode_images(model, images, device, encoder_batch):
    features = []
    for chunk in images.split(encoder_batch):
        features.append(model._encode(chunk.to(device))['backbone_fpn'][-1])
    return torch.cat(features)


def _case_features(model, entity, device, encoder_batch):
    if entity.label is not None:
        vectors = []
        for batch in entity.batches:
            frames = batch.media[0, batch.frame_valid[0]]
            frames = frames[:, batch.view_valid[0]].flatten(0, 1)
            features = _encode_images(model, frames, device, encoder_batch)
            vectors.append(features.mean(dim=(-2, -1)))
        return torch.cat(vectors).mean(dim=0)

    region_vectors = []
    for batch in entity.batches:
        valid_frames = batch.frame_valid[0] & batch.mask_valid[0].any(dim=-1)
        if not valid_frames.any():
            continue
        view = int(batch.seg_view_ids[0])
        images = batch.media[0, valid_frames, view]
        valid_masks = batch.mask_valid[0, valid_frames, :, None, None]
        masks = (batch.masks[0, valid_frames] * valid_masks).amax(dim=1)
        features = _encode_images(model, images, device, encoder_batch)
        region_means, valid_regions = mask_prototypes(features, masks.to(device))
        region_vectors.append((region_means, valid_regions))
    if not region_vectors:
        return None
    means = torch.cat([means for means, _ in region_vectors])
    valid = torch.cat([valid for _, valid in region_vectors])
    counts = valid.sum(dim=0)
    if not bool((counts > 0).all()):
        return None
    return (means * valid[..., None]).sum(dim=0) / counts[:, None]


@torch.no_grad()
def initialize_support_profiles(model, datasets, options, seed, device):
    shots = options['shots']
    encoder_batch = options['encoder_batch']
    if type(shots) is not int or shots < 1 or type(encoder_batch) is not int or encoder_batch < 1:
        raise ValueError('Support shots and encoder batch must be positive integers')

    sources = defaultdict(list)
    for name, dataset in datasets.items():
        spec = dataset.spec
        key = support_key(TASK_TO_ID[spec['task']], spec['anatomy_id'],
                          spec['label_space_id'], spec['target_ids'])
        sources[key].append((name, dataset))
    if sources.keys() != model.support_profile_indices.keys():
        raise ValueError('Support profiles differ from the reader training sources')

    model.eval()
    chosen = {}
    for key, profile_index in model.support_profile_indices.items():
        candidates = defaultdict(list)
        classification = sources[key][0][1].spec['kind'] == 'classification'
        for name, dataset in sources[key]:
            for index, (_, rows) in enumerate(dataset.groups):
                label = 0
                if classification:
                    label, valid, _ = classification_label(rows[0])
                    if not valid:
                        raise ValueError(f'{name}: training classification label is invalid')
                candidates[label].append((name, dataset, index))

        rng = random.Random(seed + profile_index)
        vectors = []
        selected = []
        for label in ((0, 1) if classification else (0,)):
            cases = candidates[label]
            rng.shuffle(cases)
            group_vectors = []
            for name, dataset, index in cases:
                entity = dataset[index]
                features = _case_features(model, entity, device, encoder_batch)
                if features is None:
                    continue
                group_vectors.append(features)
                selected.append({'source': name, 'entity': entity.key,
                                 'label': label if classification else None})
                if len(group_vectors) == shots:
                    break
            if len(group_vectors) != shots:
                raise ValueError(f'Support profile {key} has fewer than {shots} eligible cases')
            vectors.append(torch.stack(group_vectors).mean(dim=0))

        prototypes = torch.stack(vectors) if classification else vectors[0]
        model.support_prototypes[profile_index].copy_(F.normalize(prototypes, dim=-1))
        chosen[str(key)] = selected
    return chosen
