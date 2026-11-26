"""Train and evaluate example-defined classification and segmentation episodes."""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from tum_us.data.registry import classification_label
from tum_us.engine.checkpoint import load_checkpoint, save_checkpoint
from tum_us.engine.metrics import classification_metrics
from tum_us.engine.runtime import autocast_context, choose_device, seed_everything
from tum_us.generalist.config import load_config
from tum_us.generalist.data import build_endpoint_datasets
from tum_us.generalist.support_metric import SupportMetric, mask_prototypes
from tum_us.model import TUMConfig, TUMModel
from tum_us.paths import ROOT


def load_study_config(path, overrides=None):
    path = Path(path).resolve()
    options = json.loads(path.read_text(encoding='utf-8-sig'))
    reference = path.parent / options['reference_config']
    from tum_us.generalist.config import _merge
    local_overrides={'model':options.get('model',{})}
    cfg=load_config(reference,_merge(local_overrides,overrides or {}))
    if not overrides or 'output_dir' not in overrides:
        seed = cfg['training']['seed']
        cfg['output_dir'] = str(ROOT / options['output_dir'] / f'seed_{seed}')
    for key in ('shots', 'train_shots', 'queries', 'episodes_per_task', 'encoder_batch'):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    cfg['support_study'] = options
    return cfg


def _identity(dataset, index):
    key, rows = dataset.groups[index]
    ids = {f'entity:{key}'}
    for row in rows:
        if row.get('entity_uid'):
            ids.add(f"entity:{row['entity_uid']}")
        if row.get('record_id'):
            ids.add(f"record:{row['record_id']}")
        ids.update(f'record:{value}' for value in (row.get('linked_record_ids') or []))
        ids.update('media:' + path.replace('\\', '/').casefold() for path in row['sample_paths'])
    return ids


class Episodes:
    def __init__(self, dataset):
        self.dataset = dataset
        self.task = dataset.spec['task']
        self.groups = defaultdict(list)
        for index, (key, rows) in enumerate(dataset.groups):
            label = 0
            if self.task == 'image_cls':
                labels = set()
                for row in rows:
                    row_label, valid, _ = classification_label(row)
                    if valid:
                        labels.add(row_label)
                if len(labels) != 1 or not labels.issubset({0, 1}):
                    raise ValueError(f'{dataset.endpoint}/{key}: expected one binary case label')
                label = next(iter(labels))
            self.groups[label].append(index)
        if self.task == 'image_seg' and len(dataset.spec['target_ids']) != 1:
            raise ValueError(f'{dataset.endpoint}: support episodes require one mask target')

    def eligible(self, size):
        labels = (0, 1) if self.task == 'image_cls' else (0,)
        return all(len(self.groups[label]) >= size for label in labels)

    def draw(self, rng, shots, queries=0):
        if not self.eligible(shots + queries):
            raise ValueError(f'{self.dataset.endpoint}: insufficient cases for the episode')
        support_indices = []
        query_indices = []
        for label in sorted(self.groups):
            chosen = rng.sample(self.groups[label], shots + queries)
            support_indices.extend(chosen[:shots])
            query_indices.extend(chosen[shots:])
        return support_indices, query_indices


def _data(cfg, *, training):
    """Build source pools and exclude held-out labels, targets and case identities."""
    options = cfg['support_study']
    heldout = options['heldout']
    if not set(options['evaluate']).issubset(heldout):
        raise ValueError('Evaluation endpoints must be held out of training')
    specs = cfg['endpoints']
    heldout_train = build_endpoint_datasets(cfg, 'train', heldout)
    heldout_dev = build_endpoint_datasets(cfg, 'dev', heldout)
    train_ids = set()
    dev_ids = set()
    for name in heldout:
        for index in range(len(heldout_train[name])):
            train_ids.update(_identity(heldout_train[name], index))
        for index in range(len(heldout_dev[name])):
            dev_ids.update(_identity(heldout_dev[name], index))
    if train_ids & dev_ids:
        raise ValueError('Held-out support and query manifests share case or media identifiers')
    training_pools = defaultdict(list)
    excluded = []
    if training:
        heldout_label_spaces = set()
        heldout_targets = set()
        for name in heldout:
            spec = specs[name]
            if spec['task'] == 'image_cls':
                heldout_label_spaces.add(spec['label_space_id'])
            elif spec['task'] == 'image_seg':
                heldout_targets.update(spec['target_ids'])

        names = []
        for name in cfg['endpoint_sets']['readers']:
            if name in heldout:
                continue
            spec = specs[name]
            if spec['task'] == 'image_cls' and spec['label_space_id'] not in heldout_label_spaces:
                names.append(name)
            elif spec['task'] == 'image_seg' and heldout_targets.isdisjoint(spec['target_ids']):
                names.append(name)

        heldout_ids = train_ids | dev_ids
        episode_size = options['train_shots'] + options['queries']
        for dataset in build_endpoint_datasets(cfg, 'train', names).values():
            retained_groups = []
            for index, group in enumerate(dataset.groups):
                if heldout_ids.isdisjoint(_identity(dataset, index)):
                    retained_groups.append(group)
            dataset.groups = retained_groups
            episodes = Episodes(dataset)
            if episodes.eligible(episode_size):
                training_pools[episodes.task].append(episodes)
            else:
                excluded.append(dataset.endpoint)
        if not training_pools['image_cls'] or not training_pools['image_seg']:
            raise ValueError('Training requires eligible classification and segmentation sources')
    support_pools = {}
    for name in options['evaluate']:
        pool = Episodes(heldout_train[name])
        if not pool.eligible(options['shots']):
            raise ValueError(f'{name}: too few held-out training entities for the configured support size')
        support_pools[name] = pool
    return training_pools, support_pools, heldout_dev, excluded


class FrozenImages:
    def __init__(self, cfg, device):
        spec = cfg['model']
        model_config = TUMConfig(
            backbone=spec['backbone'],
            image_size=spec['image_size'],
            checkpoint=spec['checkpoint'],
        )
        model = TUMModel(model_config)
        model.requires_grad_(False)
        model.eval()
        self.encoder = model.sam.image_encoder.to(device)
        self.mean = model.image_mean.to(device)
        self.std = model.image_std.to(device)
        self.device = device
        self.amp = cfg['training']['amp']
        self.batch_size = cfg['support_study']['encoder_batch']

    @torch.no_grad()
    def __call__(self, images):
        feature_batches = []
        for start in range(0, len(images), self.batch_size):
            image_batch = torch.stack(images[start:start + self.batch_size]).to(self.device)
            with autocast_context(self.device, self.amp):
                normalized_images = (image_batch - self.mean) / self.std
                top_features = self.encoder(normalized_images)['backbone_fpn'][-1]
            feature_batches.append(top_features.float())
        return torch.cat(feature_batches)


def _examples(dataset, indices, encode, rng=None):
    """Sample one image per training entity and use all images during evaluation."""
    entities = [dataset[index] for index in indices]
    batches = []
    images_per_entity = []
    for entity in entities:
        selected = entity.batches
        if rng is not None:
            selected = [rng.choice(selected)]
        images_per_entity.append(len(selected))
        batches.extend(selected)

    images = [batch.media[0, 0, 0] for batch in batches]
    features = encode(images)
    if dataset.spec['task'] == 'image_cls':
        image_features = features.mean(dim=(-2, -1))
        entity_groups = image_features.split(images_per_entity)
        entity_features = torch.stack([group.mean(dim=0) for group in entity_groups])
        labels = torch.tensor([entity.label for entity in entities], device=features.device)
        return entity_features, labels
    if not all(bool(batch.mask_valid[0, 0, 0]) for batch in batches):
        raise ValueError(f'{dataset.endpoint}: unannotated segmentation image')
    masks = torch.stack([batch.masks[0, 0, 0] for batch in batches]).to(features.device)
    return features, masks


def _context(metric, task, features, labels):
    if task == 'image_cls':
        prototypes = torch.stack([features[labels == label].mean(0) for label in (0, 1)])
    else:
        region_means, valid = mask_prototypes(features, labels)
        valid_counts = valid.sum(dim=0)
        if not bool((valid_counts > 0).all()):
            raise ValueError('Mask examples must contain annotated foreground and background')
        region_sums = (region_means * valid[..., None]).sum(dim=0)
        prototypes = region_sums / valid_counts[:, None]
    return metric.context(prototypes, task)


def _mask_scores(metric, features, context, shape):
    pixel_features = features.permute(0, 2, 3, 1)
    region_logits = metric.score(pixel_features, context)
    foreground_logits = region_logits[..., 1] - region_logits[..., 0]
    resized_logits = F.interpolate(
        foreground_logits[:, None], size=shape, mode='bilinear', align_corners=False,
    )
    return resized_logits[:, 0]


def _loss(metric, pool, support, query, encode, rng):
    task = pool.task
    support_features, support_labels = _examples(pool.dataset, support, encode, rng)
    context = _context(metric, task, support_features, support_labels)
    query_features, query_labels = _examples(pool.dataset, query, encode, rng)
    if task == 'image_cls':
        logits = metric.score(query_features, context)
        return F.cross_entropy(logits, query_labels)
    logits = _mask_scores(metric, query_features, context, query_labels.shape[-2:])
    probability = logits.sigmoid()
    intersection = (probability * query_labels).sum((-2, -1))
    total = (probability + query_labels).sum((-2, -1))
    dice = (2 * intersection + 1) / (total + 1)
    dice_loss = (1 - dice).mean()
    pixel_loss = F.binary_cross_entropy_with_logits(logits, query_labels)
    return pixel_loss + dice_loss


def _checkpoint(cfg, variant):
    return ROOT / cfg['support_study']['checkpoint_dir'] / f"seed_{cfg['training']['seed']}" / f'{variant}.pt'


def train_study(cfg, output, variant):
    options = cfg['support_study']
    training_pools, _, _, excluded = _data(cfg, training=True)
    seed = cfg['training']['seed']
    seed_everything(seed)
    rng = random.Random(seed)
    device = choose_device(cfg['training']['device'])
    encoder = FrozenImages(cfg, device)
    metric = SupportMetric(variant=variant).to(device)
    optimizer = torch.optim.AdamW(metric.parameters(), lr=options['lr'])
    output.mkdir(parents=True, exist_ok=True)
    tasks = ('image_cls', 'image_seg')
    log_path = output / f'{variant}_training.jsonl'
    with log_path.open('w', encoding='utf-8') as log:
        for step in range(options['episodes_per_task']):
            for task in tasks:
                task_pools = training_pools[task]
                pool = task_pools[step % len(task_pools)]
                support_indices, query_indices = pool.draw(rng, options['train_shots'], options['queries'])
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(metric, pool, support_indices, query_indices, encoder, rng)
                if not bool(loss.isfinite()):
                    raise RuntimeError(f'Non-finite {task} loss at episode {step + 1}')
                loss.backward()
                optimizer.step()
                record = {
                    'episode': step + 1,
                    'task': task,
                    'source': pool.dataset.endpoint,
                    'loss': loss.item(),
                }
                log.write(json.dumps(record) + '\n')

    sources = {task: [pool.dataset.endpoint for pool in training_pools[task]] for task in tasks}
    checkpoint = save_checkpoint(
        _checkpoint(cfg, variant), metric,
        phase='support_prompt',
        config=cfg,
        registry={'variant': variant},
        optimizer=optimizer,
        global_step=len(tasks) * options['episodes_per_task'],
        extra={'sources': sources},
    )
    return {
        'checkpoint': str(checkpoint),
        'variant': variant,
        'sources': sources,
        'ineligible_sources': excluded,
        'episodes_per_task': options['episodes_per_task'],
    }


@torch.no_grad()
def evaluate_study(cfg, output, variant, checkpoint=None):
    options = cfg['support_study']
    _, support_pools, dev_datasets, _ = _data(cfg, training=False)
    device = choose_device(cfg['training']['device'])
    encoder = FrozenImages(cfg, device)
    metric = SupportMetric(variant=variant).to(device).eval()
    checkpoint = checkpoint or _checkpoint(cfg, variant)
    load_checkpoint(checkpoint, metric, expected_phase='support_prompt',
                    registry={'variant': variant}, restore_randomness=False,
                    expected_image_size=cfg['model']['image_size'])
    results = {}
    for name, pool in support_pools.items():
        rng = random.Random(cfg['training']['seed'])
        support_indices, _ = pool.draw(rng, options['shots'])
        support_features, support_labels = _examples(pool.dataset, support_indices, encoder)
        context = _context(metric, pool.task, support_features, support_labels)

        query_dataset = dev_datasets[name]
        records = []
        for index in range(len(query_dataset)):
            query_features, query_labels = _examples(query_dataset, [index], encoder)
            if pool.task == 'image_cls':
                logits = metric.score(query_features, context)
                positive_probability = logits.softmax(dim=-1)[0, 1].item()
                records.append({'label': query_labels.item(), 'probability': positive_probability})
            else:
                logits = _mask_scores(metric, query_features, context, query_labels.shape[-2:])
                predicted_mask = logits > 0
                true_mask = query_labels.bool()
                overlap = (predicted_mask & true_mask).sum(dim=(-2, -1))
                total = predicted_mask.sum(dim=(-2, -1)) + true_mask.sum(dim=(-2, -1))
                dice = torch.where(total > 0, 2 * overlap / total, 1.0)
                records.append({'dice': dice.mean().item()})

        if pool.task == 'image_cls':
            scores = classification_metrics(records)
        else:
            mean_dice = sum(row['dice'] for row in records) / len(records)
            scores = {'n': len(records), 'dice': mean_dice}
        results[name] = {
            'metrics': scores,
            'support': [pool.dataset.groups[index][0] for index in support_indices],
        }
    return {
        'checkpoint': str(checkpoint),
        'variant': variant,
        'partition': 'dev',
        'shots': options['shots'],
        'endpoints': results,
    }
