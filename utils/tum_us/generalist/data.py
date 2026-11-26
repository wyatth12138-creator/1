"""Endpoint access for source-specific, unpaired task data."""
from tum_us.transfer.data import EndpointDataset

class GeneralistEndpointDataset(EndpointDataset):
    """Attach the declared source-index semantics to every returned batch."""
    def __getitem__(self, index):
        entity = super().__getitem__(index)
        task = self.spec['task']
        semantics = 'static_image'
        if task == 'ceus_cls':
            semantics = 'stored_array_ordinal_not_acquisition_frame'
        elif task in ('video_seg', 'ceus_seg'):
            semantics = 'stored_frame_mapping; raw_acquisition_direction_not_verified'
        for batch in entity.batches:
            for metadata in batch.metadata:
                metadata['frame_index_semantics'] = semantics
        return entity

def build_endpoint_datasets(cfg, split, endpoints=None):
    names = list(cfg['endpoints']) if endpoints is None else list(endpoints)
    result={}
    for name in names:
        if name not in cfg['endpoints']: raise KeyError(f'Unknown endpoint {name}')
        spec=dict(cfg['endpoints'][name],name=name)
        field = 'train_manifest' if split == 'train' else 'final_manifest' if split == 'test' else 'dev_manifest'
        if not spec.get(field): raise ValueError(f'{name}: {field} not prepared; run prepare.py and inspect readiness report')
        if spec.get('readiness',{}).get(split) is False:
            raise ValueError(f'{name}/{split}: data not ready; inspect readiness report')
        result[name]=GeneralistEndpointDataset(spec,split,cfg)
    return result

