"""Train the complete task-utility generalist, or resume an individual stage."""
import argparse
import json
from pathlib import Path
from tum_us.generalist.config import load_config

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,help='prepared_config.json from prepare.py')
    p.add_argument('--stage',choices=['run','study','object-study','baselines','readers','collect','controller','evaluate','expand','prompt-train','prompt-evaluate'],default='run')
    p.add_argument('--variant',choices=['fixed','independent','shared'],default='shared',help='Support-prompt ablation variant')
    p.add_argument('--output'); p.add_argument('--seed',type=int); p.add_argument('--device')
    p.add_argument('--checkpoint'); p.add_argument('--endpoints',help='Comma separated explicit stage endpoints')
    p.add_argument('--image-size', type=int, help='Square model and data input size, for example 256')
    p.add_argument('--init-checkpoint', help='Initialize a new reader fit from TUM-US weights; optimizer and schedule start fresh')
    p.add_argument('--report'); a=p.parse_args(argv)
    overrides={}
    if a.image_size is not None:
        if a.image_size < 32 or a.image_size % 32:
            p.error('--image-size must be a positive multiple of 32')
        overrides.setdefault('model', {})['image_size'] = a.image_size
    if a.init_checkpoint:
        if a.stage not in ('run', 'readers'):
            p.error('--init-checkpoint applies to run or readers')
        overrides.setdefault('training', {})['init_checkpoint'] = str(Path(a.init_checkpoint).resolve())
    if a.output: overrides['output_dir']=a.output
    if a.seed is not None: overrides.setdefault('training',{})['seed']=a.seed
    if a.device: overrides.setdefault('training',{})['device']=a.device
    prompt_stage = a.stage in ('prompt-train', 'prompt-evaluate')
    if prompt_stage:
        from tum_us.generalist import support_study
        cfg = support_study.load_study_config(a.config, overrides)
    else:
        cfg=load_config(a.config,overrides)
    if a.checkpoint:
        cfg.setdefault('evaluation',{})['checkpoint']=str(Path(a.checkpoint).resolve())
        if a.stage=='collect': cfg.setdefault('utility',{})['reference_checkpoint']=str(Path(a.checkpoint).resolve())
        if a.stage in ('controller','expand'): cfg.setdefault('controller',{})['reference_checkpoint']=str(Path(a.checkpoint).resolve())
    if a.endpoints:
        key={'readers':'readers','collect':'collect','evaluate':'evaluate','expand':'expand_new'}.get(a.stage)
        if key is None: p.error('--endpoints applies to readers, collect, evaluate or expand')
        names=[n.strip() for n in a.endpoints.split(',') if n.strip()]
        if not names or any(n not in cfg['endpoints'] for n in names): p.error('Unknown or empty endpoints')
        cfg.setdefault('endpoint_sets',{})[key]=names
    output = Path(cfg['output_dir'])
    if a.stage == 'prompt-train':
        result = support_study.train_study(cfg, output, a.variant)
    elif a.stage == 'prompt-evaluate':
        result = support_study.evaluate_study(cfg, output, a.variant, a.checkpoint)
    elif a.stage=='object-study':
        from tum_us.generalist.object_study import run_object_study
        result=run_object_study(cfg,Path(cfg['output_dir']))
    elif a.stage in ('study','baselines'):
        from tum_us.generalist.study import run_study,run_reader_baselines
        result=(run_study if a.stage=='study' else run_reader_baselines)(cfg,Path(cfg['output_dir']))
    else:
        from tum_us.generalist import engine
        fn={'run':engine.run,'readers':engine.run_readers,'collect':engine.collect,'controller':engine.train_controller,
            'evaluate':engine.evaluate,'expand':engine.expand}[a.stage]
        result=fn(cfg,Path(cfg['output_dir']))
    result_name = f'{a.stage}_{a.variant}_result.json' if prompt_stage else a.stage + '_result.json'
    destination=Path(a.report) if a.report else output / result_name
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps({'stage':a.stage,'report':str(destination.resolve()),'checkpoint':result.get('checkpoint'),
                      'status':result.get('status','returned')},ensure_ascii=False))
    return result
if __name__=='__main__': main()
