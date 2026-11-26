"""Export raw CAMUS NIfTI with explicit frame correspondence, following MemSAM."""
import argparse,json,sys
from pathlib import Path
from tum_us.data.cardiac_preprocess import preprocess_camus
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('-i','--input-dir',required=True); p.add_argument('-o','--output-dir',required=True)
    p.add_argument('-f','--split-file'); p.add_argument('--size',type=int,default=256)
    p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val'])
    p.add_argument('--temporal-order',choices=['source','ed-to-es'],default='source'); p.add_argument('--mask-mode',choices=['auto','dense','ed-es'],default='auto')
    a=p.parse_args(); print(json.dumps(preprocess_camus(a.input_dir,a.output_dir,size=a.size,split_file=a.split_file,splits=tuple(a.splits),temporal_order=a.temporal_order,mask_mode=a.mask_mode),ensure_ascii=False))
