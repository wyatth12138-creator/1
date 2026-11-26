"""Recover explicit CAMUS/EchoNet storage-frame mappings from local archives."""
import argparse, json, sys
from pathlib import Path
from tum_us.data.cardiac import index_existing

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True); p.add_argument('--inventory',required=True); p.add_argument('--output',required=True)
    p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val'])
    a=p.parse_args(); print(json.dumps(index_existing(a.data_root,a.inventory,a.output,tuple(a.splits)),ensure_ascii=False))
