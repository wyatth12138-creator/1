"""Export raw EchoNet AVI and CSV traces, preserving official partitions."""
import argparse,json,sys
from pathlib import Path
from tum_us.data.cardiac_preprocess import preprocess_echonet
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('-i','--input-dir',required=True); p.add_argument('-o','--output-dir',required=True)
    p.add_argument('--size',type=int,default=112); p.add_argument('--expected-trace-rows',type=int,default=21)
    p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val'])
    a=p.parse_args(); print(json.dumps(preprocess_echonet(a.input_dir,a.output_dir,size=a.size,splits=tuple(a.splits),expected_trace_rows=a.expected_trace_rows),ensure_ascii=False))
