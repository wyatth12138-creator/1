"""Evaluate fixed readers and memory policies on explicit prepared partitions."""
import sys
from train import main
if __name__=='__main__': main(['--stage','evaluate',*sys.argv[1:]])
