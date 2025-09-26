'''
main.py
1. foundation processing
    - process review into natural language features and coresponding embedding (input args.dataset review output ...)
    - build embedding vector database (?)
    - conduct clustering
    - retrieve clusters, assign central theme

2. application stage
    - todo
'''
import faiss
import numpy as np
from pathlib import Path

import argparse

from data import prepare_data
from systems import build_system
from utils import load_or_build, readf, dumpj, loadj, dumpp, loadp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type=str, default='yelp')
    parser.add_argument('--system', type=str, default='heur') 
    # heur, sat, ou, sulm, sugar
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--dset_root', type=str, default='.dset_root')
    # benchmark
    parser.add_argument('--num_test', type=int, default=2)
    args = parser.parse_args()

    new_todo = """
    1. prepare data
    2. (core) process data => evaluate
    """
    print(new_todo)

    args.cache_dir = Path(args.cache_dir)
    args.dset_root = Path(readf(args.dset_root).strip())
    args.cache_dir.mkdir(exist_ok=True)
    args.prepared_data_path = args.cache_dir/f"prepared_{args.dset}_data.json"
    DATA = load_or_build(args.prepared_data_path, dumpj, loadj, prepare_data, args)

    System = build_system(args, DATA)

    # for city in DATA["USERS"].keys():
    city = 'saint louise'


if __name__ == '__main__':
    main()
