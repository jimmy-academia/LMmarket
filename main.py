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
    parser.add_argument('--system', type=str, default='sulm') 
    # best, sat, ou, sulm
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--dset_root', type=str, default='.dset_root')
    # benchmark
    parser.add_argument('--num_test', type=int, default=2)
    args = parser.parse_args()

    new_todo = """
    1. prepare data
    2. (core) process data => evaluate
    3. (app) equilibrium task => evaluate
    """
    print(new_todo)

    args.cache_dir = Path(args.cache_dir)
    args.dset_root = Path(readf(args.dset_root).strip())
    args.cache_dir.mkdir(exist_ok=True)
    args.prepared_data_path = args.cache_dir/f"prepared_{args.dset}_data.json"
    DATA = load_or_build(args.prepared_data_path, dumpj, loadj, prepare_data, args)
    

    print('todo: load test request')
    System = build_system(args, reviews, tests)
    mock_requests = [
    ["I’m looking for a ramen shop where the broth is rich and flavorful but the wait time isn’t too long.",
     ["ramen broth flavor", "service speed / wait time"]],
     
    ["Show me sushi places with the freshest fish and friendly staff.",
     ["sushi freshness", "staff friendliness"]],
     
    ["I want a brunch spot that has delicious pancakes but also plenty of parking nearby.",
     ["pancake taste", "parking availability"]],
    ]

    System.serve(mock_requests)
    System.evaluate()

if __name__ == '__main__':
    main()

# request (with aspect parse) -> item -> (detailed) utility score
# evaluation:
# synthetic => score
# online learning approx real world score
