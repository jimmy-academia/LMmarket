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

from debug import check
from foundation import process_data, fetch_embedder
from utils import load_or_build, readf, dumpj, loadj
from playground.reproducibility import run_reproducibility_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type=str, default='yelp')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--dset_root', type=str, default='.dset_root')
    args = parser.parse_args()

    args.cache_dir = Path(args.cache_dir)
    args.dset_root = Path(readf(args.dset_root).strip())
    args.cache_dir.mkdir(exist_ok=True)
    args.processed_data_path = args.cache_dir/f"processed_{args.dset}_data.json"
    
    DATA = load_or_build(args.processed_data_path, dumpj, loadj, process_data, args)

    # for city in DATA['USERS']:
    #     print(city, len(DATA['USERS'][city]), len(DATA['ITEMS'][city]), len(DATA['REVIEWS'][city]))
    # input('pause')

    '''
    philadelphia 1041 2536 512173
    indianapolis 362 1098 173652
    new orleans 280 1200 355692
    nashville 238 1153 233161
    tampa 304 1255 208616
    tucson 289 1100 173804
    reno 220 770 148916
    saint louis 176 752 125592
    '''
    
    city = 'saint louis'
    args.div_name = f"{args.dset}_{city.replace(' ', '_')}"

    embedder = fetch_embedder(args, DATA['REVIEWS'][city])
    run_reproducibility_experiment(embedder)



if __name__ == '__main__':
    main()