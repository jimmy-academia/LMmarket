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

from data_foundation import process_data, construct_benchmark
from systems import build_system
from utils import load_or_build, readf, dumpj, loadj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type=str, default='yelp')
    parser.add_argument('--system', type=str, default='ou') 
    # best, sat, ou
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--dset_root', type=str, default='.dset_root')
    # benchmark
    parser.add_argument('--num_test', type=int, default=10)
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
    args.testdata_path = args.cache_dir/f"test_data_{args.div_name}.json"

    reviews = DATA['REVIEWS'][city]
    tests = load_or_build(args.testdata_path, dumpj, loadj, construct_benchmark, reviews, args.num_test)
    
    todos = """
    Now we are at square 1, but we know where we want to go.
    1. Design method. Train model to disect evident unit
        We are now going into the details
            - Found model: TARGER or AMTM
            - LLM label
            - train
            (LLM in the loop guidance)
            1- validate, observe error
            2- random sample confidence score, LLM feedback
            3-> use LLM to create rule or device to retrieve more of a type of error??
            https://chatgpt.com/c/68cb3dbb-1c48-832f-8612-feb69e72e99b
    2. Baseline method.   

    Pipeline approach (proposed and baseline):
        -> review to exerpts segment
        -> cluster and retrieval model
        -> utility model
    End-to-end approach (baseline): eg sparse, dense, 
    """
    print(todos)
    System = build_system(args, reviews, tests)


if __name__ == '__main__':
    main()