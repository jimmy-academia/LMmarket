# main.py
import torch
import argparse
from pathlib import Path

from data import prepare_data
from systems import build_system
from utils import load_or_build, readf, dumpj, loadj, _ensure_dir, set_seeds, set_logging, _ensure_pathref

import logging 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dset', type=str, default='yelp')
    parser.add_argument('--system', type=str, default='react')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--logs_dir', type=str, default='cache/logs')
    parser.add_argument('--dset_root_ref', type=str, default='.dset_root')
    parser.add_argument('--openaiapi_key_ref', type=str, default='.openaiapi_key')
    return parser.parse_args()

def _resolve_args(args):
    _ensure_dir(args.cache_dir)
    _ensure_dir(args.logs_dir)
    set_seeds(args.seed)
    set_logging(args.verbose, args.logs_dir)

    args.dset_root = _ensure_pathref(args.dset_root_ref)
    args.openaiapi_key = _ensure_pathref(args.openaiapi_key_ref)
    
    if args.device > torch.cuda.device_count():
        logging.warning(f'Warning: args.device {args.device} > device count {torch.cuda.device_count()}.')
        input('args.device set to default index 0. Press anything to continue.')
        args.device = 0
    args.device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")

    return args 

def main():
    args = get_arguments()
    args = _resolve_args(args)
    data = prepare_data(args)
    system = build_system(args, data)


if __name__ == '__main__':
    main()


'''
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--retrieve_k', type=int, default=500)
    parser.add_argument('--bm25_top_m', type=int, default=3)
def _seed_defaults(args):
    args.div_name = 'default'
    args.segment_batch_size = 32
    args.min_user_location_reviews = 5
    args.react_temperature = 0
    args.react_summary_k = 3
    args.encode_batch_size = 64
    args.normalize_embeddings = True
    args.top_l_segments = 3
    args.top_docs = 3
    args.faiss_topk = 256
    args.segment_temperature = 0.1

'''