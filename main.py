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
    parser.add_argument('--system', type=str, default='dense')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--logs_dir', type=str, default='cache/logs')
    parser.add_argument('--clean_dir', type=str, default='cache/clean') # cleaned data
    parser.add_argument('--dset_root_ref', type=str, default='.dset_root')
    parser.add_argument('--openaiapi_key_ref', type=str, default='.openaiapi_key')
    parser.add_argument('--embedder_name', type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument('--normalize', type=str, default="true")

    parser.add_argument('--top_k', type=int, default=5)
    return parser.parse_args()


def _to_bool(v):
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "n"):
        return False

def _resolve_args(args):
    args.cache_dir = _ensure_dir(args.cache_dir)
    args.logs_dir = _ensure_dir(args.logs_dir)
    args.clean_dir = _ensure_dir(args.clean_dir)
    set_seeds(args.seed)
    set_logging(args.verbose, args.logs_dir)

    args.dset_root = _ensure_pathref(args.dset_root_ref)
    args.openaiapi_key = _ensure_pathref(args.openaiapi_key_ref)
    
    if args.device > torch.cuda.device_count():
        logging.warning(f'Warning: args.device {args.device} > device count {torch.cuda.device_count()}.')
        input('args.device set to default index 0. Press anything to continue.')
        args.device = 0
    args.device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")

    args.normalize = _to_bool(args.normalize)
    return args 

def main():
    args = get_arguments()
    args = _resolve_args(args)
    args.prepared_data_path = args.clean_dir/f'{args.dset}_data.json'
    data = load_or_build(args.prepared_data_path, dumpj, loadj, prepare_data, args)
    system = build_system(args, data)
    system.recommend("Find a quiet, cozy cafe with comfortable seating and good natural light that's perfect for reading a book for a few hours.")
    system.evaluate()

if __name__ == '__main__':
    main()