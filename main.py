# main.py
import torch
import argparse
from pathlib import Path

from data import prepare_data, process_data
from systems import build_system
from utils import load_or_build, readf, dumpj, loadj


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='yelp')
    parser.add_argument('--system', default='react')
    parser.add_argument('--cache_dir', default='cache')
    parser.add_argument('--dset_root', default='.dset_root')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--retrieve_k', type=int, default=500)
    parser.add_argument('--bm25_top_m', type=int, default=3)
    parser.add_argument('--device', type=int, default = 0)
    return parser.parse_args()


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

def _resolve_args(args):
    args.cache_dir = Path(args.cache_dir)
    args.cache_dir.mkdir(exist_ok=True)

    args.dset_root = Path(args.dset_root)
    if args.dset_root.is_file():
        args.dset_root = readf(args.dset_root).strip()
    else:
        print(f"Warning: place the dataset root dir in the path assigned to args.dset_root: {args.dset_root}")

    if args.device > torch.cuda.device_count():
        input(f'Warning: args.device {args.device} > device count {torch.cuda.device_count()}. Set to default index 0. Press anything to continue.')
        args.device = 0
    args.device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")

    return args 

def _build_data(args):
    prepared = args.cache_dir / f'prepared_{args.dset}_data.json'
    data = load_or_build(prepared, dumpj, loadj, prepare_data, args)
    processed = args.cache_dir / f'processed_{args.dset}_data.json'
    data['user_loc'] = load_or_build(processed, dumpj, loadj, process_data, args, data)
    data['test'] = loadj('data/test_data.json')
    return data

def _announce_city(system):
    city = system.default_city
    if city:
        print(f"[main] >>> default city set to '{city}'")
    return city

def _evaluate(system, args, city):
    if city:
        print(f"[main] >>> evaluating '{args.system}' for city '{city}' with top_k={args.top_k}")
        system.evaluate(city=city, top_k=args.top_k)
    else:
        print('[main] >>> no city data available; skipping evaluation.')

def main():
    args = get_arguments()
    _seed_defaults(args)
    args = _resolve_args(args)
    data = _build_data(args)
    system = build_system(args, data)
    city = _announce_city(system)
    _evaluate(system, args, city)


if __name__ == '__main__':
    main()
