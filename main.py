from pathlib import Path

import argparse

from data import prepare_data, process_data
from systems import build_system
from utils import load_or_build, readf, dumpj, loadj


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='yelp')
    parser.add_argument('--system', default='react')
    parser.add_argument('--cache_dir', default='cache')
    parser.add_argument('--dset_root', default='.dset_root')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--retrieve_k', type=int, default=500)
    parser.add_argument('--bm25_top_m', type=int, default=3)
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


def _resolve_cache_dir(raw):
    cache_dir = Path(raw)
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _resolve_dset_root(raw):
    path = Path(raw)
    if path.is_file():
        content = readf(path).strip()
        if content:
            return Path(content)
        return path.parent
    return path


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
    args = _parse_cli()
    _seed_defaults(args)
    args.cache_dir = _resolve_cache_dir(args.cache_dir)
    args.dset_root = _resolve_dset_root(args.dset_root)
    data = _build_data(args)
    system = build_system(args, data)
    city = _announce_city(system)
    _evaluate(system, args, city)


if __name__ == '__main__':
    main()
