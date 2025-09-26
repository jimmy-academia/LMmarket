"""LMmarket pipeline entry point."""

import argparse
from pathlib import Path

from utils import load_or_build, readf, dumpj, loadj
from pipeline import prepare_basic, process_build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='yelp')
    parser.add_argument('--cache_dir', default='cache')
    parser.add_argument('--dset_root', default='.dset_root')
    parser.add_argument('--seg_model', default='sat', choices=['sat'])
    parser.add_argument('--embed_model', default='gte-large', choices=['gte-large', 'bge-large', 'trained'])
    parser.add_argument('--clusterer', default='hdbscan', choices=['kmeans', 'hdbscan'])
    parser.add_argument('--absa_model', default='pyabsa-restaurants', choices=['pyabsa-restaurants'])
    parser.add_argument('--labeler', default='off', choices=['off', 'ctfidf'])
    parser.add_argument('--topk_opinion_units', type=int, default=64)
    parser.add_argument('--threshold_mode', default='none', choices=['none'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.cache_dir = Path(args.cache_dir)
    args.cache_dir.mkdir(exist_ok=True)
    args.dset_root = Path(readf(args.dset_root).strip())

    args.prepared_data_path = args.cache_dir / f"prepare_{args.dset}.json"
    prep = load_or_build(args.prepared_data_path, dumpj, loadj, prepare_basic, args)
    print('[main] STEP-1 = PREPARE complete -> PREP')

    args.processed_data_path = args.cache_dir / f"processed_{args.dset}.json"
    proc = load_or_build(args.processed_data_path, dumpj, loadj, process_build, args, prep)
    print('[main] STEP-2 = PROCESS complete -> PROC')

    print('[main] STEP-3 = APPS pending (PROC-only consumers)')


if __name__ == '__main__':
    main()