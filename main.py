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
import json
import numpy as np
from pathlib import Path

import argparse

from data import prepare_data, process_data
from systems import build_system
from utils import load_or_build, readf, dumpj, loadj, dumpp, loadp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type=str, default='yelp')
    parser.add_argument('--system', type=str, default='react')
    # heur, sat, ou, sulm, sugar
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--dset_root', type=str, default='.dset_root')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--retrieve_k', type=int, default=500)
    parser.add_argument('--bm25_top_m', type=int, default=3)
    # benchmark
    parser.add_argument('--num_test', type=int, default=2)
    parser.add_argument('--div_name', type=str, default=None)
    parser.add_argument('--segment_candidate_segments', type=int, default=50)
    parser.add_argument('--segment_top_m', type=int, default=3)
    parser.add_argument('--segment_encode_batch_size', type=int, default=8)
    parser.add_argument('--segment_train_samples', type=int, default=64)
    parser.add_argument('--segment_learning_rate', type=float, default=2e-5)
    parser.add_argument('--segment_checkpoint', type=str, default=None)
    parser.add_argument('--segment_model_config', type=json.loads, default=None)
    parser.add_argument('--segment_backbone', default=None)
    parser.add_argument('--segment_pooling', default=None)
    parser.add_argument('--segment_hidden_dim', type=int, default=None)
    parser.add_argument('--segment_aspect_dim', type=int, default=None)
    parser.add_argument('--segment_sentiment_dim', type=int, default=None)
    parser.add_argument('--segment_lambda_aspect', type=float, default=None)
    parser.add_argument('--segment_lambda_sentiment', type=float, default=None)
    parser.add_argument('--segment_aspect_temperature', type=float, default=None)
    parser.add_argument('--segment_sentiment_temperature', type=float, default=None)
    parser.add_argument('--segment_sentiment_loss', default=None)
    parser.add_argument('--segment_sentiment_margin', type=float, default=None)
    parser.add_argument('--segment_curvature', type=float, default=None)
    parser.add_argument('--segment_max_length', type=int, default=None)
    args = parser.parse_args()

    new_todo = """
    a. implement main approach, run training, inference (todo with systems/sugar.py)
    b. implement explanable evaluation with LLM
    8. finalize evaluation comparison
    9. introduce distance and conduct final experiment design!
    """
    print(new_todo)

    args.cache_dir = Path(args.cache_dir)
    args.cache_dir.mkdir(exist_ok=True)
    dset_root_path = Path(args.dset_root)
    if dset_root_path.is_file():
        content = readf(dset_root_path).strip()
        args.dset_root = Path(content) if content else dset_root_path.parent
    else:
        args.dset_root = dset_root_path
    prepared_data_path = args.cache_dir / f"prepared_{args.dset}_data.json"
    data = load_or_build(prepared_data_path, dumpj, loadj, prepare_data, args)
    processed_data_path = args.cache_dir / f"processed_{args.dset}_data.json"
    data['user_loc'] = load_or_build(processed_data_path, dumpj, loadj, process_data, args, data)

    test_data_path = "data/test_data.json"
    data['test'] = loadj(test_data_path)

    system = build_system(args, data)
    default_city = getattr(system, 'default_city', None)
    if default_city:
        print(f"[main] >>> default city set to '{default_city}'")

    city = default_city
    if city:
        print(f"[main] >>> evaluating '{args.system}' for city '{city}' with top_k={args.top_k}")
        system.evaluate(city=city, top_k=args.top_k)
    else:
        print("[main] >>> no city data available; skipping evaluation.")


if __name__ == '__main__':
    main()
