# main.py
import torch
import argparse
from pathlib import Path

from data import prepare_data, pick_city_data
from systems import build_system
from utils import load_or_build, readf, dumpj, loadj, _ensure_dir, set_seeds, set_logging, _ensure_pathref

import logging 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dset', type=str, default='yelp')
    parser.add_argument('--system', type=str, default='method')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--logs_dir', type=str, default='cache/logs')
    parser.add_argument('--clean_dir', type=str, default='cache/clean') # cleaned data
    parser.add_argument('--dset_root_ref', type=str, default='.dset_root')
    parser.add_argument('--openaiapi_key_ref', type=str, default='.openaiapi_key')
    parser.add_argument('--enc', type=str, default="f2llm")
    parser.add_argument('--normalize', type=str, default="true")
    parser.add_argument('--city', type=str, default=None)

    parser.add_argument('--top_k', type=int, default=5)
    return parser.parse_args()


def _to_bool(v):
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "n"):
        return False

MODEL_ALIASES = {
    "mini": "sentence-transformers/all-MiniLM-L6-v2",
    "ef": "intfloat/e5-large-v2",
    "jina": "jinaai/jina-embeddings-v3",
    "f2llm": "codefuse-ai/F2LLM-0.6B",
    "f2llm-0.6b": "codefuse-ai/F2LLM-0.6B",
    "f2llm-1.7b": "codefuse-ai/F2LLM-1.7B",
    "f2llm-4b": "codefuse-ai/F2LLM-4B",
}

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
    args.encoder_name = MODEL_ALIASES[args.enc]
    return args 

def main():
    args = get_arguments()
    args = _resolve_args(args)
    args.prepared_data_path = args.clean_dir/f'{args.dset}_data.json'
    data = load_or_build(args.prepared_data_path, dumpj, loadj, prepare_data, args)

    if args.city is not None:
        data = pick_city_data(data, args.city)
        logging.info(f"[main.py] downsizing to city: {args.city}")
        _num = lambda x: len(data[x].keys())
        logging.info("[main.py] # users {} # items {} # reviews {}".format(*map(_num, ['users', 'items', 'reviews'])))

    dict2list = lambda _dict: list(_dict.values())
    data = {k: dict2list(v) for k, v in data.items()}

    experiment_plan = """EXPERIMENTS for RETRIEVAL
    1. check how it is segmented: determine whether a segment really only has 1 aspect. does it need nearby segments for context?
    2. how to retrieve? compare using as query: aspect (keyword); aspect description; aspect definition; "find"-sentences; synonym expansion sentences
    3. what is retrieved? Checked if using similar keywords retrieve same results (when N is large). Check if non-overlaping results are important
    ///4. Embedding vector: what happen when shift emerging vector? How close are similar queries?///
    """
    print(experiment_plan)

    system = build_system(args, data)    
    system.recommend("Find a quiet, cozy cafe with comfortable seating and good natural light that's perfect for reading a book for a few hours.")
    # system.evaluate()

if __name__ == '__main__':
    main()

# I want a lively restaurant with great seafood, quick service, and outdoor seating, but it shouldn’t be too expensive.