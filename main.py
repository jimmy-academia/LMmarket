## python main.py --sample --dset yelp

import argparse
from dataset import make
from utils import version_path

from scheme import base

def main():
    print("Hello from LMmarket/main.py!")

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help="Use a small sample dataset")
    parser.add_argument('--use_feature_cache', action='store_true', help="Use cached feature")
    parser.add_argument('--feature_cache_path', type=str, default=None)
    parser.add_argument('--dset', type=str, default='yelp')
    parser.add_argument('--scheme', type=str, default='random')
    args = parser.parse_args()
    
    if args.feature_cache_path is None:
        args.feature_cache_path = version_path(f'cache/{args.dset}_feature_score.json')

    make.main(args)

    input('work on task loader!!')
    task_loader = get_task_loader(args)
    Scheme = setup_scheme(args, task_loader) # set up scheme for task
    Scheme.operate() # and record intermediate step/ final result


if __name__ == "__main__":
    main()
