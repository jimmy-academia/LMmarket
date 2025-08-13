## python main.py --sample --dset yelp

import argparse
from dataset import make
from utils import version_path

def main():
    print("Hello from LMmarket/main.py!")

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help="Use a small sample dataset")
    parser.add_argument('--use_feature_cache', action='store_true', help="Use cached feature")
    parser.add_argument('--feature_cache_path', type=str, default=None)
    parser.add_argument('--dset', type=str, default='yelp')
    args = parser.parse_args()
    
    if args.feature_cache_path is None:
        args.feature_cache_path = version_path(f'cache/{args.dset}_feature_score.json')

    make.main(args)
    


if __name__ == "__main__":
    main()
