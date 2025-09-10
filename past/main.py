## python main.py --sample --dset yelp --scheme random bm25 embed --top_m 5 --top_n 10

import argparse
from dataset import make
from loader import get_task_loader
from schemes import setup_scheme

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help="Use a small sample dataset")
    parser.add_argument('--dset', type=str, default='yelp')
    parser.add_argument('--scheme', nargs='*', type=str, help="Evaluated scheme(s)")
    parser.add_argument('--top_m', type=int, default=3, help="Select first 'm' items in ground truth ranking")
    parser.add_argument('--top_n', type=int, default=5, help="Retrieve 'n' items for evaluation")
    args = parser.parse_args()
    
    print("Hello from lmmarket!")
    make.main(args)    
    if not (args.scheme is None):
        print(f"Load benchmark dataset")
        task_loader = get_task_loader(args.dset)

        print(f"Evaluate schemes {args.scheme} with top_m = {args.top_m}, top_n = {args.top_n}")
        for scheme in args.scheme:
            Scheme = setup_scheme(args, task_loader, scheme)
            Scheme.operate()

if __name__ == "__main__":
    main()