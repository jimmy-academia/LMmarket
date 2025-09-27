from .prepare_yelp import prepare_yelp_data
from .process_yelp import process_yelp_data

def prepare_data(args):
    if args.dset == 'yelp':
        return prepare_yelp_data(args.dset_root)
    else:
        print(f'[data/__init__.py] WARNING, prepare_{args.dset}_data function not implemented for {args.dset}.')

def process_data(args, DATA):
    if args.dset == 'yelp':
        return process_yelp_data(args, DATA)
    else:
        print(f'[data/__init__.py] WARNING, process_{args.dset}_data function not implemented for {args.dset}.')
