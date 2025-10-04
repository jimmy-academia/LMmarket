import logging
from .yelp import prepare_yelp_data

def prepare_data(args):
    if args.dset == 'yelp':
        return prepare_yelp_data(args.dset_root)
    else:
        logging.error(f'[data/__init__.py] ERROR, prepare_{args.dset}_data function not implemented for {args.dset}.')
