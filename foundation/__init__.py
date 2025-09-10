from .process_yelp import process_yelp_data
from .vectorize_yelp import vectorize_yelp_embedding


def process_data(args):
    if args.dset == 'yelp':
        return process_yelp_data(args.dset_root)
    else:
        print(f'[foundation/__init__.py] WARNING, process_{args.dset}_data function not implemented.')

def vectorize_embedding(args, reviews):
    if args.dset == 'yelp':
        return vectorize_yelp_embedding(reviews)
    else:
        print(f'[foundation/__init__.py] WARNING, vectorize_{args.dset}_embedding function not implemented.')



## === deprecate ==
from .extract_yelp import extract_yelp_features


def extract_feature(args, reviews):
    if args.dset == 'yelp':
        return extract_yelp_features(reviews)
    else:
        print(f'[foundation/__init__.py] WARNING, extract_{args.dset}_feature function not implemented.')
