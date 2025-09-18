from .process_yelp import process_yelp_data

def process_data(args):
    if args.dset == 'yelp':
        return process_yelp_data(args.dset_root)
    else:
        print(f'[foundation/__init__.py] WARNING, process_{args.dset}_data function not implemented.')

## === deprecate ==
# def fetch_embedder(args, reviews):
#     if args.dset == 'yelp':
#         return Yelp_Embedder(args, reviews)
#     else:
#         print(f'[foundation/__init__.py] WARNING, vectorize_{args.dset}_embedding function not implemented.')

# from .extract_yelp import extract_yelp_features


# def extract_feature(args, reviews):
#     if args.dset == 'yelp':
#         return extract_yelp_features(reviews)
#     else:
#         print(f'[foundation/__init__.py] WARNING, extract_{args.dset}_feature function not implemented.')
