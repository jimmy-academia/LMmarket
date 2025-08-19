from .random_baseline import Random

SCHEME_DICT = {
    'random': Random
}

def setup_scheme(args):
    if args.scheme in SCHEME_DICT:
        return SCHEME_DICT[args.scheme](args)
    else:
        raise NotImplementedError(f"prompt scheme {args.scheme} not implemented")