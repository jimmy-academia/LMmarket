from .method import HyperbolicSegmentSystem
from .sparse import BM25Baseline
from .dense import DenseRetrieverBaseline


def build_system(args, data):
    if args.system == 'sparse':
        print('[systems] >>> operating BM25 retrieval baseline')
        return BM25Baseline(args, data)
    if args.system == 'dense':
        print('[systems] >>> operating dense retrieval baseline')
        return DenseRetrieverBaseline(args, data)
    if args.system == 'method':
        print('[systems] >>> operating hyperbolic segment method (mine)')
        return HyperbolicSegmentSystem(args, data)
    raise ValueError(f"Unknown system '{args.system}'")
