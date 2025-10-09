import logging
from .base import BaseSystem
from .sparse import BM25Baseline
from .dense import DenseRetrieverBaseline
from .react import ReactRetrievalBaseline

def build_system(args, data):
    if args.system == 'base':
        logging.info('[systems] >>> operating base class for init')
        return BaseSystem(args, data)
    if args.system == 'sparse':
        logging.info('[systems] >>> operating BM25 retrieval baseline')
        return BM25Baseline(args, data)
    if args.system == 'dense':
        logging.info('[systems] >>> operating dense retrieval baseline')
        return DenseRetrieverBaseline(args, data)
    if args.system == 'react':
        logging.info('[systems] >>> operating react retrieval baseline')
        return ReactRetrievalBaseline(args, data)
    if args.system == 'method':
        logging.info('[systems] >>> operating hyperbolic segment method (mine)')
        return HyperbolicSegmentSystem(args, data)
    raise ValueError(f"Unknown system '{args.system}'")
