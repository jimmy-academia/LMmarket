from .method import BestSystem, SATBaseline
from .bm25 import BM25Baseline
from .ou import OUBaseline
from .sulm import SULMBaseline
from .sat import SATBaseline

def build_system(args, reviews, tests):
    if args.system == 'best':
        print('[systems] >>> operating OUR PROPOSED METHOD')
        return BestSystem(args, reviews, tests)
    if args.system == 'sat':
        print('[systems] >>> operating Segment Any Text baseline')
        return SATBaseline(args, reviews, tests)
    if args.system == 'bm25':
        print('[systems] >>> operating BM25 retrieval baseline')
        return BM25Baseline(args, reviews)
    if args.system == 'ou':
        print('[systems] >>> operating Opinion Unit baseline')
        return OUBaseline(args, reviews, tests)
    if args.system == 'sulm':
        print('[systems] >>> operating Sentiment Utility Logistic Model baseline')
        return SULMBaseline(args, reviews, tests)