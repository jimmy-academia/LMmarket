# from .method import BestSystem
from .rm25 import BM25Baseline
from .dense import DenseRetrieverBaseline
# from .ou import OUBaseline
# from .sulm import SULMBaseline
# from .sat import SATBaseline as SegmentAnyTextBaseline


def build_system(args, data):
    # if args.system == 'best':
    #     print('[systems] >>> operating OUR PROPOSED METHOD')
    #     return BestSystem(args, data)
    # if args.system == 'sat':
        # print('[systems] >>> operating Segment Any Text baseline')
        # return SegmentAnyTextBaseline(args, data)
    if args.system == 'bm25':
        print('[systems] >>> operating BM25 retrieval baseline')
        return BM25Baseline(args, data)
    if args.system == 'dense':
        print('[systems] >>> operating dense retrieval baseline')
        return DenseRetrieverBaseline(args, data)
    # if args.system == 'ou':
    #     print('[systems] >>> operating Opinion Unit baseline')
    #     return OUBaseline(args, data)
    # if args.system == 'sulm':
    #     print('[systems] >>> operating Sentiment Utility Logistic Model baseline')
    #     return SULMBaseline(args, data)
    raise ValueError(f"Unknown system '{args.system}'")
