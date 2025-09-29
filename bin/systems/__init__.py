from .method import BestSystem, SATBaseline
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
    if args.system == 'ou':
        print('[systems] >>> operating Opinion Unit baseline')
        return OUBaseline(args, reviews, tests)
    if args.system == 'sulm':
        print('[systems] >>> operating Sentiment Utility Logistic Model baseline')
        return SULMBaseline(args, reviews, tests)