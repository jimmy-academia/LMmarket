from .method import BestSystem, SATBaseline
from .ou import OUBaseline
# SATClusterBaseline, OUClusterBaseline

def build_system(args, reviews, tests):
    if args.system == 'best':
        print('[systems] >>> operating OUR PROPOSED METHOD segmentation')
        return BestSystem(args, reviews, tests)
    if args.system == 'sat':
        print('[systems] >>> operating Segment Any Text baseline segmentation')
        return SATBaseline(args, reviews, tests)
    # if args.system == 'sat_cluster':
    #     print('[systems] >>> operating SAT baseline with clustering enhancement')
    #     return SATClusterBaseline(args, DATA, city)
    if args.system == 'ou':
        print('[systems] >>> operating Opinion Unit baseline segmentation')
        return OUBaseline(args, reviews, tests)
    # if args.system == 'ou_cluster':
    #     print('[systems] >>> operating OU baseline with clustering enhancement')
    #     return OUClusterBaseline(args, DATA, city)