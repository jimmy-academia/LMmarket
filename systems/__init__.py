from .best_system import BestSystem, SATBaseline, OUBaseline

def build_system(args, DATA, city):
    if args.system == 'best':
        print('[systems] >>> operating OUR PROPOSED METHOD segmentation')
        return BestSystem(args, DATA, city)
    if args.system == 'sat':
        print('[systems] >>> operating Segment Any Text baseline segmentation')
        return SATBaseline(args, DATA, city)
    if args.system == 'ou':
        print('[systems] >>> operating Opinion Unit baseline segmentation')
        return OUBaseline(args, DATA, city)