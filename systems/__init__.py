from .best_system import BestSystem, SATBaseline, OUBaseline

def build_system(args, DATA, city):
    if args.system == 'best':
        return BestSystem(args, DATA, city)
    if args.system == 'sat':
        return SATBaseline(args, DATA, city)
    if args.system == 'ou':
        return OUBaseline(args, DATA, city)