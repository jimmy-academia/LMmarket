import logging
from .base import BaseSystem
from .method import MainMethod

def build_system(args, data):
    if args.system == 'base':
        logging.info('[systems] >>> operating base class for init')
        return BaseSystem(args, data)
    if args.system == 'method':
        logging.info('[systems] >>> operating main method')
        return MainMethod(args, data)
    raise ValueError(f"Unknown system '{args.system}'")
