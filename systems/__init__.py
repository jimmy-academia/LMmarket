import logging
from .base import BaseSystem
from .method import MainMethod
from .horizontal import HorizontalMethod

def build_system(args, data):
    logging.info(f"[systems] >>> operating {args.system}")
    if args.system == 'base':
        return BaseSystem(args, data)
    if args.system == 'horizontal':
        return HorizontalMethod(args, data)
    if args.system == 'method':
        return MainMethod(args, data)
    raise ValueError(f"Unknown system '{args.system}'")
