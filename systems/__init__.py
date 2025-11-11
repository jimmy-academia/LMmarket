import logging
from .base import BaseSystem
from .method import MainMethod
from .aspect_match import ASPECT_MATCH_Method

def build_system(args, data):
    logging.info(f"[systems] >>> operating {args.system}")
    if args.system == 'base':
        return BaseSystem(args, data)
    if args.system == 'method':
        return MainMethod(args, data)
    if args.system == 'aspect_match':
        return ASPECT_MATCH_Method(args, data)
    raise ValueError(f"Unknown system '{args.system}'")
