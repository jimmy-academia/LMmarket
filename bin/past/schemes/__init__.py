from .random_ import Random
from .bm25 import BM25
from .embedding import Embedding

SCHEME_DICT = {
    'random': Random,
    'bm25': BM25,
    'embed': Embedding,
}

def setup_scheme(args, task_loader, scheme):
    if scheme in SCHEME_DICT:
        return SCHEME_DICT[scheme](args, task_loader)
    else:
        raise NotImplementedError(f"Scheme '{scheme}' not implemented")