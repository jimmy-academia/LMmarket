from .rankbm25 import BM25Baseline


def build_system(args, data):
    name = getattr(args, 'system', 'bm25')
    if name == 'bm25':
        print('[systems] >>> operating BM25 retrieval baseline')
        return BM25Baseline(args, data)
    raise ValueError(f'Unknown system: {name}')
