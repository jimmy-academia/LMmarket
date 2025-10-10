import torch
# import faiss
import warnings
with warnings.catch_warnings():
    # remove sentence_transformer warnings
    warnings.filterwarnings(
        "ignore",
        message=r"(?i)builtin type (swigpyobject|swigpypacked|swigvarlink) has no __module__ attribute",
        category=DeprecationWarning,
        module=r"importlib\._bootstrap",
    )
    import faiss

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import loadp, dumpp

def build_segment_embeddings(segments, args, embedding_path, batch_size=1024, show_progress=True):
    # embedder_name="sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(args.embedder_name, device=args.device)

    partial_path = embedding_path.with_name(embedding_path.name + ".partial")
    partial_save_frequency = max(len(segments)//batch_size//10, 10)
    
    start_i, embeddings = 0, []
    N = len(segments)

    if partial_path.exists():
        embeddings = loadp(partial_path)
        start_i = len(embeddings)

    it = range(start_i, N, batch_size)
    if show_progress: it = tqdm(it, desc=f"[encoder] from {start_i}", ncols=88)

    for i in it:
        batch = segments[i:i+batch_size]
        with torch.no_grad():
            batch_emb = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=args.normalize,   # IP == cosine
                show_progress_bar=False,
            )
        embeddings.extend(batch_emb)

        if ((i-start_i)//batch_size+1) % partial_save_frequency == 0:
            dumpp(partial_path, embeddings)
            if show_progress:
                it.set_postfix(note=f"saved@{len(embeddings)//batch_size}")

    matrix = np.asarray(embeddings, dtype="float32")
    matrix = np.ascontiguousarray(matrix)
    partial_path.unlink(missing_ok=True)
    return matrix

### faiss index

def faiss_dump(path, index):
    faiss.write_index(index, str(path))
    return path
    
def faiss_load(path):
    return faiss.read_index(str(path))

def _unwrap_ivf(index):
    """Return underlying IVF index to set nprobe."""
    base = index
    if isinstance(base, faiss.IndexIDMap2): base = base.index
    if isinstance(base, faiss.IndexPreTransform): base = base.index
    return base  # should be IndexIVF

def _add_with_progress(index, embs, ids=None, batch=100_000):
    for i in tqdm(range(0, len(embs), batch), desc="[faiss] batch adding", ncols=88):
        xb = embs[i:i+batch]
        if ids is not None:
            index.add_with_ids(xb, np.asarray(ids[i:i+batch], np.int64))
        else:
            index.add(xb)

def build_faiss_ivfpq_ip(embs: np.ndarray, nlist=8192, m=32, nbits=8, train_sz=400_000, ids=None, use_opq=True):
    assert embs.dtype == np.float32 and embs.ndim == 2
    N, d = embs.shape
    train = embs
    quantizer = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    if use_opq:
        opq = faiss.OPQMatrix(d, m)
        opq.train(train)
        ivfpq.train(opq.apply_py(train))
        index = faiss.IndexPreTransform(opq, ivfpq)
    else:
        ivfpq.train(train)
        index = ivfpq

    if ids is not None:
        index = faiss.IndexIDMap2(index)
        _add_with_progress(index, embs, ids=ids)
    else:
        _add_with_progress(index, embs, ids=None)

    _unwrap_ivf(index).nprobe = max(1, nlist // 80)  # ~1.25% lists
    return index

def faiss_search(index, q: np.ndarray, topk=10, nprobe=None):
    ivf = _unwrap_ivf(index)
    if nprobe is not None: ivf.nprobe = int(nprobe)
    return index.search(q.astype(np.float32, copy=False), topk)
