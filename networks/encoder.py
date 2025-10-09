import torch
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def build_segment_embeddings(segments, device=None, batch_size=1024, show_progress=True):
    if not segments: raise ValueError("empty segments")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    it = range(0, len(segments), batch_size)
    if show_progress: it = tqdm(it, desc="[encoder] Encoding segments", ncols=88)

    embeddings = []
    for i in it:
        batch = segments[i:i+batch_size]
        with torch.no_grad():
            batch_emb = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,   # IP == cosine
                show_progress_bar=False,
            )
        embeddings.extend(batch_emb)

    matrix = np.asarray(embeddings, dtype="float32")
    return matrix

### faiss index

def faiss_dump(index, path):
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

def build_faiss_ivfpq_ip(embs: np.ndarray, nlist=8192, m=32, nbits=8, train_sz=300_000, ids=None, use_opq=True):
    assert embs.dtype == np.float32 and embs.ndim == 2
    N, d = embs.shape
    train_idx = np.random.choice(N, size=min(train_sz, N), replace=False)
    train = embs[train_idx]

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
        idmap = faiss.IndexIDMap2(index)
        idmap.add_with_ids(embs, np.asarray(ids, np.int64))
        index = idmap
    else:
        index.add(embs)

    _unwrap_ivf(index).nprobe = max(1, nlist // 80)  # ~1.25% lists
    return index

def faiss_search(index, q: np.ndarray, topk=10, nprobe=None):
    ivf = _unwrap_ivf(index)
    if nprobe is not None: ivf.nprobe = int(nprobe)
    return index.search(q.astype(np.float32, copy=False), topk)
