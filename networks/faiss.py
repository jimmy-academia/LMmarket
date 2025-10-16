import warnings
with warnings.catch_warnings():
    # remove faiss warnings
    warnings.filterwarnings(
        "ignore",
        message=r"(?i)builtin type (swigpyobject|swigpypacked|swigvarlink) has no __module__ attribute",
        category=DeprecationWarning,
        module=r"importlib\._bootstrap",
    )
    import faiss
import numpy as np
from tqdm import tqdm 

from utils import dumpj, loadj

# ---------- basics ----------
def faiss_dump(path, payload):
    index, ctx = payload
    faiss.write_index(index, str(path))
    dumpj(path.with_suffix(".json"), ctx)
    return path

def faiss_load(path):
    return (faiss.read_index(str(path)), loadj(path.with_suffix(".json")))

def _add_with_progress(index, embedding, ids, batch=100_000):
    for i in tqdm(range(0, len(embedding), batch), desc="[faiss] batch adding", ncols=88):
        xb = embedding[i:i+batch]
        index.add_with_ids(xb, np.asarray(ids[i:i+batch], np.int64))
        # else:
            # index.add(xb)

# ---------- 1) build ----------
# method: "hnsw" | "ivf_flat" | "flat"
# metric: "cosine" | "ip" | "l2"

def build_faiss(
    embedding,
    method="flat",
    metric="cosine",
    M=32,
    ef_construction=400,
    nlist=None,
    nprobe=None,
    normalize_inplace=True,
    seed=42,
):
    assert embedding.ndim == 2 and embedding.dtype == np.float32, "embedding must be float32 [N,d]"
    xb = embedding if normalize_inplace else embedding.copy()
    xb = np.ascontiguousarray(xb)
    N, d = xb.shape
    rng = np.random.default_rng(seed)

    # ----- metric -----
    if metric == "cosine":
        faiss.normalize_L2(xb)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "ip":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        faiss_metric = faiss.METRIC_L2
    else:
        raise ValueError("metric must be 'cosine'|'ip'|'l2'")

    # ----- ids -----
    ids = np.arange(N, dtype=np.int64)

    # ----- index -----
    if method == "flat":
        index_base = faiss.IndexFlatIP(d) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
    elif method == "hnsw":
        index_base = faiss.IndexHNSWFlat(d, int(M))
        index_base.hnsw.efConstruction = int(ef_construction)
    elif method == "ivf_flat":
        if nlist is None:
            nlist = int(np.clip(8 * np.sqrt(N), 4096, 32768))
        quantizer = faiss.IndexFlatIP(d) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
        index_base = faiss.IndexIVFFlat(quantizer, d, int(nlist), faiss_metric)
        train_size = min(N, 1_000_000)
        train_sample = xb[rng.choice(N, size=train_size, replace=False)]
        index_base.train(train_sample)
        if nprobe is None:
            nprobe = max(64, int(nlist // 16))
        index_base.nprobe = int(min(nprobe, nlist))
    else:
        raise ValueError("method must be 'flat'|'hnsw'|'ivf_flat'")

    index = faiss.IndexIDMap2(index_base)
    _add_with_progress(index, xb, ids=ids)

    # ----- context -----
    # no numpy arrays inside ctx; only metadata / scalars
    ctx = {
        "metric": metric,
        "method": method,
        "N": N,
        "d": d,
        "nlist": nlist,
        "nprobe_default": getattr(index_base, "nprobe", None),
        "xb_sq_used": metric == "l2",
    }

    return index, ctx

def faiss_search(Xq, index, ctx, k=10, ef_search=None, nprobe=None, refine=True, embedding=None, xb_sq=None):
    """
    Perform FAISS search with optional exact rerank.
    embedding: the full corpus embedding (needed only if refine=True)
    ids: np.ndarray of vector IDs (same length as embedding)
    """
    xq = np.ascontiguousarray(Xq.astype(np.float32, copy=False))
    metric = ctx["metric"]

    # --- normalize query for cosine ---
    if metric == "cosine":
        faiss.normalize_L2(xq)

    base = index.index if isinstance(index, faiss.IndexIDMap2) else index

    # --- set search parameters ---
    if ctx["method"] == "hnsw":
        base.hnsw.efSearch = int(ef_search if ef_search is not None else max(200, 8*k))
    elif ctx["method"] == "ivf_flat":
        base.nprobe = int(nprobe if nprobe is not None else ctx["nprobe_default"])

    # --- stage 1: coarse FAISS search ---
    k1 = k * 50 if refine else k
    D1, I1 = index.search(xq, k1)

    if not refine:
        return D1[:, :k], I1[:, :k]

    # --- stage 2: exact rerank (requires full embedding) ---
    assert embedding is not None, "embedding must be provided for refine=True"
    ids = np.arange(ctx["N"], dtype=np.int64)
    xb = np.ascontiguousarray(embedding.astype(np.float32, copy=False))
    row_by_id = {int(ids[i]): i for i in range(len(ids))}

    # precompute xb_sq only if needed
    # xb_sq = (xb**2).sum(axis=1).astype(np.float32) if metric == "l2" else None

    Q, _ = xq.shape
    D = np.empty((Q, k), dtype=np.float32)
    I = np.empty((Q, k), dtype=np.int64)

    for qi in range(Q):
        cand = I1[qi][I1[qi] >= 0]
        if cand.size == 0:
            fill_val = -np.inf if metric in ("cosine", "ip") else np.inf
            D[qi].fill(fill_val)
            I[qi].fill(-1)
            continue

        rows = np.fromiter((row_by_id.get(int(x), -1) for x in cand), dtype=np.int64)
        rows = rows[rows >= 0]
        if rows.size == 0:
            fill_val = -np.inf if metric in ("cosine", "ip") else np.inf
            D[qi].fill(fill_val)
            I[qi].fill(-1)
            continue

        if metric in ("cosine", "ip"):
            s = xb[rows] @ xq[qi]
            keep = np.argpartition(s, -k)[-k:]
            ord_ = np.argsort(s[keep])[::-1]
            D[qi] = s[keep][ord_]
            I[qi] = ids[rows[keep]][ord_]
        else:  # L2
            q = xq[qi]
            qsq = float((q*q).sum())
            dots = xb[rows] @ q
            dist2 = xb_sq[rows] + qsq - 2.0 * dots
            keep = np.argpartition(dist2, k)[:k]
            ord_ = np.argsort(dist2[keep])
            D[qi] = dist2[keep][ord_]
            I[qi] = ids[rows[keep]][ord_]

    return D, I
