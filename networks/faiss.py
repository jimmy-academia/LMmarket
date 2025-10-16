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
from tqdm import tqdm  # needed for _add_with_progress

from utils import dumpj, loadj
# ---------- basics ----------
def faiss_dump(path, payload):
    index, ctx = payload
    faiss.write_index(index, str(path))
    dumpj(path.with_suffix(".json"), ctx)
    return path

def faiss_load(path):
    return faiss.read_index(str(path)), loadj(path.with_suffix(".json"))

def _unwrap_ivf(index):
    base = index
    if isinstance(base, faiss.IndexIDMap2): base = base.index
    if isinstance(base, faiss.IndexPreTransform): base = base.index
    return base  # IVF if you built IVF

def _add_with_progress(index, embedding, ids=None, batch=100_000):
    for i in tqdm(range(0, len(embedding), batch), desc="[faiss] batch adding", ncols=88):
        xb = embedding[i:i+batch]
        if ids is not None:
            index.add_with_ids(xb, np.asarray(ids[i:i+batch], np.int64))
        else:
            index.add(xb)

# ---------- 1) build ----------
# method: "hnsw" | "ivf_flat" | "flat"
# metric: "cosine" | "ip" | "l2"
def build_faiss(
    embedding,
    method="hnsw",
    metric="cosine",
    ids=None,
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

    # metric
    if metric == "cosine":
        faiss.normalize_L2(xb)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "ip":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        faiss_metric = faiss.METRIC_L2
    else:
        raise ValueError("metric must be 'cosine'|'ip'|'l2'")

    # ids & id->row map (needed for exact rerank when ids are custom)
    if ids is None:
        ids = np.arange(N, dtype=np.int64)
        row_by_id = None
    else:
        ids = np.asarray(ids, dtype=np.int64)
        assert ids.shape == (N,), "ids must match N"
        row_by_id = {int(ids[i]): i for i in range(N)}

    # base index
    if method == "flat":
        index_base = faiss.IndexFlatIP(d) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)

    elif method == "hnsw":
        index_base = faiss.IndexHNSWFlat(d, int(M))
        index_base.hnsw.efConstruction = int(ef_construction)  # HNSW is L2; cosine works via unit vectors

    elif method == "ivf_flat":
        if nlist is None:
            nlist = int(np.clip(8 * np.sqrt(N), 4096, 32768))  # recall-first default
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
    # IMPORTANT: add the (possibly normalized) xb, not the original embedding
    _add_with_progress(index, xb, ids=ids)

    ctx = {
        "embedding": xb,                # vectors inside the index (maybe normalized)
        "ids": ids,
        "row_by_id": row_by_id,         # None if ids == arange(N)
        "metric": metric,
        "faiss_metric": faiss_metric,
        "method": method,
        "N": N, "d": d,
        "nlist": nlist,
        "nprobe_default": getattr(index_base, "nprobe", None),
        "xb_sq": ((xb**2).sum(axis=1).astype(np.float32) if metric == "l2" else None),
    }
    return index, ctx

# ---------- 2) exact rerank (batch) ----------
def faiss_rerank_exact(Xq, candI, k, ctx):
    xb = ctx["embedding"]; ids = ctx["ids"]; row_by_id = ctx["row_by_id"]
    metric = ctx["metric"]; xb_sq = ctx["xb_sq"]

    Q = Xq.shape[0]
    D = np.empty((Q, k), dtype=np.float32)
    I = np.empty((Q, k), dtype=np.int64)

    for qi in range(Q):
        cand = candI[qi]
        cand = cand[cand >= 0]
        if cand.size == 0:
            D[qi].fill(-np.inf if metric in ("cosine", "ip") else np.inf)
            I[qi].fill(-1)
            continue

        if row_by_id is None:
            rows = cand
        else:
            rows = np.fromiter((row_by_id.get(int(x), -1) for x in cand), dtype=np.int64)
            rows = rows[rows >= 0]
            if rows.size == 0:
                D[qi].fill(-np.inf if metric in ("cosine", "ip") else np.inf)
                I[qi].fill(-1)
                continue

        if metric in ("cosine", "ip"):
            s = xb[rows] @ Xq[qi]
            keep = np.argpartition(s, -k)[-k:]
            ord_ = np.argsort(s[keep])[::-1]
            D[qi] = s[keep][ord_]
            I[qi] = ids[rows[keep]][ord_]
        else:
            q = Xq[qi]
            qsq = float((q*q).sum())
            dots = xb[rows] @ q
            dist2 = xb_sq[rows] + qsq - 2.0 * dots
            keep = np.argpartition(dist2, k)[:k]
            ord_ = np.argsort(dist2[keep])
            D[qi] = dist2[keep][ord_]
            I[qi] = ids[rows[keep]][ord_]
    return D, I

# ---------- 3) search (stage-1 + optional rerank) ----------
def faiss_search(index, Xq, ctx, k=50, refine=True, refine_k_factor=50, ef_search=None, nprobe=None):
    xq = Xq.astype(np.float32, copy=False)
    xq = np.ascontiguousarray(xq)
    if ctx["metric"] == "cosine":
        faiss.normalize_L2(xq)

    base = index.index if isinstance(index, faiss.IndexIDMap2) else index
    if ctx["method"] == "hnsw":
        base.hnsw.efSearch = int(ef_search if ef_search is not None else max(200, 8*k))
    elif ctx["method"] == "ivf_flat":
        np_override = nprobe if nprobe is not None else ctx["nprobe_default"]
        if np_override is not None:
            base.nprobe = int(np_override)

    k1 = k * refine_k_factor if refine else k
    D1, I1 = index.search(xq, k1)
    if not refine:
        return D1[:, :k], I1[:, :k]
    return faiss_rerank_exact(xq, I1, k, ctx)
