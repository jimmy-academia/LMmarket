import faiss
import numpy as np


def build_index(vectors, ids, cfg):
    dim = vectors.shape[1]
    index_type = cfg.get("type")
    if index_type == "HNSW":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.add_with_ids(vectors.astype("float32"), np.array(ids, dtype="int64"))
        return index
    if index_type == "IVF-PQ":
        quantizer = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIVFPQ(quantizer, dim, cfg["ivf_nlist"], cfg["pq_m"], 8, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors.astype("float32"))
        index.add_with_ids(vectors.astype("float32"), np.array(ids, dtype="int64"))
        return index
    raise ValueError("index type must be 'HNSW' or 'IVF-PQ'")


def search_index(index, queries, k=10):
    return index.search(queries.astype("float32"), k)
