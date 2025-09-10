import json
from pathlib import Path

import torch
import numpy as np

import faiss
from blingfire import text_to_sentences
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

import warnings
# ---------------- Config ----------------

@dataclass
class Config:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256
    max_chars: int = 300
    min_merge: int = 40
    normalize: bool = True
    index_type: str = 'hnsw'

# -------------- Main function -------------

def vectorize_yelp_embedding(reviews):
    '''
    reviews is a list of dict, dict['text'] = 'the review text'
    '''
    cfg = Config()

    reviews = [r['text'] for r in reviews]
    chunks = [split_to_spans(rtext, cfg.max_chars, cfg.min_merge) for rtext in reviews]
    flat, offsets = flatten_with_offsets(chunks)
    vecs = embed_texts(flat, cfg.model_name, cfg.batch_size, cfg.normalize)
    index = build_index(vecs, cfg)
    index = faiss.index_gpu_to_cpu(index)

    meta = {
        "model_name": cfg.model_name,
        "dim": int(vecs.shape[1]),
        "n_vectors": int(vecs.shape[0]),
        "n_reviews": len(chunks),
        "normalize": bool(cfg.normalize),
        "paths": {
            "embeddings": str("embeddings.npy"),
            "index": str("faiss.index"),
        },
    }
    return meta, index, vecs, offsets


# -------------- Quick search --------------

def load_index_and_search(bundle, queries, top_k=10):
    index = faiss.read_index(bundle["paths"]["faiss_index"])
    D, I = index.search(queries.astype(np.float32), top_k)
    return I, D

# -------------- Modules --------------

def split_to_spans(text, max_chars, min_merge):
    sents = text_to_sentences(text.strip()).split("\n")
    spans, buf = [], ""
    for s in sents:
        if not s: continue
        if buf and len(buf) + 1 + len(s) > max_chars:
            spans.append(buf); buf = s
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf: spans.append(buf)

    # Merge tiny spans
    out, i = [], 0
    while i < len(spans):
        if len(spans[i]) < min_merge and i + 1 < len(spans):
            out.append(spans[i] + " " + spans[i+1]); i += 2
        else:
            out.append(spans[i]); i += 1
    return out

def flatten_with_offsets(chunks):
    """Flatten list-of-lists and record start/end offsets for each group."""
    flat = [s for review in chunks for s in review]
    lengths = [len(review) for review in chunks]
    offsets = np.cumsum([0] + lengths)  # shape (n_reviews+1,)
    return flat, offsets

def embed_texts(texts, model_name, batch_size, normalize=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        warnings.warn("CUDA is not available, using CPU instead.", RuntimeWarning)


    model = SentenceTransformer(model_name, device=device)
    with torch.no_grad():
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=True,
            normalize_embeddings=normalize,
        )
    return embs.cpu().numpy().astype("float32")

# -------------- FAISS index --------------
def build_index(vecs, cfg):
    d = vecs.shape[1]
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, d)  # inner product (use L2-normalized vecs for cosine)
    index.add(vecs)
    return index

def build_index_cpu(vecs, cfg):
    d = vecs.shape[1]
    if cfg.index_type.upper() == "HNSW":
        index = faiss.IndexHNSWFlat(d, cfg["hnsw_M"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg["hnsw_efConstruction"]
        index.add(vecs)
        return index
    if cfg.index_type.upper() == "IVF-PQ":
        quant = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIVFPQ(quant, d, cfg["ivf_nlist"], cfg["pq_m"], 8, faiss.METRIC_INNER_PRODUCT)
        n = vecs.shape[0]
        if n <= cfg["train_sample"]:
            train_x = vecs
        else:
            rs = np.random.RandomState(123)
            train_x = vecs[rs.choice(n, size=cfg["train_sample"], replace=False)]
        index.train(train_x)
        index.add(vecs)
        return index
    raise ValueError("index_type must be 'HNSW' or 'IVF-PQ'")


