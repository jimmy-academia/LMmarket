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

F2_MODEL_ALIASES = {
    "F2LLM-0.6B": "codefuse-ai/F2LLM-0.6B",
    "F2LLM-1.7B": "codefuse-ai/F2LLM-1.7B",
    "F2LLM-4B": "codefuse-ai/F2LLM-4B",
}

def _resolve_embedder_name(name):
    if name in F2_MODEL_ALIASES:
        return F2_MODEL_ALIASES[name]
    return name

class F2Encoder:
    def __init__(self, model_name, device):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device if device else "cpu")
        self.model_name = _resolve_embedder_name(model_name)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        load_kwargs = {}
        if self.device.type == "cuda":
            load_kwargs["torch_dtype"] = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False):
        import torch.nn.functional as F

        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype="float32")
        seqs = []
        eos = self.tokenizer.eos_token or ""
        for t in texts:
            seqs.append(t + eos)
        batch = self.tokenizer(
            seqs,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
        eos_pos = batch["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(len(seqs), device=self.device)
        emb = hidden[idx, eos_pos]
        if normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=1)
        emb = emb.float()
        if convert_to_numpy:
            return emb.cpu().numpy()
        return emb

def get_text_encoder(model_name, device):
    resolved = _resolve_embedder_name(model_name)
    if resolved.startswith("codefuse-ai/F2LLM"):
        return F2Encoder(resolved, device)
    return SentenceTransformer(resolved, device=device)

def build_segment_embeddings(segments, args, embedding_path, batch_size=1024, show_progress=True):
    # embedder_name="sentence-transformers/all-MiniLM-L6-v2"
    model = get_text_encoder(args.embedder_name, args.device)

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
