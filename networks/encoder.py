import logging
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F

from tqdm import tqdm

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
    # import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from utils import loadp, dumpp


def mute_flash_attn_warning():
    class _FlashFilter(logging.Filter):
        def filter(self, record):
            return "flash_attn is not installed" not in record.getMessage()

    # Hit the usual sources
    for name in ("py.warnings", "transformers", "sentence_transformers"):
        logging.getLogger(name).addFilter(_FlashFilter())

    # Also attach to existing handlers in case the warning
    # is bubbling to root via handlers only
    for h in logging.getLogger().handlers:
        h.addFilter(_FlashFilter())

def get_text_encoder(encoder_name, device):
    encoderclass = F2Encoder if encoder_name.startswith("codefuse-ai/F2LLM") else SentenceTransformer
    if "jina" in encoder_name:
        mute_flash_attn_warning()
        encoderclass = partial(encoderclass, trust_remote_code=True)
    model = encoderclass(encoder_name, device = device)
    if "jina" in encoder_name:
        model = JinaAdapter(model)
    return model


class JinaAdapter:
    """SentenceTransformer subclass for Jina embeddings with automatic prompt selection."""

    def __init__(self, model):
        # always trust remote code for Jina models
        self.model = model
    def encode(self, *args, **kwargs):
        # handle encode(is_query, texts, ...) OR encode(texts, is_query=False, ...)
        is_query = True
        if args and isinstance(args[0], bool):
            is_query = args[0]
            args = args[1:]
        if "is_query" in kwargs:
            is_query = kwargs.pop("is_query")

        prompt_name = "retrieval.query" if is_query else "retrieval.passage"
        kwargs.setdefault("prompt_name", prompt_name)
        return self.model.encode(*args, **kwargs)


class F2Encoder:
    def __init__(self, model_name, device):
        
        self.device = torch.device(device if device else "cpu")
        self.model_name = model_name
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        load_kwargs = {}
        if self.device.type == "cuda":
            load_kwargs["torch_dtype"] = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False):

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
