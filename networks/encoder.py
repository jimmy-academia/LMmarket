import os
import logging
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from utils import dumpp, loadp


_ENCODER_STATE = {}


def _prepare_model():
    global _ENCODER_STATE
    if _ENCODER_STATE:
        return _ENCODER_STATE
    model_name = "NovaSearch/stella_en_1.5B_v5"
    query_prompt_s2p = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
    query_prompt_s2s = "Instruct: Retrieve semantically similar text.\nQuery: "
    vector_dim = 1024
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="cuda",
        torch_dtype=torch.float16,
    ).eval()
    vector_linear_directory = f"2_Dense_{vector_dim}"
    ckpt_path = hf_hub_download(repo_id=model_name, filename=os.path.join(vector_linear_directory, "pytorch_model.bin"))
    state = torch.load(ckpt_path, map_location="cpu")
    cleaned = {k.replace("linear.", ""): v for k, v in state.items()}
    proj = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
    proj.load_state_dict(cleaned, strict=True)
    proj = proj.to(device=next(model.parameters()).device, dtype=torch.float16).eval()
    _ENCODER_STATE = {
        "model_name": model_name,
        "tokenizer": tokenizer,
        "model": model,
        "proj": proj,
        "vector_dim": vector_dim,
        "query_prompt_s2p": query_prompt_s2p,
        "query_prompt_s2s": query_prompt_s2s,
    }
    return _ENCODER_STATE


def _tokenize_with_progress(texts, max_length=160, chunk_size=1024):
    state = _prepare_model()
    tokenizer = state["tokenizer"]
    result = {"input_ids": [], "attention_mask": [], "length": []}
    total = len(texts)
    for i in tqdm(range(0, total, chunk_size), ncols=88, desc=f"[{state['model_name']}] Tokenizing", leave=False):
        chunk = texts[i:i + chunk_size]
        enc = tokenizer(
            chunk,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_length=True,
            return_tensors=None,
        )
        result["input_ids"].extend(enc["input_ids"])
        result["attention_mask"].extend(enc["attention_mask"])
        result["length"].extend(enc["length"])
    return result


def _model_encode(texts, isquery=False, batch_size=512, max_length=256, normalize=True, query_task="s2s", return_numpy=True):
    state = _prepare_model()
    model_name = state["model_name"]
    tokenizer = state["tokenizer"]
    model = state["model"]
    proj = state["proj"]
    if isquery:
        if query_task == "s2p":
            prefix = state["query_prompt_s2p"]
        elif query_task == "s2s":
            prefix = state["query_prompt_s2s"]
        else:
            raise ValueError("query_task must be 's2p' or 's2s'")
    else:
        prefix = ""
    single = False
    if type(texts) is str:
        single = True
        texts = [texts]
    if prefix:
        texts = [prefix + t for t in texts]
    device = next(model.parameters()).device
    len_enc = _tokenize_with_progress(texts, max_length=max_length, chunk_size=1024)
    lengths = len_enc["length"]
    idxs = np.argsort(lengths)
    bucket_bins = (64, 96, 128, 160)

    def which_bin(L):
        for b in bucket_bins:
            if L <= b:
                return b
        return bucket_bins[-1]

    buckets = defaultdict(list)
    for i in tqdm(idxs, ncols=88, desc=f"[{model_name}] sorting buckets...", leave=False):
        buckets[which_bin(lengths[i])].append(i)
    vecs = [None] * len(texts)
    with torch.inference_mode():
        for _, inds in tqdm(buckets.items(), ncols=88, desc=f"[{model_name}] encode", leave=False):
            for s in range(0, len(inds), batch_size):
                sub = inds[s:s + batch_size]
                sub_texts = [texts[i] for i in sub]
                batch = tokenizer(
                    sub_texts,
                    padding="longest",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                for k, v in batch.items():
                    batch[k] = v.pin_memory()
                    batch[k] = batch[k].to(device, non_blocking=True)
                last_hidden = model(**batch)[0]
                mask = batch["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
                sent_vec = (last_hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
                sent_vec = sent_vec.to(proj.weight.dtype)
                vec = proj(sent_vec)
                if normalize:
                    vec = F.normalize(vec, p=2, dim=1)
                if return_numpy:
                    vec = vec.detach().cpu().numpy()
                for j, idx in enumerate(sub):
                    vecs[idx] = vec[j] if return_numpy else vec[j:j + 1]
    if single:
        return vecs[0]
    if return_numpy:
        return np.stack(vecs, axis=0)
    return torch.cat(vecs, dim=0)


def build_segment_embeddings(segments, output_path, flush_size=10240):
    output_path = Path(output_path)
    partial_path = Path(str(output_path) + ".partial")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    seen = set()
    if partial_path.exists():
        logging.info(f"[encoder] resuming embeddings from {partial_path}")
        cached = loadp(partial_path)
        stored = cached.get("entries") if cached else None
        if stored:
            entries = stored
            for entry in entries:
                sid = entry.get("segment_id")
                if sid:
                    seen.add(sid)
    def flush(records, texts):
        if not texts:
            return
        embeds = _model_encode(texts)
        for idx, vector in enumerate(embeds):
            item = records[idx]
            item["embedding"] = vector.tolist()
            entries.append(item)
            sid = item.get("segment_id")
            if sid:
                seen.add(sid)
        dumpp(partial_path, {"entries": entries})
    batch_records = []
    batch_texts = []
    for record in tqdm(segments, ncols=88, desc="[encoder] embed segments"):
        text = record.get("text")
        if not text:
            continue
        sid = record.get("segment_id")
        if sid and sid in seen:
            continue
        info = {
            "segment_id": sid,
            "review_id": record.get("review_id"),
            "item_id": record.get("item_id"),
            "text": text,
        }
        batch_records.append(info)
        batch_texts.append(text)
        if len(batch_texts) >= flush_size:
            flush(batch_records, batch_texts)
            batch_records = []
            batch_texts = []
    if batch_texts:
        flush(batch_records, batch_texts)
    partial_path.unlink(missing_ok=True)
    if not entries:
        return {"entries": [], "index": None, "matrix": None, "dimension": None}
    entry_map = {}
    for entry in entries:
        sid = entry.get("segment_id")
        if sid and sid not in entry_map:
            entry_map[sid] = entry
    ordered = []
    used = set()
    for record in segments:
        sid = record.get("segment_id")
        if not sid or sid in used:
            continue
        entry = entry_map.get(sid)
        if entry:
            ordered.append(entry)
            used.add(sid)
    entries = ordered
    if not entries:
        return {"entries": [], "index": None, "matrix": None, "dimension": None}
    matrix = np.asarray([entry.get("embedding") for entry in entries], dtype="float32")
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    serialized = faiss.serialize_index(index)
    return {
        "entries": entries,
        "index": serialized,
        "matrix": matrix,
        "dimension": dim,
    }


def apply_segment_embeddings(payload):
    data = payload or {}
    entries = data.get("entries") or []
    matrix = data.get("matrix")
    if matrix is None and entries:
        vectors = []
        for entry in entries:
            vector = entry.get("embedding")
            if vector is None:
                continue
            vectors.append(vector)
        if vectors:
            matrix = np.asarray(vectors, dtype="float32")
    index_bytes = data.get("index")
    index = faiss.deserialize_index(index_bytes) if index_bytes else None
    dimension = data.get("dimension")
    if matrix is not None and dimension is None:
        dimension = matrix.shape[1]
    return matrix, index, entries, dimension
