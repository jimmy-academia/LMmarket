import logging
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F

from tqdm import tqdm

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


