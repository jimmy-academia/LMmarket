import os
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

class Encoder:
    def _prepare_model(self):
        # Drop-in switch to NovaSearch/stella_en_1.5B_v5 with 1024-d vectors (8-bit).
        # Uses the Transformers path (fast + quantized) and the official projection layer from 2_Dense_1024.
        
        self._model_name = "NovaSearch/stella_en_1.5B_v5"
        self._query_prompt_s2p = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
        self._query_prompt_s2s = "Instruct: Retrieve semantically similar text.\nQuery: "
        self._vector_dim = 1024  # default dimension recommended by the authors

        # 8-bit weights + BF16 activations for speed/VRAM
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            quantization_config=bnb_cfg,
            device_map="cuda",
            torch_dtype=torch.float16,
        ).eval()

        # Load the official projection layer weights for 1024-d from repo folder 2_Dense_1024
        vector_linear_directory = f"2_Dense_{self._vector_dim}"
        ckpt_path = hf_hub_download(repo_id=self._model_name, filename=os.path.join(vector_linear_directory, "pytorch_model.bin"))
        state = torch.load(ckpt_path, map_location="cpu")
        # The file uses keys like 'linear.weight' / 'linear.bias' — strip the 'linear.' prefix.
        state = {k.replace("linear.", ""): v for k, v in state.items()}

        # Build the projection layer and move to CUDA if available
        self._proj = torch.nn.Linear(in_features=self.model.config.hidden_size, out_features=self._vector_dim)
        self._proj.load_state_dict(state, strict=True)
        
        model_device = next(self.model.parameters()).device
        self._proj = self._proj.to(device=model_device, dtype=torch.float16).eval()

    def _model_encode(self, texts, isquery=False, batch_size=512, max_length=256, normalize=True, return_numpy=True, query_task="s2s"):
        """
        Encode with Stella v5 (8-bit) and return 1024-d embeddings.
        mode: 'passage' (no prompt) or 'query' (adds prompt). For queries, choose task via query_task:
              - 's2p' (sentence->passage retrieval, recommended for RAG)
              - 's2s' (sentence->sentence similarity)
        """
        if isquery:
            if query_task == "s2p":
                prefix = self._query_prompt_s2p
            elif query_task == "s2s":
                prefix = self._query_prompt_s2s
            else:
                raise ValueError("query_task must be 's2p' or 's2s'")
        else:
            prefix = ""

        single = isinstance(texts, str)
        if single:
            texts = [texts]
        if prefix:
            texts = [prefix + t for t in texts]

        device = next(self.model.parameters()).device

        # ---- quick length pass (no padding) to build buckets ----
        len_enc = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_length=True,
        )
        lengths = len_enc["length"]
        idxs = np.argsort(lengths)  # shortest -> longest

        def which_bin(L):
            for b in bucket_bins:
                if L <= b:
                    return b
            return bucket_bins[-1]

        buckets = defaultdict(list)
        for i in idxs:
            buckets[which_bin(lengths[i])].append(i)

        # ---- encode per bucket ----
        vecs = [None] * len(texts)
        with torch.inference_mode():
            for _, inds in buckets.items():
                for s in range(0, len(inds), batch_size):
                    sub = inds[s:s + batch_size]
                    sub_texts = [texts[i] for i in sub]

                    # Tokenize to pinned CPU memory
                    batch = self.tokenizer(
                        sub_texts,
                        padding="longest",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    # Pin & non_blocking copy to GPU
                    for k, v in batch.items():
                        batch[k] = v.pin_memory()
                        batch[k] = batch[k].to(device, non_blocking=True)

                    # Forward + mean pool
                    last_hidden = self.model(**batch)[0]                       # [B, L, H]
                    mask = batch["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
                    sent_vec = (last_hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6)

                    # Match projection dtype (fp16/bf16) and project to 1024-d
                    sent_vec = sent_vec.to(self._proj.weight.dtype)
                    vec = self._proj(sent_vec)

                    if normalize:
                        vec = F.normalize(vec, p=2, dim=1)

                    if return_numpy:
                        vec = vec.detach().cpu().numpy()

                    for j, i in enumerate(sub):
                        vecs[i] = vec[j] if return_numpy else vec[j:j+1]

        if single:
            return vecs[0]
        if return_numpy:
            return np.stack(vecs, axis=0)
        return torch.cat(vecs, dim=0)

class dep_nv_Encoder:
    def _prepare_model(self):
        self._model_name = "nvidia/NV-Embed-v2"
        self._query_instruction = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "

        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=bnb_cfg,
            device_map="auto",
            attn_implementation="eager",
        )
        self.model.eval()

    def _model_encode(self, texts, isquery=False, batch_size=128, max_length=256, normalize=True, return_numpy=True, num_workers=8, squeeze_single=True):

        instruction = self._query_instruction if isquery else ""

        single = isinstance(texts, str)
        if single:
            texts = [texts]

        # Call NVIDIA's high-throughput encoder with desired return type
        embs = self.model._do_encode(
            texts,
            batch_size=batch_size,
            instruction=instruction,
            max_length=max_length,
            num_workers=num_workers,
            return_numpy=return_numpy,
        )

        # Optional L2 normalization (works for both numpy and torch)
        if normalize:
            if return_numpy:
                norms = np.linalg.norm(embs, ord=2, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                embs = embs / norms
            else:
                embs = F.normalize(embs, p=2, dim=1)

        # Squeeze back to 1D if a single string was provided
        if single and squeeze_single:
            embs = embs[0]

        return embs
