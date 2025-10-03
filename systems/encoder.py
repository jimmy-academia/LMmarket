class Encoder:
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
