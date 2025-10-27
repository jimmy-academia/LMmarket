import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Dictionary mapping shorthand -> full Hugging Face model name
MODEL_MAP = {
    # LLaMA family (Meta)
    "llama1b": "meta-llama/Llama-3.2-1B",
    "llama3b": "meta-llama/Llama-3.2-3B",
    "llama7b": "meta-llama/Llama-3.1-7B",

    # OpenLLaMA family
    "openllama3b": "openlm-research/open_llama_3b_v2",
    "openllama7b": "openlm-research/open_llama_7b_v2",

    # Gemma family (Google)
    "gemma2b": "google/gemma-2-2b-it",
    "gemma4b": "google/gemma-2-4b-it",
    "gemma12b": "google/gemma-2-12b-it",
    "gemma27b": "google/gemma-2-27b-it",

    # Phi family (Microsoft)
    "phi3mini": "microsoft/phi-3-mini-4k-instruct",  # 3.8B
    "phi3small": "microsoft/phi-3-small-8k-instruct",  # 7B
}


class SmallLM:
    """
    Unified small language model loader that handles multiple model families safely.
    Automatically adjusts tokenizer mode (legacy/use_fast) and dtype for each model type.
    """

    def __init__(self, model_key: str, device: torch.device, dtype: torch.dtype = torch.float16):
        if model_key not in MODEL_MAP:
            raise ValueError(f"Unknown model key: {model_key}. Options: {list(MODEL_MAP.keys())}")

        full_name = MODEL_MAP[model_key]
        print(f"⏳ Loading model: {full_name} ...")

        # Decide tokenizer behavior
        is_openllama = "openllama" in full_name.lower() or "open_llama" in full_name.lower()
        use_fast = not is_openllama
        legacy = True if is_openllama else None  # only pass legacy flag for OpenLLaMA

        # Adjust dtype for certain model families
        if "gemma" in full_name.lower() and dtype == torch.float16:
            dtype = torch.bfloat16

        # Load tokenizer with proper settings
        tokenizer_kwargs = {"use_fast": use_fast}
        if legacy is not None:
            tokenizer_kwargs["legacy"] = legacy  # only applied for OpenLLaMA

        self.tokenizer = AutoTokenizer.from_pretrained(full_name, **tokenizer_kwargs)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(full_name, torch_dtype=dtype).to(device)
        self.eos_token_id = self.tokenizer.eos_token_id

        print(f"✅ Model loaded successfully on device: {self.model.device} (dtype={dtype})")

    def __call__(self, prompt: str, max_new_tokens=128, temperature=0.0, top_p=0.9):
        """Generates text from a prompt using the wrapped model + tokenizer."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
