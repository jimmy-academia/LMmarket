import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from api import user_struct, system_struct

model_id_dict = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi": "microsoft/Phi-3.5-mini-instruct"
}

class Summarizer:
    def __init__(self, args):
        self.model_name = args.summarizer_name
        model_id = model_id_dict[self.model_name]
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map=args.device)
