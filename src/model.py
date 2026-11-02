from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_settings

cfg = get_settings()


@lru_cache(maxsize=1)  # синглтон
def load_tokenizer():
    """отступы слева для батчевой генерации"""
    return AutoTokenizer.from_pretrained(cfg.model.id, padding_side="left")


@lru_cache(maxsize=1)
def load_base_model():
    """4 bit, чтобы 3B влезла в T4"""
    return AutoModelForCausalLM.from_pretrained(
        cfg.model.id,
        quantization_config=cfg.bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )
