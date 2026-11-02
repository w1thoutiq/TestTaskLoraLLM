from contextlib import nullcontext  # для прогона модели без адаптера
from functools import lru_cache

from peft import PeftModel
from transformers import pipeline, set_seed

from config import get_settings
from model import load_base_model, load_tokenizer

cfg = get_settings()


class StyleModel:
    """одна модель в памяти на все 4 варианта"""

    def __init__(self):
        self.model = PeftModel.from_pretrained(load_base_model(), cfg.paths.adapter).eval()
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=load_tokenizer())

    def generate(self, batch: list[list[dict]], use_adapter: bool = True) -> list[str]:
        """без адаптера — ответы базовой модели для сравнения"""
        set_seed(cfg.seed)
        with nullcontext() if use_adapter else self.model.disable_adapter():
            out = self.pipe(batch, batch_size=cfg.test.batch_size, return_full_text=False, **cfg.test.gen)
        return [o[0]["generated_text"].strip() for o in out]


@lru_cache(maxsize=1)  # синглтон
def load_style_model() -> StyleModel:
    return StyleModel()
