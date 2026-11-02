from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

ROOT = Path(__file__).parent.parent


class Section(BaseModel):
    """для паттерна фейлфаст при лишних аттрибутах"""

    model_config = ConfigDict(extra="forbid")


class Paths(Section):
    """глобальные пути"""
    data: Path
    adapter: Path
    checkpoints: Path
    artifacts: Path

    # от корня проекта, чтобы скрипты работали из любой папки
    @field_validator("*")
    @classmethod
    def _from_root(cls, p: Path) -> Path:
        return p if p.is_absolute() else ROOT / p


class Model(Section):
    id: str
    bnb: dict[str, Any]


class Data(Section):
    test_file: str
    val_per_type: int


class Style(Section):
    train_file: str
    instruction: str
    signature: str | None = None


class Train(Section):
    lora: dict[str, Any]
    sft: dict[str, Any]


class Test(Section):
    fewshot_k: int
    batch_size: int
    output_variant: str
    gen: dict[str, Any]


class Settings(Section):
    """собирает настройки из config.yaml"""
    seed: int
    paths: Paths
    model: Model
    data: Data
    system_prompt: str = ""
    styles: dict[int, Style]
    train: Train
    test: Test
    serve: dict[str, Any]

    # для ленивых импортов
    @cached_property
    def bnb_config(self):
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(**self.model.bnb)

    @cached_property
    def peft_config(self):
        from peft import LoraConfig
        return LoraConfig(**self.train.lora)

    @cached_property
    def sft_config(self):
        from trl import SFTConfig
        return SFTConfig(**self.train.sft, output_dir=str(self.paths.checkpoints), seed=self.seed)


@lru_cache(maxsize=1)  # синглтон
def get_settings() -> Settings:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return Settings.model_validate(yaml.safe_load(f))

