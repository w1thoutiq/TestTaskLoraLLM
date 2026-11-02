import pandas as pd
from datasets import Dataset
from peft.tuners.tuners_utils import cast_adapter_dtype
from trl import SFTTrainer

from config import get_settings
from data import load_data
from model import load_base_model, load_tokenizer
from prompts import build_messages

cfg = get_settings()


def to_dataset(examples: pd.DataFrame) -> Dataset:
    """prompt/completion, чтобы loss шёл только по ответу"""
    return Dataset.from_list([
        {"prompt": build_messages(e.style, e.source),
         "completion": [{"role": "assistant", "content": e.target}]}
        for e in examples.itertuples()
    ])


def main():
    """обучение адаптера на оба стиля сразу"""
    data = load_data()
    train, val = data[data.split == "train"], data[data.split == "val"]
    print(f"train={len(train)} val={len(val)}")

    trainer = SFTTrainer(
        model=load_base_model(),
        args=cfg.sft_config,
        train_dataset=to_dataset(train),
        eval_dataset=to_dataset(val),
        peft_config=cfg.peft_config,
        processing_class=load_tokenizer(),
    )
    # fp32 из-за t4
    cast_adapter_dtype(trainer.model, trainer.model.active_adapter)
    trainer.train()

    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trainer.state.log_history).to_csv(cfg.paths.artifacts / "train_log.csv", index=False)

    trainer.save_model(cfg.paths.adapter)
    print(f"adapter saved to {cfg.paths.adapter}")


if __name__ == "__main__":
    main()
