import pandas as pd

from config import get_settings
from data import load_data
from generation import StyleModel, load_style_model
from metrics import METRICS, score
from prompts import build_messages

cfg = get_settings()

VARIANTS = {
    "base_prompt": (False, False),
    "base_fewshot": (False, True),
    "lora": (True, False),
    "lora_fewshot": (True, True),
}


def pick_shots(train: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """фиксированные примеры"""
    return {s: g.sample(cfg.test.fewshot_k, random_state=cfg.seed) for s, g in train.groupby("style")}


def generate(model: StyleModel, inputs: pd.DataFrame, shots: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """все 4 варианта на одних входах"""
    gens = []
    for variant, (adapter, fewshot) in VARIANTS.items():
        batch = [build_messages(r.style, r.source, shots[r.style] if fewshot else None) for r in inputs.itertuples()]
        gens.append(inputs.assign(variant=variant, output=model.generate(batch, use_adapter=adapter)))
    return pd.concat(gens, ignore_index=True)


def aggregate_metrics(gens: pd.DataFrame) -> pd.DataFrame:
    """таблица для ответа"""
    report = gens.groupby(["split", "variant", "style"])[METRICS].mean()
    return report.reindex(list(VARIANTS), level="variant").round(3)


def posts_txt(posts: pd.DataFrame) -> str:
    """формат итоговых файлов по тз"""
    return "\n\n".join(f"### {r.id}\n{r.output}" for r in posts.sort_values("id").itertuples()) + "\n"


def save_results(gens: pd.DataFrame, final: pd.DataFrame) -> None:
    """весь вывод на диск в одном месте"""
    out = cfg.paths.artifacts
    out.mkdir(parents=True, exist_ok=True)
    gens.to_json(out / "generations_all.json", orient="records", force_ascii=False, indent=1)
    for style, posts in final.groupby("style"):
        (out / f"outputs_type{style}.txt").write_text(posts.pipe(posts_txt), encoding="utf-8")


def main():
    """сравнение вариантов и итоговые посты"""
    data = load_data()
    shots = data[data.split == "train"].pipe(pick_shots)
    gens = generate(load_style_model(), data[data.split != "train"], shots).pipe(score)
    print(gens.pipe(aggregate_metrics).to_string())

    final = gens[(gens.split == "test") & (gens.variant == cfg.test.output_variant)]
    save_results(gens, final)


if __name__ == "__main__":
    main()
