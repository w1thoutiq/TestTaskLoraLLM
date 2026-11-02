import pandas as pd

from config import get_settings

cfg = get_settings()


def load_data() -> pd.DataFrame:
    """выдает все выборки в одном фрейме"""
    posts = pd.concat(
        [pd.read_json(cfg.paths.data / st.train_file, lines=True).assign(style=s) for s, st in cfg.styles.items()],
        ignore_index=True,
    ).rename(columns={"input": "source", "output": "target"})
    val = posts.groupby("style").sample(cfg.data.val_per_type, random_state=cfg.seed)
    test = pd.read_json(cfg.paths.data / cfg.data.test_file, lines=True).rename(columns={"text": "source"})
    data = pd.concat([
        posts.drop(val.index).assign(split="train"),
        val.assign(split="val"),
        *(test.assign(style=s, split="test") for s in cfg.styles),
    ], ignore_index=True)
    return data[["split", "style", "id", "source", "target"]].astype({"id": "Int64"})
