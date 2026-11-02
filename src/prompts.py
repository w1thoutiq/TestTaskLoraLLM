import pandas as pd

from config import get_settings

cfg = get_settings()


def user_message(style: int, source: str) -> str:
    """формат как в трейне"""
    return cfg.styles[style].instruction + "\n\n" + source


def build_messages(style: int, source: str, shots: pd.DataFrame | None = None) -> list[dict]:
    """один диалог для обучения, теста"""
    msgs = [{"role": "system", "content": cfg.system_prompt}] if cfg.system_prompt else []
    if shots is not None:
        for s in shots.itertuples():
            msgs += [
                {"role": "user", "content": user_message(s.style, s.source)},
                {"role": "assistant", "content": s.target},
            ]
    msgs.append({"role": "user", "content": user_message(style, source)})
    return msgs
