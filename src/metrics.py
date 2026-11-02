import re

import pandas as pd
from sacrebleu.metrics import CHRF

from config import get_settings

cfg = get_settings()

# стикеры
EMOJI_START = re.compile(r"^[\U0001F000-\U0001FFFF☀-➿⬀-⯿]")
EMOJI_ANY = re.compile(r"[\U0001F000-\U0001FFFF]")
# сломанные токены (китайский язык и пр)
CJK = re.compile(r"[一-鿿]")
LATIN_WORD = re.compile(r"[A-Za-zÀ-ÿ]{4,}")

SIGNATURE = cfg.styles[1].signature
_chrf = CHRF()


def style_ok(style: int, text: str) -> bool:
    """проверка формы поста, работает чисто на эмпирике, что я глазами разглядел в стилях"""
    starts_emoji = bool(EMOJI_START.match(text))
    if style == 1:
        return text.rstrip().endswith(SIGNATURE) and not starts_emoji
    return starts_emoji and "\n\n" in text and SIGNATURE not in text


def foreign(text: str) -> bool:
    """ловит срывы токенов в китайский и английский"""
    latin = LATIN_WORD.findall(text.replace(SIGNATURE, ""))
    return bool(CJK.search(text)) or len(latin) >= 3


def chrf(hyp: str, ref: str) -> float:
    return _chrf.sentence_score(hyp, [ref]).score


def emoji_count(text: str) -> int:
    """для сравнения с живыми постами"""
    return len(EMOJI_ANY.findall(text))


METRICS = ["chrF", "len_ratio", "style_ok", "foreign", "hashtags", "emoji_count", "chars"]


def score(gens: pd.DataFrame) -> pd.DataFrame:
    """метрики на каждый ответ"""
    chars = gens.output.str.len()
    return gens.assign(
        chrF=[chrf(o, t) if pd.notna(t) else float("nan") for o, t in zip(gens.output, gens.target)],
        len_ratio=chars / gens.target.str.len(),
        style_ok=[style_ok(s, o) for s, o in zip(gens["style"], gens.output)],
        foreign=gens.output.map(foreign),
        hashtags=gens.output.str.contains(r"#\w"),
        emoji_count=gens.output.map(emoji_count),
        chars=chars,
    )
