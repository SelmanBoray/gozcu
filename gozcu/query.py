"""Türkçe sorgu ayrıştırıcı — zamansal ifadeyi görsel ifadeden ayırır.

"dün gece siteye giren beyaz Transit" → görsel: "siteye giren beyaz transit",
zaman filtresi: [dün 20:00, bugün 06:00].
KRİTİK KURAL: Zaman kelimeleri ASLA embedlenmez — zaman ifadesi Qdrant
filtresine, görsel ifade vektöre gider (en büyük kalite kaldıracı — ARCHITECTURE.md §4).
MVP: kural tabanlı regex. Faz 2: lokal LLM ayrıştırıcı (Qwen3-4B, Ollama).
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable


@dataclass
class ParsedQuery:
    """Ayrıştırılmış sorgu: embedlenecek görsel metin + Qdrant filtreleri."""

    visual_text: str                 # embedlenecek kısım ("beyaz transit")
    ts_from: float | None = None     # epoch (saniye)
    ts_to: float | None = None
    camera_id: str | None = None     # MVP'de hep None; Faz 2'de "kamera 3" ayrıştırılacak
    time_phrase: str | None = None   # kullanıcıya "şu aralıkta aradım" diyebilmek için


# ── Yardımcılar ──

def _day(now: datetime, days_ago: int) -> datetime:
    """N gün önceki günün 00:00'ı."""
    return (now - timedelta(days=days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)


def _clock_window(now: datetime, m: re.Match) -> tuple[datetime, datetime]:
    """'(dün) saat 22(:30) civarı' → o saatin ±45 dakikası."""
    days_ago = 1 if m.group(1) else 0
    hour = int(m.group(2))
    minute = int(m.group(3) or 0)
    center = _day(now, days_ago) + timedelta(hours=hour, minutes=minute)
    return center - timedelta(minutes=45), center + timedelta(minutes=45)


# ── Zaman kalıpları: sıralama önemli — en özgül olan en üstte ──
# Her giriş: (regex, (now, match) -> (başlangıç, bitiş))

_Span = Callable[[datetime, re.Match], tuple[datetime, datetime]]

TIME_PATTERNS: list[tuple[re.Pattern, _Span]] = [
    # "dün saat 22 civarı", "saat 14:30 gibi", "dün saat 3'te"
    (
        re.compile(
            r"(dün\s+)?saat\s+(\d{1,2})(?::(\d{2}))?(?:'?\w*)?"
            r"\s*(?:civarı(?:nda)?|gibi|sularında|sıralarında)?"
        ),
        _clock_window,
    ),
    # "son 3 saat", "son 12 saatte"
    (
        re.compile(r"son\s+(\d+)\s+saat\w*"),
        lambda now, m: (now - timedelta(hours=int(m.group(1))), now),
    ),
    # "son 30 dakika"
    (
        re.compile(r"son\s+(\d+)\s+dakika\w*"),
        lambda now, m: (now - timedelta(minutes=int(m.group(1))), now),
    ),
    # "dün gece" → dün 20:00 – bugün 06:00
    (
        re.compile(r"dün\s+gece\w*"),
        lambda now, m: (_day(now, 1) + timedelta(hours=20), _day(now, 0) + timedelta(hours=6)),
    ),
    # "dün sabah" → dün 05:00–12:00
    (
        re.compile(r"dün\s+sabah\w*"),
        lambda now, m: (_day(now, 1) + timedelta(hours=5), _day(now, 1) + timedelta(hours=12)),
    ),
    # "dün öğle(n)" → dün 11:00–15:00
    (
        re.compile(r"dün\s+öğle\w*"),
        lambda now, m: (_day(now, 1) + timedelta(hours=11), _day(now, 1) + timedelta(hours=15)),
    ),
    # "dün akşam" → dün 17:00–23:00
    (
        re.compile(r"dün\s+akşam\w*"),
        lambda now, m: (_day(now, 1) + timedelta(hours=17), _day(now, 1) + timedelta(hours=23)),
    ),
    # "dün" → dünün tamamı
    (
        re.compile(r"\bdün\b"),
        lambda now, m: (_day(now, 1), _day(now, 0)),
    ),
    # "bu gece" → bugün 20:00 – yarın 06:00
    (
        re.compile(r"bu\s+gece\w*"),
        lambda now, m: (_day(now, 0) + timedelta(hours=20), _day(now, -1) + timedelta(hours=6)),
    ),
    # "bu sabah" → bugün 05:00–12:00
    (
        re.compile(r"bu\s+sabah\w*"),
        lambda now, m: (_day(now, 0) + timedelta(hours=5), _day(now, 0) + timedelta(hours=12)),
    ),
    # "bu akşam" → bugün 17:00–23:00
    (
        re.compile(r"bu\s+akşam\w*"),
        lambda now, m: (_day(now, 0) + timedelta(hours=17), _day(now, 0) + timedelta(hours=23)),
    ),
    # "bugün" → bugün 00:00 – şimdi
    (
        re.compile(r"\bbugün\b"),
        lambda now, m: (_day(now, 0), now),
    ),
]

# ── Dolgu kelimeler: görsel anlam taşımaz, embedlenmeden önce atılır ──
FILLER_RE = re.compile(
    r"\b(bul|göster|ara|listele|getir|bana|acaba|var\s*mı(ydı)?|"
    r"ne\s*zaman|kaçta|hangi\s+saatte)\b"
)


def parse_query(raw_query: str, now: datetime | None = None) -> ParsedQuery:
    """Ham Türkçe sorguyu görsel metin + zaman aralığına ayırır."""
    now = now or datetime.now()

    # ── 1. Türkçe'ye uygun küçük harfe indir (İ→i, I→ı) — kalıplar küçük harf ──
    remaining = raw_query.strip().replace("İ", "i").replace("I", "ı").lower()

    # ── 2. Zaman ifadesini yakala, metinden çıkar ──
    ts_from: float | None = None
    ts_to: float | None = None
    time_phrase: str | None = None
    for pattern, span_fn in TIME_PATTERNS:
        match = pattern.search(remaining)
        if match:
            start, end = span_fn(now, match)
            ts_from, ts_to = start.timestamp(), end.timestamp()
            time_phrase = match.group(0).strip()
            remaining = pattern.sub(" ", remaining, count=1)
            break

    # ── 3. Dolgu kelimeleri ve noktalamayı temizle → saf görsel metin ──
    cleaned = FILLER_RE.sub(" ", remaining)
    cleaned = re.sub(r"[?!.,;]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return ParsedQuery(
        visual_text=cleaned,
        ts_from=ts_from,
        ts_to=ts_to,
        time_phrase=time_phrase,
    )
