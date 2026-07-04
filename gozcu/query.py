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


# ── Nesne sınıfı niyeti: sorgu hangi YOLO sınıfını istiyor? (bulunamadı kapısı) ──
# Amaç: korpusta HİÇ olmayan bir nesne (ör. bisiklet) arandığında CLIP skoruna değil,
# YOLO tespitine dayanan güvenilir "bulunamadı" sinyali vermek. Detay: ARCHITECTURE.md §4b

_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def _fold(s: str) -> str:
    """Türkçe karakterleri ASCII'ye indir — çekim eklerine rağmen kök eşleşsin."""
    return s.translate(_ASCII).lower()


# (folded_kök, YOLO sınıf adı) — prefix eşleşir: arabalar→araba, insanı→insan, otobüsler→otobüs
_CLASS_ROOTS: list[tuple[str, str]] = [
    ("insan", "insan"), ("adam", "insan"), ("kisi", "insan"), ("kadin", "insan"),
    ("erkek", "insan"), ("cocuk", "insan"), ("yaya", "insan"), ("surucu", "insan"),
    ("bisiklet", "bisiklet"),
    ("araba", "araba"), ("otomobil", "araba"), ("sedan", "araba"),
    ("motosiklet", "motosiklet"), ("motorsiklet", "motosiklet"),
    ("otobus", "otobüs"), ("minibus", "otobüs"), ("midibus", "otobüs"),
    ("kamyonet", "kamyon"), ("kamyon", "kamyon"),
]
# Kısa/riskli kökler yalnız tam kelime eşleşir (yanlış prefix eşleşmesini önler)
_CLASS_EXACT: dict[str, str] = {"suv": "araba", "tir": "kamyon", "tira": "kamyon"}
# Belirli sınıf değil, herhangi bir taşıt ("araç geçti") — taşıt varsa boş dönme
_GENERIC_VEHICLE_ROOTS = ("arac", "vasita", "tasit")
VEHICLE_CLASSES = {"bisiklet", "araba", "motosiklet", "otobüs", "kamyon"}


@dataclass
class ObjectIntent:
    """Sorgunun istediği YOLO sınıfları — bulunamadı kapısı için."""

    required: set[str]      # sorguda geçen BELİRLİ sınıflar (ör. {"bisiklet"})
    generic_vehicle: bool   # yalnız genel "araç/taşıt" geçti (belirli sınıf yok)


def extract_object_intent(visual_text: str) -> ObjectIntent:
    """Görsel metinden istenen nesne sınıflarını çıkar (çekim ekine dayanıklı)."""
    required: set[str] = set()
    generic = False
    for raw_tok in _fold(visual_text).split():
        tok = raw_tok.strip(".,;:!?()[]\"'")
        if not tok:
            continue
        hit = None
        for root, cls in _CLASS_ROOTS:
            if tok.startswith(root):
                hit = cls
                break
        if hit is None:
            hit = _CLASS_EXACT.get(tok)
        if hit is not None:
            required.add(hit)
        elif tok.startswith(_GENERIC_VEHICLE_ROOTS):
            generic = True
    return ObjectIntent(required=required, generic_vehicle=generic)


# ── Sahne vs nesne niyeti: durum ekiyle (case morphology), pozisyonla DEĞİL ──
# "araçlarLA dolu açık otopark" → araç=INSTR (ortam), otopark=NOM → SAHNE
# "otoparkTA yürüyen insan"     → otopark=LOC (ortam), insan=NOM → NESNE
# Head = son NOMİNATİF (eksiz) içerik ismi. -DA/-DAn/-lA/-In ekli kelime adjunct'tır.
# Detay: ARCHITECTURE.md §7 (Olgu B — kırpık seli sahne karesini gömüyordu)

# Yer/sahne kökleri (prefix eşleşir). Kısa/belirsiz kökler (kat, park) kasten dışarıda.
_SCENE_ROOTS: tuple[str, ...] = (
    "otopark", "garaj", "cadde", "sokak", "kavsak", "kampus", "otoyol", "otoban",
    "kaldirim", "meydan", "bahce", "avlu", "koridor", "merdiven", "peron", "durak",
    "giris", "cikis", "bina", "alan", "yol", "manzara", "sahne", "goruntu", "kamera",
    "kayit",
)
# Adjunct durum ekleri (folded): locative -DA, ablative -DAn, instrumental -lA, genitive -In.
# Accusative -i/-u KASITEN yok — belirli nesneyi işaretler, head olabilir.
_CASE_SUFFIXES: tuple[str, ...] = (
    "da", "de", "ta", "te", "dan", "den", "tan", "ten", "la", "le", "yla", "yle",
    "in", "un", "nin", "nun", "nda", "nde", "ndan", "nden",
)


def _classify_token(tok: str) -> tuple[str | None, bool]:
    """Token'ı (tür, adjunct_mu) olarak sınıfla. tür: 'object'|'scene'|None."""
    for root, _cls in _CLASS_ROOTS:
        if tok.startswith(root):
            return ("object", tok[len(root):].endswith(_CASE_SUFFIXES))
    if tok in _CLASS_EXACT:
        return ("object", False)
    for root in _GENERIC_VEHICLE_ROOTS:
        if tok.startswith(root):
            return ("object", tok[len(root):].endswith(_CASE_SUFFIXES))
    for root in _SCENE_ROOTS:
        if tok.startswith(root):
            return ("scene", tok[len(root):].endswith(_CASE_SUFFIXES))
    return (None, False)


# ── Türkçe → İngilizce görsel-kelime çevirisi (Faz 2: VLM'in Türkçesine güvenme) ──
# Küçük VLM İngilizce talimatı çok daha güvenilir takip eder. Deterministik, +0 VRAM,
# tam lokal (KVKK). Kök prefix eşleşir (çekim ekine dayanıklı). Detay: ARCHITECTURE.md §8

_TR_EN: list[tuple[str, str]] = [
    # nesneler
    ("insan", "person"), ("adam", "man"), ("kadin", "woman"), ("cocuk", "child"),
    ("kisi", "person"), ("yaya", "pedestrian"), ("surucu", "driver"), ("bebek", "baby"),
    ("araba", "car"), ("otomobil", "car"), ("kamyonet", "pickup truck"), ("kamyon", "truck"),
    ("otobus", "bus"), ("minibus", "minibus"), ("motosiklet", "motorcycle"),
    ("bisiklet", "bicycle"), ("kopek", "dog"), ("kedi", "cat"), ("suv", "SUV"),
    # renkler
    ("siyah", "black"), ("beyaz", "white"), ("mavi", "blue"), ("kirmizi", "red"),
    ("sari", "yellow"), ("yesil", "green"), ("gri", "gray"), ("gumus", "silver"),
    ("koyu", "dark"), ("turuncu", "orange"), ("mor", "purple"), ("kahverengi", "brown"),
    ("pembe", "pink"), ("lacivert", "navy blue"),
    # sahne / yer
    ("otopark", "parking lot"), ("garaj", "garage"), ("cadde", "street"), ("sokak", "street"),
    ("kavsak", "intersection"), ("kampus", "campus"), ("otoyol", "highway"), ("yol", "road"),
    ("kaldirim", "sidewalk"), ("meydan", "square"), ("bina", "building"), ("durak", "bus stop"),
    # hava / durum / eylem
    ("gece", "at night"), ("gunduz", "during day"), ("yagmur", "rain"), ("kar", "snow"),
    ("semsiye", "umbrella"), ("trafik", "traffic"), ("yuruyen", "walking"), ("kosan", "running"),
    ("giden", "moving"), ("duran", "stopped"), ("park", "parked"), ("telefon", "phone"),
    ("bosaltan", "unloading"), ("bekleyen", "waiting"), ("arac", "vehicle"), ("tasit", "vehicle"),
    ("gezdiren", "walking"), ("suren", "riding"), ("binen", "boarding"), ("konus", "talking"),
    ("kapli", "covered"), ("tasiyan", "carrying"), ("iten", "pushing"), ("kosan", "running"),
]
# Anlam taşımayan bağlaçlar (çeviriden atılır)
_TR_SKIP = {"ve", "ile", "bir", "bu", "su", "cok", "dolu", "olan", "halindeki", "arasinda",
            "gecen", "giren", "cikan", "ustunde", "yaninda", "onunde", "arkasinda", "renkli"}


# Renk kökleri (VLM color_match tetikler) ve VLM'in çözdüğü "zor" kavramlar
# (YOLO sınıfı olmayan nesne / hava / öznitelik — CLIP tek başına garantilemiyor)
_COLOR_ROOTS = ("siyah", "beyaz", "mavi", "kirmizi", "sari", "yesil", "gri", "gumus",
                "koyu", "turuncu", "mor", "kahverengi", "pembe", "lacivert")
_HARD_CONCEPT_ROOTS = ("kopek", "kedi", "yagmur", "kar", "semsiye", "semsiy")


def has_color(visual_text: str) -> bool:
    """Sorguda renk öznitelik kelimesi var mı? (VLM'de color_match sorulur)"""
    toks = _fold(visual_text).split()
    return any(t.startswith(_COLOR_ROOTS) for t in toks)


def needs_vlm(visual_text: str) -> bool:
    """VLM doğrulaması tetiklenmeli mi? Yalnız CLIP'in zorlandığı yer: renk + zor kavram.

    Nesne/sahne sorguları zaten yüksek recall → VLM vergisi ödenmez (koşullu, AI Engineer S3).
    """
    toks = _fold(visual_text).split()
    return any(t.startswith(_COLOR_ROOTS) or t.startswith(_HARD_CONCEPT_ROOTS) for t in toks)


def translate_visual(visual_text: str) -> str:
    """Türkçe görsel ifadeyi İngilizce kelime dizisine çevir (VLM prompt'u için).

    Kelime bazlı, kök-prefix eşleşmeli; bilinmeyen kelime aynen geçer (isim vb.).
    Gramer değil anahtar-kelime hedefli — VLM'e 'şunları içeriyor mu' diye sorulacak.
    """
    out: list[str] = []
    for raw in _fold(visual_text).split():
        tok = raw.strip(".,;:!?()[]\"'")
        if not tok or tok in _TR_SKIP:
            continue
        en = next((e for root, e in _TR_EN if tok.startswith(root)), None)
        out.append(en if en is not None else tok)
    # tekrarları koru (sıra önemli), boşsa orijinali döndür
    return " ".join(out) if out else visual_text


def scene_or_object_intent(visual_text: str) -> str:
    """Sorgu niyeti: 'scene' | 'object' | 'neutral'.

    Head = son NOMİNATİF (adjunct eki olmayan) içerik ismi; onun türü niyeti verir.
    Sahne-niyeti YALNIZ head bir sahne kelimesiyse tetiklenir — sahne kelimesinin
    sadece geçmesiyle değil (S5: 'otopark bariyeri' tuzağı). Kararsızsa 'neutral'.
    """
    head_type: str | None = None
    for raw in _fold(visual_text).split():
        tok = raw.strip(".,;:!?()[]\"'")
        if not tok:
            continue
        typ, is_adjunct = _classify_token(tok)
        if typ is not None and not is_adjunct:  # nominatif → head adayı, sonuncusu kazanır
            head_type = typ
    return head_type or "neutral"


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
