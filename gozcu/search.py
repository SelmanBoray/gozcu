"""Arama hattı — sorgu ayrıştır, embedle, Qdrant'ta ara, sonuçları birleştir.

Embedder ve FrameStore tembel tekil (lazy singleton) olarak yüklenir:
model ~2 GB olduğundan yalnızca ilk aramada belleğe alınır.
"""

from dataclasses import dataclass

from gozcu.config import settings
from gozcu.query import ParsedQuery, parse_query

# ── Tembel tekiller ──
_embedder = None
_store = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from gozcu.embedder import Embedder
        _embedder = Embedder(device=settings.device)
    return _embedder


def get_store():
    global _store
    if _store is None:
        from gozcu.store import FrameStore
        _store = FrameStore()
    return _store


@dataclass
class SearchOutcome:
    """Arama sonucu + kullanıcıya gösterilecek bağlam bilgisi."""

    results: list[dict]          # store.search çıktısı (payload + score)
    parsed: ParsedQuery          # hangi görsel metin / zaman aralığı kullanıldı
    time_filter_dropped: bool    # zaman filtresi sıfır sonuç verdi, filtresiz tekrarlandı


def _dedup_and_group(hits: list[dict], top_k: int) -> list[dict]:
    """Kırpık + kare aynı kareyi gösterebilir; aynı yürüyüş art arda kareler doldurabilir.

    1. (video, kare) başına en yüksek skorlu tek sonuç.
    2. Aynı videoda `group_window_s` penceresi içinde tek sonuç (en iyisi).
    """
    best: dict = {}
    for h in hits:  # Qdrant skora göre sıralı döner
        key = (h["video_id"], h["frame_idx"])
        if key not in best:
            best[key] = h

    kept: list[dict] = []
    for h in sorted(best.values(), key=lambda x: -x["score"]):
        if any(
            k["video_id"] == h["video_id"]
            and abs(k["offset_s"] - h["offset_s"]) < settings.group_window_s
            for k in kept
        ):
            continue
        kept.append(h)
        if len(kept) >= top_k:
            break
    return kept


def search(raw_query: str, top_k: int | None = None, source: str | None = None) -> SearchOutcome:
    """Türkçe sorgu → sıralı eşleşme listesi.

    source: "frame"|"crop"|None — kaynak filtresi (eval ablation'ı; None → prod hattı).
    """
    top_k = top_k or settings.default_top_k
    fetch_k = top_k * settings.search_overfetch  # tekilleştirme payı

    # ── 1. Ayrıştır: görsel metin + zaman filtresi ──
    parsed = parse_query(raw_query)

    # ── 2. Görsel metni embedle (boş kaldıysa ham sorguya geri düş) ──
    text = parsed.visual_text or raw_query
    vector = get_embedder().encode_text(text)

    # ── 3. Filtreli vektör araması ──
    results = get_store().search(
        vector, top_k=fetch_k,
        ts_from=parsed.ts_from, ts_to=parsed.ts_to,
        camera_id=parsed.camera_id, source=source,
    )

    # ── 4. Zaman filtresi hiçbir şey bulamadıysa filtresiz tekrar dene ──
    time_filter_dropped = False
    if not results and parsed.ts_from is not None:
        results = get_store().search(
            vector, top_k=fetch_k, camera_id=parsed.camera_id, source=source
        )
        time_filter_dropped = True

    # ── 5. Kare tekilleştirme + zaman kümeleme ──
    results = _dedup_and_group(results, top_k)

    return SearchOutcome(results=results, parsed=parsed, time_filter_dropped=time_filter_dropped)
