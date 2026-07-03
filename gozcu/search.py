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


def search(raw_query: str, top_k: int | None = None) -> SearchOutcome:
    """Türkçe sorgu → sıralı eşleşme listesi."""
    top_k = top_k or settings.default_top_k

    # ── 1. Ayrıştır: görsel metin + zaman filtresi ──
    parsed = parse_query(raw_query)

    # ── 2. Görsel metni embedle (boş kaldıysa ham sorguya geri düş) ──
    text = parsed.visual_text or raw_query
    vector = get_embedder().encode_text(text)

    # ── 3. Filtreli vektör araması ──
    results = get_store().search(
        vector, top_k=top_k,
        ts_from=parsed.ts_from, ts_to=parsed.ts_to,
        camera_id=parsed.camera_id,
    )

    # ── 4. Zaman filtresi hiçbir şey bulamadıysa filtresiz tekrar dene ──
    time_filter_dropped = False
    if not results and parsed.ts_from is not None:
        results = get_store().search(vector, top_k=top_k, camera_id=parsed.camera_id)
        time_filter_dropped = True

    return SearchOutcome(results=results, parsed=parsed, time_filter_dropped=time_filter_dropped)
