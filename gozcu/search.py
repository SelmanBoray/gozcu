"""Arama hattı — sorgu ayrıştır, embedle, Qdrant'ta ara, sonuçları birleştir."""

from dataclasses import dataclass


@dataclass
class SearchResult:
    """Kullanıcıya gösterilecek tek eşleşme."""

    video_id: str
    camera_id: str
    ts: float
    score: float
    thumb_path: str


def search(raw_query: str, top_k: int | None = None) -> list[SearchResult]:
    """Türkçe sorgu → sıralı eşleşme listesi."""
    # ── 1. query.parse_query: görsel metin + zaman/kamera filtreleri ──
    # ── 2. embedder.encode_text: görsel metni embedle ──
    # ── 3. store.search: filtreli vektör araması ──
    # ── 4. SearchResult listesine dönüştür ──
    raise NotImplementedError("Faz 1 — sıradaki implementasyon adımı")
