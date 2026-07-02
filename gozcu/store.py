"""Qdrant katmanı — şema kurulumu, upsert, filtreli arama.

Koleksiyon: `frames`, 1024-dim, Cosine, HNSW varsayılan + int8 scalar kuantizasyon.
Şema detayı: ARCHITECTURE.md §3
"""

import numpy as np

from gozcu.sampler import FrameRecord


class FrameStore:
    """Lokal (embedded) Qdrant üzerinde kare indeksi."""

    def __init__(self) -> None:
        # ── Qdrant client (lokal path modu) + koleksiyon yoksa oluştur ──
        # ── Payload indeksleri: camera_id (keyword), ts (float, range) ──
        # ── Faz 2 rezerve alanlar: track_ids, plates, yolo_classes ──
        raise NotImplementedError("Faz 1 — sıradaki implementasyon adımı")

    def upsert(self, records: list[FrameRecord], vectors: np.ndarray, thumb_paths: list[str]) -> None:
        """Kare kayıtlarını vektörleriyle birlikte indekse yazar."""
        raise NotImplementedError

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        ts_from: float | None = None,
        ts_to: float | None = None,
        camera_id: str | None = None,
    ) -> list[dict]:
        """Vektör araması + opsiyonel zaman aralığı / kamera filtresi."""
        raise NotImplementedError
