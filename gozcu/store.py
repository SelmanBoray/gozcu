"""Qdrant katmanı — şema kurulumu, upsert, filtreli arama.

Koleksiyon: `frames`, 1024-dim, Cosine. Lokal (embedded) mod; sunucu moduna
geçişte şema aynı kalır. Şema detayı: ARCHITECTURE.md §3
"""

import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from gozcu.config import settings
from gozcu.sampler import FrameRecord

COLLECTION = "frames"


def _point_id(video_id: str, frame_idx: int) -> str:
    """Deterministik nokta kimliği — aynı kare iki kez indekslenirse üzerine yazar."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"gozcu:{video_id}:{frame_idx}"))


class FrameStore:
    """Lokal (embedded) Qdrant üzerinde kare indeksi."""

    def __init__(self) -> None:
        # ── Client + koleksiyon kurulumu ──
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(settings.qdrant_path))
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=settings.embed_dim, distance=models.Distance.COSINE
                ),
            )
            # ── Payload indeksleri (lokal mod yok sayabilir; sunucu modunda etkili) ──
            try:
                self.client.create_payload_index(
                    COLLECTION, "camera_id", models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    COLLECTION, "ts", models.PayloadSchemaType.FLOAT
                )
            except Exception:
                pass

    def upsert(
        self, records: list[FrameRecord], vectors: np.ndarray, thumb_paths: list[str]
    ) -> None:
        """Kare kayıtlarını vektörleriyle birlikte indekse yazar."""
        points = [
            models.PointStruct(
                id=_point_id(rec.video_id, rec.frame_idx),
                vector=vec.tolist(),
                payload={
                    "video_id": rec.video_id,
                    "video_path": rec.video_path,
                    "camera_id": rec.camera_id,
                    "ts": rec.ts,
                    "offset_s": rec.offset_s,
                    "frame_idx": rec.frame_idx,
                    "motion_score": rec.motion_score,
                    "phash": rec.phash,
                    "thumb_path": thumb,
                    # Faz 2 rezerve: track_ids, plates, yolo_classes
                },
            )
            for rec, vec, thumb in zip(records, vectors, thumb_paths)
        ]
        self.client.upsert(COLLECTION, points=points)

    def search(
        self,
        vector: np.ndarray,
        top_k: int,
        ts_from: float | None = None,
        ts_to: float | None = None,
        camera_id: str | None = None,
    ) -> list[dict]:
        """Vektör araması + opsiyonel zaman aralığı / kamera filtresi."""
        # ── Filtre kurulumu ──
        must: list[models.Condition] = []
        if ts_from is not None or ts_to is not None:
            must.append(
                models.FieldCondition(key="ts", range=models.Range(gte=ts_from, lte=ts_to))
            )
        if camera_id is not None:
            must.append(
                models.FieldCondition(key="camera_id", match=models.MatchValue(value=camera_id))
            )
        query_filter = models.Filter(must=must) if must else None

        # ── Arama ──
        response = self.client.query_points(
            COLLECTION,
            query=vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [{**point.payload, "score": point.score} for point in response.points]

    def count(self) -> int:
        """İndeksteki toplam kare sayısı."""
        return self.client.count(COLLECTION).count
