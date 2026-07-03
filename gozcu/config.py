"""Gözcü yapılandırması — tüm yollar, model kimlikleri ve eşikler tek yerde.

Eşik değerlerinin gerekçeleri için: ARCHITECTURE.md
"""

from pathlib import Path

from pydantic_settings import BaseSettings

# ── Proje kökü: yollar çalışma dizininden bağımsız olsun ──
_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── Yollar ──
    data_dir: Path = _ROOT / "data"
    thumbs_dir: Path = _ROOT / "data" / "thumbnails"
    qdrant_path: Path = _ROOT / "data" / "qdrant_storage"

    # ── Embedding modeli ──
    model_id: str = "jinaai/jina-clip-v2"
    embed_dim: int = 1024
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    batch_size_gpu: int = 16
    batch_size_cpu: int = 4

    # ── Kare örnekleme ──
    # Eşikler 3 Temmuz 2026 VIRAT teşhisiyle yeniden ayarlandı — uzaktaki küçük
    # özneler (40 px insan) eski değerlerde tamamen eleniyordu. AI Engineer onaylı.
    sample_fps: float = 2.0             # aday kare oranı
    motion_width: int = 320             # hareket kapısı çözünürlüğü
    motion_height: int = 180
    motion_pixel_thresh: int = 10       # normalize absdiff "piksel değişti" eşiği (eski: 25)
    motion_keep_ratio: float = 0.0005   # değişen piksel oranı eşiği (eski: 0.005)
    dedup_change_ratio: float = 0.001   # son tutulan kareye göre birikimli değişim eşiği
    global_change_ratio: float = 0.25   # üstü "küresel olay" (ışık/AGC) — tek kare tut
    min_blob_px: int = 5                # bağlı bileşen gürültü filtresi (yağmur/gren)
    osd_freq_thresh: float = 0.5        # sürekli değişen piksel maskesi (yanık OSD saati)
    max_keep_per_hour: int = 600        # kamera başına oran sınırı (çapalar muaf)
    anchor_interval_s: float = 60.0     # hareketsiz de olsa çapa karesi aralığı

    # ── Küçük resimler ──
    thumb_width: int = 480
    thumb_quality: int = 80

    # ── Arama ──
    default_top_k: int = 12


settings = Settings()
