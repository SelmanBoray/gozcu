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
    sample_fps: float = 2.0            # aday kare oranı
    motion_width: int = 320            # hareket kapısı çözünürlüğü
    motion_height: int = 180
    motion_pixel_thresh: int = 25      # absdiff "piksel değişti" eşiği
    motion_keep_ratio: float = 0.005   # değişen piksel oranı bu eşiği aşarsa kareyi tut
    anchor_interval_s: float = 60.0    # hareketsiz de olsa çapa karesi aralığı
    phash_hamming_max: int = 6         # dedup: bu mesafenin altı "aynı kare" sayılır
    phash_window: int = 10             # dedup: son N tutulan kareyle karşılaştır

    # ── Küçük resimler ──
    thumb_width: int = 480
    thumb_quality: int = 80

    # ── Arama ──
    default_top_k: int = 12


settings = Settings()
