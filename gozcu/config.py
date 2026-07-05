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

    # ── YOLO kırpık embedding (Faz 1.5) ──
    yolo_model: str = "yolo11m.pt"
    yolo_imgsz: int = 1280           # küçük özne için çözünürlük model boyutundan önemli
    yolo_conf_person: float = 0.15   # düşük eşik: indekste olmayan tespit kurtarılamaz
    yolo_conf_vehicle: float = 0.25
    crop_margin: float = 0.20        # bbox çevresi bağlam payı
    crop_min_h: int = 16             # altı gürültü lekesi — embedlenmez
    crop_tiny_h: int = 32            # altı 'tiny' etiketi alır (eval ölçümü için)
    static_iou: float = 0.85         # önceki kareyle IoU üstü = statik nesne, atla
    max_crops_per_frame: int = 24    # statik bastırma SONRASI, insan öncelikli sınır
    crop_thumb_px: int = 160

    # ── Küçük resimler ──
    thumb_width: int = 480
    thumb_quality: int = 80

    # ── Arama ──
    default_top_k: int = 12
    search_overfetch: int = 4        # kare tekilleştirme öncesi top_k × N getir
    group_window_s: float = 8.0      # aynı videoda bu pencere içinde tek sonuç
    # Sahne-niyetli sorguda kareye z-normalize yumuşak boost (Olgu B — ARCHITECTURE.md §7).
    # Yalnız intent=="scene"; nesne-niyeti nötr. z-skoru: skor-boşluğundan kalibre edildi.
    scene_boost_lambda: float = 1.0

    # ── Faz 2: VLM doğrulayıcı (retrieve-then-verify, YES/NO VQA — ARCHITECTURE.md §8) ──
    # qwen2.5vl:3b: non-thinking (qwen3-vl'in sonsuz-düşünme çöküşü yok), 37/37 layer GPU'da
    # CLIP ile sığar (ölçüldü), atomik yes/no'da rengi kusursuz ayırır. Teşhis:
    # experiments/2026-07-05_vlm_latency/
    vlm_model: str = "qwen2.5vl:3b"       # 3.2GB — CLIP ile eşzamanlı GPU'ya sığar (8GB kart)
    vlm_url: str = "http://localhost:11434/api/chat"
    vlm_keep_alive: str = "30m"           # her sorgu arası model boşaltma (swap thrash) önle
    vlm_top_n: int = 8                    # yalnız top-N aday doğrulanır (latency)
    vlm_drop_below: float = 0.3           # negasyon: eşleşme-güveni bu altındaysa düşür (absent)
    vlm_beta: float = 0.5                 # öznitelik rerank ağırlığı (skor-boşluğundan kalibre)
    vlm_timeout_s: float = 20.0           # atomik yes/no ~2-5s; 20s cold-load + baskı payı


settings = Settings()
