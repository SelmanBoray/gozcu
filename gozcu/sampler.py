"""Video → örneklenmiş kare akışı.

2 fps aday kare + hareket kapısı + çapa karesi + pHash dedup.
Algoritma detayı: ARCHITECTURE.md §2
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class FrameRecord:
    """İndekslenecek tek bir karenin tüm meta verisi."""

    video_id: str
    camera_id: str
    ts: float           # epoch UTC (saniye)
    frame_idx: int
    motion_score: float  # değişen piksel oranı (0.0–1.0)
    phash: str           # 64-bit perceptual hash (hex)
    image: np.ndarray    # BGR kare (embedder ve thumbs için)


def sample_video(video_path: Path, camera_id: str, base_ts: float) -> Iterator[FrameRecord]:
    """Videoyu çözüp indekslemeye değer kareleri üretir.

    base_ts: videonun ilk karesinin epoch UTC zamanı.
    DİKKAT: dosya mtime'ına güvenme — OSD saati ile çapraz doğrula (Risk 2).
    """
    # ── 1. PyAV ile 2 fps aday kare çözümleme ──
    # ── 2. Hareket kapısı: 320×180 gri + blur + absdiff > 25, oran > 0.005 ──
    # ── 3. 60 sn'de bir çapa karesi (hareketten bağımsız) ──
    # ── 4. pHash dedup: son 10 kareyle Hamming ≤ 6 ise atla ──
    raise NotImplementedError("Faz 1 — sıradaki implementasyon adımı")
