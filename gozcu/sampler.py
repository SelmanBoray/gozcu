"""Video → örneklenmiş kare akışı.

2 fps aday kare + hareket kapısı + çapa karesi + pHash dedup.
Algoritma detayı: ARCHITECTURE.md §2
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import cv2
import imagehash
import numpy as np
from PIL import Image

from gozcu.config import settings


@dataclass
class FrameRecord:
    """İndekslenecek tek bir karenin tüm meta verisi."""

    video_id: str
    video_path: str      # viewer'ın videoyu açabilmesi için mutlak yol
    camera_id: str
    ts: float            # epoch UTC (saniye)
    offset_s: float      # videonun başından itibaren saniye (seek için)
    frame_idx: int
    motion_score: float  # değişen piksel oranı (0.0–1.0)
    phash: str           # 64-bit perceptual hash (hex)
    image: np.ndarray    # BGR kare (embedder ve thumbs için)


def get_video_duration(video_path: Path) -> float:
    """Videonun süresini saniye cinsinden döner."""
    with av.open(str(video_path)) as container:
        if container.duration is not None:
            return container.duration / av.time_base
        stream = container.streams.video[0]
        return float(stream.duration * stream.time_base)


def video_start_ts(video_path: Path) -> float:
    """Videonun ilk karesinin tahmini epoch zamanı: mtime - süre.

    DİKKAT: DVR dosya zamanları yalan söyleyebilir — OSD saati ile
    çapraz doğrulama Faz 2 işi (ARCHITECTURE.md Risk 2).
    """
    return video_path.stat().st_mtime - get_video_duration(video_path)


def sample_video(video_path: Path, camera_id: str, base_ts: float) -> Iterator[FrameRecord]:
    """Videoyu çözüp indekslemeye değer kareleri üretir."""
    # ── Durum değişkenleri ──
    step = 1.0 / settings.sample_fps
    next_sample_t = 0.0
    prev_small: np.ndarray | None = None
    last_keep_t = float("-inf")
    recent_hashes: deque = deque(maxlen=settings.phash_window)
    video_id = video_path.stem

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for frame_idx, frame in enumerate(container.decode(stream)):
            # ── 1. 2 fps aday kare seçimi ──
            t = frame.time if frame.time is not None else frame_idx / (stream.average_rate or 25)
            if t < next_sample_t:
                continue
            next_sample_t = t + step

            image = frame.to_ndarray(format="bgr24")

            # ── 2. Hareket kapısı: 320×180 gri + blur + absdiff ──
            small = cv2.resize(image, (settings.motion_width, settings.motion_height))
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            small = cv2.GaussianBlur(small, (5, 5), 0)

            if prev_small is None:
                motion_score = 1.0  # ilk kare her zaman aday
            else:
                diff = cv2.absdiff(small, prev_small)
                motion_score = float((diff > settings.motion_pixel_thresh).mean())
            prev_small = small

            # ── 3. Çapa karesi: hareketsiz de olsa 60 sn'de bir tut ──
            is_anchor = (t - last_keep_t) >= settings.anchor_interval_s
            if motion_score <= settings.motion_keep_ratio and not is_anchor:
                continue

            # ── 4. pHash dedup (çapa kareleri dedup'ı atlar — kapsama garantisi) ──
            phash = imagehash.phash(Image.fromarray(small))
            if not is_anchor and any(phash - h <= settings.phash_hamming_max for h in recent_hashes):
                continue
            recent_hashes.append(phash)
            last_keep_t = t

            yield FrameRecord(
                video_id=video_id,
                video_path=str(video_path.resolve()),
                camera_id=camera_id,
                ts=base_ts + t,
                offset_s=t,
                frame_idx=frame_idx,
                motion_score=motion_score,
                phash=str(phash),
                image=image,
            )
