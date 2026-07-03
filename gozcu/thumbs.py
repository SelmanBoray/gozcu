"""Küçük resim yazıcı — arama sonuçlarında gösterilecek 480 px JPEG'ler."""

from pathlib import Path

import cv2
import numpy as np

from gozcu.config import settings


def write_thumb(image: np.ndarray, video_id: str, frame_idx: int) -> Path:
    """Kareyi 480 px genişliğe ölçekleyip JPEG (q=80) olarak kaydeder, yolunu döner."""
    # ── Hedef klasör: data/thumbnails/<video_id>/ ──
    out_dir = settings.thumbs_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Ölçekle ve yaz ──
    height, width = image.shape[:2]
    scale = settings.thumb_width / width
    thumb = cv2.resize(image, (settings.thumb_width, max(1, round(height * scale))))
    out_path = out_dir / f"{frame_idx}.jpg"
    cv2.imwrite(str(out_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, settings.thumb_quality])
    return out_path
