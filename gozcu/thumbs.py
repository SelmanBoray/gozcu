"""Küçük resim yazıcı — arama sonuçlarında gösterilecek 480 px JPEG'ler."""

from pathlib import Path

import numpy as np


def write_thumb(image: np.ndarray, video_id: str, frame_idx: int) -> Path:
    """Kareyi 480 px genişliğe ölçekleyip JPEG (q=80) olarak kaydeder, yolunu döner."""
    # ── Ölçekle + JPEG yaz: data/thumbnails/<video_id>/<frame_idx>.jpg ──
    raise NotImplementedError("Faz 1 — sıradaki implementasyon adımı")
