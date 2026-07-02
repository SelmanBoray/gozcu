"""jina-clip-v2 sarmalayıcı — görüntü ve Türkçe metin için ortak embedding uzayı.

GPU'da fp16, CPU'da ONNX int8 (aynı 1024-dim vektörler → tek ortak indeks).
Model seçimi gerekçesi: ARCHITECTURE.md §1
"""

import numpy as np


class Embedder:
    """Toplu görüntü/metin encode; cihazı (cuda/cpu) otomatik seçer."""

    def __init__(self, device: str = "auto") -> None:
        # ── Model yükleme: HF transformers, trust_remote_code, fp16 (GPU) / ONNX int8 (CPU) ──
        raise NotImplementedError("Faz 1 — sıradaki implementasyon adımı")

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        """BGR kare listesi → (N, 1024) float32, L2-normalize."""
        raise NotImplementedError

    def encode_text(self, text: str) -> np.ndarray:
        """Türkçe sorgu metni → (1024,) float32, L2-normalize. Çeviri YOK — doğrudan embed."""
        raise NotImplementedError
