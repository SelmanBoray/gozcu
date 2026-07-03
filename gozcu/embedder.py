"""jina-clip-v2 sarmalayıcı — görüntü ve Türkçe metin için ortak embedding uzayı.

GPU'da fp16; CPU'da fp32 (ONNX int8 yolu ileride eklenecek — aynı vektör uzayı).
Model seçimi gerekçesi: ARCHITECTURE.md §1
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from gozcu.config import settings


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)


class Embedder:
    """Toplu görüntü/metin encode; cihazı (cuda/cpu) otomatik seçer."""

    def __init__(self, device: str = "auto") -> None:
        # ── Cihaz seçimi ──
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = settings.batch_size_gpu if device == "cuda" else settings.batch_size_cpu

        # ── Model yükleme ──
        self.model = AutoModel.from_pretrained(settings.model_id, trust_remote_code=True)
        if device == "cuda":
            self.model = self.model.to("cuda", dtype=torch.float16)
        self.model.eval()

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        """BGR kare listesi → (N, 1024) float32, L2-normalize."""
        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
        with torch.inference_mode():
            vectors = self.model.encode_image(pil_images, batch_size=self.batch_size)
        return _l2_normalize(np.asarray(vectors))

    def encode_text(self, text: str) -> np.ndarray:
        """Türkçe sorgu metni → (1024,) float32, L2-normalize. Çeviri YOK — doğrudan embed."""
        with torch.inference_mode():
            vectors = self.model.encode_text([text])
        return _l2_normalize(np.asarray(vectors))[0]
