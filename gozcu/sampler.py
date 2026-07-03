"""Video → örneklenmiş kare akışı.

2 fps aday kare + hareket kapısı + birikimli değişim dedup + çapa karesi.
pHash dedup KALDIRILDI: global hash, uzaktaki küçük özneye yapısal olarak kör
(VIRAT kampüs testinde 44 kareyi 1'e indiriyordu — Hamming=0). Yerine son
TUTULAN kareye göre birikimli değişim: yürüyen kişi fark biriktirir, gürültü
biriktirmez. Teşhis: experiments/2026-07-03_gercek_cctv_testi/
Algoritma detayı: ARCHITECTURE.md §2
"""

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
    phash: str           # 64-bit perceptual hash (yalnız metadata/hata ayıklama)
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


# ── Değişim maskesi yardımcıları ──

def _normalize(gray: np.ndarray) -> np.ndarray:
    """Ortalama-normalizasyon: AGC/pozlama kayması tüm kareyi 'değişti' saymasın."""
    g = gray.astype(np.float32)
    return g - g.mean()


def _change_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """İki normalize kare arasındaki ham değişim maskesi (bool)."""
    return np.abs(a - b) > settings.motion_pixel_thresh


def _blob_filter(mask: np.ndarray) -> np.ndarray:
    """Gürültü filtresi: küçük bağlı bileşenleri at (yağmur/gren blob değildir)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= settings.min_blob_px:
            keep[labels == i] = True
    return keep


def sample_video(video_path: Path, camera_id: str, base_ts: float) -> Iterator[FrameRecord]:
    """Videoyu çözüp indekslemeye değer kareleri üretir."""
    # ── Durum değişkenleri ──
    step = 1.0 / settings.sample_fps
    next_sample_t = 0.0
    prev_norm: np.ndarray | None = None    # önceki aday (hareket skoru için)
    ref_norm: np.ndarray | None = None     # son tutulan kare (dedup referansı)
    change_freq: np.ndarray | None = None  # EMA: sürekli değişen pikseller (OSD saati vb.)
    last_keep_t = float("-inf")
    hour_window = -1
    kept_in_hour = 0
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

            # ── 2. Küçültme + blur + ortalama-normalizasyon ──
            small = cv2.resize(image, (settings.motion_width, settings.motion_height))
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            small = cv2.GaussianBlur(small, (5, 5), 0)
            norm = _normalize(small)

            # ── 3. OSD maskesi: son adaylarda sürekli değişen pikseller (yanık saat vb.) ──
            if prev_norm is not None:
                raw_change = _change_mask(norm, prev_norm)
                if change_freq is None:
                    change_freq = raw_change.astype(np.float32)
                else:
                    change_freq = 0.95 * change_freq + 0.05 * raw_change
            osd_mask = change_freq > settings.osd_freq_thresh if change_freq is not None else None

            # ── 4. Hareket kapısı: önceki adaya göre blob-filtreli değişim oranı ──
            if prev_norm is None:
                motion_score = 1.0  # ilk kare her zaman aday
            else:
                mask = raw_change.copy()
                if osd_mask is not None:
                    mask &= ~osd_mask
                motion_score = float(_blob_filter(mask).mean())
            prev_norm = norm

            is_anchor = (t - last_keep_t) >= settings.anchor_interval_s
            if motion_score <= settings.motion_keep_ratio and not is_anchor:
                continue

            # ── 5. Küresel olay koruması: ışık/AGC sıçraması → tek kare tut, referansı sıfırla ──
            is_global_event = False
            if ref_norm is not None:
                ref_change = _change_mask(norm, ref_norm)
                if float(ref_change.mean()) > settings.global_change_ratio:
                    is_global_event = True

                # ── 6. Birikimli değişim dedup: son tutulan kareden yeterince farklı mı? ──
                elif not is_anchor:
                    if osd_mask is not None:
                        ref_change &= ~osd_mask
                    if float(_blob_filter(ref_change).mean()) <= settings.dedup_change_ratio:
                        continue

            # ── 7. Oran sınırı: kamera başına saatte en çok N kare (çapalar muaf) ──
            window = int(t // 3600)
            if window != hour_window:
                hour_window, kept_in_hour = window, 0
            if kept_in_hour >= settings.max_keep_per_hour and not (is_anchor or is_global_event):
                continue
            kept_in_hour += 1

            # ── 8. Kareyi tut ──
            ref_norm = norm
            last_keep_t = t
            phash = imagehash.phash(Image.fromarray(small))

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
