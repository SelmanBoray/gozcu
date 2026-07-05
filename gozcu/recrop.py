"""Faz 2 — orijinal videodan yüksek çözünürlüklü yeniden-kırpma (VLM doğrulaması için).

Sorun: kırpık thumbnail'leri uzak öznede 36×36 px olabiliyor — VLM'in renk/detay
doğrulaması için (siyah SUV? yağmur var mı?) yetersiz. Çözüm: payload zaten
`video_path` + `offset_s` + normalize `bbox` tutuyor; orijinal kareden yüksek
çözünürlüklü bağlam-paylı bir kırpık yeniden çıkarılır.

Kaynak videolar mevcut (tüm korpus için doğrulandı). ffmpeg ile hızlı seek.
"""

import io
import subprocess

from PIL import Image, ImageDraw

from gozcu.config import settings


def frame_at(video_path: str, offset_s: float) -> Image.Image:
    """Videonun `offset_s` saniyesindeki kareyi tam çözünürlükte döndür (ffmpeg seek)."""
    # ── -ss girişten ÖNCE: anahtar-kareye hızlı seek (yaklaşık ama hızlı) ──
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{offset_s:.3f}",
         "-i", video_path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=False,
    )
    if not proc.stdout:
        raise RuntimeError(f"Kare çıkarılamadı: {video_path} @ {offset_s}s — {proc.stderr[:200]}")
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


def crop_bbox(
    frame: Image.Image,
    bbox_norm: list[float],
    margin: float = 0.25,
    out_size: int = 384,
) -> Image.Image:
    """Normalize bbox'ı bağlam payıyla kırp, kareye tamamla, out_size'a ölçekle (VLM girdisi).

    Bağlam payı: VLM'in özneyi çevresiyle görmesi doğrulamayı iyileştirir (yağmur/kar
    sahne bağlamı, rengin ışıkta görünümü). Kare aspect: VLM'ler kare girdide tutarlı.
    """
    w, h = frame.size
    x1, y1, x2, y2 = bbox_norm
    # ── Piksele çevir + bağlam payı ──
    bw, bh = (x2 - x1) * w, (y2 - y1) * h
    px1 = (x1 * w) - bw * margin
    py1 = (y1 * h) - bh * margin
    px2 = (x2 * w) + bw * margin
    py2 = (y2 * h) + bh * margin
    # ── Kareye tamamla (uzun kenara göre), sınıra kırp ──
    side = max(px2 - px1, py2 - py1)
    cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
    sx1 = max(0, int(cx - side / 2))
    sy1 = max(0, int(cy - side / 2))
    sx2 = min(w, int(cx + side / 2))
    sy2 = min(h, int(cy + side / 2))
    crop = frame.crop((sx1, sy1, sx2, sy2))
    # ── VLM için ölçekle (küçükse büyüt — detay yaratmaz ama VLM giriş boyutunu karşılar) ──
    return crop.resize((out_size, out_size), Image.LANCZOS)


def vlm_image_for_hit(hit: dict, out_size: int = 384) -> Image.Image:
    """(Eski) tight-kırpık — geriye dönük uyum. Yeni doğrulama: vlm_frame_for_hit."""
    frame = frame_at(hit["video_path"], hit["offset_s"])
    if hit.get("source") == "crop" and hit.get("bbox"):
        return crop_bbox(frame, hit["bbox"], out_size=out_size)
    w, h = frame.size
    scale = settings.thumb_width / max(w, h)
    return frame.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _cap_side(frame: Image.Image, max_side: int) -> Image.Image:
    """Uzun kenarı max_side'a indir (küçükse dokunma) — token/latency'yi tutar."""
    w, h = frame.size
    if max(w, h) <= max_side:
        return frame
    s = max_side / max(w, h)
    return frame.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def draw_bbox(frame: Image.Image, bbox_norm: list[float]) -> Image.Image:
    """Normalize bbox'ı parlak dikdörtgenle çiz (referring-style — VLM'e 'hangi özne' der)."""
    frame = frame.copy()
    w, h = frame.size
    x1, y1, x2, y2 = bbox_norm
    lw = max(3, int(max(w, h) * 0.006))
    ImageDraw.Draw(frame).rectangle(
        [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)],
        outline=(255, 40, 40), width=lw)
    return frame


def vlm_frame_for_hit(hit: dict, draw_box: bool = True, max_side: int | None = None) -> Image.Image:
    """Doğrulama görüntüsü: sınırlandırılmış TAM-KARE (küçük kırpık yerine tam bağlam).

    draw_box: kırpık aday ise bbox'ı çiz (renk/öznitelik → 'kutudaki nesne'). Zor-kavram
    (köpek insan-bbox dışında olabilir) sorgusunda draw_box=False → kutusuz tüm sahne.
    Detay: ARCHITECTURE.md §8 (AI Engineer b-kararı).
    """
    max_side = max_side or settings.vlm_frame_max_side
    frame = _cap_side(frame_at(hit["video_path"], hit["offset_s"]), max_side)
    if draw_box and hit.get("source") == "crop" and hit.get("bbox"):
        return draw_bbox(frame, hit["bbox"])
    return frame
