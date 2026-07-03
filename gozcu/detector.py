"""YOLO insan/araç tespiti — Faz 1.5 kırpık embedding hattı.

Tam-kare embedding'in küçük özne sulanması ölçülüp kesinleşince (3 Temmuz 2026,
experiments/2026-07-03_gercek_cctv_testi/) öne çekildi: tespit kırpıkları ayrı
vektör olarak indekslenir, kırpıkta özne karenin tamamıdır — sulanma yok.
Tasarım kararları AI Engineer onaylı: imgsz=1280 (küçük özne için çözünürlük
model boyutundan önemli), düşük güven eşiği (geri alınamaz karar — indekste
olmayan tespit kurtarılamaz), orijinal kareden kırpma, statik nesne bastırma.
"""

from dataclasses import dataclass

import numpy as np

from gozcu.config import settings

# ── COCO sınıf kimliği → Türkçe etiket ──
CLASSES = {0: "insan", 1: "bisiklet", 2: "araba", 3: "motosiklet", 5: "otobüs", 7: "kamyon"}
VEHICLE_IDS = {1, 2, 3, 5, 7}


@dataclass
class Detection:
    """Tek bir tespit — bbox orijinal kare pikseli cinsinden."""

    cls_id: int
    cls_name: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    tiny: bool = False                       # yükseklik < crop_tiny_h (eval etiketi)


def _iou(a: tuple, b: tuple) -> float:
    """İki bbox'ın kesişim/birleşim oranı."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class Detector:
    """yolo11m sarmalayıcı — tam karede insan/araç tespiti."""

    def __init__(self, device: str = "auto") -> None:
        # ── Cihaz seçimi + model yükleme ──
        import torch
        from ultralytics import YOLO

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = YOLO(settings.yolo_model)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """BGR kare → güvene göre sıralı tespit listesi (kare başına üst sınırlı)."""
        result = self.model.predict(
            image,
            imgsz=settings.yolo_imgsz,
            conf=settings.yolo_conf_person,  # en düşük eşik; araçlar aşağıda ayrıca süzülür
            classes=list(CLASSES),
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            cls_id, conf = int(box.cls), float(box.conf)
            if cls_id in VEHICLE_IDS and conf < settings.yolo_conf_vehicle:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            box_h = y2 - y1
            if box_h < settings.crop_min_h:
                continue  # gürültü lekesi — embedlemeye değmez
            detections.append(Detection(
                cls_id=cls_id, cls_name=CLASSES[cls_id], conf=conf,
                bbox=(x1, y1, x2, y2), tiny=box_h < settings.crop_tiny_h,
            ))

        # Kare başına sınır BURADA UYGULANMAZ — statik bastırmadan sonra uygulanır
        # (park halindeki 15+ araba güven sıralamasını doldurup insanları kesiyordu).
        return detections


def suppress_static(detections: list[Detection], prev_detections: list[Detection]) -> list[Detection]:
    """Önceki tutulan karedeki aynı-sınıf kutuyla IoU yüksekse atla.

    Otopark karesindeki 20-40 park halindeki araç her karede yeniden tespit edilir —
    her statik nesne yalnız İLK görünümünde bir vektör alır.
    """
    return [
        d for d in detections
        if not any(
            p.cls_id == d.cls_id and _iou(p.bbox, d.bbox) > settings.static_iou
            for p in prev_detections
        )
    ]


def cap_crops(detections: list[Detection]) -> list[Detection]:
    """Kare başına kırpık sınırı — İNSAN ÖNCELİKLİ, sonra güvene göre.

    CCTV aramada insan en nadir ve en kıymetli sınıf; salt güven sıralaması
    park halindeki araçların insanları kesmesine yol açıyordu (3 Temmuz bulgusu).
    """
    ranked = sorted(detections, key=lambda d: (d.cls_id != 0, -d.conf))
    return ranked[: settings.max_crops_per_frame]


def crop_image(image: np.ndarray, det: Detection) -> np.ndarray:
    """%20 bağlam payı + kareye tamamlama, ORİJİNAL çözünürlükten kırp.

    Kareye tamamlama: jina işlemcisinin kare resize'ı 3:1 yaya kutusunu esnetmesin.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = det.bbox
    bw, bh = x2 - x1, y2 - y1

    # ── Bağlam payı ──
    x1, x2 = x1 - bw * settings.crop_margin, x2 + bw * settings.crop_margin
    y1, y2 = y1 - bh * settings.crop_margin, y2 + bh * settings.crop_margin

    # ── Kareye tamamla, sınırlara kıstır ──
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1 = max(0, int(round(cx - side / 2)))
    y1 = max(0, int(round(cy - side / 2)))
    x2 = min(w, int(round(cx + side / 2)))
    y2 = min(h, int(round(cy + side / 2)))
    return image[y1:y2, x1:x2]
