"""Hipotez: UI default top_k=12 ama vlm_top_n=8 → rank 9-12 DOĞRULANMADAN ekleniyor
(kuyruk sızıntısı). 'kırmızı kıyafetli adam'da bu 4 kart kırmızı araba → sızıyor.

top_k=12 ile koş, her sonucun _vlm var mı (doğrulandı) yoksa None mı (kuyruk) göster.
"""

from gozcu.config import settings
from gozcu.search import search


def main() -> None:
    print(f"vlm_top_n={settings.vlm_top_n}  default_top_k={settings.default_top_k}\n")
    o = search("kırmızı kıyafetli adam", top_k=12, use_vlm=True)
    print(f"gösterilen={len(o.results)}  elenen={len(o.vlm_filtered)}\n")
    for i, h in enumerate(o.results):
        v = h.get("_vlm")
        if "_vlm" not in h:
            tag = "KUYRUK (doğrulanmadı!)"
        elif v is None:
            tag = "VLM-hata"
        else:
            tag = f"doğrulandı present={v['object_present']} color={v['color_match']}"
        print(f"  rank {i:2} {h.get('yolo_class','frame'):7} {h['score']:.3f}  {tag}")


if __name__ == "__main__":
    main()
