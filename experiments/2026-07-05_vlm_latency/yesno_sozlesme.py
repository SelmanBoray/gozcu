"""YES/NO VQA sözleşmesi — verify katmanının doğru daralması (JSON değil, öznitelik-başına
yes/no). Kanıt (renk_gorebiliyor_mu.py): qwen2.5vl:3b düz yes/no'da rengi kusursuz ayırıyor,
JSON şemasında rubber-stamp'liyor.

Sözleşme: sorguyu (nesne, renk) ikilisine ayır; iki ayrı yes/no sorusu sor.
  object_present = "Is there a {obj} in this image?" == yes
  color_match    = "Is the {obj} {color}?" == yes   (renk yoksa None)
  confidence     = 1.0 if object_present else 0.0    (ayrım booleanlarda)

Ölç: ayrım (red vs blue) + latency (JSON'un 16.8s'ine karşı).
"""

import statistics
import time

import requests

from gozcu.config import settings
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import _encode

MODEL = "qwen2.5vl:3b"


def yesno(img_b64: str, question: str) -> tuple[bool, float]:
    """Tek yes/no sorusu → (cevap_evet_mi, latency)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": question + " Answer only 'yes' or 'no'.",
                      "images": [img_b64]}],
        "stream": False, "keep_alive": settings.vlm_keep_alive,
        "options": {"temperature": 0, "num_predict": 4},  # tek kelime yeter
    }
    t0 = time.perf_counter()
    r = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
    r.raise_for_status()
    ans = r.json()["message"]["content"].strip().lower()
    return ans.startswith("y"), time.perf_counter() - t0


def verify(img_b64: str, obj: str, color: str | None) -> dict:
    """YES/NO sözleşmeli tam verdict + toplam latency."""
    present, dt1 = yesno(img_b64, f"Is there a {obj} in this image?")
    dt2 = 0.0
    color_match = None
    if color and present:
        color_match, dt2 = yesno(img_b64, f"Is the {obj} {color} colored?")
    return {"object_present": present, "color_match": color_match,
            "confidence": 1.0 if present else 0.0, "dt": dt1 + dt2}


def main() -> None:
    outcome = run_search("kırmızı araba", top_k=12, use_vlm=False)
    imgs = []
    for h in outcome.results[:9]:
        try:
            imgs.append(_encode(vlm_image_for_hit(h, out_size=384)))
        except Exception:
            pass

    for obj, color, beklenti in [("car", "red", "POZITIF: present+color T"),
                                 ("car", "blue", "NEGATIF: color F"),
                                 ("dog", None, "NEGATIF: present F")]:
        present_ok = cmatch_true = 0
        lat = []
        for b in imgs:
            v = verify(b, obj, color)
            present_ok += v["object_present"]
            cmatch_true += (v["color_match"] is True)
            lat.append(v["dt"])
        print(f'"{color or ""} {obj}".strip() ({beklenti})')
        print(f"   present {present_ok}/{len(imgs)}  color_match=True {cmatch_true}/{len(imgs)}  "
              f"| latency med={statistics.median(lat):.1f}s max={max(lat):.1f}s\n")


if __name__ == "__main__":
    main()
