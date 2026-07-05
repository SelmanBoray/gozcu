"""Ayırt-etme (discrimination) testi — qwen2.5vl:3b gerçekten BAKIYOR mu, yoksa
her şeyi onaylıyor mu (rubber-stamp)? Bir verify katmanı için asıl kapı budur.

Aynı 9 araba kırpığında 3 sorgu:
  - "red car"   → POZITIF (çoğu true beklenir)
  - "dog"       → NEGATIF (hepsi false beklenir — kırpıklar araba)
  - "blue truck"→ NEGATIF/kısmi (mavi kamyon değil kırmızı araba → false beklenir)

Hepsi 'true/1.0' dönerse model ayırt etmiyor → verify anlamsız → moondream'e bak.
"""

import json
import time

import requests

from gozcu.config import settings
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import _encode, _prompt

MODEL = "qwen2.5vl:3b"
SCHEMA = {
    "type": "object",
    "properties": {
        "object_present": {"type": "boolean"},
        "color_match": {"type": ["boolean", "null"]},
        "confidence": {"type": "number"},
    },
    "required": ["object_present", "color_match", "confidence"],
}


def call(img_b64: str, english: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": _prompt(english, True), "images": [img_b64]}],
        "format": SCHEMA, "stream": False, "keep_alive": settings.vlm_keep_alive,
        "options": {"temperature": 0},
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
        r.raise_for_status()
        v = json.loads(r.json()["message"]["content"])
        return {"dt": time.perf_counter() - t0, "present": v.get("object_present"),
                "conf": v.get("confidence")}
    except Exception as e:
        return {"err": str(e)[:40]}


def main() -> None:
    outcome = run_search("kırmızı araba", top_k=12, use_vlm=False)
    imgs = []
    for h in outcome.results[:9]:
        try:
            imgs.append(_encode(vlm_image_for_hit(h, out_size=384)))
        except Exception:
            pass
    for english, beklenti in [("red car", "POZITIF"), ("dog", "NEGATIF"),
                              ("blue truck", "NEGATIF")]:
        present_count = 0
        confs = []
        for b in imgs:
            r = call(b, english)
            if "err" not in r:
                present_count += bool(r["present"])
                confs.append(r["conf"])
        avg = sum(confs) / len(confs) if confs else 0
        print(f'"{english}" ({beklenti}):  present {present_count}/{len(imgs)}  '
              f"ort-conf {avg:.2f}")


if __name__ == "__main__":
    main()
