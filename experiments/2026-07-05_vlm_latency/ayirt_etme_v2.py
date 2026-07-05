"""Ayırt-etme v2 — TAM verdict (object_present + color_match + confidence) her görüntüde.

v1 hatası: color_match yazdırılmadı; oysa renk ayrımını o yapar ve fusion onu kullanır.
Ayrıca kırpıkların gerçekte NE olduğunu (yolo_class) yazdır — 'blue truck' 9/9 present
gerçekten yanlış mı yoksa kırpıklar kamyon mu?

Kritik soru: color_match kırmızı vs mavi'yi ayırıyor mu? Ayırıyorsa verify işe yarar.
"""

import json

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
    try:
        r = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
        r.raise_for_status()
        v = json.loads(r.json()["message"]["content"])
        return v
    except Exception as e:
        return {"err": str(e)[:40]}


def main() -> None:
    outcome = run_search("kırmızı araba", top_k=12, use_vlm=False)
    hits = outcome.results[:9]
    imgs, labels = [], []
    for h in hits:
        try:
            imgs.append(_encode(vlm_image_for_hit(h, out_size=384)))
            labels.append(f"{h.get('yolo_class','?')}")
        except Exception:
            pass
    for english in ["red car", "blue car", "dog"]:
        print(f'\n=== "{english}" ===')
        for lbl, b in zip(labels, imgs):
            v = call(b, english)
            if "err" in v:
                print(f"  {lbl:10} HATA {v['err']}")
            else:
                print(f"  {lbl:10} present={str(v.get('object_present')):5} "
                      f"color={str(v.get('color_match')):5} conf={v.get('confidence')}")


if __name__ == "__main__":
    main()
