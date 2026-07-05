"""Sıkı renk promptu mavi/teal yanlış-kabulünü kesiyor mu, kırmızı/siyah/beyaz'ı bozmuyor mu?
A/B: mevcut vs sıkı ("predominantly {color}, teal/yeşil/gri değil"). AI Engineer: VLM mavi zayıf.
"""

import requests

from gozcu.config import settings
from gozcu.query import extract_vqa_targets
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import _encode


def ask(img_b64: str, q: str) -> bool | None:
    payload = {"model": settings.vlm_model,
               "messages": [{"role": "user", "content": q + " Answer only 'yes' or 'no'.",
                             "images": [img_b64]}],
               "stream": False, "keep_alive": settings.vlm_keep_alive,
               "options": {"temperature": 0, "num_predict": 4}}
    try:
        r = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"].strip().lower().startswith("y")
    except Exception:
        return None


def q_cur(obj, color):
    return f"Is the {obj} {color} in color?"


def q_strict(obj, color):
    return (f"Is the {obj} predominantly {color}? Answer 'no' if its main color is a "
            f"different or similar-but-distinct shade (e.g. teal, green, gray, silver, white).")


def crops(query, n=6):
    obj, color = extract_vqa_targets(query)
    out = run_search(query, top_k=n, use_vlm=False)
    imgs = []
    for h in out.results[:n]:
        try:
            imgs.append((h.get("yolo_class", "?"), _encode(vlm_image_for_hit(h, out_size=384))))
        except Exception:
            pass
    return obj, color, imgs


def main() -> None:
    for query, hedef in [("mavi kamyonet", "NEGATIF (True azalmalı)"),
                         ("kırmızı araba", "POZITIF (True kalmalı)"),
                         ("siyah SUV araç", "POZITIF (True kalmalı)")]:
        obj, color, imgs = crops(query, 6)
        cur = sum(1 for _, b in imgs if ask(b, q_cur(obj, color)))
        strict = sum(1 for _, b in imgs if ask(b, q_strict(obj, color)))
        print(f'"{query}" ({hedef}): mevcut True={cur}/{len(imgs)}  sıkı True={strict}/{len(imgs)}')


if __name__ == "__main__":
    main()
