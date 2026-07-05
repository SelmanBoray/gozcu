"""Varlık sorusu yes-bias'ı: sıkı prompt yanlış-pozitifi kesiyor mu, doğru-pozitifi
bozuyor mu? A/B — aynı kırpıklarda mevcut vs sıkı presence promptu.

Yanlış-pozitif vakaları: "dog" (insan kırpığında evet demişti), "man" (araba kırpığında).
Doğru-pozitif kontrol: "man" (insan kırpığında evet olmalı), "car" (araba kırpığında evet).
"""

import requests

from gozcu.config import settings
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import _encode

MODEL = settings.vlm_model


def ask(img_b64: str, q: str) -> bool | None:
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": q, "images": [img_b64]}],
               "stream": False, "keep_alive": settings.vlm_keep_alive,
               "options": {"temperature": 0, "num_predict": 4}}
    try:
        r = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"].strip().lower().startswith("y")
    except Exception as e:
        return f"HATA {str(e)[:20]}"


def q_current(obj: str) -> str:
    art = "an" if obj[0] in "aeiou" else "a"
    return f"Is there {art} {obj} in this image? Answer only 'yes' or 'no'."


def q_strict(obj: str) -> str:
    art = "an" if obj[0] in "aeiou" else "a"
    return (f"Look at this image carefully. Is {art} {obj} actually present and clearly "
            f"visible? Answer only 'yes' or 'no'. If you mainly see other things "
            f"(people, vehicles, background) and no {obj}, answer 'no'.")


def crops_for(query: str, n: int = 6) -> list:
    out = run_search(query, top_k=n, use_vlm=False)
    imgs = []
    for h in out.results[:n]:
        try:
            imgs.append((h.get("yolo_class", "?"), _encode(vlm_image_for_hit(h, out_size=384))))
        except Exception:
            pass
    return imgs


def main() -> None:
    # köpek sorgusu insan kırpıkları getirir → "dog" yanlış-pozitif testi + "man" doğru-pozitif
    dog_crops = crops_for("köpek gezdiren adam", 5)
    print("İNSAN kırpıkları (köpek sorgusundan):")
    for cls, b in dog_crops:
        print(f"  {cls:6} | dog: cur={ask(b, q_current('dog'))!s:5} strict={ask(b, q_strict('dog'))!s:5}"
              f"  || man(kontrol): cur={ask(b, q_current('man'))!s:5} strict={ask(b, q_strict('man'))!s:5}")

    # kırmızı kıyafetli adam araba getirir → "man" yanlış-pozitif + "car" doğru-pozitif
    red_crops = crops_for("kırmızı kıyafetli adam", 6)
    print("\nARABA/insan kırpıkları (kırmızı kıyafetli adam sorgusundan):")
    for cls, b in red_crops:
        print(f"  {cls:6} | man: cur={ask(b, q_current('man'))!s:5} strict={ask(b, q_strict('man'))!s:5}"
              f"  || car(kontrol): cur={ask(b, q_current('car'))!s:5} strict={ask(b, q_strict('car'))!s:5}")


if __name__ == "__main__":
    main()
