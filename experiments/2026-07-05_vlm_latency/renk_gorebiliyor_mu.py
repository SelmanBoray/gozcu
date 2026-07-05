"""qwen2.5vl:3b rengi HİÇ görebiliyor mu, yoksa JSON/şema çerçevesi mi rubber-stamp'e
zorluyor? Düz (şemasız) soru: "What color is the main vehicle? One word."

Doğru renk derse → model görüyor, sorun verify-JSON framing (düzeltilebilir).
Diyemezse → model bu iş için umutsuz (yes-bias derin), moondream de kurtarmaz.
"""

import requests

from gozcu.config import settings
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import _encode

MODEL = "qwen2.5vl:3b"


def ask(img_b64: str, q: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": q, "images": [img_b64]}],
        "stream": False, "keep_alive": settings.vlm_keep_alive,
        "options": {"temperature": 0},
    }
    try:
        r = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"].strip().replace("\n", " ")[:60]
    except Exception as e:
        return f"HATA {str(e)[:30]}"


def main() -> None:
    outcome = run_search("kırmızı araba", top_k=12, use_vlm=False)
    imgs = []
    for h in outcome.results[:9]:
        try:
            imgs.append(_encode(vlm_image_for_hit(h, out_size=384)))
        except Exception:
            pass
    print("Düz renk sorusu (şemasız):\n")
    for i, b in enumerate(imgs):
        color = ask(b, "What color is the main vehicle in this image? Answer with one word.")
        yn = ask(b, "Is the main vehicle in this image blue? Answer only 'yes' or 'no'.")
        print(f"  img{i}: renk={color!r}   mavi-mi={yn!r}")


if __name__ == "__main__":
    main()
