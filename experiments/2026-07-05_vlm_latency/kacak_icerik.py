"""Kaçak üretim NE üretiyor + görüntüye-özgü mü (deterministik) yoksa rastgele mi?

Her görüntüyü 2× çağır (base config). Yakala: eval_count, ham içerik ilk 160 karakter,
JSON geçerli mi. Amaç: hata deterministik mi (aynı görüntü hep patlıyor mu) + kaçak
çıktının şekli ne (thinking trace? tekrar? uzun string?).
"""

import json
import time

import requests

from gozcu.config import settings
from gozcu.query import translate_visual
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import _encode, _prompt


def call(img_b64: str, english: str) -> dict:
    payload = {
        "model": settings.vlm_model,
        "messages": [{"role": "user", "content": _prompt(english, True), "images": [img_b64]}],
        "format": "json", "stream": False, "keep_alive": settings.vlm_keep_alive,
        "options": {"temperature": 0},
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
        r.raise_for_status()
        body = r.json()
        content = body["message"]["content"]
        try:
            json.loads(content)
            valid = True
        except Exception:
            valid = False
        return {"dt": time.perf_counter() - t0, "gen": body.get("eval_count"),
                "ptok": body.get("prompt_eval_count"), "valid": valid,
                "raw": content[:160]}
    except Exception as e:
        return {"dt": time.perf_counter() - t0, "gen": None, "err": str(e)[:60]}


def main() -> None:
    english = translate_visual("kırmızı araba")
    outcome = run_search("kırmızı araba", top_k=12, use_vlm=False)
    imgs = []
    for hit in outcome.results[:9]:
        try:
            imgs.append(_encode(vlm_image_for_hit(hit, out_size=384)))
        except Exception:
            pass
    print(f"{len(imgs)} görüntü × 2 tekrar\n")
    for idx, b in enumerate(imgs):
        for rep in (1, 2):
            r = call(b, english)
            if "err" in r:
                print(f"img{idx} r{rep}: {r['dt']:5.1f}s  HTTP-HATA {r['err']}")
            else:
                tag = "OK  " if r["valid"] else "BOZUK"
                print(f"img{idx} r{rep}: {r['dt']:5.1f}s  ptok={r['ptok']:>4} "
                      f"gen={r['gen']:>4}  {tag}  raw={r['raw']!r}")
        print()


if __name__ == "__main__":
    main()
