"""Hipotez: HATALAR = qwen3 thinking-modu kaçak üretimi (n_ctx=4096 dolup truncate),
VRAM sayfalaması DEĞİL. Log kanıtı: eval 2905 token, n_tokens=4095 truncated=1.

Test: aynı 9 görüntü, 2 config yan yana:
  A = mevcut  (temperature 0)                         → baseline (kaçak üretim bekleniyor)
  B = think:false + num_predict:200  (emniyet kapağı) → hata 0, latency düşük bekleniyor

Karar dayanağı: deney_notu.md
"""

import json
import statistics
import time

import requests

from gozcu.config import settings
from gozcu.query import translate_visual
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import _encode, _prompt

QUERY = "kırmızı araba"


def call(img_b64: str, english: str, fix: bool) -> dict:
    payload = {
        "model": settings.vlm_model,
        "messages": [{"role": "user", "content": _prompt(english, True), "images": [img_b64]}],
        "format": "json", "stream": False, "keep_alive": settings.vlm_keep_alive,
        "options": {"temperature": 0},
    }
    if fix:
        payload["think"] = False                 # thinking trace'i kapat (kök neden)
        payload["options"]["num_predict"] = 200  # emniyet kapağı (JSON ~40 token)
    t0 = time.perf_counter()
    try:
        r = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
        r.raise_for_status()
        body = r.json()
        dt = time.perf_counter() - t0
        json.loads(body["message"]["content"])  # geçerli JSON mi?
        return {"ok": True, "dt": dt, "ptok": body.get("prompt_eval_count"),
                "gen": body.get("eval_count")}
    except Exception as e:
        return {"ok": False, "dt": time.perf_counter() - t0,
                "gen": None, "err": str(e)[:50]}


def main() -> None:
    english = translate_visual(QUERY)
    outcome = run_search(QUERY, top_k=12, use_vlm=False)
    imgs = []
    for hit in outcome.results[:9]:
        try:
            imgs.append(_encode(vlm_image_for_hit(hit, out_size=384)))
        except Exception:
            pass
    print(f"{len(imgs)} görüntü, sorgu EN={english!r}\n")

    for label, fix in [("A mevcut     ", False), ("B think:false", True)]:
        rows = [call(b, english, fix) for b in imgs]
        ok = [r for r in rows if r["ok"]]
        lat = [r["dt"] for r in ok]
        gens = [r["gen"] for r in ok if r["gen"] is not None]
        errs = len(rows) - len(ok)
        line = f"{label} | başarılı {len(ok)}/{len(rows)}  HATA {errs}"
        if lat:
            line += (f"  | latency med={statistics.median(lat):.1f}s "
                     f"max={max(lat):.1f}s")
        if gens:
            line += f"  | gen-token med={int(statistics.median(gens))} max={max(gens)}"
        print(line)


if __name__ == "__main__":
    main()
