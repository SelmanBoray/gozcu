"""Co-residency + doğruluk kapısı — AI Engineer karar testi (B: non-thinking VLM).

Aynı 9 altın kırpık, CLIP GPU'da resident'ken. Model swap: qwen2.5vl:3b.
- format = GERÇEK JSON şeması (sadece "json" değil — non-thinking modelde sözleşmeyi garanti)
- tüm think hack'leri YOK (think:false / /no_think / num_predict cap → hepsi çıkarıldı)
- temperature 0

Kapı kuralı (AI Engineer):
  9/9 geçerli JSON + done_reason hepsi 'stop' (hiç 'length' yok)  → ship
  herhangi 'length' / boş içerik / OOM                            → moondream'e düş

Kullanım: python -m experiments.2026-07-05_vlm_latency.coresidency_kapisi [model]
"""

import json
import statistics
import sys
import time

import requests

from gozcu.config import settings
from gozcu.query import translate_visual
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import _encode, _prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5vl:3b"

# ── Gerçek JSON şeması: non-thinking modelde çıktı sözleşmesini fiilen garanti eder ──
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
        "format": SCHEMA,              # ← şema kısıtı (think hack YOK)
        "stream": False,
        "keep_alive": settings.vlm_keep_alive,
        "options": {"temperature": 0},
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
        r.raise_for_status()
        body = r.json()
        dt = time.perf_counter() - t0
        content = body.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content); valid = True
        except Exception:
            parsed, valid = None, False
        return {"dt": dt, "gen": body.get("eval_count"), "ptok": body.get("prompt_eval_count"),
                "done": body.get("done_reason"), "valid": valid,
                "raw": content[:70], "parsed": parsed}
    except Exception as e:
        return {"dt": time.perf_counter() - t0, "err": str(e)[:60]}


def main() -> None:
    english = translate_visual("kırmızı araba")
    outcome = run_search("kırmızı araba", top_k=12, use_vlm=False)
    hits = outcome.results[:9]
    imgs = []
    for h in hits:
        try:
            imgs.append(_encode(vlm_image_for_hit(h, out_size=384)))
        except Exception:
            pass
    print(f"MODEL={MODEL}  |  {len(imgs)} görüntü, EN='red car'\n")

    rows, lat, ok, bad_done = [], [], 0, 0
    for i, b in enumerate(imgs):
        r = call(b, english)
        rows.append(r)
        if "err" in r:
            print(f"[{i}] {r['dt']:5.1f}s  HTTP-HATA {r['err']}")
            continue
        lat.append(r["dt"]); ok += r["valid"]
        if r["done"] != "stop":
            bad_done += 1
        mark = "OK " if r["valid"] else "BOZUK"
        print(f"[{i}] {r['dt']:5.1f}s  done={r['done']:<6} ptok={r['ptok']:>4} "
              f"gen={r['gen']:>4}  {mark}  {r['raw']!r}")

    print("\n" + "=" * 55)
    print(f"Geçerli JSON: {ok}/{len(imgs)}  |  done!='stop': {bad_done}")
    if lat:
        print(f"Latency  med={statistics.median(lat):.1f}s  min={min(lat):.1f}s  "
              f"max={max(lat):.1f}s")
    # ── Kapı kararı ──
    passed = ok == len(imgs) and bad_done == 0
    print(f"\nKAPI: {'GEÇTI → ship' if passed else 'KALDI → moondream fallback'}")


if __name__ == "__main__":
    main()
