"""img3 (deterministik başarısız) üzerinde tam message nesnesini dök: content vs thinking
alanı, done_reason, eval_count. think:false gerçekten thinking'i kapatıyor mu?

3 config: base / think:false / think:false+num_predict:200. Hangisi geçerli JSON döndürür?
"""

import json

import requests

from gozcu.config import settings
from gozcu.query import translate_visual
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import _encode, _prompt


def probe(img_b64: str, english: str, extra: dict, label: str) -> None:
    payload = {
        "model": settings.vlm_model,
        "messages": [{"role": "user", "content": _prompt(english, True), "images": [img_b64]}],
        "format": "json", "stream": False, "keep_alive": settings.vlm_keep_alive,
        "options": {"temperature": 0},
    }
    payload.update(extra)
    try:
        r = requests.post(settings.vlm_url, json=payload, timeout=60)
        r.raise_for_status()
        body = r.json()
        msg = body.get("message", {})
        content = msg.get("content", "")
        thinking = msg.get("thinking") or ""
        try:
            json.loads(content); valid = "GEÇERLI-JSON"
        except Exception:
            valid = "boş/bozuk"
        print(f"── {label} ──")
        print(f"   done_reason={body.get('done_reason')}  eval_count={body.get('eval_count')}")
        print(f"   content ({len(content)} kr): {content[:120]!r}  → {valid}")
        print(f"   thinking ({len(thinking)} kr): {thinking[:120]!r}")
    except Exception as e:
        print(f"── {label} ── HATA: {str(e)[:80]}")


def main() -> None:
    english = translate_visual("kırmızı araba")
    outcome = run_search("kırmızı araba", top_k=12, use_vlm=False)
    hit = outcome.results[3]  # deterministik başarısız img3
    img = _encode(vlm_image_for_hit(hit, out_size=384))
    print(f"img3: {hit.get('video_id')} src={hit.get('source')} "
          f"cls={hit.get('yolo_class')}\n")
    probe(img, english, {}, "A base (temperature 0)")
    probe(img, english, {"think": False}, "B think:false")
    probe(img, english, {"think": False, "options": {"temperature": 0, "num_predict": 200}},
          "C think:false + num_predict:200")


if __name__ == "__main__":
    main()
