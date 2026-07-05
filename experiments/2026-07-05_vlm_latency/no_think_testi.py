"""qwen3 soft-switch: prompt'a '/no_think' ekleyince thinking kapanıyor mu?

`think` API parametresi qwen3-vl için bu Ollama'da çalışmıyor (think_alani.py kanıtı).
qwen3-native yol: user mesajının sonuna '/no_think'. 3 başarısız + 2 başarılı görüntüde
test — hepsi geçerli JSON + düşük latency döndürüyor mu?
"""

import json
import time

import requests

from gozcu.config import settings
from gozcu.query import translate_visual
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import _encode, _prompt


def call(img_b64: str, english: str, no_think: bool) -> dict:
    prompt = _prompt(english, True)
    if no_think:
        prompt = prompt + "\n/no_think"
    payload = {
        "model": settings.vlm_model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "format": "json", "stream": False, "keep_alive": settings.vlm_keep_alive,
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
            json.loads(content); valid = True
        except Exception:
            valid = False
        return {"dt": dt, "gen": body.get("eval_count"), "valid": valid,
                "raw": content[:70]}
    except Exception as e:
        return {"dt": time.perf_counter() - t0, "gen": None, "err": str(e)[:50]}


def main() -> None:
    english = translate_visual("kırmızı araba")
    outcome = run_search("kırmızı araba", top_k=12, use_vlm=False)
    # img3,5,8 = deterministik başarısız · img4,2 = başarılı (regresyon kontrolü)
    picks = [3, 5, 8, 4, 2]
    imgs = {i: _encode(vlm_image_for_hit(outcome.results[i], out_size=384)) for i in picks}
    for tag, nt in [("base    ", False), ("/no_think", True)]:
        print(f"\n=== {tag} ===")
        ok = 0
        for i in picks:
            r = call(imgs[i], english, nt)
            if "err" in r:
                print(f"  img{i}: {r['dt']:5.1f}s  HATA {r['err']}")
            else:
                mark = "OK " if r["valid"] else "BOZUK"
                ok += r["valid"]
                print(f"  img{i}: {r['dt']:5.1f}s  gen={r['gen']:>4}  {mark}  {r['raw']!r}")
        print(f"  → geçerli {ok}/{len(picks)}")


if __name__ == "__main__":
    main()
