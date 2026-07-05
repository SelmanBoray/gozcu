"""VLM çağrı güvenilirliği — temiz baseline (flash on, q8_0 YOK, default f16 KV).

Amaç: q8_0 leverini geri aldıktan sonra hata oranı 0'a döndü mü + latency dağılımını
doğrula. Gerçek kod yolunu kullanır: CLIP top-N aday → recrop → verify_hit (VLM).

Çıktı: her çağrı için wall-clock + prompt_eval token (spike kaynağı) + verdict/HATA.
Karar dayanağı: experiments/2026-07-05_vlm_latency/deney_notu.md
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

# ── Renk sorgusu: VLM'i koşullu tetikleyen tip (needs_vlm=True) ──
QUERY = "kırmızı araba"


def timed_verify(hit: dict, english: str) -> dict:
    """verify_hit'in içini açık zamanlar — prompt_eval token + wall-clock döner."""
    try:
        img = vlm_image_for_hit(hit, out_size=384)
    except Exception as e:
        return {"err": f"recrop: {e}"}
    payload = {
        "model": settings.vlm_model,
        "messages": [{"role": "user", "content": _prompt(english, True),
                      "images": [_encode(img)]}],
        "format": "json", "stream": False, "keep_alive": settings.vlm_keep_alive,
        "options": {"temperature": 0},
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
        r.raise_for_status()
        body = r.json()
        dt = time.perf_counter() - t0
        content = body["message"]["content"]
        verdict = json.loads(content)  # JSON geçerli mi? (q8_0 bunu bozuyordu)
        return {
            "dt": dt,
            "prompt_eval": body.get("prompt_eval_count"),
            "eval": body.get("eval_count"),
            "ok": True,
            "present": verdict.get("object_present"),
            "conf": verdict.get("confidence"),
        }
    except Exception as e:
        return {"dt": time.perf_counter() - t0, "err": str(e)[:80]}


def main() -> None:
    print(f"Sorgu: {QUERY!r}  →  EN: {translate_visual(QUERY)!r}")
    outcome = run_search(QUERY, top_k=12, use_vlm=False)
    hits = outcome.results[:12]
    print(f"CLIP {len(hits)} aday getirdi. VLM ile doğrulanıyor…\n")
    english = translate_visual(QUERY)

    rows, lat, errs = [], [], 0
    for i, hit in enumerate(hits):
        res = timed_verify(hit, english)
        rows.append(res)
        if res.get("ok"):
            lat.append(res["dt"])
            print(f"[{i:2}] {res['dt']:5.2f}s  ptok={res['prompt_eval']:>4}  "
                  f"present={res['present']}  conf={res['conf']}")
        else:
            errs += 1
            print(f"[{i:2}] {res.get('dt', 0):5.2f}s  HATA: {res.get('err')}")

    print("\n" + "=" * 50)
    print(f"Çağrı: {len(hits)}  |  Başarılı: {len(lat)}  |  HATA: {errs}")
    if lat:
        print(f"Latency  medyan={statistics.median(lat):.2f}s  "
              f"min={min(lat):.2f}s  max={max(lat):.2f}s  "
              f"ort={statistics.mean(lat):.2f}s")
        ptoks = [r["prompt_eval"] for r in rows if r.get("ok")]
        print(f"prompt_eval token  min={min(ptoks)}  max={max(ptoks)}  "
              f"medyan={int(statistics.median(ptoks))}")


if __name__ == "__main__":
    main()
