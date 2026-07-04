"""Faz 2 — VLM doğrulayıcı (qwen3-vl:2b, Ollama). retrieve-then-verify.

CLIP top-N adayı, tam-çözünürlüklü görüntüyle (recrop) VLM'e sorulur. VLM yapılandırılmış
JSON döner: {object_present, color_match, confidence}. Yes-bias'a karşı öznitelik-başına
JSON + reddetme-yanlı prompt. Sorgu İngilizceye çevrilir (VLM'in Türkçesine güvenme).

Kullanım: yalnız öznitelik/negasyon sorgularında (search tarafında koşullu tetiklenir).
Detay: ARCHITECTURE.md §8
"""

import base64
import io
import json

import requests

from gozcu.config import settings
from gozcu.recrop import vlm_image_for_hit


def _encode(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _prompt(english_desc: str, ask_color: bool) -> str:
    """Reddetme-yanlı, öznitelik-başına JSON isteyen İngilizce doğrulama promptu."""
    color_line = (
        '"color_match": true only if the described color clearly matches the main object, '
        if ask_color else '"color_match": null, '
    )
    return (
        "You are verifying whether a security-camera image matches a search description.\n"
        f'Search description: "{english_desc}".\n'
        "Look carefully. Reply with ONLY a JSON object:\n"
        '{"object_present": true only if the described main subject is clearly visible, '
        f"{color_line}"
        '"confidence": your certainty from 0.0 to 1.0 that the FULL description matches '
        "this image (1.0 = clearly matches, 0.0 = clearly does NOT match)}\n"
        "Be strict: only high confidence when clearly and unambiguously matching. "
        "If the described subject is absent, set object_present false AND confidence near 0."
    )


def verify_hit(hit: dict, english_desc: str, ask_color: bool) -> dict | None:
    """Bir adayı VLM ile doğrula. Döndürür: {object_present, color_match, confidence}.

    Ollama erişilemez/hata → None (çağıran VLM'i atlar, CLIP sıralaması korunur).
    """
    try:
        img = vlm_image_for_hit(hit, out_size=384)
    except Exception:
        return None
    payload = {
        "model": settings.vlm_model,
        "messages": [{"role": "user", "content": _prompt(english_desc, ask_color),
                      "images": [_encode(img)]}],
        "format": "json",
        "stream": False,
        "keep_alive": settings.vlm_keep_alive,
        "options": {"temperature": 0},
    }
    # ── Tek retry: CLIP+VLM eşzamanlı GPU baskısında ara sıra timeout/500 olur ──
    verdict = None
    for _attempt in range(2):
        try:
            resp = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
            resp.raise_for_status()
            verdict = json.loads(resp.json()["message"]["content"])
            break
        except Exception:
            continue
    if verdict is None:
        return None
    # ── Normalleştir ──
    return {
        "object_present": bool(verdict.get("object_present")),
        "color_match": verdict.get("color_match"),  # true/false/None
        "confidence": float(verdict.get("confidence") or 0.0),
    }


def is_available() -> bool:
    """Ollama servisi ayakta ve model yüklü mü? (search tarafı VLM'i koşullu tetikler)"""
    try:
        base = settings.vlm_url.rsplit("/api/", 1)[0]
        tags = requests.get(f"{base}/api/tags", timeout=3).json()
        return any(settings.vlm_model.split(":")[0] in m.get("name", "")
                   for m in tags.get("models", []))
    except Exception:
        return False
