"""Faz 2 — VLM doğrulayıcı (qwen2.5vl:3b, Ollama). retrieve-then-verify, YES/NO VQA.

CLIP top-N adayı, tam-çözünürlüklü görüntüyle (recrop) VLM'e sorulur. Sözleşme:
öznitelik-başına ATOMİK yes/no soru — nesne var mı? + (renk sorgusunda) rengi doğru mu?

Neden yes/no, JSON değil: küçük VLM'ler çok-öznitelikli reddetme-yanlı JSON promptunda
her şeyi onaylıyor (rubber-stamp — kırmızı arabaya "mavi araba? evet" diyor) ve önceki
qwen3-vl:2b thinking-modu belirsiz kırpıklarda sonsuz düşünüp boş JSON dönüyordu. Aynı
model atomik yes/no sorulunca rengi KUSURSUZ ayırıyor (kırmızı 9/9, mavi 0/9). Verify
katmanının zaten yalnız boolean'a ihtiyacı var. Teşhis: experiments/2026-07-05_vlm_latency/

Türkçe→İngilizce çeviri query.extract_vqa_targets'te (VLM'in Türkçesine güvenme).
Detay: ARCHITECTURE.md §8
"""

import base64
import io

import requests

from gozcu.config import settings
from gozcu.recrop import vlm_image_for_hit

# ── Sayılamayan kavramlar: "a rain" değil "rain" (dilbilgisi VLM'i şaşırtmasın) ──
_UNCOUNTABLE = {"rain", "snow", "traffic", "fog", "smoke"}

# ── İnsan-tipi nesneler: renk GÖVDEYE değil GİYSİYE işaret eder ──
# "kırmızı kıyafetli adam" = kırmızı giyen adam; "Is the man red?" YANLIŞ (kimse kırmızı
# değil) → "Is the man wearing red?" DOĞRU. Araçta renk gövdedir (araba kırmızıdır).
_PERSON_OBJECTS = {"person", "man", "woman", "child", "pedestrian", "driver", "baby"}


def _encode(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _presence_q(obj_en: str) -> str:
    """'Is there a car in this image?' — sayılamayanlarda artikel düşürülür."""
    if obj_en in _UNCOUNTABLE:
        return f"Is there {obj_en} visible in this image?"
    article = "an" if obj_en[:1].lower() in "aeiou" else "a"
    return f"Is there {article} {obj_en} in this image?"


def _yesno(img_b64: str, question: str) -> bool | None:
    """Tek atomik yes/no VLM sorusu → True/False, hata/timeout → None.

    num_predict:4 — tek kelime yeter, kaçak üretimi imkânsız kılar (thinking modelinin
    sonsuz-düşünme çöküşünün yapısal panzehiri).
    """
    payload = {
        "model": settings.vlm_model,
        "messages": [{"role": "user",
                      "content": question + " Answer only 'yes' or 'no'.",
                      "images": [img_b64]}],
        "stream": False,
        "keep_alive": settings.vlm_keep_alive,
        "options": {"temperature": 0, "num_predict": 4},
    }
    # ── Tek retry: CLIP+VLM eşzamanlı GPU baskısında ara sıra timeout/500 olur ──
    for _attempt in range(2):
        try:
            resp = requests.post(settings.vlm_url, json=payload, timeout=settings.vlm_timeout_s)
            resp.raise_for_status()
            ans = resp.json()["message"]["content"].strip().lower()
            return ans.startswith("y")
        except Exception:
            continue
    return None


def verify_hit(hit: dict, obj_en: str | None, color_en: str | None) -> dict | None:
    """Bir adayı yes/no VQA ile doğrula. Döndürür: {object_present, color_match, confidence}.

    object_present: nesne görünür mü. color_match: rengi doğru mu (renk yoksa None).
    confidence: 1.0 (nesne var) / 0.0 (yok) — ayrım booleanlarda, füzyon bunu kullanır.
    Recrop/Ollama hatası → None (çağıran VLM'i atlar, CLIP sıralaması korunur).
    """
    try:
        img_b64 = _encode(vlm_image_for_hit(hit, out_size=384))
    except Exception:
        return None
    obj = obj_en or "object"
    present = _yesno(img_b64, _presence_q(obj))
    if present is None:
        return None  # VLM erişilemedi → dokunma
    # ── Renk yalnız nesne varsa ve renk sorulmuşsa; nesne yoksa renk anlamsız ──
    # İnsanlarda renk giysiye işaret eder (kırmızı kıyafetli adam ≠ kırmızı adam).
    color_match = None
    if color_en and present:
        if obj in _PERSON_OBJECTS:
            q = f"Is this {obj} wearing {color_en} clothing?"
        else:
            q = f"Is the {obj} {color_en} in color?"
        color_match = _yesno(img_b64, q)
    return {
        "object_present": present,
        "color_match": color_match,
        "confidence": 1.0 if present else 0.0,
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
