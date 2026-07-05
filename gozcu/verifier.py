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
from gozcu.recrop import vlm_frame_for_hit, vlm_image_for_hit

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


def _article(obj_en: str) -> str:
    return "an" if obj_en[:1].lower() in "aeiou" else "a"


def _crop_presence_q(obj_en: str) -> str:
    """Renk yolu (tight-kırpık, özne kadrajı doldurur): aday bu kırpıkta aranan nesne mi."""
    if obj_en in _UNCOUNTABLE:
        return f"Is there {obj_en} visible in this image?"
    return f"Is there {_article(obj_en)} {obj_en} in this image?"


def _scene_presence_q(obj_en: str) -> str:
    """Zor-kavram yolu (kutusuz TAM-KARE): nesne sahnenin HERHANGİ yerinde mi (köpek
    insan-bbox dışında olabilir → tight-kırpık göremez). Sayılamayanlarda artikel düşer."""
    if obj_en in _UNCOUNTABLE:
        return f"Is there {obj_en} visible anywhere in this image?"
    return f"Is there {_article(obj_en)} {obj_en} visible anywhere in this image?"


def _color_q(obj_en: str, color_en: str, is_person: bool) -> str:
    """Kırpıktaki öznenin rengi. İnsanda renk giysiye işaret eder (gövde değil)."""
    if is_person:
        return f"Is this {obj_en} wearing {color_en} clothing?"
    return f"Is the {obj_en} {color_en} in color?"


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

    HİBRİT görüntü (ölçümle en iyi — experiments/2026-07-05_vlm_latency):
    - **Renk/öznitelik → tight-kırpık** (özne kadrajı doldurur, renk net, hızlı; tam-kare
      küçültmesi uzak aracı görünmez kılıp gerçek eşleşmeyi eliyordu).
    - **Zor-kavram (köpek/yağmur) → kutusuz TAM-KARE** (nesne kırpık dışında olabilir; tam
      sahne yanlış-pozitifi çözer: "köpek gezdiren adam"→bulunamadı, önceki 1 yanlış-pozitif gitti).

    confidence: 1.0 (nesne var) / 0.0 (yok). Recrop/Ollama hatası → None (çağıran atlar).
    """
    obj = obj_en or "object"
    try:
        if color_en:
            img_b64 = _encode(vlm_image_for_hit(hit, out_size=384))       # tight-kırpık
        else:
            img_b64 = _encode(vlm_frame_for_hit(hit, draw_box=False))     # kutusuz tam-kare
    except Exception:
        return None

    if color_en:
        present = _yesno(img_b64, _crop_presence_q(obj))
        if present is None:
            return None
        color_match = _yesno(img_b64, _color_q(obj, color_en, obj in _PERSON_OBJECTS)) \
            if present else None
    else:
        present = _yesno(img_b64, _scene_presence_q(obj))  # zor-kavram: tüm sahne
        if present is None:
            return None
        color_match = None

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


def warmup() -> bool:
    """Modeli önceden GPU'ya yükle (dummy inference) → gerçek ilk sorgu asla cold olmaz.
    Uygulama açılışında bir kez çağrılır (viewer). Ollama yoksa sessizce False."""
    try:
        payload = {
            "model": settings.vlm_model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False, "keep_alive": settings.vlm_keep_alive,
            "options": {"num_predict": 1},
        }
        r = requests.post(settings.vlm_url, json=payload, timeout=90)
        return r.status_code == 200
    except Exception:
        return False
