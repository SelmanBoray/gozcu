"""Arama hattı — sorgu ayrıştır, embedle, Qdrant'ta ara, sonuçları birleştir.

Embedder ve FrameStore tembel tekil (lazy singleton) olarak yüklenir:
model ~2 GB olduğundan yalnızca ilk aramada belleğe alınır.
"""

from dataclasses import dataclass

from gozcu.config import settings
from gozcu.query import (
    VEHICLE_CLASSES,
    ParsedQuery,
    extract_object_intent,
    parse_query,
    scene_or_object_intent,
)

# ── Tembel tekiller ──
_embedder = None
_store = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from gozcu.embedder import Embedder
        _embedder = Embedder(device=settings.device)
    return _embedder


def get_store():
    global _store
    if _store is None:
        from gozcu.store import FrameStore
        _store = FrameStore()
    return _store


@dataclass
class SearchOutcome:
    """Arama sonucu + kullanıcıya gösterilecek bağlam bilgisi."""

    results: list[dict]          # store.search çıktısı (payload + score)
    parsed: ParsedQuery          # hangi görsel metin / zaman aralığı kullanıldı
    time_filter_dropped: bool    # zaman filtresi sıfır sonuç verdi, filtresiz tekrarlandı
    not_found_reason: str | None = None  # bulunamadı kapısı tetiklendiyse gerekçe


def _intent_rerank(hits: list[dict], intent: str, lam: float) -> None:
    """Sıralama skorunu (`_rank`) yerinde yaz. Ham cosine (`score`) gösterim için korunur.

    Sahne-niyetinde: z-normalize cosine + kareye λ boost (kırpık selini dengeler, Olgu B).
    Diğer niyetlerde (object/neutral): dokunma — kırpık selini nesne sorgusu için LEHE bırak.
    z-normalizasyon: cosine aralığı sorgudan sorguya değişir; z-skor λ'yı sorgu-bağımsız yapar.
    """
    if not hits:
        return
    if intent != "scene" or lam <= 0:
        for h in hits:
            h["_rank"] = h["score"]
        return
    scores = [h["score"] for h in hits]
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    if std < 1e-6:  # tüm skorlar eşit → boost anlamsız
        for h in hits:
            h["_rank"] = h["score"]
        return
    for h in hits:
        z = (h["score"] - mean) / std
        h["_rank"] = z + (lam if h.get("source") == "frame" else 0.0)


def _dedup_and_group(hits: list[dict], top_k: int) -> list[dict]:
    """Kırpık + kare aynı kareyi gösterebilir; aynı yürüyüş art arda kareler doldurabilir.

    Sıralama `_rank` üzerinden (niyet-boost sonrası); yoksa ham `score`.
    1. (video, kare) başına en yüksek `_rank`'li tek sonuç.
    2. Aynı videoda `group_window_s` penceresi içinde tek sonuç (en iyisi).
    """
    def rank(h: dict) -> float:
        return h.get("_rank", h["score"])

    best: dict = {}
    for h in hits:
        key = (h["video_id"], h["frame_idx"])
        if key not in best or rank(h) > rank(best[key]):
            best[key] = h

    kept: list[dict] = []
    for h in sorted(best.values(), key=lambda x: -rank(x)):
        if any(
            k["video_id"] == h["video_id"]
            and abs(k["offset_s"] - h["offset_s"]) < settings.group_window_s
            for k in kept
        ):
            continue
        kept.append(h)
        if len(kept) >= top_k:
            break
    return kept


def search(raw_query: str, top_k: int | None = None, source: str | None = None) -> SearchOutcome:
    """Türkçe sorgu → sıralı eşleşme listesi.

    source: "frame"|"crop"|None — kaynak filtresi (eval ablation'ı; None → prod hattı).
    """
    top_k = top_k or settings.default_top_k
    fetch_k = top_k * settings.search_overfetch  # tekilleştirme payı

    # ── 1. Ayrıştır: görsel metin + zaman filtresi ──
    parsed = parse_query(raw_query)
    text = parsed.visual_text or raw_query

    # ── 1b. Bulunamadı kapısı (yalnız prod hattında; ablation ham retrieval'ı ölçer) ──
    # Sorgu, korpusta HİÇ tespit edilmemiş bir YOLO sınıfı istiyorsa CLIP skoruna
    # bakmadan boş dön — "kırmızı bisiklet" gibi yok olan nesneye makul-ama-yanlış
    # sonuç sunmayı önler (eval'de negatif örtüşme bulgusunun hedefli hafifletmesi).
    if source is None:
        intent = extract_object_intent(text)
        available = get_store().available_object_classes()
        missing = intent.required - available
        if missing:
            reason = (f"Korpusta '{', '.join(sorted(missing))}' tespit edilmedi — "
                      f"bu nesne kayıtlarda yok.")
            return SearchOutcome(results=[], parsed=parsed,
                                 time_filter_dropped=False, not_found_reason=reason)
        if intent.generic_vehicle and not (available & VEHICLE_CLASSES):
            return SearchOutcome(results=[], parsed=parsed, time_filter_dropped=False,
                                 not_found_reason="Korpusta hiç taşıt tespit edilmedi.")

    # ── 2. Görsel metni embedle (boş kaldıysa ham sorguya geri düş) ──
    vector = get_embedder().encode_text(text)

    # ── 3. Filtreli vektör araması ──
    results = get_store().search(
        vector, top_k=fetch_k,
        ts_from=parsed.ts_from, ts_to=parsed.ts_to,
        camera_id=parsed.camera_id, source=source,
    )

    # ── 4. Zaman filtresi hiçbir şey bulamadıysa filtresiz tekrar dene ──
    time_filter_dropped = False
    if not results and parsed.ts_from is not None:
        results = get_store().search(
            vector, top_k=fetch_k, camera_id=parsed.camera_id, source=source
        )
        time_filter_dropped = True

    # ── 5. Niyet-koşullu yeniden sıralama (sahne-niyeti → kareye z-normalize boost) ──
    # Yalnız birleşik hatta (source=None) anlamlı; tek-kaynak ablation'da boost no-op.
    intent = scene_or_object_intent(text)
    _intent_rerank(results, intent, settings.scene_boost_lambda)

    # ── 6. Kare tekilleştirme + zaman kümeleme (_rank üzerinden sıralar) ──
    results = _dedup_and_group(results, top_k)

    return SearchOutcome(results=results, parsed=parsed, time_filter_dropped=time_filter_dropped)
