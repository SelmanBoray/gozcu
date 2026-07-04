"""Arama hattı — sorgu ayrıştır, embedle, Qdrant'ta ara, sonuçları birleştir.

Embedder ve FrameStore tembel tekil (lazy singleton) olarak yüklenir:
model ~2 GB olduğundan yalnızca ilk aramada belleğe alınır.
"""

from dataclasses import dataclass, field

from gozcu.config import settings
from gozcu.query import (
    VEHICLE_CLASSES,
    ParsedQuery,
    extract_object_intent,
    has_color,
    needs_vlm,
    parse_query,
    scene_or_object_intent,
    translate_visual,
)

# ── Tembel tekiller ──
_embedder = None
_store = None
_vlm_available: bool | None = None  # Ollama erişilebilirliği (süreç başına bir kez)


def _vlm_ready() -> bool:
    global _vlm_available
    if _vlm_available is None:
        from gozcu.verifier import is_available
        _vlm_available = is_available()
    return _vlm_available


def vlm_available() -> bool:
    """Ollama + VLM hazır mı (viewer VLM bölümünü gösterip göstermeyeceğine karar verir)."""
    return _vlm_ready()


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
    not_found_reason: str | None = None  # bulunamadı kapısı / VLM tetiklendiyse gerekçe
    vlm_applied: bool = False    # VLM doğrulaması uygulandı mı (viewer rozeti için)
    vlm_filtered: list[dict] = field(default_factory=list)  # VLM'in elediği adaylar (expander)


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


def _apply_vlm(results: list[dict], visual_text: str) -> tuple[list[dict], str | None]:
    """Faz 2 — top-N adayı VLM ile doğrula. İki mod (renk var mı?):

    - **Negasyon** (renk yok, ör. köpek/yağmur): eşleşme-güveni `vlm_drop_below` altındaki
      adayı DÜŞÜR (aranan konsept görüntüde yok). Hepsi düşerse → bulunamadı.
    - **Öznitelik** (renk var, ör. siyah SUV): DÜŞÜRME (renk güvenilmez, AI Engineer) —
      yalnız sınırlı rerank `z(cos) + β·conf·[color_match]`.
    - VLM hatası (None): dokunma, CLIP sıralaması korunur (halüsinasyon/erişim koruması).
    """
    from gozcu.verifier import verify_hit

    en = translate_visual(visual_text)
    ask_color = has_color(visual_text)
    head, tail = results[: settings.vlm_top_n], results[settings.vlm_top_n:]

    survivors: list[dict] = []
    filtered: list[dict] = []
    for h in head:
        v = verify_hit(h, en, ask_color)
        h["_vlm"] = v
        # Negasyon modu: yüksek-güvenle "eşleşmiyor" → düşür (renk modunda düşürme)
        if v and not ask_color and v["confidence"] < settings.vlm_drop_below:
            filtered.append(h)
            continue
        survivors.append(h)

    # ── Negasyonda head tümü düştüyse: konsept korpusta yok → bulunamadı ──
    if not ask_color and head and not survivors:
        return [], filtered, "VLM: tanımlanan sahne/nesne görüntülerde doğrulanamadı."

    # ── Sınırlı rerank (z-normalize cosine + eşleşme bonusu) ──
    if survivors:
        scores = [h["score"] for h in survivors]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5 or 1e-9
        for h in survivors:
            v = h.get("_vlm")
            bonus = 0.0
            if v:
                if ask_color:  # öznitelik: renk eşleşmesini ödüllendir
                    signal = 1.0 if v["color_match"] else 0.0
                else:          # negasyon-sonrası: eşleşme güvenini ödüllendir
                    signal = 1.0
                bonus = settings.vlm_beta * v["confidence"] * signal
            h["_vrank"] = (h["score"] - mean) / std + bonus
        survivors.sort(key=lambda x: -x["_vrank"])
    return survivors + tail, filtered, None


def refine_vlm(outcome: SearchOutcome) -> SearchOutcome:
    """CLIP sonucunu VLM ile rafine et — progressive/async ikinci faz (ARCHITECTURE.md §8).

    Viewer önce `search(use_vlm=False)` ile CLIP'i ANINDA gösterir, sonra bunu çağırıp
    VLM-doğrulanmış sonucu günceller. needs_vlm değil / VLM hazır değil / sonuç yok →
    outcome'u AYNEN döner (ucuz no-op, çağrı güvenli).
    """
    text = outcome.parsed.visual_text or ""
    if not outcome.results or not needs_vlm(text) or not _vlm_ready():
        return outcome
    results, filtered, nf = _apply_vlm(outcome.results, text)
    return SearchOutcome(results=results, parsed=outcome.parsed,
                         time_filter_dropped=outcome.time_filter_dropped, not_found_reason=nf,
                         vlm_applied=True, vlm_filtered=filtered)


def search(raw_query: str, top_k: int | None = None, source: str | None = None,
           use_vlm: bool = True) -> SearchOutcome:
    """Türkçe sorgu → sıralı eşleşme listesi.

    source: "frame"|"crop"|None — kaynak filtresi (eval ablation'ı; None → prod hattı).
    use_vlm: Faz 2 VLM doğrulaması (yalnız prod + renk/zor-kavram sorgusu + Ollama ayakta).
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

    outcome = SearchOutcome(results=results, parsed=parsed,
                            time_filter_dropped=time_filter_dropped)

    # ── 7. Faz 2 VLM doğrulama (senkron; viewer progressive için refine_vlm'i ayrı çağırır) ──
    if source is None and use_vlm:
        outcome = refine_vlm(outcome)
    return outcome
