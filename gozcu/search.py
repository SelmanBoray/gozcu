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
    extract_vqa_targets,
    has_color,
    needs_vlm,
    parse_query,
    scene_or_object_intent,
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


def verify_top_n(results: list[dict], visual_text: str, on_verdict=None) -> None:
    """Top-N adayı VLM ile doğrula, `h['_vlm']` doldur. Her verdict sonrası (streaming için)
    `on_verdict(i, hit)` çağır. Füzyon YOK — yalnız verdict toplama.

    YES/NO VQA sözleşmesi: sorgu (nesne, renk) hedeflerine indirgenir; verifier atomik
    sorar (rubber-stamp/thinking-loop panzehiri — ARCHITECTURE.md §8)."""
    from gozcu.verifier import verify_hit

    obj_en, color_en = extract_vqa_targets(visual_text)
    for i, h in enumerate(results[: settings.vlm_top_n]):
        h["_vlm"] = verify_hit(h, obj_en, color_en)
        if on_verdict is not None:
            on_verdict(i, h)


def _fuse_verdicts(results: list[dict], visual_text: str) -> tuple[list[dict], list[dict], str | None]:
    """Toplanmış `h['_vlm']` verdict'lerinden füzyon. Eşleşmeyen aday ANA IZGARADAN çıkar
    (elenenler expander'ında görünür); hiç gerçek eşleşme yoksa → bulunamadı.

    - **Nesne yokluğu → DÜŞÜR (her iki modda):** yes/no VQA ile nesne-varlığı GÜVENİLİR.
      `confidence < vlm_drop_below` (nesne yok) → düşür.
    - **Renk uymadı → DÜŞÜR (renk sorgusunda):** qwen2.5vl renk ayrımı ölçülmüş güvenilir
      (red/blue 9/9↔0/9) → `color_match is False` de eleme. Böylece "kırmızı kıyafetli
      adam" korpusta kırmızı giyen yoksa kırmızı ARABA göstermez, bulunamadı der. (Eski
      "renk güvenilmez → rerank-only" tasarımı qwen3-vl JSON rubber-stamp içindi, artık geçersiz.)
    - VLM hatası (None): dokunma (CLIP sıralaması korunur).
    """
    ask_color = has_color(visual_text)
    # Kuyruk (vlm_top_n dışı) DOĞRULANMAZ → VLM modunda gösterilmez (sızıntı önle):
    # "kırmızı kıyafetli adam"da rank 9-12 kırmızı arabalar doğrulanmadan sızıyordu.
    # vlm_top_n=default_top_k olduğundan varsayılan sorguda kuyruk zaten boş.
    head = results[: settings.vlm_top_n]

    survivors: list[dict] = []
    filtered: list[dict] = []
    for h in head:
        v = h.get("_vlm")
        if v:
            absent = v["confidence"] < settings.vlm_drop_below       # nesne yok
            wrong_color = ask_color and v.get("color_match") is False  # renk uymadı
            if absent or wrong_color:
                filtered.append(h)
                continue
        survivors.append(h)

    if head and not survivors:
        reason = ("VLM: aranan renk/nesne bileşimi görüntülerde doğrulanamadı."
                  if ask_color else "VLM: tanımlanan nesne görüntülerde doğrulanamadı.")
        return [], filtered, reason

    if survivors:
        scores = [h["score"] for h in survivors]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5 or 1e-9
        for h in survivors:
            v = h.get("_vlm")
            bonus = 0.0
            if v:
                signal = (1.0 if v["color_match"] else 0.0) if ask_color else 1.0
                bonus = settings.vlm_beta * v["confidence"] * signal
            h["_vrank"] = (h["score"] - mean) / std + bonus
        survivors.sort(key=lambda x: -x["_vrank"])
    return survivors, filtered, None  # yalnız doğrulanmış survivor'lar (kuyruk sızıntısı yok)


def _apply_vlm(results: list[dict], visual_text: str) -> tuple[list[dict], list[dict], str | None]:
    """Batch: doğrula + füzyon (CLI/senkron). Streaming için stream_verify kullanılır."""
    verify_top_n(results, visual_text)
    return _fuse_verdicts(results, visual_text)


def stream_verify(outcome: SearchOutcome, on_verdict=None) -> SearchOutcome:
    """Per-item streaming rafine: her verdict sonrası `on_verdict(i, hit)` (viewer canlı rozet),
    sonra füzyon → rafine SearchOutcome. needs_vlm/VLM-hazır değilse aynen döner."""
    text = outcome.parsed.visual_text or ""
    if not outcome.results or not needs_vlm(text) or not _vlm_ready():
        return outcome
    verify_top_n(outcome.results, text, on_verdict)
    results, filtered, nf = _fuse_verdicts(outcome.results, text)
    return SearchOutcome(results=results, parsed=outcome.parsed,
                         time_filter_dropped=outcome.time_filter_dropped, not_found_reason=nf,
                         vlm_applied=True, vlm_filtered=filtered)


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
