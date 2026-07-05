"""Arama hattı — sorgu ayrıştır, embedle, Qdrant'ta ara, sonuçları birleştir.

Embedder ve FrameStore tembel tekil (lazy singleton) olarak yüklenir:
model ~2 GB olduğundan yalnızca ilk aramada belleğe alınır.
"""

import threading
import time
from collections import defaultdict
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

# ── VLM erişilebilirliği: SELF-HEALING (kalıcı cache DEĞİL) ──
# Eski bug: "bir kez False → hep False" (Ollama çökünce/gelince UI kilitleniyordu).
# Çözüm: kısa TTL (yeniden yoklama) + hata-anında reset (çökme sonrası ilk sorgu iyileşir).
_vlm_available: bool | None = None
_vlm_checked_at: float = 0.0
_VLM_TTL_S: float = 30.0


def _vlm_ready(force: bool = False) -> bool:
    global _vlm_available, _vlm_checked_at
    now = time.time()
    if force or _vlm_available is None or (now - _vlm_checked_at) > _VLM_TTL_S:
        from gozcu.verifier import is_available
        _vlm_available = is_available()
        _vlm_checked_at = now
    return _vlm_available


def _mark_vlm_down() -> None:
    """Verify çağrıları toptan başarısız oldu (Ollama çökmüş olabilir) → cache'i geçersiz kıl:
    bir sonraki sorgu yeniden yoklar (çökme sonrası otomatik iyileşme)."""
    global _vlm_available, _vlm_checked_at
    _vlm_available = False
    _vlm_checked_at = 0.0


def vlm_available() -> bool:
    """Ollama + VLM hazır mı (viewer VLM bölümünü gösterip göstermeyeceğine karar verir)."""
    return _vlm_ready()


def get_embedder():
    """Sorgu-anı embedder — CPU'da (settings.query_device). GPU'yu VLM'e bırakır:
    sorgu yalnız metin embed'ler, görüntü vektörleri zaten Qdrant'ta. Co-residency OOM'u
    yapısal çözüm (ARCHITECTURE.md §8). İndeksleme ayrı süreç, GPU kullanır (cli.py)."""
    global _embedder
    if _embedder is None:
        from gozcu.embedder import Embedder
        _embedder = Embedder(device=settings.query_device)
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
    vlm_unavailable: bool = False  # VLM toptan erişilemedi (Ollama çökük) → global banner


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


def _make_event(video_id: str, hits: list[dict]) -> dict:
    """Bir olay: temsilci (en yüksek skor) + ilk/son görülme + kare sayısı."""
    rep = max(hits, key=lambda h: h["score"])
    return {"video_id": video_id, "rep": rep, "count": len(hits),
            "first_ts": min(h["ts"] for h in hits), "last_ts": max(h["ts"] for h in hits)}


def cluster_events(results: list[dict], gap_s: float | None = None) -> list[dict]:
    """Doğrulanmış sonuçları video + zaman-yakınlığıyla OLAYlara kümele (zaman grounding).

    KİMLİK TAKİBİ YOK — tracking olmadan "girdi/çıktı" iddia edilemez. Bunun yerine dürüst
    "görülme aralığı": aynı videoda ardışık hit arası boşluk > gap_s ise yeni olay. Her olay
    ilk/son görülme + kare sayısı taşır. Saf sunum katmanı (retrieval/verify'a dokunmaz).
    Detay: ARCHITECTURE.md §9 (AI Engineer #3).
    """
    gap_s = gap_s if gap_s is not None else settings.event_gap_s
    by_video: dict[str, list[dict]] = defaultdict(list)
    for h in results:
        by_video[h["video_id"]].append(h)
    events: list[dict] = []
    for video, hits in by_video.items():
        hits = sorted(hits, key=lambda h: h["ts"])
        cur = [hits[0]]
        for h in hits[1:]:
            if h["ts"] - cur[-1]["ts"] > gap_s:  # boşluk büyük → yeni olay
                events.append(_make_event(video, cur))
                cur = [h]
            else:
                cur.append(h)
        events.append(_make_event(video, cur))
    events.sort(key=lambda e: -e["rep"]["score"])  # en iyi eşleşen olay üstte
    return events


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


def _build_verified_outcome(outcome: SearchOutcome, text: str) -> SearchOutcome:
    """verify_top_n SONRASI: toptan-çökme tespiti + füzyon → rafine SearchOutcome.

    Toptan çökme: doğrulanan top-N'in HEPSİ None ve VLM artık erişilemez (Ollama çöktü) →
    `vlm_unavailable` (global banner), per-kart 'doğrulanamadı' gösterme, CLIP sırasını koru.
    Tekil None (recrop/timeout) ise füzyonda korunur (kart rozeti)."""
    verified = outcome.results[: settings.vlm_top_n]
    if verified and all(h.get("_vlm") is None for h in verified) and not _vlm_ready(force=True):
        _mark_vlm_down()
        for h in outcome.results:
            h.pop("_vlm", None)  # rozet gösterme; banner gösterilecek
        return SearchOutcome(results=outcome.results, parsed=outcome.parsed,
                             time_filter_dropped=outcome.time_filter_dropped,
                             vlm_applied=False, vlm_unavailable=True)
    results, filtered, nf = _fuse_verdicts(outcome.results, text)
    return SearchOutcome(results=results, parsed=outcome.parsed,
                         time_filter_dropped=outcome.time_filter_dropped, not_found_reason=nf,
                         vlm_applied=True, vlm_filtered=filtered)


def stream_verify(outcome: SearchOutcome, on_verdict=None) -> SearchOutcome:
    """Per-item streaming rafine (SENKRON — CLI/test). Viewer artık start_stream_job kullanır
    (worker thread, ana thread bloke olmaz). needs_vlm/VLM-hazır değilse aynen döner."""
    text = outcome.parsed.visual_text or ""
    if not outcome.results or not needs_vlm(text) or not _vlm_ready():
        return outcome
    verify_top_n(outcome.results, text, on_verdict)
    return _build_verified_outcome(outcome, text)


# ── Streaming DECOUPLE: worker thread doğrular, ana (Streamlit) thread bloke OLMAZ ──
# Sorun: senkron stream_verify ~40s ana thread'i bloke edip tarayıcıyı donduruyordu.
# Çözüm (AI Engineer): worker thread hit['_vlm']'i doldurur (st.* YOK → context sorunu yok),
# viewer st.fragment(run_every) ile job'ı poller. Verdict'ler module-level store'da (Lock'lu).
_stream_jobs: dict = {}
_stream_reg_lock = threading.Lock()


class StreamJob:
    """Arka plan VLM doğrulama işi. Worker hit['_vlm']'e yazar (GIL-atomik); fragment poller."""

    def __init__(self, outcome: SearchOutcome) -> None:
        self.outcome = outcome
        self.lock = threading.Lock()
        self.done = False
        self.result: SearchOutcome | None = None

    def progress(self) -> tuple[int, bool, SearchOutcome | None]:
        """(doğrulanan_sayısı, bitti_mi, füzyonlanmış_sonuç|None). Reader kilit altında kopya."""
        head = self.outcome.results[: settings.vlm_top_n]
        n = sum(1 for h in head if "_vlm" in h)
        with self.lock:
            return n, self.done, self.result


def start_stream_job(job_id: str, outcome: SearchOutcome) -> StreamJob | None:
    """VLM worker thread'ini bir kez başlat. needs_vlm/VLM-hazır değilse None (viewer CLIP-only)."""
    text = outcome.parsed.visual_text or ""
    if not outcome.results or not needs_vlm(text) or not _vlm_ready():
        return None
    with _stream_reg_lock:
        existing = _stream_jobs.get(job_id)
        if existing is not None:
            return existing
        job = StreamJob(outcome)
        _stream_jobs[job_id] = job

    def _worker() -> None:
        from gozcu.verifier import verify_hit
        obj_en, color_en = extract_vqa_targets(text)
        for h in outcome.results[: settings.vlm_top_n]:
            h["_vlm"] = verify_hit(h, obj_en, color_en)  # GIL-atomik atama
        fused = _build_verified_outcome(outcome, text)
        with job.lock:
            job.result = fused
            job.done = True

    threading.Thread(target=_worker, daemon=True).start()
    return job


def get_stream_job(job_id: str) -> StreamJob | None:
    with _stream_reg_lock:
        return _stream_jobs.get(job_id)


def clear_stream_job(job_id: str) -> None:
    with _stream_reg_lock:
        _stream_jobs.pop(job_id, None)


def refine_vlm(outcome: SearchOutcome) -> SearchOutcome:
    """CLIP sonucunu VLM ile rafine et — progressive/async ikinci faz (ARCHITECTURE.md §8).

    Viewer önce `search(use_vlm=False)` ile CLIP'i ANINDA gösterir, sonra bunu çağırıp
    VLM-doğrulanmış sonucu günceller. needs_vlm değil / VLM hazır değil / sonuç yok →
    outcome'u AYNEN döner (ucuz no-op, çağrı güvenli).
    """
    text = outcome.parsed.visual_text or ""
    if not outcome.results or not needs_vlm(text) or not _vlm_ready():
        return outcome
    verify_top_n(outcome.results, text)
    return _build_verified_outcome(outcome, text)


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
