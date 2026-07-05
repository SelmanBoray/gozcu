"""Streamlit arayüzü — sorgu → CLIP ANINDA → VLM per-item CANLI doğrulama (streaming).

Çalıştırma: python -m gozcu ui   (veya: streamlit run gozcu/viewer.py)

Faz 2 async rafine (ARCHITECTURE.md §8b): CLIP sonucu ~1s'de gösterilir; renk/negasyon
sorgusunda VLM her adayı SIRAYLA doğrular ve o kartın rozetini CANLI günceller (⏳→✅/🚫).
Bitince final reflow (yeniden sıralama + elenenler expander). Cache: session_state
(streaming canlı render gerektirdiği için @cache_data kullanılamaz; Oynat tıklaması
VLM'i yeniden koşmasın diye sonuç sorgu-anahtarlı saklanır).
"""

import sys
from datetime import datetime
from pathlib import Path

# ── `streamlit run` script klasöründen başlatır; paket importu için kökü ekle ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from gozcu.config import settings
from gozcu.query import needs_vlm
from gozcu.search import search as run_search
from gozcu.search import stream_verify, vlm_available

st.set_page_config(page_title="Gözcü", page_icon="👁️", layout="wide")


@st.cache_resource(show_spinner=False)
def _warmup_vlm() -> bool:
    """Açılışta VLM'i ARKA PLANDA ısıt (thread → UI'ı bloke etmez). Kullanıcı sorguyu
    yazarken model yüklenir; ilk gerçek sorgu cold olmaz. cache_resource: süreçte bir kez."""
    import threading
    from gozcu.verifier import warmup
    threading.Thread(target=warmup, daemon=True).start()
    return True


@st.cache_data(show_spinner=False)
def clip_search(query: str, top_k: int):
    """CLIP-only arama (hızlı) — anında gösterim + rerun'da yeniden koşmasın."""
    return run_search(query, top_k=top_k, use_vlm=False)


@st.cache_data(show_spinner=False)
def _card_image(thumb_path: str, bbox: tuple | None, dim: bool = False):
    """Kart görseli: kırpık aday ise TAM-KARE thumb üzerine eşleşen özneyi kutuyla işaretle
    (kullanıcı 'hangi araç/kişi' eşleşti anında görsün). dim=True → soluklaştır (streaming'de
    reddedilen aday 'eleniyor' görünür, 'sonuç' değil)."""
    from PIL import Image

    from gozcu.recrop import draw_bbox
    img = Image.open(thumb_path).convert("RGB")
    if bbox:
        img = draw_bbox(img, list(bbox))
    if dim:
        img = Image.blend(img, Image.new("RGB", img.size, (55, 55, 55)), 0.6)
    return img


# ── Aday VLM'ce reddedildi mi (nesne yok / renk uymadı) — streaming'de soluklaştırma için ──
def _is_rejected(hit: dict) -> bool:
    v = hit.get("_vlm")
    if not v:
        return False
    return v.get("color_match") is False or not v.get("object_present")


# ── Rozet: VLM verdict'inden (streaming ⏳ / doğrulandı ✅ / elendi 🚫) ──
def _verdict_badge(hit: dict) -> str:
    if "_vlm" not in hit:
        return ""                          # henüz doğrulanmadı (streaming ⏳ ayrı ele alınır)
    v = hit["_vlm"]
    if v is None:
        return " · ⚠️ doğrulanamadı"       # VLM hata/timeout — kart asılı kalmasın
    if v.get("color_match") is True:
        return " · ✅ renk doğru"
    if v.get("color_match") is False:      # renk soruldu ama uymadı — dürüst göster
        return " · 🚫 renk uymadı"
    if v.get("object_present") and v["confidence"] >= settings.vlm_drop_below:
        return f" · ✅ VLM ({v['confidence']:.0%})"
    return " · 🚫 eşleşmedi"


def render_grid(results: list[dict], done: int | None = None, buttons: bool = True) -> None:
    """Sonuç kartları. done=None → final (rozetler _vlm'den, Oynat butonlu).
    done=int → streaming: ilk `done` top-N kartı verdictli, kalan top-N ⏳, kuyruk düz."""
    n = settings.vlm_top_n
    cols = st.columns(4)
    for i, hit in enumerate(results):
        with cols[i % 4]:
            when = datetime.fromtimestamp(hit["ts"]).strftime("%d.%m %H:%M:%S")
            bbox = tuple(hit["bbox"]) if hit.get("source") == "crop" and hit.get("bbox") else None
            # ── streaming'de doğrulanıp reddedilen kartı soluklaştır ("eleniyor") ──
            streaming_done = done is not None and i < n and i < done
            dim = streaming_done and _is_rejected(hit)
            st.image(_card_image(hit["thumb_path"], bbox, dim), use_container_width=True)
            if done is not None and i < n:
                badge = " · ⏳" if i >= done else _verdict_badge(hit)
            elif done is not None:
                badge = ""  # kuyruk (top-N dışı) — doğrulanmadı
            else:
                badge = _verdict_badge(hit)
            if hit.get("source") == "crop":
                st.caption(f"**{hit['score']:.3f}** · 🎯 {hit['yolo_class']} "
                           f"({hit['yolo_conf']:.2f}) · {hit['video_id']} · {when}{badge}")
                st.image(hit["crop_thumb"], width=90)
            else:
                st.caption(f"**{hit['score']:.3f}** · {hit['video_id']} · {when}{badge}")
            if buttons and st.button("▶ Oynat", key=f"play_{i}_{hit['video_id']}_{hit['frame_idx']}"):
                st.session_state["selected"] = hit
                st.rerun()


def parse_info(outcome, note: str) -> None:
    parsed = outcome.parsed
    info = f'Görsel sorgu: **"{parsed.visual_text or "—"}"**'
    if parsed.ts_from is not None:
        fmt = "%d.%m.%Y %H:%M"
        info += (f' · Zaman: **"{parsed.time_phrase}"** → '
                 f"{datetime.fromtimestamp(parsed.ts_from):{fmt}}–"
                 f"{datetime.fromtimestamp(parsed.ts_to):{fmt}}")
    if note:
        info += f" · {note}"
    st.markdown(info)
    if outcome.time_filter_dropped:
        st.warning("Zaman aralığında sonuç yok — tüm arşivde arandı.")


def render_outcome(outcome, note: str) -> None:
    """Final görünüm: parse özeti + (bulunamadı / ızgara) + elenenler expander."""
    parse_info(outcome, note)
    if outcome.not_found_reason:
        st.info(f"🔍 **Bulunamadı** — {outcome.not_found_reason}")
    elif not outcome.results:
        st.error("Sonuç yok. Önce `python -m gozcu index <klasör>` ile indeksleyin.")
    else:
        sel = st.session_state.get("selected")
        if sel is not None:
            when = datetime.fromtimestamp(sel["ts"]).strftime("%d.%m.%Y %H:%M:%S")
            st.subheader(f"▶ {sel['video_id']} — {when}")
            st.video(sel["video_path"], start_time=int(sel["offset_s"]))
        render_grid(outcome.results, done=None, buttons=True)
    if outcome.vlm_filtered:
        with st.expander(f"🚫 VLM elenenler ({len(outcome.vlm_filtered)}) — CLIP getirdi, "
                         f"VLM eşleştirmedi"):
            for h in outcome.vlm_filtered:
                st.image(h["thumb_path"], width=160)
                st.caption(f"{h['video_id']} · CLIP skor {h['score']:.3f} · VLM: eşleşme yok")


def render_final(outcome) -> None:
    """Final render: VLM çöktüyse (vlm_unavailable) banner + CLIP; değilse VLM-doğrulandı notu."""
    if outcome.vlm_unavailable:
        st.warning("⚠️ VLM doğrulama sırasında Ollama erişilemez oldu — yalnız CLIP sıralaması "
                   "gösteriliyor.")
        render_outcome(outcome, "")
    else:
        render_outcome(outcome, "🔍 VLM ile doğrulandı")


# ── Sayfa ──
st.title("👁️ Gözcü")
st.caption("Kamera arşivinde Türkçe doğal dil arama — her şey lokalde, hiçbir veri dışarı çıkmaz.")

_warmup_vlm()  # açılışta VLM'i arka planda ısıt (bir kez, bloke etmez)

with st.form("arama"):
    col_q, col_k = st.columns([5, 1])
    query = col_q.text_input(
        "Sorgu", placeholder='örn: "siyah SUV" · "köpek gezdiren adam" · "dün gece giren araç"',
        label_visibility="collapsed")
    top_k = col_k.slider("Sonuç", 4, 24, settings.default_top_k, 4)
    submitted = st.form_submit_button("Ara", use_container_width=True)

if submitted and query:
    st.session_state["query"] = query
    st.session_state["top_k"] = top_k
    st.session_state.pop("selected", None)

q = st.session_state.get("query")
if q:
    k = st.session_state.get("top_k", settings.default_top_k)
    clip_outcome = clip_search(q, k)
    wants_vlm = bool(clip_outcome.results) and needs_vlm(clip_outcome.parsed.visual_text or "")
    vlm_up = vlm_available()
    will_verify = wants_vlm and vlm_up
    vkey = f"vlm::{q}::{k}"
    slot = st.empty()

    # ── VLM istendi ama erişilemez (Ollama kapalı/çökük) → global banner + yalnız CLIP ──
    if wants_vlm and not vlm_up:
        st.warning("⚠️ VLM doğrulayıcı kapalı (Ollama erişilemedi) — yalnız CLIP sıralaması "
                   "gösteriliyor. Ollama'yı başlatınca renk/negasyon doğrulaması devreye girer.")

    if not will_verify:
        with slot.container():
            render_outcome(clip_outcome, "")
    elif vkey in st.session_state:  # VLM daha önce koştu → önbellekten anında
        with slot.container():
            render_final(st.session_state[vkey])
    else:  # ── PER-ITEM STREAMING (ilk kez) ──
        topn = clip_outcome.results[: settings.vlm_top_n]
        prog = st.progress(0.0, text=f"VLM 0/{len(topn)} doğrulanıyor…")

        def paint(done: int) -> None:
            with slot.container():
                parse_info(clip_outcome, "⚡ hızlı CLIP · VLM canlı doğruluyor…")
                render_grid(clip_outcome.results, done=done, buttons=False)

        paint(0)  # hepsi ⏳

        def on_verdict(i: int, hit: dict) -> None:
            paint(i + 1)  # i. kartın rozeti dolar
            prog.progress((i + 1) / len(topn), text=f"VLM {i + 1}/{len(topn)} doğrulandı")

        refined = stream_verify(clip_outcome, on_verdict)
        st.session_state[vkey] = refined
        prog.empty()
        with slot.container():  # final reflow: yeniden sıralama + elenenler + Oynat
            render_final(refined)
