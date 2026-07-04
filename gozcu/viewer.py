"""Streamlit arayüzü — sorgu → CLIP sonucu ANINDA → VLM ile arkadan doğrula (progressive).

Çalıştırma: python -m gozcu ui   (veya: streamlit run gozcu/viewer.py)

Faz 2 async rafine (ARCHITECTURE.md §8): CLIP sonucu ~1s'de gösterilir; renk/negasyon
sorgusunda VLM doğrulaması arkadan koşup sonucu YERİNDE günceller (elenenler expander'a).
Cache ZORUNLU: her rerun (widget dokunuşu) VLM'i yeniden koşmasın.
"""

import sys
from datetime import datetime
from pathlib import Path

# ── `streamlit run` script klasöründen başlatır; paket importu için kökü ekle ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from gozcu.config import settings
from gozcu.query import needs_vlm
from gozcu.search import refine_vlm
from gozcu.search import search as run_search
from gozcu.search import vlm_available

st.set_page_config(page_title="Gözcü", page_icon="👁️", layout="wide")


# ── Önbellekli arama: rerun'lar (Oynat tıklaması vb.) VLM'i yeniden koşmasın ──
@st.cache_data(show_spinner=False)
def clip_search(query: str, top_k: int):
    """CLIP-only arama (hızlı, VLM yok) — anında gösterim için."""
    return run_search(query, top_k=top_k, use_vlm=False)


@st.cache_data(show_spinner=False)
def vlm_refine(query: str, top_k: int):
    """CLIP + VLM doğrulama (yavaş, sorgu başına bir kez; sonra önbellekten)."""
    return refine_vlm(run_search(query, top_k=top_k, use_vlm=False))


# ── Tek sonucu kart olarak çiz (VLM rozetleriyle) ──
def render_grid(results: list[dict], reranked: bool) -> None:
    cols = st.columns(4)
    for i, hit in enumerate(results):
        with cols[i % 4]:
            when = datetime.fromtimestamp(hit["ts"]).strftime("%d.%m %H:%M:%S")
            st.image(hit["thumb_path"], use_container_width=True)
            v = hit.get("_vlm")
            badge = ""
            if v:  # VLM doğruladı
                if v.get("color_match") is True:
                    badge = " · ✅ renk doğru"
                elif v.get("object_present"):
                    badge = f" · ✅ VLM ({v['confidence']:.0%})"
            if hit.get("source") == "crop":
                st.caption(f"**{hit['score']:.3f}** · 🎯 {hit['yolo_class']} "
                           f"({hit['yolo_conf']:.2f}) · {hit['video_id']} · {when}{badge}")
                st.image(hit["crop_thumb"], width=90)
            else:
                st.caption(f"**{hit['score']:.3f}** · {hit['video_id']} · {when}{badge}")
            if st.button("▶ Oynat", key=f"play_{i}_{hit['video_id']}_{hit['frame_idx']}"):
                st.session_state["selected"] = hit
                st.rerun()


def render_outcome(outcome, title_note: str) -> None:
    """Parse özeti + sonuç ızgarası + (VLM varsa) elenenler expander'ı."""
    parsed = outcome.parsed
    info = f'Görsel sorgu: **"{parsed.visual_text or "—"}"**'
    if parsed.ts_from is not None:
        fmt = "%d.%m.%Y %H:%M"
        info += (f' · Zaman: **"{parsed.time_phrase}"** → '
                 f"{datetime.fromtimestamp(parsed.ts_from):{fmt}}–"
                 f"{datetime.fromtimestamp(parsed.ts_to):{fmt}}")
    if title_note:
        info += f" · {title_note}"
    st.markdown(info)
    if outcome.time_filter_dropped:
        st.warning("Zaman aralığında sonuç yok — tüm arşivde arandı.")

    if outcome.not_found_reason:
        st.info(f"🔍 **Bulunamadı** — {outcome.not_found_reason}")
    elif not outcome.results:
        st.error("Sonuç yok. Önce `python -m gozcu index <klasör>` ile indeksleyin.")
    else:
        # ── Seçili sonucu videoda o andan oynat ──
        sel = st.session_state.get("selected")
        if sel is not None:
            when = datetime.fromtimestamp(sel["ts"]).strftime("%d.%m.%Y %H:%M:%S")
            st.subheader(f"▶ {sel['video_id']} — {when}")
            st.video(sel["video_path"], start_time=int(sel["offset_s"]))
        render_grid(outcome.results, outcome.vlm_applied)

    # ── VLM'in elediği CLIP adayları (şeffaflık — sessizce kaybolmasın) ──
    if outcome.vlm_filtered:
        with st.expander(f"🚫 VLM elenenler ({len(outcome.vlm_filtered)}) — CLIP getirdi, "
                         f"VLM eşleştirmedi"):
            for h in outcome.vlm_filtered:
                st.image(h["thumb_path"], width=160)
                st.caption(f"{h['video_id']} · CLIP skor {h['score']:.3f} · VLM: eşleşme yok")


# ── Sayfa ──
st.title("👁️ Gözcü")
st.caption("Kamera arşivinde Türkçe doğal dil arama — her şey lokalde, hiçbir veri dışarı çıkmaz.")

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
    st.session_state.pop("selected", None)  # yeni sorgu → eski seçimi temizle

q = st.session_state.get("query")
if q:
    k = st.session_state.get("top_k", settings.default_top_k)
    clip_outcome = clip_search(q, k)

    # ── VLM tetiklenecek mi? (renk/zor-kavram + Ollama ayakta) ──
    will_verify = bool(clip_outcome.results) and needs_vlm(
        clip_outcome.parsed.visual_text or "") and vlm_available()

    slot = st.empty()  # yerinde değiştirme için placeholder (container() tekrar çağrısı içeriği değiştirir)
    if will_verify:
        # 1) CLIP sonucunu ANINDA göster (slot'a), 2) VLM'i arkadan koş, 3) YERİNDE değiştir
        with slot.container():
            render_outcome(clip_outcome, "⚡ hızlı CLIP sonucu · VLM doğruluyor…")
        est = int(min(k, settings.vlm_top_n) * 5)
        with st.status(f"🔍 VLM ile doğrulanıyor (~{est} sn)…", expanded=False) as status:
            refined = vlm_refine(q, k)
            status.update(label="✅ VLM doğrulaması tamam", state="complete")
        with slot.container():  # aynı yeri VLM sonucuyla değiştir (iki bölüm değil)
            render_outcome(refined, "🔍 VLM ile doğrulandı")
    else:
        with slot.container():
            render_outcome(clip_outcome, "")
