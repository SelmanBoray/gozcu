"""Streamlit arayüzü — sorgu kutusu → küçük resim ızgarası → tıkla, videoyu o andan izle.

Çalıştırma: python -m gozcu ui   (veya: streamlit run gozcu/viewer.py)
"""

import sys
from datetime import datetime
from pathlib import Path

# ── `streamlit run` script klasöründen başlatır; paket importu için kökü ekle ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from gozcu.config import settings
from gozcu.search import search as run_search

# ── Sayfa düzeni ──
st.set_page_config(page_title="Gözcü", page_icon="👁️", layout="wide")
st.title("👁️ Gözcü")
st.caption("Kamera arşivinde Türkçe doğal dil arama — her şey lokalde, hiçbir veri dışarı çıkmaz.")

# ── Sorgu kutusu + top-k seçici ──
col_query, col_k = st.columns([5, 1])
with col_query:
    query = st.text_input(
        "Sorgu",
        placeholder='örn: "dün gece giren beyaz Transit" veya "bahçede yürüyen kişi"',
        label_visibility="collapsed",
    )
with col_k:
    top_k = st.slider("Sonuç", min_value=4, max_value=48, value=settings.default_top_k, step=4)

if query:
    # ── Arama (model ilk sorguda yüklenir — search modülü tembel tekil tutar) ──
    with st.spinner("Aranıyor..."):
        outcome = run_search(query, top_k=top_k)
    parsed = outcome.parsed

    # ── Ayrıştırma özeti ──
    info = f'Görsel sorgu: **"{parsed.visual_text or query}"**'
    if parsed.ts_from is not None:
        fmt = "%d.%m.%Y %H:%M"
        info += (
            f' · Zaman filtresi: **"{parsed.time_phrase}"** → '
            f"{datetime.fromtimestamp(parsed.ts_from):{fmt}} — "
            f"{datetime.fromtimestamp(parsed.ts_to):{fmt}}"
        )
    st.markdown(info)
    if outcome.time_filter_dropped:
        st.warning("Zaman aralığında sonuç bulunamadı — filtre kaldırılıp tüm arşivde arandı.")

    if not outcome.results:
        st.error("Sonuç yok. Önce `python -m gozcu index <klasör>` ile arşivi indeksleyin.")
    else:
        # ── Seçilen sonucu videoda o andan oynat ──
        selected = st.session_state.get("selected")
        if selected is not None:
            when = datetime.fromtimestamp(selected["ts"]).strftime("%d.%m.%Y %H:%M:%S")
            st.subheader(f"▶ {selected['video_id']} — {when}")
            st.video(selected["video_path"], start_time=int(selected["offset_s"]))

        # ── Sonuç ızgarası: 4 sütun, thumb + skor + zaman + oynat düğmesi ──
        cols = st.columns(4)
        for i, hit in enumerate(outcome.results):
            with cols[i % 4]:
                when = datetime.fromtimestamp(hit["ts"]).strftime("%d.%m %H:%M:%S")
                st.image(hit["thumb_path"], use_container_width=True)
                st.caption(f"**{hit['score']:.3f}** · {hit['video_id']} · {when}")
                if st.button("▶ Oynat", key=f"play_{i}"):
                    st.session_state["selected"] = hit
                    st.rerun()
