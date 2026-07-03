"""İlk indeksleme sonrası Türkçe arama kalite testi.

Çalıştırma (proje kökünden):
    .venv/Scripts/python.exe -X utf8 experiments/2026-07-03_faz1_ilk_indeksleme/arama_testi.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gozcu.search import search

# ── Test sorguları: görsel çeşitlilik + zaman filtresi geri düşüş senaryosu ──
QUERIES = [
    "kapalı otoparkta siyah SUV",
    "farları açık araba",
    "telefonla konuşan sürücü",
    "otoparkta park etmiş araç",
    "boş otopark koridoru",
    "dün gece otoparka giren siyah araç",  # zaman filtresi boş dönmeli → geri düşüş uyarısı
]

for query in QUERIES:
    outcome = search(query, top_k=5)
    parsed = outcome.parsed
    print(f'\n=== SORGU: "{query}" ===')
    print(f'  görsel metin: "{parsed.visual_text}"')
    if parsed.ts_from is not None:
        fmt = "%d.%m.%Y %H:%M"
        print(
            f'  zaman filtresi: "{parsed.time_phrase}" → '
            f"{datetime.fromtimestamp(parsed.ts_from):{fmt}} — "
            f"{datetime.fromtimestamp(parsed.ts_to):{fmt}}"
        )
    if outcome.time_filter_dropped:
        print("  UYARI: zaman aralığında sonuç yok — filtresiz tekrar arandı")
    for i, hit in enumerate(outcome.results, 1):
        print(
            f"  {i}. {hit['score']:.3f}  {hit['video_id']}  "
            f"offset={hit['offset_s']:5.1f}s  {Path(hit['thumb_path']).name}"
        )
