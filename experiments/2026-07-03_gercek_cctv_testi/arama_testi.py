"""Gerçek CCTV kaydıyla küçük özne (Risk 1) testi.

Veri: VIRAT otopark (720p, 30-80 px insanlar), VIRAT kampüs (1080p, uzak yayalar),
UCF trafik (320x240, çok küçük araçlar), UCF gece (düşük ışık).
Sorgular AI Engineer video içeriği analizine göre seçildi.

Çalıştırma (proje kökünden):
    .venv/Scripts/python.exe -X utf8 experiments/2026-07-03_gercek_cctv_testi/arama_testi.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gozcu.search import search

# ── Test sorguları: hedef video parantez içinde — sıralama başarısı elle kontrol edilecek ──
QUERIES = [
    # VIRAT otopark — küçük insanlar araçların arasında
    "park halindeki arabaların arasında yürüyen kişi",   # (VIRAT_otopark)
    "arabasına binen adam",                              # (VIRAT_otopark)
    "otoparktan çıkan beyaz araç",                       # (VIRAT_otopark)
    # VIRAT kampüs — uzak yayalar (en zor sulanma senaryosu)
    "kaldırımda yürüyen sırt çantalı kişi",              # (VIRAT_kampus)
    "birlikte yürüyen iki kişi",                         # (VIRAT_kampus)
    "uzakta tek başına yürüyen insan",                   # (VIRAT_kampus)
    # UCF trafik — 320x240'ta minik araçlar
    "caddede giden mavi kamyonet",                       # (ucf_trafik)
    "yolda ilerleyen beyaz otobüs",                      # (ucf_trafik)
    "trafikte bekleyen kırmızı arabalar",                # (ucf_trafik)
    # UCF gece — düşük ışık
    "gece karanlıkta binaya giren kişi",                 # (ucf_gece)
    # Zamansal geri düşüş
    "dün akşam 8'den sonra otoparka giren araçlar",
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
            f"  {i}. {hit['score']:.3f}  {hit['video_id']:<14}  "
            f"offset={hit['offset_s']:6.1f}s  {Path(hit['thumb_path']).name}"
        )
