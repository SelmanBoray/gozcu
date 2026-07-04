# Gözcü Eval — 2026-07-04 17:43

- Sorgu: 5 (skorlanabilir 3, advisory 2, zaman 0, negatif 0)
- İndeks: 4391 vektör · top_k=10 · model jinaai/jina-clip-v2

> **n≈20 uyarısı (AI Engineer S5):** Bu sayılar GÖSTERGE + regresyon temeli, istatistiksel benchmark değil. R@5 için %95 CI yarı-genişliği ≈±0.15 — 0.6 vs 0.9 ayırt edilir, 0.82 vs 0.88 EDİLMEZ. Asıl değer sabit sette eşleştirilmiş öncesi/sonrası karşılaştırmada.

## Manşet (skorlanabilir, birleşik hat)

| Metrik | Değer | %95 CI | Hedef |
|---|---|---|---|
| Recall@1 | 0.333 | [0.061, 0.792] | |
| Recall@5 | 0.333 | [0.061, 0.792] | |
| Recall@10 | 0.667 | [0.208, 0.939] | |
| MRR | 0.381 | | |
| Makro R@5 (kategori eşit ağırlık) | 0.25 | | 0.8 |

## Faz 1.5 marjinal değeri (kırpık embedding)

- Birleşik R@5: **0.333** · Yalnız-kare R@5: 0.667 · Yalnız-kırpık R@5: 0.0
- **Marjinal katkı (birleşik − kare): -0.333** (hedef ≥ 0.0: ❌)

## Golden-frame (gözle doğrulanmış tek kare)

- n=2 · Recall@1=0.0 (hedef 0.66) · Recall@5=0.0 · MRR=0.071
  - `v2_gold_meva_yuruyen` sıra=KAÇTI (dedup/kümeleme)
  - `v2_gold_meva_otopark` sıra=7

## Negatif ayrımcılık (S4) + bulunamadı kapısı

- **Kapı yakalama oranı: None** (yakalanan: —)
- ✅ Yanlış kapı yok — hiçbir pozitif sorgu takılmadı
- Kapıya takılmayan negatiflerin top-1 skorları: []
- Pozitif top-1 skorları: [0.332, 0.375, 0.421]

| negatif | tip | kapı | top-1 skor | top-1 sınıf |
|---|---|---|---|---|

## Zaman ayrıştırma (Boşluk 1 — 'gece' embed mi parse mi?)

| id | görsel metin (embedlenen) | zaman ifadesi (parse) | filtre düştü |
|---|---|---|---|

## Tüm sorgular

| id | tip | sorgu | top-1 | sıra |
|---|---|---|---|---|
| v2_gold_meva_yuruyen | golden | kaldırımda yürüyüp uzaklaşan insan | meva_hastane1 | ❌ |
| v2_gold_meva_otopark | golden | araçlarla dolu açık otopark | ucf_trafik | 7 |
| v2_scene_kampus | scene | kampüs yürüyüş yolları | VIRAT_kampus | 1 |
| v2_adv_kamyonet | advisory | beyaz kamyonet | meva_hastane2 | — |
| v2_adv_bina_arac | advisory | tuğla bina önünde araç | video_1 | — |
