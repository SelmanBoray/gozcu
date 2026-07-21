# Vel'Koz Eval — 2026-07-21 18:06

- Sorgu: 22 (skorlanabilir 12, advisory 4, zaman 2, negatif 4)
- İndeks: 4395 vektör · top_k=10 · model jinaai/jina-clip-v2

> **n≈20 uyarısı (AI Engineer S5):** Bu sayılar GÖSTERGE + regresyon temeli, istatistiksel benchmark değil. R@5 için %95 CI yarı-genişliği ≈±0.15 — 0.6 vs 0.9 ayırt edilir, 0.82 vs 0.88 EDİLMEZ. Asıl değer sabit sette eşleştirilmiş öncesi/sonrası karşılaştırmada.

## Manşet (skorlanabilir, birleşik hat)

| Metrik | Değer | %95 CI | Hedef |
|---|---|---|---|
| Recall@1 | 0.917 | [0.646, 0.985] | |
| Recall@5 | 0.917 | [0.646, 0.985] | |
| Recall@10 | 0.917 | [0.646, 0.985] | |
| MRR | 0.917 | | |
| Makro R@5 (kategori eşit ağırlık) | 0.933 | | 0.8 |

## Faz 1.5 marjinal değeri (kırpık embedding)

- Birleşik R@5: **0.917** · Yalnız-kare R@5: 0.667 · Yalnız-kırpık R@5: 0.917
- **Marjinal katkı (birleşik − kare): +0.250** (hedef ≥ 0.0: ✅)

## Golden-frame (gözle doğrulanmış tek kare)

- n=3 · Recall@1=1.0 (hedef 0.66) · Recall@5=1.0 · MRR=1.0
  - `gold_telefon_surucu` sıra=1
  - `gold_yuruyen_insan` sıra=1
  - `gold_parkta_yuruyen` sıra=1

## Negatif ayrımcılık (S4) + bulunamadı kapısı

- **Kapı yakalama oranı: 1.0** (yakalanan: ['neg_bisiklet', 'neg_kopek', 'neg_yagmur', 'neg_kar'])
- ⚠️ **YANLIŞ KAPI** (pozitif takıldı): ['scene_karanlik_yol']
- Kapıya takılmayan negatiflerin top-1 skorları: []
- Pozitif top-1 skorları: [0.282, 0.349, 0.363, 0.363, 0.364, 0.366, 0.366, 0.377, 0.384, 0.408, 0.452]

| negatif | tip | kapı | top-1 skor | top-1 sınıf |
|---|---|---|---|---|
| neg_bisiklet | class_absent | ✅ boş | None | None |
| neg_kopek | class_absent | ✅ boş | None | None |
| neg_yagmur | attribute_absent | ✅ boş | None | None |
| neg_kar | attribute_absent | ✅ boş | None | None |

## Zaman ayrıştırma (Boşluk 1 — 'gece' embed mi parse mi?)

| id | görsel metin (embedlenen) | zaman ifadesi (parse) | filtre düştü |
|---|---|---|---|
| scene_gece | gece çekilmiş kamera görüntüsü | — | hayır |
| time_dun_gece | otoparka giren araç | dün gece | evet |
| time_son_saat | geçen kamyon | son 3 saatte | evet |

## Tüm sorgular

| id | tip | sorgu | top-1 | sıra |
|---|---|---|---|---|
| obj_otobus | class | otobüs | ucf_trafik | 1 |
| obj_motosiklet | class | motosiklet | video_3 | 1 |
| obj_kamyon | class | kamyon | ucf_trafik | 1 |
| scene_gece | scene | gece çekilmiş kamera görüntüsü | ucf_gece | 1 |
| scene_karanlik_yol | scene | karanlıkta ilerleyen araçlar | — | ❌ |
| scene_kapali_otopark | scene | kapalı otopark katı | video_2 | 1 |
| scene_yogun_trafik | scene | yoğun trafik akışı | ucf_trafik | 1 |
| scene_acik_otopark | scene | açık hava otopark alanı | VIRAT_otopark | 1 |
| beh_uzakta_yuruyen | scene | uzakta tek başına yürüyen insan | VIRAT_otopark | 1 |
| gold_telefon_surucu | golden | telefonla konuşan sürücü | video_2 | 1 |
| gold_yuruyen_insan | golden | yürüyen insan | VIRAT_otopark | 1 |
| gold_parkta_yuruyen | golden | park halindeki arabaların arasında yürüyen kişi | VIRAT_otopark | 1 |
| adv_siyah_suv | advisory | siyah SUV araç | VIRAT_otopark | — |
| adv_mavi_kamyonet | advisory | mavi kamyonet | ucf_trafik | — |
| adv_beyaz_arac | advisory | beyaz renkli araba | ucf_trafik | — |
| adv_direksiyon | advisory | direksiyon başındaki sürücü | video_3 | — |
| time_dun_gece | zaman | dün gece otoparka giren araç | VIRAT_otopark | — |
| time_son_saat | zaman | son 3 saatte geçen kamyon | ucf_trafik | — |
| neg_bisiklet | negatif | kırmızı bisiklet süren çocuk | — | — |
| neg_kopek | negatif | köpek gezdiren adam | — | — |
| neg_yagmur | negatif | yağmurda şemsiyeli insanlar | — | — |
| neg_kar | negatif | karla kaplı sokak | — | — |
