# C1 baseline — korpus büyütme ÖNCESİ mühür (21 Temmuz 2026)

Korpus zenginleştirmeden (task #4) ÖNCE dondurulan referans. AI Engineer'ın "eklemeli
sürümleme" planının 1. adımı: büyütme sonrası "bozdum mu / iyileştirdim mi"yi bu sabit
noktaya karşı ölçeceğiz. Dondurulmuş sorgu setleri (`queries.yaml`, `queries_faz2.yaml`)
byte-frozen kalır; yeni içerik ayrı `v2` setine yazılır.

## Korpus (C1) — `C1_manifest.json`
- **13 video · 4395 vektör** (3560 kırpık + 835 kare)
- Videolar: VIRAT_kampus, VIRAT_otopark, meva_hastane1/2, meva_idari, meva_okul1/2,
  meva_otobus, ucf_gece, ucf_trafik, video_1/2/3
- YOLO sınıf envanteri: **araba, insan, kamyon, motosiklet, otobüs** (bisiklet YOK →
  `neg_bisiklet` envanter kapısını test eder)

## Retrieval eval (queries.yaml, 22 sorgu) — `sonuc.json` / `rapor.md`
- **R@1=0.917 · R@5=0.917 · R@10=0.917 · MRR=0.917** (skorlanabilir, birleşik)
- Makro R@5=0.933
- **Faz 1.5 marjini (birleşik−kare) R@5: +0.250** (kırpık embedding hâlâ yardım ediyor)
- 4 negation + 3 advisory-absent hepsi `[∅]`/not_found (envanter kapısı + discrimination)

## VLM eval (queries_faz2.yaml, 12 sorgu) — `../2026-07-21_hard_negatif_eval/`
- **11/12 · ayrım 2/2** (araç sızıntısı 0)
- `hn_kirmizi_kiyafet` non-vacuous (6 araba adayı elendi); `hn_siyah_kiyafet` **vacuous ⚠**
  — büyütme SONRASI bu ⚠'nin kaybolması birincil iyileşme ölçütü.
- `mavi kamyonet` dokümante tavan FAIL (skoru şişirmiyoruz).

## Büyütme sonrası verdict ölçütü (AI Engineer Q5)
- **"Bozmadı"** = guard eksenleri C1'in McNemar/CI bandında + `false_gates`=0 + 5 negation/absent
  hâlâ not_found + faz2 positive_ctrl & forbid korundu.
- **"İyileşti"** = `hn_*` non-vacuous (⚠ gitti) + yeni v2 person-color pozitifi geçti + v1 recall
  +distractor'a dayanıklı.
