# Deney: Statik lot zenginleştirme — kare-açlığını gidermek

**Tarih:** 4 Temmuz 2026
**Amaç:** Olgu B'nin ikinci bileşeni (kare-açlığı): meva_okul2/otobus statik lot →
her biri 1 kare vermişti, sahne-niyeti boost'unun yükseltecek karesi yoktu.

## Ne yapıldı

meva_okul2 ve meva_otobus, MEVA S3 kaynağından **60 sn → 180 sn** yeniden çekildi
(ffmpeg -t 180). Yeniden indekslendi.

- meva_okul2: **1 → 3 kare** (0/60/120 sn çapa kareleri), 10 kırpık (aynı — statik
  bastırma park araçlarını çapalar arası eliyor, doğru).
- meva_otobus: **1 → 3 kare**, 7 kırpık.
- İndeks: 4391 → 4395 vektör.

## Sonuç — hedefe ulaşıldı, ama pinpoint golden onu gizliyor

**Doğrudan ölçüm ("araçlarla dolu açık otopark", meva_okul2 sahne-karesi sırası):**
| aşama | λ=0 | λ=1.0 |
|---|---|---|
| zenginleşme öncesi (1 kare) | 8 | 6 |
| zenginleşme sonrası (3 kare) | 7 | **5** |

meva_okul2 artık **top-5'te** (rank 5). Zenginleşme (3 kare) + boost birlikte
sahneyi rank 8'den rank 5'e taşıdı.

**Kırpık-seli tamamen elendi:** v2'de Faz 1.5 marjini (birleşik − yalnız-kare) R@5
**−0.333 → 0.000.** Yani kareler artık kırpıkların altında gömülmüyor — boost'un
etkisi kare-açlığı confound'u kalkınca temiz görünür oldu.

**v1 kilitli: sıfır regresyon** (R@1=0.917, R@5=1.0).

## Dürüst nüans — pinpoint golden statik sahne için yanlış metrik (Olgu A'nın tekrarı)

v2_gold_meva_otopark golden'ı `frame_idx=0`'a sabitli. Zenginleşmeyle eklenen çapa
karesi (60/120 sn) rank 5'i alıyor, spesifik frame 0 değil → golden metriği "rank 9"
diyor, oysa SAHNE top-5'te. Bu, Olgu A ile aynı **etiket-granülaritesi** sorunu:
statik/tekrarlı sahne için pinpoint frame golden yanlış — doğru metrik "doğru video VE
top-k'da source==frame" (dev bataryasının kullandığı sahne-kare metriği).

**Ders:** golden_frame yalnız BENZERSİZ an için (yürüyen kişi, telefonlu sürücü);
statik SAHNE için video+kare metriği kullanılmalı. v2 dondurulmuş kalıyor (pre-registration),
ama v2_gold_meva_otopark'ın doğru okuması "sahne rank 8→5", pinpoint "rank 9" değil.

## Kalan durum — doğru davranış

meva_okul2 rank 5'te; top-4 = VIRAT_otopark (kare+kırpık) + ucf_trafik kırpık.
VIRAT_otopark "açık otopark"ın **meşru daha iyi eşleşmesi** → onu meva_okul2'nin
altına itmek yanlış olurdu. Rank 5 bu korpus için tavan; boost + zenginleşme
üstüne düşeni yaptı, kalanı doğru sıralama.

## Sonraki adım
Olgu B kapandı (sıralama + veri bileşenleri). Sıradaki: **Faz 2 VLM re-rank** —
kalan negatif örtüşme (köpek/yağmur/kar), öznitelik doğrulama (renk sorguları),
Olgu A'nın kümeleme nüansı.
