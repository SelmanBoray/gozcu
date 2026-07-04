# Deney: Eval seti — ilk ölçüm (Risk 3 kapatıldı)

**Tarih:** 4 Temmuz 2026
**Amaç:** ARCHITECTURE.md Risk 3 — "eval seti yoksa karar da yok". 22 Türkçe
sorgu → ground-truth çifti kuruldu, hat ölçüldü, gelecek her model/eşik kararının
regresyon temeli oluşturuldu.

## Tasarım (AI Engineer incelemesi sonrası — metriği etiket tipine eşle)

Naif "40 sorgu, video-düzeyi recall" tasarımını AI Engineer 3 yerde kırdı, düzeltildi:

1. **Golden-frame birincil, ikincil değil.** Ubik sınıflar (insan, araba — her
   videoda) video-düzeyinde triviyal %100 verir, hiçbir sıralama ölçmez. Bunlar +
   davranış sorguları için **gözle doğrulanmış tek kare** (golden) zorunlu. Video-düzeyi
   yalnızca sınıf/sahne ve `|relevant| ≤ 2` ise skorlanabilir.
2. **Öznitelik GT'si otomatik doğrulanamaz (Boşluk 2).** "siyah SUV" için yalnız
   Teknofest Togg'unun siyah olduğunu biliyoruz; VIRAT/ucf_trafik araç renkleri
   etiketsiz. Nitekim "siyah SUV" top-1 = VIRAT_otopark'ta **meşru bir koyu SUV**
   (küçük resim doğrulandı). Renk/öznitelik sorguları `advisory` — recall'a girmez,
   yalnız top-1 gözle denetlenir.
3. **Kare vs kırpık ayrı koşulmalı.** Her sorgu 3 kez: `source=frame`, `source=crop`,
   birleşik. Faz 1.5'in marjinal değeri = R@5(birleşik) − R@5(yalnız-kare).
   (Bunun için `store.search`/`search`'e `source` filtresi eklendi.)

Set **sonuçlara bakılmadan donduruldu** (pre-registration); golden frame_idx'ler
yalnızca küçük resim gözle doğrulanarak etiketlendi (skora güvenilmedi).
Koşmadan hedefler ilan edildi: golden R@1 ≥ 0.66, skorlanabilir R@5 ≥ 0.80,
Faz 1.5 marjini ≥ 0.

## Sonuçlar

### Güçlü — Faz 1.5'in değeri artık bir sayı
- **Marjinal katkı: R@5 0.667 (yalnız-kare) → 1.0 (birleşik) = +0.333.**
  Yalnız-kırpık tek başına 0.917. Kırpık embedding nesne sorgularını *taşıyor*;
  ablation bunu kanıtlıyor. Faz 1.5'i haklı çıkaran tek ölçüm buydu.
- **Golden R@1 = 1.0** — 3 gözle-doğrulanmış kare (telefonlu sürücü, asfaltta
  yürüyen, araç yanındaki kişi) de birinci sırada.
- Skorlanabilir birleşik: R@1=0.917, R@5=1.0, MRR=0.958. Makro R@5=1.0.

### Boşluk 1 çözüldü — "gece" sınır davranışı doğru
- `scene_gece` ("gece çekilmiş..."): "gece" **embed edildi** (parse'a takılmadı) →
  ucf_gece rank-1. ✅
- `time_dun_gece`: "dün gece" **parse edildi** (ts filtresi), görsel = "otoparka
  giren araç". `time_son_saat`: filtre boş → **filtresiz geri düşüş çalıştı**. ✅

### KRİTİK DÜRÜST SINIR — negatif ayrımcılık çöktü
- Pozitif top-1 min = 0.282; negatif top-1 max = **0.36** (`neg_kopek`).
  **Örtüşme: -0.078.** Tek mutlak skor eşiği korpustaki ile olmayanı AYIRMAZ.
- Sebep: negatifler sorgunun *insan/araç* parçasına eşleşiyor — "köpek gezdiren
  **adam**" → insan (0.36); "kırmızı bisiklet süren **çocuk**" → insan; "**karla**
  kaplı sokak" → kamyon (0.277). Kavramın tamamı (köpek, kar, kırmızı bisiklet)
  yok ama parçası var.
- **Sonuç: sistemin güvenilir bir "bulunamadı" sinyali yok.** Korpusta olmayan
  bir şey arandığında makul görünen ama yanlış sonuç, gerçek sorgularla benzer
  skorla döner. Bu, AI Engineer'ın S4'te öngördüğü ve **Faz 2'nin (VLM re-rank /
  doğrulayıcı) somut gerekçesidir.** Kısa vadeli hafifletme: nesne sorgularında
  `yolo_class` sertlik filtresi (kırmızı bisiklet → bisiklet sınıfı hiç yok → boş).

## Kendini kandırmama notları (S5)
- Video-düzeyi R@5=1.0 **cesaret verici ama şişmiş**: korpustaki 4 sahne (gece /
  kapalı garaj / trafik / açık otopark) görsel olarak çok farklı, ayırması kolay.
  Gerçek zorluk golden pinpoint (R@1=1.0, iyi) ve negatif ayrım (çöktü) tarafında.
- n=12 skorlanabilir → R@5 %95 CI = [0.757, 1.0]. Bu bir **gösterge**, benchmark
  değil. Değeri regresyon temeli olmasında: sonraki değişiklik bu sabit sette
  eşleştirilmiş (McNemar) karşılaştırılacak.
- Kategori metrikleri (n≈3-4) anekdot; arıza avı için, iddia için değil.

## Sonraki adımlar
1. **"Bulunamadı" mekanizması** — negatif örtüşme en yüksek öncelikli açık. Faz 2
   VLM doğrulayıcı ya da nesne sorgularında yolo_class sertlik kapısı.
2. Korpusu büyüt (özellikle görsel olarak benzer sahneler) — video-düzeyi şişmesini
   kırmak ve gerçek ayrım gücünü ölçmek için.
3. VIRAT kampüs için insanlı daha uzun segment (şu an 0 insan — veri sınırı).

**Çıktılar:** `rapor.md` (otomatik), `sonuc.json` (tam per-query veri), `queries.yaml`
(dondurulmuş set). Runner: `eval/run_eval.py`.
