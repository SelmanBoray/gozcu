# Deney: Faz 1.5 — YOLO kırpık embedding (sulanma çözümü)

**Tarih:** 3 Temmuz 2026 (gece)
**Amaç:** Ölçülüp kesinleşen Risk 1'i (tam-kare embedding'in küçük özne sulanması)
YOLO insan/araç kırpıklarını ayrı vektör olarak indeksleyerek çözmek.

## Tasarım (AI Engineer onaylı)

- yolo11m @ imgsz=1280 — küçük özne için çözünürlük model boyutundan önemli
- Düşük güven eşiği (insan 0.15 / araç 0.25), `yolo_conf` payload'da → arama anında
  filtrelenebilir; indekste olmayan tespit kurtarılamaz (geri alınamaz karar)
- %20 bağlam payı + kareye tamamlama + ORİJİNAL çözünürlükten kırpma
- Statik nesne bastırma: önceki tutulan kareyle aynı-sınıf IoU > 0.85 → atla
  (park halindeki araç yalnız ilk görünümünde vektör alır)
- Tek koleksiyon, `source: frame|crop`; skor ofseti yok (kırpık-kare yanlılığı
  büyük ölçüde kendi kendini düzeltiyor — nesne sorgusu kırpığı, sahne sorgusu kareyi seçer)
- Arama: kare tekilleştirme + 8 sn zaman kümeleme (aynı yürüyüş top-10'u doldurmasın)

## Yolda bulunan 2 hata

1. **CPU-only torchvision:** ultralytics kurulumu PyPI'dan CUDA'sız wheel çekti →
   `torchvision::nms` GPU'da patladı. Çözüm: cu126 indeksinden yeniden kurulum
   (requirements.txt'e not düşüldü).
2. **Kırpık sınırı insanları kesiyordu:** kare başına 15 sınırı güven sıralamasıyla
   ve statik bastırmadan ÖNCE uygulanıyordu — otoparkta 15+ park halindeki araç
   (conf 0.65+) sıralamayı doldurup 0.57'lik insanları eliyordu. VIRAT kırpıklarında
   SIFIR insan vardı. Çözüm: sınır bastırma SONRASI + insan öncelikli (`cap_crops`).
   Sonuç: otopark insan kırpığı 0 → **132**.

## Sonuçlar — önce/sonra

| Sorgu | Faz 1 (yalnız kare) | Faz 1.5 (kırpık) |
|---|---|---|
| "otoparkta yürüyen insan" | — | **0.403 VIRAT insan kırpığı #1** ✅ |
| "uzakta tek başına yürüyen insan" | ucf_gece yakın plan kazandı (0.228), hedef 0.155'te kayıp | **İlk 3'ün 3'ü de VIRAT insan kırpığı (0.363)** ✅ görsel doğrulandı |
| "park halindeki arabaların arasında yürüyen kişi" | Togg garajı kazandı (0.322), hedef 0.263 | **VIRAT insan kırpığı #1 (0.349)** ✅ |
| "caddede giden mavi kamyonet" | doğruydu | hâlâ doğru, skor 0.306→**0.416** |

Görsel doğrulama: "uzakta tek başına yürüyen insan" 1. sonucu = boş asfaltta tek
başına yürüyen adam kırpığı; "otoparkta yürüyen insan" 1. sonucu = beyaz aracın
kapısındaki adam. Birebir.

## İndeks istatistikleri

647 kare + 3.323 kırpık = 3.970 vektör. Kırpık dağılımı:
- VIRAT_otopark: 132 insan, 91 araba, 26 kamyon (statik bastırma çalışıyor: 59 karede 20+ park halindeki araç ≈ 91 tekil)
- ucf_trafik: 1.713 araba, 731 kamyon, 516 otobüs, 40 insan (yoğun trafik — beklenen)
- VIRAT_kampus: 38 araba, 2 kamyon, **0 insan** — boru hattı hatası DEĞİL: ham YOLO
  testi imgsz=1920 & conf=0.05'te bile insan bulamıyor; bu 22 sn'lik segmentte tespit
  edilebilir insan yok (<10 px veya hiç). Bilinen tırmanma yolu: SAHI döşeme (4-5×
  maliyet) — eval seti gerektirirse.

## Sonraki adım

Eval seti (30-50 Türkçe sorgu → doğru kare çifti) — artık 7 videoluk korpus ve
çalışan kırpık hattı var. Kampüs için insanlı daha uzun VIRAT segmenti eklenebilir.
