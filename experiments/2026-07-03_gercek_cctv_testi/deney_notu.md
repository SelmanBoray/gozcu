# Deney: Gerçek CCTV kaydıyla küçük özne testi (Risk 1)

**Tarih:** 3 Temmuz 2026
**Veri:** VIRAT otopark (720p, 31 sn), VIRAT kampüs (1080p, 22 sn), UCF trafik
(320×240, 2:20, yanık OSD saatli), UCF gece (320×240, düşük ışık). Tümü araştırma
amaçlı yayınlanmış gerçek gözetim kayıtları (AI Engineer link doğrulamalı seçti).

## Bulgu 1 — Örnekleyici tasarım hatası bulundu ve düzeltildi 🔴→✅

İlk indekslemede VIRAT otopark 31 sn'den **3 kare**, kampüs 22 sn'den **1 kare** verdi.
Teşhis (`teshis_orneklyici_kapilari.txt`):

- **Hareket kapısı:** 288 piksel değişim şartı, uzak yayanın ~94 pikselinin 3 katıydı;
  piksel eşiği 25 + blur, 3 piksellik insanı tamamen siliyordu.
- **pHash dedup YAPISAL kör:** 320×180 küçültmenin global 64-bit hash'i, sadece minik
  insan hareket ettiğinde Hamming=0 — hamming≤2'de bile kampüste 44 kareyi 1'e indirdi.
  Parametreyle çözülemez; mekanizma değişti.

**Yeni tasarım** (AI Engineer onaylı, sampler.py yeniden yazıldı):
ortalama-normalize absdiff (AGC/bulut koruması) + bağlı bileşen gürültü filtresi
(yağmur/gren blob değildir) + son TUTULAN kareye göre birikimli değişim dedup
(yürüyen insan fark biriktirir, gürültü biriktirmez) + küresel olay koruması
+ OSD maskesi + saatlik oran sınırı. pHash artık yalnız metadata.

**Sonuç:** otopark 3→**59**, kampüs 1→**31**, trafik 117→279, gece 39→236 kare.
Yürüyen insanlı kareler artık indekste (735.jpg'de 3 ayrı yaya — görsel doğrulandı).

## Bulgu 2 — Arama sonuçları (11 Türkçe sorgu)

| Sorgu | Sonuç |
|---|---|
| "caddede giden mavi kamyonet" | ✅ **birebir** — kare merkezinde mavi kamyon (görsel doğrulandı) |
| "yolda ilerleyen beyaz otobüs" | ~ otobüs bulundu, renk kısmen (turkuaz otobüs + beyaz minibüs) |
| "trafikte bekleyen kırmızı arabalar" | ✅ makul küme |
| "gece karanlıkta binaya giren kişi" | ✅ doğru video (gece, bina önü figür) |
| "dün akşam 8'den sonra..." | ✅ zaman ayrıştı + filtresiz geri düşüş uyarısı |
| "park halindeki arabaların arasında yürüyen kişi" | ❌ Togg garajı kazandı (0.322), VIRAT otopark 0.263'te kaldı |
| "kaldırımda yürüyen sırt çantalı kişi" | ❌ ucf_gece kazandı (0.220), kampüs 0.196'da |
| "uzakta tek başına yürüyen insan" | ❌ ucf_gece (0.228, yakın plan figür), kampüs 0.155 |

## Bulgu 3 — Risk 1 KESİNLEŞTİ, ama nüansla

Tam-kare embedding küçük özneye kör değil, **zayıf**: "yürüyen kişi" sorgusu VIRAT
otopark İÇİNDE doğru kareyi 1. sıraya koyuyor (24.5 sn, üç yayalı kare — video içi
sıralama doğru). Ama videolar ARASI yarışta 0.02–0.07 puanlık sulanma boşluğu,
özneyi büyük gösteren alakasız videoya yenilmesine yetiyor.

**Karar: Faz 1.5 öne çekildi** — YOLO insan/araç kırpıkları ayrı vektör olarak
indekslenecek (aynı kareye işaret eder). Mimari zaten bunu öngörmüştü; kanıt tamam.

## Yan bulgular

- Kitware VIRAT sunucusu HTTP Range'i yok sayıyor (206 der, baştan yollar) + ~150 KB/sn
  → büyük klip yerine aynı sahnelerin küçük segmentleri indirildi.
- İndeksleme hızı: 605 kare / 52.9 sn (GPU, mpeg4 CPU çözme dahil).
- UCF trafikte OSD saati var — OSD maskesi devrede ama etkinliği ayrıca ölçülmedi.

## Sonraki adım

1. Faz 1.5: YOLO-crop embedding (kırpık vektörler + `source: "crop"` payload alanı).
2. Eval seti bu 7 videoyla başlatılabilir: sorgu→doğru kare çiftleri artık elde.
