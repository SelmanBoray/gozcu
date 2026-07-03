# Deney: Faz 1 ilk indeksleme + Türkçe arama testi

**Tarih:** 3 Temmuz 2026
**Veri:** `C:/teknofest/testverisi/` — 3 video (kapalı otopark, siyah Togg SUV, gece/loş ışık)
**Donanım:** RTX 4070 Laptop (cuda), jina-clip-v2 fp16

## İndeksleme

| Video | Tutulan kare | Süre | Hız |
|---|---|---|---|
| video_1.mp4 | 12 | 4.0 sn | 3.0 kare/sn |
| video_2.mp4 | 12 | 3.0 sn | 4.0 kare/sn |
| video_3.mp4 | 13 | 3.0 sn | 4.4 kare/sn |

Toplam 37 kare, 10.3 sn. Hareket kapısı + pHash dedup kısa kliplerde beklendiği gibi
agresif eleme yaptı (~7 sn video → 12-13 kare).

## Arama sonuçları (top-5, tam çıktı: arama_sonuclari.txt)

| Sorgu | Sonuç | Değerlendirme |
|---|---|---|
| "telefonla konuşan sürücü" | İlk 3 sonuç video_2 (sürücü telefonda) | ✅ **görsel doğrulandı** — davranış düzeyinde eşleşme |
| "boş otopark koridoru" | video_3 uzak/boş kareler + video_1 sonu | ✅ görsel doğrulandı |
| "dün gece otoparka giren siyah araç" | Zaman filtresi boş → filtresiz geri düşüş + uyarı | ✅ tasarlandığı gibi |
| "kapalı otoparkta siyah SUV" | Tüm videolardan araçlı kareler | ✅ makul |
| "farları açık araba" | Farlı ön kareler üstte | ✅ makul |

## Bulgular

1. **Türkçe doğrudan embedding çalışıyor** — çeviri katmanı olmadan davranış
   düzeyinde ("telefonla konuşan") ayrım yapabildi. SigLIP2 yedeğine şu an gerek yok.
2. **Skorlar dar bantta** (0.35–0.39 genel sorgularda): tüm kareler aynı garaj +
   aynı araç olduğundan normal. Ayırt edici sorguda ("telefon") skor düşük (0.29)
   ama sıralama doğru — mutlak skor değil sıralama önemli.
3. **Zaman filtresi + geri düşüş** uçtan uca doğru: "dün gece" ayrıştırıldı,
   aralıkta veri yokunca uyarıyla filtresiz arandı.
4. **Risk 1 (küçük özne sulanması) henüz test edilemedi** — araç karede büyük.
   Uzak/küçük insan içeren gerçek CCTV kaydıyla eval seti şart (Risk 3).

## Sonraki adım

30–50 sorguluk eval seti; mümkünse gerçek dış mekan CCTV kaydı ekle
(küçük özne + gündüz/gece çeşitliliği).
