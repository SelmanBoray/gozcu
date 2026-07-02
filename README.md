# Gözcü 👁️

**Güvenlik kamerası arşivinde Türkçe doğal dil arama motoru.**

> "Dün gece siteye giren beyaz Transit ne zaman çıktı?"
> "Kırmızı montlu adamı bul."

Saatlerce arşiv izlemek yerine, arşive soru sorarsın — Gözcü ilgili anları saniyeler içinde bulur.

## Neden?

- Site yönetimleri, AVM'ler ve fabrikalar bir olay olduğunda **saatlerce video izliyor**.
- Yurt dışı muadili **Conntour**, Mart 2026'da General Catalyst + Y Combinator'dan **$7M** yatırım aldı — pazar doğrulanmış.
- Türkiye'de muadili yok; **Türkçe sorgu desteği** kimsede yok.
- **KVKK avantajı:** Gözcü tamamen lokal çalışır — görüntü binadan çıkmaz.

## Nasıl çalışır? (mimari özet)

```
Video arşivi ──> Akıllı kare örnekleme (hareket bazlı)
           ──> Görsel embedding (multilingual CLIP/SigLIP)
           ──> Vektör veritabanı (Qdrant, lokal)

Türkçe sorgu ──> Embedding ──> Top-k eşleşme (kare + zaman damgası)
             ──> VLM doğrulama + araç takibi ile giriş/çıkış zamanı (Faz 2)
```

Detaylı kararlar için: [ARCHITECTURE.md](ARCHITECTURE.md)

## Fazlar

| Faz | Kapsam | Durum |
|-----|--------|-------|
| 1 — MVP Offline İndeksleyici | Video klasörü → aranabilir indeks → Türkçe sorgu → kare + zaman | 🔨 Geliştiriliyor |
| 2 — VLM Doğrulama + Zaman Aralığı | Adayları VLM ile doğrula; araç takibi + plaka ile giriş/çıkış zaman çifti | ⏳ Planlandı |
| 3 — Gerçek Pilot | Bir sitenin 1 haftalık arşivi, gerçek sorgularla saha testi | ⏳ Planlandı |

## Kurulum

```bash
pip install -r requirements.txt
```

## Proje günlüğü

Gelişmeler için: [PROGRESS.md](PROGRESS.md)
