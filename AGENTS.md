# AGENTS.md — Gözcü Projesi Çalışma Kuralları

Bu dosya, projede çalışan tüm AI agentlerin uyması gereken kuralları tanımlar.

## Proje özeti

Gözcü: güvenlik kamerası arşivinde Türkçe doğal dil arama motoru. Lokal-first (KVKK — görüntü müşteri binasından çıkmaz), ucuz donanım hedefi (orta seviye GPU + CPU fallback). Detay: `README.md` ve `ARCHITECTURE.md`.

## Çalışma kuralları

1. **Dokümantasyon otomatik güncellenir:** Her teknik adımdan/işlemden sonra sormadan `AGENTS.md` (kural değiştiyse) ve `PROGRESS.md` (her zaman) güncellenir.
2. **Teknik kararlar tek başına verilmez:** Mimari/model/kütüphane seçimi gibi teknik kararlarda AI Engineer agenti çağrılır.
3. **Kod yorumları konu başlıkları şeklinde yazılır:** Kod blokları `# ── Bölüm adı ──` tarzı başlık yorumlarıyla ayrılır; satır satır açıklama değil, bölüm başlığı mantığı.
4. **Gerekli pip paketleri sorulmadan kurulur.**
5. **Test disiplini:** Her anlamlı değişiklikte eldeki tüm test videolarıyla test edilir; çıktılar `experiments/` altında sınıflandırılmış klasörlere kaydedilir (`experiments/<tarih>_<konu>/`).
6. **Kalite > hız:** Örnekleme/FPS optimizasyonları tespit kalitesini bozamaz.
7. **Dil:** Dokümantasyon ve kullanıcıya dönük metinler Türkçe; kod tanımlayıcıları (değişken/fonksiyon adları) İngilizce.

## Dizin yapısı

```
gozcu/          → Python paketi (pipeline modülleri)
data/           → test videoları (git'e girmez)
experiments/    → test çıktıları (git'e girmez)
ARCHITECTURE.md → mimari kararlar ve gerekçeleri
PROGRESS.md     → proje günlüğü
```
