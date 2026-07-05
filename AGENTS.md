# AGENTS.md — Vel'Koz Projesi Çalışma Kuralları

Bu dosya, projede çalışan tüm AI agentlerin uyması gereken kuralları tanımlar.

## Proje özeti

Vel'Koz: güvenlik kamerası arşivinde Türkçe doğal dil arama motoru. Lokal-first (KVKK — görüntü müşteri binasından çıkmaz), ucuz donanım hedefi (orta seviye GPU + CPU fallback). Detay: `README.md` ve `ARCHITECTURE.md`.

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
eval/           → dondurulmuş sorgu seti (queries.yaml) + koşucu (run_eval.py)
data/           → test videoları + indeks (git'e girmez)
experiments/    → test çıktıları (deney notları git'e girer, görüntü girmez)
ARCHITECTURE.md → mimari kararlar ve gerekçeleri
PROGRESS.md     → proje günlüğü
```

## Eval disiplini

Retrieval kalitesi `eval/queries.yaml` (dondurulmuş, pre-registration) üzerinden
`eval/run_eval.py` ile ölçülür. Metodoloji: ARCHITECTURE.md §6b. Kurallar: sorgu
seti sonuca bakmadan dondurulur; golden frame_idx'ler küçük resim gözle doğrulanır;
her model/eşik değişikliği bu sabit sette eşleştirilmiş (McNemar) karşılaştırılır.
