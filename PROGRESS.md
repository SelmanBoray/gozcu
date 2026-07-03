# PROGRESS.md — Gözcü Proje Günlüğü

## 3 Temmuz 2026 — Faz 1 MVP kodu tamamlandı

- Ortam kuruldu: `.venv` + CUDA torch (torch 2.12.1+cu126, `cuda: True` — RTX 4070 Laptop).
- **Tüm 8 modül gerçek koda çevrildi** (iskelet kalmadı):
  - `config.py` — tüm eşikler tek yerde; yollar proje köküne sabitlendi (çalışma dizininden bağımsız).
  - `sampler.py` — PyAV çözme, 2 fps aday, hareket kapısı, 60 sn çapa karesi, pHash dedup.
  - `embedder.py` — jina-clip-v2, GPU fp16, toplu görüntü/metin encode, L2-normalize.
  - `thumbs.py` — 480 px JPEG q=80 küçük resim yazıcı.
  - `store.py` — Qdrant lokal mod, `frames` koleksiyonu, deterministik nokta ID (yeniden indeksleme idempotent), zaman/kamera filtreli arama.
  - `query.py` — Türkçe zamansal ayrıştırıcı: "dün gece", "bu sabah", "son N saat", "dün saat 22 civarı" kalıpları + dolgu kelime temizliği. Zaman kelimeleri asla embedlenmez.
  - `search.py` — arama hattı: ayrıştır → embedle → filtreli ara; zaman filtresi boş dönerse filtresiz tekrar + kullanıcıya uyarı.
  - `cli.py` — typer: `index`, `search`, `stats`, `ui` komutları.
  - `viewer.py` — Streamlit: sorgu kutusu → 4 sütun küçük resim ızgarası → "Oynat" ile videoyu o saniyeden izleme.
- Sorgu ayrıştırıcı 6 örnek sorguyla test edildi — zaman aralıkları ve görsel metin ayrımı doğru.
- Payload şemasına `video_path` ve `offset_s` eklendi (viewer'ın videoyu doğru saniyeden açabilmesi için) — ARCHITECTURE.md tablosu güncellendi.
- İlk indeksleme: 3 Teknofest test videosu (`C:/teknofest/testverisi/`) — sonuçlar `experiments/2026-07-03_faz1_ilk_indeksleme/`.

**Sıradaki adım:** Türkçe arama kalite kontrolü (test sorguları), ardından 30–50 sorguluk eval seti (Risk 3).

## 2 Temmuz 2026 — Proje başlangıcı

- Proje kararı verildi: CCTV arşivinde Türkçe doğal dil arama (Conntour muadili, Türkiye'de rakipsiz). LoL AI Koç projesiyle paralel yürüyecek.
- Repo kuruldu: `PycharmProjects/gozcu`, GitHub'a bağlandı.
- Temel dokümantasyon yazıldı: README, AGENTS.md, ARCHITECTURE.md.
- AI Engineer agentiyle mimari kararlar netleştirildi (embedding modeli, kare örnekleme stratejisi, Qdrant şeması, Türkçe sorgu hattı) — detaylar `ARCHITECTURE.md`.
- Paket iskeleti oluşturuldu.

**Sıradaki adım:** Faz 1 MVP — kare örnekleme modülünün implementasyonu ve Teknofest test videolarıyla ilk indeksleme denemesi.
