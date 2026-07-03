# PROGRESS.md — Gözcü Proje Günlüğü

## 3 Temmuz 2026 (akşam) — Gerçek CCTV testi: örnekleyici yeniden tasarlandı, Risk 1 kesinleşti

- AI Engineer link-doğrulamalı gerçek CCTV verisi seçti: VIRAT otopark + kampüs
  (araştırma lisanslı), UCF trafik + gece. `data/cctv_test/` altına indirildi.
- **Kritik hata bulundu:** VIRAT kampüs 22 sn'den 1 kare verdi. Teşhis: pHash dedup
  global hash olduğu için 3 piksellik insana yapısal kör (Hamming=0); hareket kapısı
  eşikleri de küçük özne için 3-7 kat yüksekti.
- **Örnekleyici yeniden tasarlandı** (AI Engineer inceleme + onay): ortalama-normalize
  absdiff, bağlı bileşen gürültü filtresi, birikimli değişim dedup (pHash emekli),
  küresel olay koruması, OSD maskesi, saatlik oran sınırı. Detay: ARCHITECTURE.md §2.
- Sonuç: kampüs 1→31, otopark 3→59 kare. Yürüyen insanlı kareler artık indekste.
- 11 Türkçe sorgu testi + görsel doğrulama: "mavi kamyonet" birebir isabet; gece ve
  trafik sorguları doğru; ama küçük özne sorgularında **Risk 1 kesinleşti** — video içi
  sıralama doğru, videolar arası 0.02–0.07 puanlık sulanma yakın plan videoya yeniriyor.
- **Karar: Faz 1.5 (YOLO-crop embedding) öne çekildi — sıradaki iş.**
- Detaylar: `experiments/2026-07-03_gercek_cctv_testi/deney_notu.md`

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
- Öngörülen risk gerçekleşti ve çözüldü: jina-clip-v2 uzak kodu transformers 5.12 ile uyumsuz (`clip_loss` import hatası) → `transformers<4.50` sabitlendi.
- **İlk indeksleme başarılı:** 3 Teknofest test videosu (kapalı otopark, siyah Togg SUV), 37 kare, GPU'da 10.3 sn.
- **Türkçe arama uçtan uca doğrulandı** (6 sorgu, detay: `experiments/2026-07-03_faz1_ilk_indeksleme/deney_notu.md`):
  - "telefonla konuşan sürücü" → ilk 3 sonuç doğru video, kare görsel olarak doğrulandı (sürücü telefonda). **Çeviri olmadan davranış düzeyinde eşleşme.**
  - "dün gece otoparka giren siyah araç" → zaman filtresi ayrıştı, aralıkta veri yokunca uyarıyla filtresiz geri düşüş çalıştı.
  - Skorlar genel sorgularda dar bantta (0.35–0.39) — tek mekân/tek araç verisinde beklenen; sıralama doğru.

**Sıradaki adım:** 30–50 sorguluk eval seti (Risk 3); küçük özneli gerçek dış mekân CCTV kaydıyla Risk 1 (tam-kare embedding sulanması) testi.

## 2 Temmuz 2026 — Proje başlangıcı

- Proje kararı verildi: CCTV arşivinde Türkçe doğal dil arama (Conntour muadili, Türkiye'de rakipsiz). LoL AI Koç projesiyle paralel yürüyecek.
- Repo kuruldu: `PycharmProjects/gozcu`, GitHub'a bağlandı.
- Temel dokümantasyon yazıldı: README, AGENTS.md, ARCHITECTURE.md.
- AI Engineer agentiyle mimari kararlar netleştirildi (embedding modeli, kare örnekleme stratejisi, Qdrant şeması, Türkçe sorgu hattı) — detaylar `ARCHITECTURE.md`.
- Paket iskeleti oluşturuldu.

**Sıradaki adım:** Faz 1 MVP — kare örnekleme modülünün implementasyonu ve Teknofest test videolarıyla ilk indeksleme denemesi.
