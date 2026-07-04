# PROGRESS.md — Gözcü Proje Günlüğü

## 4 Temmuz 2026 (öğleden sonra) — Bulunamadı kapısı: "bisiklet yok" artık boş dönüyor

- Eval'in en kritik açığı (güvenilir "bulunamadı" sinyali yok) hedefli çözüldü.
  Skor eşiği yerine **YOLO envanteri**: sorgu tespit edilebilir bir sınıf istiyorsa
  ve korpusta o sınıf hiç yoksa → CLIP'e sormadan boş dön.
- `query.extract_object_intent` (Türkçe eş anlamlı + ASCII-fold, çekim ekine dayanıklı),
  `store.available_object_classes` (önbellekli envanter), `search` kapısı (yalnız prod
  hattı; ablation atlar), cli/viewer "BULUNAMADI" mesajı.
- **Ölçüldü (aynı dondurulmuş set, eşleştirilmiş):** pozitif metrikler DEĞİŞMEDİ
  (R@1=0.917, R@5=1.0 — yanlış kapı = 0), negatif yakalama 0.25 (`neg_bisiklet` boş).
- Dürüst sınır: kapı yalnız "tespit edilebilir sınıf yok"u çözer; köpek (YOLO sınıfı
  değil) + yağmur/kar (öznitelik) geçti → kalan örtüşme -0.078 Faz 2 VLM'e kalır.
- Uçtan uca: `search "kırmızı bisiklet süren çocuk"` → "Korpusta 'bisiklet' tespit
  edilmedi". Detay: `experiments/2026-07-04_bulunamadi_kapisi/`

**Sıradaki adım:** Faz 2 VLM re-rank (öznitelik/hava/eylem doğrulaması) ya da tespit
sınıflarını genişletme (köpek/kedi → kapı kapsamı büyür).

## 4 Temmuz 2026 — Eval seti kuruldu (Risk 3 kapandı) + Faz 1.5'in değeri ölçüldü

- AI Engineer eval tasarımını inceledi, naif "video-düzeyi recall"i 3 yerde kırdı:
  golden-frame ubik/davranış için birincil; renk GT'si otomatik doğrulanamaz →
  advisory; kare vs kırpık ayrı koşulmalı. Buna göre 22 sorgu, tiered `queries.yaml`
  **dondurularak** kuruldu (golden frame_idx'ler küçük resim GÖZLE doğrulandı).
- `store.search`/`search`'e `source` filtresi eklendi (frame/crop ablation için).
  `eval/run_eval.py`: 3 yönlü koşu, Wilson CI, golden fail_attribution, Gap 1 denetimi.
- **Kanıt — Faz 1.5'in değeri artık bir sayı:** R@5 yalnız-kare 0.667 → birleşik
  **1.0** = **+0.333 marjinal katkı** (yalnız-kırpık 0.917). Kırpık nesne sorgularını
  taşıyor. Golden R@1=1.0 (3 gözle-doğrulanmış kare de rank-1).
- **Boşluk 1 çözüldü:** "gece" embed edildi (→ucf_gece rank-1), "dün gece" parse
  edildi, "son 3 saat" filtresiz geri düştü. Sınır davranışı doğru.
- **KRİTİK DÜRÜST SINIR:** negatif ayrımcılık çöktü — min-pozitif 0.282 < max-negatif
  0.36 ("köpek gezdiren **adam**" → insan'a eşleşti). **Güvenilir "bulunamadı" sinyali
  yok.** Faz 2 VLM doğrulayıcının somut gerekçesi. Kısa vade: yolo_class sertlik kapısı.
- Dürüst çerçeve: video-düzeyi R@5=1.0 şişmiş (4 sahne görsel olarak çok farklı);
  gerçek zorluk golden pinpoint + negatif ayrımda. n=12 → CI [0.757, 1.0], gösterge.
- Detay: `experiments/2026-07-04_eval/deney_notu.md`

**Sıradaki adım:** "Bulunamadı" mekanizması (yolo_class sertlik kapısı / VLM) — en
yüksek öncelikli açık. Ardından korpus büyütme (benzer sahnelerle gerçek ayrım ölçümü).

## 3 Temmuz 2026 (gece) — Faz 1.5 tamamlandı: YOLO kırpık embedding sulanmayı çözdü

- AI Engineer tasarım danışması → yolo11m@1280, düşük güven eşiği, %20 pay + kare
  kırpma, statik IoU bastırma, tek koleksiyon `source: frame|crop`.
- `detector.py` yazıldı; store/search/cli/viewer kırpık desteği aldı; arama tarafına
  kare tekilleştirme + 8 sn zaman kümeleme eklendi.
- 2 hata bulunup çözüldü: (1) CPU-only torchvision → cu126'dan yeniden kuruldu;
  (2) kırpık sınırı güven sıralamasıyla insanları kesiyordu (VIRAT'ta 0 insan kırpığı)
  → sınır statik bastırma sonrası + insan öncelikli yapıldı → 132 insan kırpığı.
- **Önce/sonra:** "uzakta tek başına yürüyen insan" Faz 1'de yanlış videoya kayıptı
  (0.228 vs 0.155); şimdi ilk 3 sonucun 3'ü de doğru insan kırpığı (0.363) — kırpık
  küçük resimleriyle görsel doğrulandı. İndeks: 647 kare + 3.323 kırpık.
- Dürüst sınır: VIRAT kampüs segmentinde insan kırpığı yok — ham YOLO bile bulamıyor
  (<10 px); boru hattı değil veri sınırı. Gerekirse SAHI döşeme (Faz 2 eval kararı).
- Detay: `experiments/2026-07-03_faz15_yolo_crop/deney_notu.md`

**Sıradaki adım:** Eval seti (30–50 Türkçe sorgu → doğru kare) — korpus ve hat hazır.

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
