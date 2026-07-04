# PROGRESS.md — Gözcü Proje Günlüğü

## 4 Temmuz 2026 (gece) — Olgu B çözüldü: sahne-niyetli frame boost (durum-eki tabanlı)

- AI Engineer tasarımı: niyet sinyali **durum ekiyle** (case morphology), pozisyonla değil —
  -DA/-lA/-In ekli isim adjunct, head = son nominatif isim. "araçlarLA...otopark"→sahne,
  "otoparkTA...insan"→nesne. `query.scene_or_object_intent`. Sınıflandırıcı 12/12 dev.
- Mekanizma: **z-normalize yumuşak frame-boost, yalnız sahne-niyetinde** (`_intent_rerank`,
  λ=1.0). Nesne-niyeti nötr (Faz 1.5 korunur). Hard filtre değil — cross-video relevans ezilmez.
- λ skor-boşluğundan kalibre, dev'de seçilip kilitli sette doğrulandı (dev/kilitli ayrı).
- **Ölçüldü:** dev kırpık-seli düzeliyor (kamyonlu_otopark sahne-kare sırası 10→4, garaj 6→3;
  3 iyileşme, 0 regresyon). **v1 kilitli SIFIR regresyon** (R@5=1.0 sabit), nesne+noobj
  kontroller regres etmiyor.
- **Dürüst nüans:** v2_gold_meva_otopark düzelmedi (8→6) çünkü Olgu B'nin ikinci bileşeni
  kare-açlığı (meva_okul2 statik lot=1 kare) + VIRAT_otopark meşru daha iyi cevap. Boost
  cross-video relevansı doğru şekilde EZMİYOR; kare-açlığı ayrı iş. `experiments/2026-07-04_olgu_b_frame_boost/`

**Sıradaki adım:** statik lotlara daha uzun MEVA segmenti (kare-açlığı) ya da Faz 2 VLM.

## 4 Temmuz 2026 (gece) — Statik lot zenginleştirme: kare-açlığı giderildi (Olgu B kapandı)

- meva_okul2 + meva_otobus MEVA'dan 60→180 sn yeniden çekildi: her biri **1→3 kare**.
- **Ölçüldü:** meva_okul2 "araçlarla dolu açık otopark" sahne-kare sırası **8→5**
  (zenginleşme + boost); kırpık-seli tamamen elendi (v2 Faz 1.5 marjini −0.333→0.000).
  v1 kilitli sıfır regresyon.
- **Nüans (Olgu A tekrarı):** v2 pinpoint golden (frame_idx=0) "rank 9" diyor ama SAHNE
  top-5'te — çapa karesi (60/120sn) rank 5'i alıyor. Ders: statik sahne için pinpoint
  golden yanlış, video+kare metriği doğru. Kalan (VIRAT üstte) = meşru doğru sıralama.
- Detay: `experiments/2026-07-04_lot_zenginlestirme/`

**Sıradaki adım:** Faz 2 VLM re-rank (negatif örtüşme + öznitelik + Olgu A kümeleme).

## 4 Temmuz 2026 (akşam) — Korpus büyütüldü (MEVA): eval artık ayrımcı, 2 gerçek olgu çıktı

- AI Engineer kaynak seçti: **MEVA KF1** (VIRAT halefi, S3 public, HTTP Range destekli —
  Kitware'in no-Range sorununu çözdü). ffmpeg ile URL'den ilk 60 sn çekilip normalize edildi.
- **6 farklı sabit kamera eklendi** (hastane/okul/otobüs/idari açık alan + iç mekan):
  korpus 7→**13 video**, 4080→**4391 vektör**, +54 yeni sahnede insan kırpığı. Statik lotlar
  1 kare verdi (hareket kapısı doğru; park araçları kırpık olarak yakalandı).
- **v1 (kilitli 22 sorgu) çeldiricilere rağmen SAĞLAM:** R@1=0.917, R@5=1.0 DEĞİŞMEDİ.
  MEVA kırpık düzeyinde rekabet ediyor ama doğru cevap kazanıyor → R@5=1.0 artık güvenilir,
  şişme "sahte" değilmiş, embedding gerçekten ayırıyor. (`experiments/2026-07-04_eval_meva/`)
- **v2 (pre-registered, eklemeli) 2 gerçek olgu ifşa etti** (R@1=0.333 — artık zor):
  - **Olgu A:** 8sn kümeleme spesifik golden'ı yutuyor ama DOĞRU video zirvede (retrieval
    hatası değil; `fail_attribution`=dedup/kümeleme doğru işaretledi — golden etiketim spesifikti).
  - **Olgu B (gerçek zaaf):** "araçlarla dolu otopark" sahne sorgusu, araç kırpıklarının seli
    altında rank 8'e gömülüyor. Faz 1.5 marjini v2'de −0.333 (kare > birleşik). Skor-ofsetsiz
    birleşik indeksin sınırı. (`experiments/2026-07-04_eval_v2/`)
- Detay: `experiments/2026-07-04_korpus_buyutme/deney_notu.md`

**Sıradaki adım:** Olgu B — sahne-niyetli sorgularda kareyi kırpığa tercih (niyet tespiti);
AI Engineer danışması gerekir. Gece/trafik için UA-DETRAC (Kaggle token gerekiyor).

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
