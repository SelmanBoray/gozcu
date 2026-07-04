# ARCHITECTURE.md — Mimari Kararlar

> AI Engineer danışmasıyla netleştirildi — 2 Temmuz 2026.
> Her karar: **seçim + gerekçe + yedek plan** formatında.

## 1. Embedding modeli: `jinaai/jina-clip-v2` (doğrudan Türkçe)

- **Seçim:** Sorgu çeviri YOK — Türkçe sorgu doğrudan embedlenir. Model 89 dil destekli XLM-RoBERTa metin kulesi + EVA02-L görüntü kulesi (512×512 giriş — geniş açı otopark planlarında küçük araç/insan için avantaj). Vektör boyutu **1024** (tam boyut; bu ölçekte depolama sorun değil, şimdiden kırpma yok).
- **VRAM:** fp16'da ~2 GB → 6 GB kartta YOLO ile yan yana çalışır.
- **CPU fallback:** Aynı modelin yayınlanmış **ONNX int8** ağırlıkları (onnxruntime). Aynı 1024-dim vektörler → GPU'lu ve CPU'lu makine aynı indeksi paylaşır. İndeksleme offline/gecelik iş olduğu için CPU'da ~1 kare/sn kabul edilebilir.
- **Yedek plan:** Türkçe recall zayıf çıkarsa → sorguyu çevir + `google/siglip2-so400m-patch14-384` (1152-dim). Karar his ile değil, **eval seti ölçümüyle** verilecek (bkz. Risk 3).

## 2. Kare örnekleme stratejisi

> **3 Temmuz 2026 revizyonu:** İlk tasarımdaki pHash dedup, gerçek CCTV testinde
> (VIRAT kampüs) uzaktaki küçük özneye yapısal olarak kör çıktı — global 64-bit hash,
> 3 piksellik insan hareketinde Hamming=0 verip 44 kareyi 1'e indirdi. Mekanizma
> değiştirildi. Teşhis: `experiments/2026-07-03_gercek_cctv_testi/`

1. PyAV ile çözümle, **2 fps** aday kare (0.5 sn'de bir).
2. **Küçültme + normalizasyon:** 320×180 gri + 5×5 Gaussian blur + **ortalama-normalizasyon**
   (AGC pompalaması/bulut geçişi tüm kareyi "değişti" saymasın).
3. **OSD maskesi:** son adaylarda >%50 sıklıkla değişen pikseller (yanık DVR saati vb.)
   tüm sayımlardan dışlanır.
4. **Hareket kapısı:** önceki adayla normalize absdiff **> 10**; değişim maskesine
   **bağlı bileşen filtresi** (≥ 5 px — yağmur/gren blob değildir, insan blobdur);
   kalan oran **> 0.0005** ise aday tutulur.
5. **Birikimli değişim dedup (pHash yerine):** son TUTULAN kareye göre aynı maskeli
   değişim oranı **> 0.001** olmalı. Gerekçe: yürüyen uzak insan kareler boyunca fark
   biriktirir ve eşiği aşar; sıfır-ortalamalı gürültü birikmez. pHash yalnız metadata.
6. **Küresel olay koruması:** referansa göre ham değişim > %25 ise (ışık/pozlama
   sıçraması) tek kare tutulur, referans sıfırlanır — akış tutulmaz.
7. Hareket olmasa da **60 sn'de bir çapa karesi** zorla tut (sahne kapsaması).
8. **Oran sınırı:** kamera başına saatte en çok **600** kare (çapalar muaf) — patolojik
   kamera (yağmur, bozuk sensör) felaketi zarif bozulmaya çevrilir.
9. Tutulanları toplu embedle (GPU batch 16 / CPU batch 4); **480 px JPEG (q=80)** küçük resim yaz.

**Doğrulama (VIRAT/UCF gerçek CCTV):** otopark 3→59, kampüs 1→31 kare; yürüyen
insanlı kareler artık indekste.

**İndeks matematiği:** 2 fps = 7.200 aday/saat. Otopark kamerası gerçekçi ortalama **~300–500 tutulan kare/saat/kamera ≈ 8–12k/gün/kamera**. Pilot (1 hafta × 4 kamera) ≈ **250–350k vektör ≈ 1–1.4 GB** (kuantizasyon öncesi) — rahat.

## 3. Vektör DB: Qdrant (lokal)

Koleksiyon `frames`: boyut **1024**, mesafe **Cosine**, HNSW **varsayılanları** (m=16, ef_construct=100 — birkaç milyon vektöre kadar yeterli) + **scalar int8 kuantizasyon (always_ram=true)** ile RAM ~4× düşer.

| Payload alanı | Tip | Not |
|---|---|---|
| `video_id` | keyword | |
| `video_path` | keyword | mutlak yol — viewer videoyu buradan açar |
| `camera_id` | keyword | **indeksli** |
| `ts` | float (epoch UTC) | **indeksli**, aralık filtresi |
| `offset_s` | float | videonun başından saniye — viewer `start_time` için |
| `frame_idx` | int | |
| `motion_score` | float | |
| `phash` | keyword | |
| `thumb_path` | keyword | |
| `track_ids`, `plates`, `yolo_classes` | int[] / keyword[] | **Faz 2 için rezerve** |

## 4. Türkçe sorgu hattı

- **MVP: kural bazlı ayırıcı.** Regex + `dateparser` (tr) zamansal ifadeleri ("dün gece", "bu sabah 8'den sonra") ayıklar → Qdrant `ts` aralık filtresi. Kalan görsel ifade ("beyaz Transit") embedlenir. **Zaman kelimeleri asla embedlenmez — en büyük kalite kaldıracı bu.**
- **Faz 2:** Lokal LLM sorgu ayrıştırıcı (Qwen3-4B, Ollama, ~3 GB q4) → `{visual_tr, visual_en, time_range, camera}` JSON; iki dil varyantı da embedlenip max skor alınır.
- **KVKK notu:** Sorgular görüntü içermez ama **kişisel veri içerebilir** (plaka, isim). Bu yüzden bulut çeviri API'si kullanılmaz — "hiçbir veri dışarı çıkmaz" hem satış argümanı hem tasarım ilkesi. 4B lokal model sorgu çevirisi için fazlasıyla yeterli.

## 4b. Bulunamadı kapısı (yolo_class sertlik kapısı — 4 Temmuz 2026)

Eval bulgusu: sistemin güvenilir "bulunamadı" sinyali yok — CLIP kosinüsü sorgular
arası kalibre değil, korpusta olmayan nesne (kırmızı bisiklet) makul-ama-yanlış
sonuç döndürüyordu (min-poz 0.282 < max-neg 0.36). Çözüm skor eşiği DEĞİL, **YOLO
envanteri:** sorgu tespit edilebilir bir sınıf istiyorsa (`query.extract_object_intent`,
Türkçe eş anlamlı + ASCII-fold ile çekim ekine dayanıklı) ve o sınıf korpusta hiç
yoksa (`store.available_object_classes`) → CLIP'e sormadan boş dön.

- **Yalnız prod hattında** (`source=None`); ablation koşuları kapıyı atlar (ham
  retrieval ölçümü korunur).
- **Kalibrasyon yok** — envanter kontrolü, eşik değil. Korpus büyüdükçe otomatik uyarlanır.
- **Ölçülen etki:** pozitif metrikler değişmedi (yanlış kapı = 0), negatif yakalama
  0.25 (`neg_bisiklet` boş döndü). Kapsam: yalnız "tespit edilebilir sınıf yok" vakası.
  Öznitelik/hava (yağmur, kar) ve tespit edilemeyen nesne (köpek) → Faz 2 VLM'e kalır.
- Detay: `experiments/2026-07-04_bulunamadi_kapisi/`

## 5. Modül yapısı (düz paket — derin iç içe klasör yok)

| Modül | Sorumluluk |
|---|---|
| `config.py` | pydantic-settings: yollar, model id'leri, tüm eşikler |
| `sampler.py` | video çözme + 2fps örnekleme + hareket kapısı + birikimli değişim dedup → FrameRecord akışı |
| `detector.py` | **Faz 1.5:** yolo11m insan/araç tespiti, statik bastırma, insan öncelikli sınır, kırpma |
| `embedder.py` | jina-clip-v2 sarmalayıcı, toplu görüntü/metin encode, cuda/cpu otomatik |
| `thumbs.py` | kare + kırpık küçük resim yazıcı |
| `store.py` | Qdrant şema kurulumu, kare + kırpık upsert, filtreli arama |
| `query.py` | Türkçe zamansal ayrıştırıcı → (görsel_metin, filtreler) |
| `search.py` | sorgu embed + Qdrant çağrısı + kare tekilleştirme + zaman kümeleme |
| `viewer.py` | Streamlit: sorgu kutusu → küçük resim ızgarası → tıkla, videoyu o saniyeden aç |
| `cli.py` | typer: `gozcu index <klasör>`, `gozcu search "<sorgu>"`, `stats`, `ui` |

## 5b. Faz 1.5 — YOLO kırpık embedding (3 Temmuz 2026'da uygulandı)

Risk 1 ölçülüp kesinleşince öne çekildi. Tutulan her karede yolo11m (imgsz=1280 —
küçük özne için çözünürlük model boyutundan önemli) insan/araç tespit eder; kırpıklar
(%20 bağlam payı + kareye tamamlama, orijinal çözünürlükten) aynı jina-clip-v2 ile
embedlenip AYNI koleksiyona `source:"crop"` olarak yazılır — ana kareye işaret eder.

- Güven eşiği düşük (insan 0.15 / araç 0.25), `yolo_conf` payload'da: arama anında
  yükseltilebilir; indekste olmayan tespit kurtarılamaz.
- **Statik bastırma:** önceki tutulan kareyle aynı-sınıf IoU > 0.85 → atla. Park
  halindeki araç yalnız ilk görünümünde vektör alır.
- **İnsan öncelikli sınır (kare başına 24):** salt güven sıralaması park halindeki
  araçların insanları kesmesine yol açıyordu — insan CCTV aramada en nadir/kıymetli sınıf.
- Skor ofseti yok: nesne sorguları kırpığı, sahne sorguları kareyi doğal seçiyor.
- Doğrulama: "uzakta tek başına yürüyen insan" → ilk 3 sonuç da doğru insan kırpığı
  (önce: yakın planlı yanlış videoya kayıptı). `experiments/2026-07-03_faz15_yolo_crop/`
- **Bilinen sınır (4 Temmuz 2026, korpus büyütme v2'de ölçüldü):** "skor ofseti yok →
  kendi kendini düzeltir" varsayımı, sahne sorgusu bir NESNE KELİMESİ içerince bozuluyor.
  "araçlarla dolu açık otopark" → "araç" kırpıklarının seli doğru sahne karesini rank 8'e
  itiyor (Faz 1.5 marjini o sorguda −0.333). Çözüm adayı: sahne-niyetli sorgularda (yer
  adı) kareyi tercih / niyet tespiti. `experiments/2026-07-04_korpus_buyutme/`

## 6. En büyük 3 risk

1. **Tam-kare embedding küçük özneleri sulandırır** — kırmızı montlu adam 1080p geniş planda 40 px'dir; recall hayal kırıklığı yaratırsa Faz 2'deki YOLO crop-embedding'i (insan/araç kırpıklarını ayrı vektör olarak indeksle) **Faz 1.5 olarak öne çek**.
   **→ 3 Temmuz 2026: KESİNLEŞTİ ve AYNI GÜN ÇÖZÜLDÜ.** Video içi sıralama doğruydu
   ama videolar arası 0.02–0.07 puanlık sulanma küçük özneli kareyi yeniriyordu.
   Faz 1.5 (YOLO kırpık embedding, §5b) uygulandı: "uzakta tek başına yürüyen insan"
   artık ilk 3'te de doğru insan kırpığını getiriyor. Kalan sınır: ~10 px altı insan
   YOLO tespit tabanının da altında (VIRAT kampüs segmenti) — gerekirse SAHI döşeme.
2. **Zaman damgası gerçeği** — DVR dosya adları/mtimes yalan söyler; taban zaman yanlışsa her "dün gece" cevabı yanlış. Erken doğrula: görüntüye gömülü OSD saatini OCR ile çapraz kontrol et.
3. **Eval seti yoksa karar da yok** — 1. haftada test videolarından **30–50 Türkçe sorgu → doğru kare** çifti oluştur; her model/eşik kararı (SigLIP2 yedeğine geçiş dahil) buna karşı ölçülür. Ayrıca: embedding ve küçük resimler de KVKK kapsamında kişisel veridir — saklama/silme politikası ilk günden tasarlanır.
   **→ 4 Temmuz 2026: KAPATILDI.** 22 sorgu tiered eval (`eval/queries.yaml`, dondurulmuş),
   koşucu `eval/run_eval.py`. Metrik etiket tipine eşlendi: sınıf/sahne → video-düzeyi
   Recall@k, ubik/davranış → gözle doğrulanmış golden-frame, renk → advisory (recall
   dışı). Kare vs kırpık ayrı koşulur (`source` filtresi eklendi). İlk ölçüm:
   skorlanabilir R@5=1.0, golden R@1=1.0, **Faz 1.5 marjini +0.333**. **Açık bulgu:
   negatif ayrımcılık çöktü (min-poz 0.282 < max-neg 0.36) — güvenilir "bulunamadı"
   sinyali yok, Faz 2 VLM doğrulayıcı gerekçesi.** Detay: `experiments/2026-07-04_eval/`.

### 6b. Eval metodolojisi (4 Temmuz 2026)

| Sorgu tipi (`gt_type`) | Metrik | Neden |
|---|---|---|
| `class` (otobüs, motosiklet) | video-düzeyi Recall@k | kesin sınıf sayımından türetilir, `|relevant|≤2-3` |
| `scene` (gece, kapalı otopark) | video-düzeyi Recall@k | sahne-özel, otomatik doğrulanabilir |
| `golden` (telefonlu sürücü, yürüyen insan) | tek kareye Recall@1/MRR | ubik sınıf video-düzeyinde triviyal; kare gözle doğrulandı |
| `advisory` (siyah SUV, mavi kamyonet) | yalnız top-1 denetim | renk GT'si otomatik doğrulanamaz — recall'a girmez |
| `zaman` (dün gece, son 3 saat) | parse + geri düşüş | zaman ayrıştırma sınır davranışı |
| `negatif` (kırmızı bisiklet, kar) | ayrımcılık (skor dağılımı) | mutlak eşik yok; CLIP kosinüsü sorgular arası kalibre değil |

Kurallar: set sonuca bakmadan **dondurulur** (pre-registration); her agregata **Wilson
%95 CI** basılır (n≈12 → gösterge, benchmark değil); değişiklikler sabit sette
**eşleştirilmiş (McNemar)** karşılaştırılır, bağımsız oran değil.

## 7. Niyet-koşullu sıralama (Olgu B çözümü — 4 Temmuz 2026)

Korpus büyütme v2'nin bulduğu Olgu B: sahne sorgusu bir NESNE KELİMESİ içerince
("araçlarla dolu açık otopark"), o nesnenin kırpıkları sele dönüp doğru sahne karesini
gömüyor. "Skor ofseti yok → kendi kendini düzeltir" varsayımının sınırı.

**Niyet sinyali = durum eki (case morphology), pozisyon değil** (`query.scene_or_object_intent`).
Türkçe serbest sözcük dizilişli ama çekim eki role'ü kodlar: -DA/-DAn/-lA/-In ekli isim
adjunct'tır, head = son NOMİNATİF içerik ismi. "araçlar**LA** ... otopark"→otopark=head=sahne;
"otopark**TA** ... insan"→insan=head=nesne. Sahne-niyeti yalnız head sahne kelimesiyse
(sadece geçmesiyle değil — "otopark bariyeri" tuzağı). Sınıflandırıcı doğruluğu ayrı
raporlanır (12/12 dev).

**Mekanizma = z-normalize yumuşak frame-boost, YALNIZ sahne-niyetinde** (`search._intent_rerank`):
overfetch havuzunda `_rank = z(cos) + λ·[source==frame]`, `λ=1.0` (`scene_boost_lambda`).
Nesne-niyeti nötr (Faz 1.5 kazanımı korunur). Hard filtre DEĞİL (recall uçurumu riski) —
yumuşak: güçlü kırpık cosine'i hâlâ kazanır, meşru cross-video relevans ezilmez.

- **λ kalibrasyonu:** metrikten değil skor-boşluğundan; robustluk bandı (λ/2, λ, 1.5λ)
  boyunca kazanç monoton; dev'de seçilip kilitli sette doğrulandı.
- **Ölçülen etki:** dev kırpık-seli vakaları düzeliyor (kamyonlu_otopark kare sırası
  10→4, garaj 6→3; 3 iyileşme 0 regresyon). v1 kilitli **sıfır regresyon** (R@5=1.0 sabit),
  nesne+noobj kontroller regres etmiyor.
- **Sınır:** Olgu B'nin kare-açlığı bileşeni (statik lot=1 kare) boost'un işi değil —
  ayrı iş (daha uzun segment). Detay: `experiments/2026-07-04_olgu_b_frame_boost/`

## 8. Faz 2 — VLM doğrulayıcı (retrieve-then-verify, 4 Temmuz 2026)

Eval'in CLIP ile kapatılamayan açıkları: negasyon örtüşmesi (köpek/yağmur/kar — kapının
çözemediği hava/öznitelik/tespit-dışı nesne) ve renk doğrulama (siyah SUV, mavi kamyonet).
Çözüm: CLIP top-N adayı, küçük bir VLM ile TAM ÇÖZÜNÜRLÜKTE doğrulanır.

- **Model: `qwen3-vl:2b`** (Ollama, 1.9GB Q4_K_M). 8GB kartta CLIP (~2GB) ile eşzamanlı
  sığan tek sağlam seçenek (qwen2.5vl:3b Ollama'da CPU'ya düşüyor, 4b 6GB ister).
  `keep_alive:30m` — swap thrash önle.
- **VLM'in Türkçesine güvenilmez** → sorgunun görsel kelimeleri deterministik sözlükle
  İngilizceye çevrilir (`query.translate_visual`; mavi→blue, köpek→dog). +0 VRAM, lokal.
- **Tam çözünürlük:** kırpık thumb 36px olabiliyor (renk/detay yok) → orijinal videodan
  bbox ile yeniden kırpılır (`recrop.py`). Renk/detay thumbnail'de kaybolur.
- **Koşullu tetik:** yalnız renk/zor-kavram sorgusu (`query.needs_vlm`) + Ollama ayakta.
  Nesne/sahne sorguları zaten yüksek recall → VLM vergisi ödenmez.
- **Yapılandırılmış verdict** (yes-bias'a karşı): `{object_present, color_match, confidence}`,
  reddetme-yanlı İngilizce prompt (`format:json`, temp 0).
- **Füzyon (VLM CLIP'i ezmez, düzeltir):** yüksek-güvenli red (`object_present=false &
  conf>τ`) → düşür; öznitelik → sınırlı rerank `z(cos) + β·conf·[match]`. VLM hatası →
  dokunma (CLIP sıralaması korunur). Ayarlar: `vlm_top_n`, `vlm_confidence_tau`, `vlm_beta`.
- **Renk'e körü körüne güvenme:** VLM'ler mavi/cyan'da zayıf (~%56 F1) → önce renk-precision
  ölçülür; düşükse renk advisory-with-confidence kalır, hard-filtre edilmez.
- **LLM sorgu-ayrıştırıcı (Qwen3-4B) ERTELENDİ:** kural-bazlı çalışıyor + 4B+2B+CLIP
  aynı VRAM'e sığmaz.
- **Ölçüldü (gözle doğrulandı):** negasyon güçlü — "köpek gezdiren adam"→BULUNAMADI,
  yağmur 0/6 kabul (kapının çözemediğini VLM kapattı); renk — VLM CLIP'in hatalarını
  düzeltiyor ("mavi kamyonet"te CLIP 4 aday, VLM 1 gerçek maviyi onayladı, 3 sahteyi
  reddetti; siyah/beyaz/mavi 3 renk de gözle doğru). Pozitif kontrol otobüs 5/5, kamyon
  4/4 (yanlış-reddetme yok). Dürüst sınır: kar 2/6 false-accept (açık zemin→"kar"),
  küçük/bulanık insan over-reject (ama koşullu tetik bunları dışlıyor).
- **Güvenilirlik:** CLIP+VLM 8GB'de eşzamanlı → çağrı spike'ları; `vlm_timeout_s=45` +
  tek retry ile tek-sorgu (prod) 0/8 hata. Detay: `experiments/2026-07-04_faz2_vlm/`.
