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

- **Model: `qwen2.5vl:3b`** (Ollama, 3.2GB Q4_K_M) — **non-thinking**. 8GB kartta CLIP
  (~2GB) ile eşzamanlı GPU'ya sığar: `offloaded 37/37 layers to GPU` (ölçüldü, CPU offload
  yok). `keep_alive:30m`. **Neden qwen3-vl:2b DEĞİL:** o bir thinking modeli — belirsiz
  kırpıklarda sonsuz düşünme döngüsüne girip n_ctx'i doldurur, boş JSON döner (%33 hata,
  deterministik); `think:false`/`/no_think`/`num_predict` cap hiçbiri bu Ollama build'inde
  çözmedi. Teşhis: `experiments/2026-07-05_vlm_latency/`.
- **Sözleşme: YES/NO VQA (JSON DEĞİL).** Küçük VLM'ler çok-öznitelikli reddetme-yanlı JSON
  şemasında **rubber-stamp'liyor** — kırmızı arabaya "mavi araba? evet, conf 1.0" der (renk
  sıfatını yok sayar). AMA aynı model düz yes/no'da rengi KUSURSUZ ayırır (kırmızı 9/9, mavi
  0/9). Bu yüzden sorgu (nesne, renk) hedeflerine indirgenir (`query.extract_vqa_targets`)
  ve verifier **iki ayrı atomik yes/no** sorar: "Is there a {obj}?" + "Is the {obj} {color}?".
  `num_predict:4` kaçak üretimi yapısal olarak imkânsız kılar (thinking-loop panzehiri).
- **VLM'in Türkçesine güvenilmez** → hedefler deterministik sözlükle İngilizceye çevrilir
  (`extract_vqa_targets`; mavi→blue, köpek→dog). Zor kavram (köpek) sınıf isminden (insan)
  öncelikli — CLIP'in kaçırdığı asıl özne odur. +0 VRAM, lokal.
- **Tam çözünürlük:** kırpık thumb 36px olabiliyor (renk/detay yok) → orijinal videodan
  bbox ile yeniden kırpılır (`recrop.py`, 384px). Renk/detay thumbnail'de kaybolur.
- **Koşullu tetik:** yalnız renk/zor-kavram sorgusu (`query.needs_vlm`) + Ollama ayakta.
  Nesne/sahne sorguları zaten yüksek recall → VLM vergisi ödenmez.
- **Renk insanlarda GİYSİYE işaret eder:** "kırmızı kıyafetli adam" ≠ kırmızı gövdeli adam.
  İnsan-tipi nesnede (`verifier._PERSON_OBJECTS`) renk sorusu "Is this man **wearing** red
  clothing?"; araçta "Is the car red in color?" (gövde). Yoksa herkes "no" döner (kimse
  kırmızı değil) ve kırmızı giyen kişi hiç öne çıkmaz.
- **Verdict:** `{object_present, color_match, confidence}`; `confidence` = 1.0 (nesne var) /
  0.0 (yok) — ayrım booleanlarda. VLM hatası → None (dokunma).
- **Füzyon = FİLTRE (VLM CLIP'in yanlış adayını ANA IZGARADAN çıkarır):** eşleşmeyen aday
  elenir (elenenler expander'ında görünür), hepsi elenirse → bulunamadı. Elenme koşulu:
  **nesne yok** (`confidence<vlm_drop_below`, her iki modda) **VEYA renk uymadı**
  (`color_match is False`, renk sorgusunda — qwen2.5vl renk ayrımı ölçülmüş güvenilir
  red/blue 9/9↔0/9). Böylece "kırmızı kıyafetli adam" korpusta kırmızı giyen yoksa kırmızı
  ARABA göstermez → bulunamadı; kırmızı arabalar expander'da. Kalan survivor'lar
  `z(cos)+β·conf` ile rerank. **`vlm_top_n=default_top_k`** → gösterilen her aday doğrulanır
  (kuyruk sızıntısı yok). VLM hatası (None) → dokunma. Ayarlar: `vlm_top_n`, `vlm_drop_below`,
  `vlm_beta`. NOT: eski "renk güvenilmez→rerank-only, düşürme yok" tasarımı qwen3-vl JSON
  rubber-stamp içindi; model+sözleşme değişince renk hard-filtreye terfi etti.
- **Doğrulama görüntüsü HİBRİT (ölçümle kalibre, 5 Tmz):** küçük/bulanık kırpıkta varlık
  yes-bias'ı (insanda "köpek", arabada "adam" yanlış-pozitif) + çok-nesne kapsamı (köpek
  insan-bbox'ı dışında) tight-kırpıkta çözülemiyordu. Çözüm cinse göre görüntü seç:
  - **Renk/öznitelik → tight-kırpık** (özne kadrajı doldurur, renk net, hızlı). Tam-kare
    denendi ama 768px'e küçültülünce uzak araç görünmez olup gerçek eşleşmeyi eledi
    (siyah SUV 7→2 fazla-eleme, gözle doğrulandı) → tight-kırpık korundu.
  - **Zor-kavram (köpek/yağmur) → kutusuz TAM-KARE** (`recrop.vlm_frame_for_hit`, uzun kenar
    `vlm_frame_max_side=768`): VLM tüm sahneyi görür → "bu görüntüde köpek yok" diyebilir.
    Ölçüldü: "köpek gezdiren adam" tight-kırpıkta 1 yanlış-pozitif → tam-karede **bulunamadı** (0).
- **LLM sorgu-ayrıştırıcı (Qwen3-4B) ERTELENDİ:** kural-bazlı çalışıyor + 4B+3B+CLIP
  aynı VRAM'e sığmaz.
- **Ölçüldü (5 Tmz, deterministik):** ayrım kusursuz — "red car" present+color 9/9, "blue
  car" color 0/9 (renk ayrımı), "dog" present 0/9 (nesne ayrımı). Uçtan uca: "köpek gezdiren
  insan"→hedef `dog`, CLIP'in 5 adayından **4 yanlış-pozitif elendi**, 1 köpekli kaldı.
- **Güvenilirlik:** non-thinking + `num_predict:4` → **%100 geçerli** (thinking-loop yok).
  Latency atomik yes/no ~0.3-2s (warm); renk sorgusu kart başına 2 çağrı, top-12 ~30-50s →
  per-item streaming maskeler. `vlm_timeout_s=30` (yalnız emniyet, warmup cold-path'i yok eder).
- **VRAM co-residency + zarif-çökme (5 Tmz):** 8GB'da CLIP (~2GB) + qwen2.5vl (2.9GB) + arka
  plan GPU uygulamaları ara sıra OOM → Ollama çöker. CLIP'i CPU'ya almak DENENDİ (jina text
  tower CPU'da ~13s/çağrı — interaktif aramayı öldürür, elendi). Çözüm: çökmeyi ÖNLEMEK
  yerine ZARİFÇE KURTARMAK — (1) **self-healing `is_available`**: kısa TTL + hata-anında reset
  ("bir kez False→hep False" bug'ı bitti, çökme sonrası ilk sorgu iyileşir); (2) **ayrık UX**:
  tekil verify-fail = kart rozeti, TOPTAN VLM-down = global banner "VLM kapalı → yalnız CLIP"
  (`SearchOutcome.vlm_unavailable`); (3) **açılış warmup** (arka plan thread, ilk sorgu cold
  değil); (4) **supervised Ollama** (masaüstü uygulaması server'ı otomatik yeniden başlatır).

### 8b. Async rafine — progressive render (viewer, 5 Temmuz 2026)

VLM latency ~4-6s/görüntü (bottleneck token DEĞİL, **görüntü prefill'i** — num_predict/boyut
düşürme fayda etmedi), top-8 ~30s. İnteraktif için bloklamamalı. AI Engineer: gerçek thread
DEĞİL, **progressive render** (tek-kullanıcı demo için thread kırılganlığı gereksiz).

- `search()` bölündü: `search(use_vlm=False)` (CLIP, ~0.1s) + public `refine_vlm(outcome)`
  (VLM ikinci faz). `search(use_vlm=True)` = ikisi (CLI senkron kalır).
- **Viewer — per-item streaming:** CLIP kartları ANINDA `⏳` ile gösterilir; `stream_verify`
  her adayı SIRAYLA doğrular ve `on_verdict(i,hit)` callback'i o kartın rozetini CANLI
  doldurur (⏳→✅ renk/✅ conf/🚫). Bitince final reflow (yeniden sıralama + elenenler
  `expander`'da — "VLM elenenler", CLIP ham recall'ı görünsün). `search.py` bölündü:
  `verify_top_n` (callback'li) + `_fuse_verdicts` (füzyon) + public `stream_verify`.
- **Cache:** `@st.cache_data` (CLIP) + **`session_state` (VLM sonucu, sorgu-anahtarlı)** —
  streaming canlı render gerektirdiği için VLM tarafı cache_data yerine session_state'te;
  Oynat tıklaması/rerun VLM'i yeniden koşmaz. `st.form` submit — keystroke başına arama yok.
- **Güvenilirlik (streaming):** yavaş bir VLM çağrısı görünür ilerlemeyi kısa süre
  durdurabilir; `vlm_timeout_s=20` bu takılmayı kısaltır (yes/no ~2-5s; hang'ler 20s'de
  kapanır → o kart "doğrulanamadı", akış devam). Not: model 5 Tmz qwen2.5vl:3b'ye geçti,
  yes/no VQA sözleşmesi — §8. Streaming mimarisi aynen geçerli.
- Prompt sıkılaştırıldı: "açıklamanın HER parçası görünmeli" → negasyon kısmi-eşleşme
  false-accept'i düzeldi (köpek/yağmur → BULUNAMADI, öznitelik bozulmadan).
- Paralel VLM YOK (Ollama tek model, KV slot şişmesi → OOM riski). Sıralı + streaming.
