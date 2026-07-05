# PROGRESS.md — Gözcü Proje Günlüğü

## 5 Temmuz 2026 (gece-6) — Otonom loop: streaming decouple (freeze çözüldü) + YOLO-skip latency

- Selman "tüm önerileri uygula → düşün → devam et, loop'ta çalış" dedi. AI Engineer'a
  streaming deseni + sonraki-round refleksiyonu danışıldı. Kritik meta: **loop'un objektif
  bir eval skoruna ihtiyacı var** (yoksa his'le sürüklenir) → eval sıradaki #1.
- **Streaming render-block ÇÖZÜLDÜ (freeze gitti):** senkron `stream_verify` ~40s ana thread'i
  bloke edip tarayıcıyı donduruyordu. Çözüm (AI Engineer deseni): **worker thread** hit['_vlm']'i
  doldurur (st.* YOK → context sorunu yok, module-level store + Lock), viewer **`st.fragment(
  run_every='0.5s')`** ile poller; bitince `st.rerun` → cached dal (polling durur). `search.py`:
  `StreamJob`/`start_stream_job`/`get_stream_job`/`clear_stream_job`. **TARAYICIYLA DOĞRULANDI:**
  "kırmızı araba" → verification boyunca tarayıcı YANIT VERDİ (screenshot donmadı), kartlar canlı
  doldu, temiz final (Oynat). Fragment dış-container'a yazamaz hatası düzeltildi (kendi alanına render).
- **Latency: renk sorgusunda YOLO-presence-skip** (AI Engineer #4): aday kırpığın YOLO sınıfı
  hedefle eşleşiyorsa (araba=car) presence VLM çağrısı atlanır → renk 2-sorgu 99→77s. Bonus:
  siyah SUV 7→8 (bir presence false-negative de düzeldi).
- **Town Centre elendi:** artık yalnız torrent/Kaggle (erişilemez) + gizlilik nedeniyle kaldırılmış
  (KVKK ürünü için ironik). Korpus → Selman'ın kendi klipleri (bekliyor).
- **Sıradaki round (AI Engineer refleksiyonu):** 1) eval harness (metrik — pazarlık dışı),
  2) sorgu-ayrıştırıcı→LLM, 5) hard-negatif+kalibrasyon, 3) zaman-aralığı grounding, 4) adaptif top_n.

## 5 Temmuz 2026 (gece-5) — UX cilası: bbox-vurgulu kart görseli (3/4, kısmi)

- **Kart görseli artık bbox-vurgulu:** kırpık aday için tam-kare thumb üzerine eşleşen özneyi
  KIRMIZI KUTUYLA işaretle (`viewer._card_image` + `recrop.draw_bbox`). Kullanıcı "hangi
  araç/kişi eşleşti" anında görüyor — minik kırpık/meşgul sahne yerine bağlamlı. **Tarayıcıyla
  doğrulandı:** "siyah SUV" → her kartta dark SUV'un etrafında kutu + "✅ renk doğru". Net kazanç.
- **Streaming'de reddedileni soluklaştırma** (`_card_image(dim=True)` + `_is_rejected`): doğrulanıp
  reddedilen kart "eleniyor" görünür ('sonuç' değil).
- **BİLİNEN SINIR (gelecek iş):** senkron `stream_verify` ~40s ana thread'i bloke ediyor →
  verification sırasında tarayıcı donuyor (CDP screenshot timeout). AI Engineer çözümü: worker
  thread + `st.fragment(run_every=0.5s)`. Yüksek riskli rework olduğu için ayrı, odaklı bir
  oturuma bırakıldı — mevcut akış DOĞRU sonuç üretiyor (yalnız verification boyunca janky).
  Latency kaynağı: vlm_top_n=12 × ~2 çağrı; vlm_top_n düşürülerek de kısaltılabilir.

## 5 Temmuz 2026 (gece-4) — VLM doğruluğu: hibrit doğrulama görüntüsü (2/4)

- AI Engineer (b): kırpık yerine tam-kare+bbox doğrula (küçük-kırpık yanlış-pozitifi +
  çok-nesne kapsamı). Uyguladım AMA ölçüm bir uyarı gösterdi: tam-kare renk sorgularını
  FAZLA ELİYOR (siyah SUV 7→2, gözle doğrulandı — kutulu araç açıkça siyah SUV ama VLM
  768px'e küçültülmüş meşgul sahnede "kutuda araba yok" dedi). Latency da 46s'e çıktı.
- **Karar: HİBRİT (ölçümle en iyi):**
  - **Renk/öznitelik → tight-kırpık** (özne kadrajı doldurur, renk net, hızlı — kanıtlı iyi).
  - **Zor-kavram (köpek/yağmur) → kutusuz tam-kare** (nesne kırpık dışında olabilir).
- **Sonuç:** "köpek gezdiren adam" → **BULUNAMADI** (önceki 1 yanlış-pozitif GİTTİ, tam-kare
  sayesinde). "kırmızı araba" 7/7, "siyah SUV" 7 (fazla-eleme yok). "kırmızı kıyafetli adam"
  bulunamadı. `recrop.vlm_frame_for_hit`/`draw_bbox` eklendi (bbox kart görselinde de kullanılacak).
  Detay: ARCHITECTURE.md §8, `experiments/2026-07-05_vlm_latency/tam_kare_gorsel.py`.

## 5 Temmuz 2026 (gece-3) — "Mükemmelleştirme" turu: Kararlılık (AI Engineer planı, 1/4)

- Selman "mükemmel hale getirelim" dedi, 4 alan seçildi (UX/kararlılık/VLM doğruluğu/korpus).
  AI Engineer'a danışıldı, sıra: **Kararlılık → Korpus → VLM doğruluğu (tam-kare) → UX**.
- **Kök sorun:** Ollama %91 VRAM'de arka plan GPU uygulamalarıyla OOM çöküyor → tüm kartlar
  "doğrulanamadı" (bugün bizzat çarpıldı). AI Engineer: CLIP'i CPU'ya al (GPU'yu VLM'e bırak).
- **CLIP→CPU DENENDİ, ELENDİ:** jina-clip-v2 text tower CPU'da sabit **~13s/çağrı** (561M×512
  token, thread/MKL fark etmiyor) — interaktif aramayı öldürür. GPU'da 286ms. VRAM ölçümü:
  CLIP CPU'da → 7445→5932 MiB (co-residency çözülür) AMA 13s bedeli kabul edilemez.
- **Çözüm — çökmeyi ÖNLE değil ZARİFÇE KURTAR** (CLIP GPU'da kalır):
  - **Self-healing `is_available`** (`search.py`): kısa TTL (30s) + hata-anında reset. "Bir
    kez False→hep False" bug'ı bitti; çökme/geri-gelme otomatik iyileşir.
  - **Ayrık çökme UX'i:** tekil verify-fail = kart rozeti; TOPTAN VLM-down = global banner
    "VLM kapalı → yalnız CLIP" (`SearchOutcome.vlm_unavailable`, `viewer.render_final`).
  - **Açılış warmup** (`verifier.warmup` + `viewer._warmup_vlm` arka plan thread) → ilk sorgu cold değil.
  - **timeout 20→30s** (yalnız emniyet). Supervised Ollama (masaüstü uygulaması auto-restart).
- **Test:** Ollama down iken → vlm_available False, CLIP-only sonuç, kartlarda _vlm yok, banner
  tetiklenir. Ollama up → normal. GPU encode warm 286ms. Detay: ARCHITECTURE.md §8.

## 5 Temmuz 2026 (gece-2) — "kırmızı kıyafetli adam" hâlâ araba gösteriyordu: 3 UI/füzyon düzeltmesi

- Selman bir önceki turdan sonra hâlâ kırmızı araba görüyordu. **Tarayıcıyla bizzat
  reprodüksiyon** (Chrome) kök nedeni gösterdi: sorun 3 katmanlı, hepsi düzeltildi + görsel doğrulandı.
- **Kuyruk sızıntısı:** `default_top_k=12` ama `vlm_top_n=8` → rank 9-12 DOĞRULANMADAN
  ekleniyordu (kırmızı arabalar). `vlm_top_n=12` (=default) + `_fuse_verdicts` doğrulanmamış
  kuyruğu döndürmez.
- **Renk-uymayan ana ızgarada kalıyordu:** füzyon `present=False`'u eliyordu ama `color_match
  is False`'u tutuyordu → renk uymayan kartlar "🚫 renk uymadı" rozetiyle görünmeye devam
  ediyordu. Renk artık ölçülmüş güvenilir (qwen2.5vl red/blue 9/9↔0/9) → renk sorgusunda
  `color_match is False` de elenir. Böylece hiç gerçek eşleşme yoksa → **bulunamadı** (eski
  "renk güvenilmez→rerank-only" tasarımı qwen3-vl JSON rubber-stamp içindi, artık geçersiz).
- **Bulunamadı mesajı renk-farkında:** "aranan renk/nesne bileşimi doğrulanamadı."
- **TARAYICIYLA DOĞRULANDI:** "kırmızı kıyafetli adam" → ana görünüm "🔍 Bulunamadı",
  7 kırmızı araba "VLM elenenler (7)" expander'ında. Regresyon yok: kırmızı araba 7/7,
  siyah SUV 1 eledi. Kalan UX notu: streaming ~30s CLIP adaylarını ⏳ ile gösterir (final
  reflow doğru) — kabul edilebilir. Detay: ARCHITECTURE.md §8.

## 5 Temmuz 2026 (gece) — Selman'ın 2 sorgusu: renk-giysi + füzyon düzeltmeleri

- Selman "köpek gezdiren adam" ve "kırmızı kıyafetli adam" aradı, ikisi de yanlış gösterdi.
  Gerçek hattan teşhis edildi (2 kök sorun + 1 model tavanı):
- **BUG 1 (düzeltildi) — renk insanlarda giysiye işaret eder:** "kırmızı kıyafetli adam"
  VQA hedefi `('man','red')` → VLM'e "Is the man red?" soruyordu (kimse kırmızı değil→herkes
  "no"). `verifier._PERSON_OBJECTS` eklendi: insan-tipinde "Is this man wearing red clothing?".
- **BUG 2 (düzeltildi) — füzyon renk modunda nesne-yokluğunu düşürmüyordu:** CLIP "kırmızı"ya
  takılıp kırmızı ARABALAR getirdi; VLM doğru `present=False` dedi ama attribute mod düşürmüyordu
  → 7 araba gösterildi. Eski "renk modunda düşürme" tasarımı JSON rubber-stamp güvenilmezken
  doğruydu; yes/no ile nesne-varlığı GÜVENİLİR → `_fuse_verdicts` her iki modda `conf<drop_below`
  düşürür. Sonuç: 7 araba → 5 elendi. Regresyon yok (kırmızı araba 7/7, siyah SUV 1 eşleşmeyeni eledi).
- **Rozet dürüstlüğü:** renk uymayınca "✅ VLM" değil "🚫 renk uymadı" (`viewer._verdict_badge`).
- **MODEL TAVANI (kabul):** qwen2.5vl:3b küçük/bulanık kırpıkta varlık sorusunda yes-bias'lı
  (insanda "köpek var" / arabada "adam var" yanlış-pozitif). Sıkı prompt denendi → gerçek
  uzak-insanları da reddediyor (false-negative'e çeviriyor), temiz çözüm yok → tavan kabul.
  Ayrıca korpusta HİÇ köpek yok → "köpek gezdiren adam" doğası gereği ~bulunamadı + gürültü.
  Detay: `experiments/2026-07-05_vlm_latency/varlik_prompt_ab.py`.

## 5 Temmuz 2026 (akşam) — VLM güvenilirlik: thinking modeli atıldı, yes/no VQA'ya geçildi

- **Sorun teşhis edildi (deterministik):** `qwen3-vl:2b` bir *thinking* modeli — belirsiz
  kırpıklarda sonsuz düşünme döngüsüne girip n_ctx=4096'yı doldurur, boş `content` döner
  (%33 hata). ~25s "timeout"lar aslında 2905-token kaçak *generation*. Her görüntü 2×→birebir
  aynı. `think:false`/`/no_think`/`num_predict` cap hiçbiri bu Ollama build'inde çözmedi.
- **AI Engineer kararı: B — non-thinking VLM'e geç.** Bir doğrulayıcının işi sınırsız akıl
  yürütme değil, sınırlı-süreli deterministik sınıflandırma; thinking modeli yanlış araç.
- **`qwen2.5vl:3b` (non-thinking)** — co-residency kapısı: 9/9 geçerli JSON, done hepsi
  `stop`, **`offloaded 37/37 layers to GPU`** (CPU offload yok, CLIP ile sığdı).
- **Ama JSON şeması RUBBER-STAMP'liyor:** kırmızı arabaya "mavi araba? evet, conf 1.0" der.
  Düz yes/no'da ise rengi KUSURSUZ ayırır (kırmızı 9/9, mavi 0/9). Model kör değil — JSON
  çerçevesi bozuyor. → **YES/NO VQA sözleşmesi**: sorgu (nesne, renk) hedeflerine indirgenir,
  verifier iki atomik yes/no sorar; `num_predict:4` kaçağı yapısal imkânsız kılar.
- **Sonuç her eksende kazanç:** reliability %100, ayrım kusursuz (red/blue, dog), latency
  4.8s/2.3s (JSON'un 16.8s'inden 3.5× hızlı). Uçtan uca "köpek gezdiren insan"→hedef `dog`,
  4 yanlış-pozitif elendi, 1 köpekli kaldı.
- **Kod:** `query.extract_vqa_targets` (yeni), `verifier.py` yes/no'ya yeniden yazıldı,
  `search.verify_top_n` yeni imza, `config` model+timeout(40→20). q8_0 env var registry'den
  silindi. Detay: `experiments/2026-07-05_vlm_latency/deney_notu.md` · ARCHITECTURE.md §8

## 5 Temmuz 2026 (öğle) — Per-item streaming: VLM rozetleri canlı doluyor

- `search.py` bölündü: `verify_top_n` (callback'li, her verdict'te `on_verdict(i,hit)`) +
  `_fuse_verdicts` (füzyon) + public `stream_verify`. `_apply_vlm`=ikisi (CLI batch).
- **Viewer per-item streaming:** CLIP kartları ⏳ ile gelir; her VLM verdict'i o kartın
  rozetini CANLI doldurur (⏳→✅ renk doğru/✅ conf/🚫), sonra final reflow. Cache VLM
  tarafında `@cache_data` yerine `session_state` (streaming canlı render gerektiriyor).
- **TARAYICIYLA GÖRSEL DOĞRULANDI** (Chrome): "siyah SUV araç" → kart 1 "✅ renk doğru"
  belirdi, kartlar 2-4 ⏳; sonra kart 1,3,4 "✅ renk doğru", kart 2 çözüldü. Rozetler tek
  tek canlı doldu. Ayrıca "köpek gezdiren adam" → CLIP insanlar → BULUNAMADI + "VLM
  elenenler (5)" expander (önceki batch doğrulamasında).
- Güvenilirlik: yavaş çağrı (spike) akışı kısa durdurabilir → `vlm_timeout_s=30` (21s
  spike absorbe, hang 30s'de kapanır, kart rozetsiz akış devam). Detay: ARCHITECTURE.md §8b

## 5 Temmuz 2026 — Async VLM rafine: progressive render (interaktif latency çözüldü)

- AI Engineer: gerçek thread değil **progressive render** (tek-kullanıcı demo). VLM latency
  ~4-6s/görüntü — bottleneck token DEĞİL görüntü prefill'i (num_predict:48 denendi, latency
  düşmedi + JSON'u kesti → geri alındı; out_size 384/256/224 fark etmiyor, VLM içeride resize).
- `search()` bölündü: `search(use_vlm=False)` (CLIP hızlı) + public `refine_vlm(outcome)`.
  `SearchOutcome`'a `vlm_applied` + `vlm_filtered` (elenenler expander için).
- **Viewer yeniden yazıldı:** `st.form` submit (keystroke rerun yok), `@st.cache_data`
  (Oynat tıklaması VLM'i yeniden koşmasın — kritik), CLIP `st.empty` slot'a anında →
  `st.status` altında VLM → aynı slot yerinde değişir, elenenler expander'da.
- **Prompt sıkılaştırıldı** ("açıklamanın HER parçası görünmeli"): köpek/yağmur → BULUNAMADI
  (önce 1 kısmi-eşleşme survivor kalıyordu), siyah SUV öznitelik bozulmadan (color_match 0.7-1.0).
- Doğrulandı: tüm modüller derleniyor, Streamlit HTTP 200 başlıyor, SearchOutcome pickle'lanıyor
  (cache güvenli). CLI senkron kalır. Paralel VLM yok (Ollama tek model, OOM). Detay: ARCHITECTURE.md §8b

## 4 Temmuz 2026 (gece) — Faz 2 VLM doğrulayıcı: negasyon çözüldü, retrieve-then-verify

- AI Engineer tasarımı: `qwen3-vl:2b` (Ollama, 1.9GB — CLIP ile eşzamanlı sığan tek
  sağlam seçenek), koşullu (yalnız renk/zor-kavram sorgusu), tam-çözünürlükte doğrula.
- **Kurulan hat:** Ollama v0.31.1 kuruldu (winget) + qwen3-vl:2b (100% GPU, keep_alive
  30dk). Modüller: `recrop.py` (orijinalden bbox yüksek-res kırpma — 36px thumb yetersiz),
  `query.translate_visual` (TR→EN, VLM'in Türkçesine güvenme), `verifier.py` (yapılandırılmış
  JSON verdict), `search._apply_vlm` (negasyon=filtre / öznitelik=rerank).
- **Doğrulandı (net görüntüde VLM güçlü):** beyaz araba renk=True conf=1.0, aynı arabaya
  "siyah"→renk=False; telefonlu sürücü davranış conf=1.0; köpek→False.
- **Uçtan uca:** "köpek gezdiren adam" → **BULUNAMADI** (VLM tüm adayları reddetti — CLIP+kapının
  çözemediği negatif örtüşme çözüldü). "otobüs"→VLM tetiklemez (0s, vergi yok).
- **Yol boyu düzeltilen hata:** VLM reddedince conf=0.0 (P(present)) → filtre iki moda ayrıldı:
  negasyon (düşük-conf düşür, hepsi düşerse bulunamadı) / öznitelik (renk rerank, drop yok).
- Latency ~4-6s/görüntü → koşullu tetik. LLM sorgu-ayrıştırıcı ERTELENDİ (VRAM). Detay: ARCHITECTURE.md §8
- **Ölçüldü + gözle doğrulandı:** negasyon güçlü (köpek→BULUNAMADI, yağmur 0/6); **renk — VLM
  CLIP'in hatalarını düzeltiyor** ("mavi kamyonet"te CLIP 4 aday, VLM 1 gerçek maviyi onayladı,
  3 sahteyi reddetti; 3 renk de gözle doğru — tam-res recrop mavi-korkusunu yendi). Pozitif
  kontrol otobüs 5/5, kamyon 4/4. Dürüst sınır: kar 2/6 false-accept (açık zemin→"kar").
- **Güvenilirlik düzeltildi:** ilk marathon %24 hata (CLIP+VLM 8GB spike) → timeout 20→45s +
  retry → tek-sorgu (prod) 0/8. Detay: `experiments/2026-07-04_faz2_vlm/`.

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
