# ARCHITECTURE.md — Mimari Kararlar

> AI Engineer danışmasıyla netleştirildi — 2 Temmuz 2026.
> Her karar: **seçim + gerekçe + yedek plan** formatında.

## 1. Embedding modeli: `jinaai/jina-clip-v2` (doğrudan Türkçe)

- **Seçim:** Sorgu çeviri YOK — Türkçe sorgu doğrudan embedlenir. Model 89 dil destekli XLM-RoBERTa metin kulesi + EVA02-L görüntü kulesi (512×512 giriş — geniş açı otopark planlarında küçük araç/insan için avantaj). Vektör boyutu **1024** (tam boyut; bu ölçekte depolama sorun değil, şimdiden kırpma yok).
- **VRAM:** fp16'da ~2 GB → 6 GB kartta YOLO ile yan yana çalışır.
- **CPU fallback:** Aynı modelin yayınlanmış **ONNX int8** ağırlıkları (onnxruntime). Aynı 1024-dim vektörler → GPU'lu ve CPU'lu makine aynı indeksi paylaşır. İndeksleme offline/gecelik iş olduğu için CPU'da ~1 kare/sn kabul edilebilir.
- **Yedek plan:** Türkçe recall zayıf çıkarsa → sorguyu çevir + `google/siglip2-so400m-patch14-384` (1152-dim). Karar his ile değil, **eval seti ölçümüyle** verilecek (bkz. Risk 3).

## 2. Kare örnekleme stratejisi

1. PyAV ile çözümle, **2 fps** aday kare (0.5 sn'de bir).
2. **Hareket kapısı:** 320×180 gri + 5×5 Gaussian blur + önceki adayla absdiff. Piksel "değişti" eşiği **> 25**; `motion_score` = değişen piksel oranı; **> 0.005** ise tut.
3. Hareket olmasa da **60 sn'de bir çapa karesi** zorla tut (sahne kapsaması, gece stabilitesi).
4. **Dedup:** 64-bit pHash; son **10** tutulan kareyle Hamming mesafesi **≤ 6** ise at.
5. Tutulanları toplu embedle (GPU batch 16 / CPU batch 4); **480 px JPEG (q=80)** küçük resim yaz.

**İndeks matematiği:** 2 fps = 7.200 aday/saat. Otopark kamerası gerçekçi ortalama **~300–500 tutulan kare/saat/kamera ≈ 8–12k/gün/kamera**. Pilot (1 hafta × 4 kamera) ≈ **250–350k vektör ≈ 1–1.4 GB** (kuantizasyon öncesi) — rahat.

## 3. Vektör DB: Qdrant (lokal)

Koleksiyon `frames`: boyut **1024**, mesafe **Cosine**, HNSW **varsayılanları** (m=16, ef_construct=100 — birkaç milyon vektöre kadar yeterli) + **scalar int8 kuantizasyon (always_ram=true)** ile RAM ~4× düşer.

| Payload alanı | Tip | Not |
|---|---|---|
| `video_id` | keyword | |
| `camera_id` | keyword | **indeksli** |
| `ts` | float (epoch UTC) | **indeksli**, aralık filtresi |
| `frame_idx` | int | |
| `motion_score` | float | |
| `phash` | keyword | |
| `thumb_path` | keyword | |
| `track_ids`, `plates`, `yolo_classes` | int[] / keyword[] | **Faz 2 için rezerve** |

## 4. Türkçe sorgu hattı

- **MVP: kural bazlı ayırıcı.** Regex + `dateparser` (tr) zamansal ifadeleri ("dün gece", "bu sabah 8'den sonra") ayıklar → Qdrant `ts` aralık filtresi. Kalan görsel ifade ("beyaz Transit") embedlenir. **Zaman kelimeleri asla embedlenmez — en büyük kalite kaldıracı bu.**
- **Faz 2:** Lokal LLM sorgu ayrıştırıcı (Qwen3-4B, Ollama, ~3 GB q4) → `{visual_tr, visual_en, time_range, camera}` JSON; iki dil varyantı da embedlenip max skor alınır.
- **KVKK notu:** Sorgular görüntü içermez ama **kişisel veri içerebilir** (plaka, isim). Bu yüzden bulut çeviri API'si kullanılmaz — "hiçbir veri dışarı çıkmaz" hem satış argümanı hem tasarım ilkesi. 4B lokal model sorgu çevirisi için fazlasıyla yeterli.

## 5. Modül yapısı (düz paket — derin iç içe klasör yok)

| Modül | Sorumluluk |
|---|---|
| `config.py` | pydantic-settings: yollar, model id'leri, tüm eşikler |
| `sampler.py` | video çözme + 2fps örnekleme + hareket kapısı + pHash dedup → FrameRecord akışı |
| `embedder.py` | jina-clip-v2 sarmalayıcı, toplu görüntü/metin encode, cuda/cpu otomatik |
| `thumbs.py` | küçük resim yazıcı |
| `store.py` | Qdrant şema kurulumu, upsert, filtreli arama |
| `query.py` | Türkçe zamansal ayrıştırıcı → (görsel_metin, filtreler) |
| `search.py` | sorgu embed + Qdrant çağrısı + sonuç birleştirme (video, ts, skor, thumb) |
| `viewer.py` | Streamlit: sorgu kutusu → küçük resim ızgarası → tıkla, videoyu o saniyeden aç |
| `cli.py` | typer: `gozcu index <klasör>`, `gozcu search "<sorgu>"` |

## 6. En büyük 3 risk

1. **Tam-kare embedding küçük özneleri sulandırır** — kırmızı montlu adam 1080p geniş planda 40 px'dir; recall hayal kırıklığı yaratırsa Faz 2'deki YOLO crop-embedding'i (insan/araç kırpıklarını ayrı vektör olarak indeksle) **Faz 1.5 olarak öne çek**.
2. **Zaman damgası gerçeği** — DVR dosya adları/mtimes yalan söyler; taban zaman yanlışsa her "dün gece" cevabı yanlış. Erken doğrula: görüntüye gömülü OSD saatini OCR ile çapraz kontrol et.
3. **Eval seti yoksa karar da yok** — 1. haftada test videolarından **30–50 Türkçe sorgu → doğru kare** çifti oluştur; her model/eşik kararı (SigLIP2 yedeğine geçiş dahil) buna karşı ölçülür. Ayrıca: embedding ve küçük resimler de KVKK kapsamında kişisel veridir — saklama/silme politikası ilk günden tasarlanır.
