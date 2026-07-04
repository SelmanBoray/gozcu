# Deney: Faz 2 VLM doğrulayıcı — negasyon + renk (qwen3-vl:2b)

**Tarih:** 4 Temmuz 2026
**Amaç:** Eval'in CLIP ile kapatılamayan iki açığı — negasyon örtüşmesi (köpek/yağmur/kar,
kapının çözemediği) ve renk doğrulama (siyah SUV, mavi kamyonet) — retrieve-then-verify
ile kapatmak. Model: `qwen3-vl:2b` (Ollama, 1.9GB, 100% GPU, keep_alive 30dk).

## Tasarım (AI Engineer)

- **Koşullu tetik:** yalnız renk/zor-kavram sorgusu (`query.needs_vlm`). Nesne/sahne
  zaten yüksek recall → VLM vergisi yok.
- **TR→EN çeviri:** VLM'in Türkçesine güvenme; deterministik sözlük (`translate_visual`).
- **Tam çözünürlük:** 36px thumb yerine orijinalden bbox ile yeniden kırp (`recrop.py`).
- **Yapılandırılmış verdict** (yes-bias'a karşı): `{object_present, color_match, confidence}`,
  reddetme-yanlı İngilizce prompt.
- **Füzyon:** negasyon → düşük-conf düşür (hepsi düşerse bulunamadı); öznitelik → renk
  rerank (drop YOK — renk güvenilmez, hard-filtre etme).

## Sonuçlar (gözle doğrulanmış)

### Negasyon — güçlü (kapının çözemediğini VLM çözüyor)
- **"köpek gezdiren adam" → BULUNAMADI** (0/4 kabul). CLIP+kapı çözememişti (köpek YOLO
  sınıfı değil, adam→insan var) — VLM "köpek yok" diyerek örtüşmeyi kapattı.
- "yağmurda şemsiyeli insanlar" → 0/6 kabul (tümü reddedildi). ✓
- **Dürüst sınır:** "karla kaplı sokak" → 2/6 kabul (false-accept). Görüntü:
  zeminde açık beton/tuz lekeleri "kar"a benziyor, VLM conf=1.0 hallüsine etti.
  Ambiguous vaka — VLM öznitelik/hava'da kusursuz değil.

### Renk — VLM, CLIP'in renk hatalarını düzeltiyor (asıl değer)
- **"mavi kamyonet": CLIP 4 "mavi" aday getirdi, VLM yalnız 1'ini onayladı** (gerçek
  mavi ISUZU kamyon — gözle doğrulandı), 3 sahte-maviyi renk-red etti. Tam da Faz 2'nin
  amacı: CLIP renk garantilemiyor, VLM ayıklıyor.
- "siyah SUV" 5/6, "beyaz araba" 5/6 renk-eşleşti. Gözle doğrulama: koyu SUV (VIRAT)✓,
  beyaz Toyota (meva_hastane1)✓, mavi ISUZU✓ — **üç renk de doğru.**
- **AI Engineer'ın mavi-korkusu (~%56 F1) bu vakalarda bitmedi** — çünkü tam-res recrop
  (36px thumb değil) VLM'e gerçek rengi gösteriyor. Yine de renk hard-filtre EDİLMİYOR
  (rerank-only), güvenli tarafta.

### Pozitif kontrol — yanlış-reddetme yok
- "otobüs" 5/5, "kamyon" 4/4 kabul. ✓
- "yürüyen insan" 1/4 — VLM küçük/bulanık insan kırpıklarını fazla reddediyor. AMA bu
  sorgu VLM TETİKLEMİYOR (renk/zor-kavram yok) → **koşullu tetiğin neden şart olduğunun
  kanıtı.** Plain nesne sorgusunu VLM'e verseydik recall düşerdi.

## Güvenilirlik — timeout ayarı

İlk marathon eval'de %24 çağrı hatası. Teşhis: CLIP (2GB) + VLM (2GB) 8GB GPU'da
eşzamanlı → bazı çağrılar spike yapıyor (biri 21.7s ölçüldü), 20s timeout'ta patlıyor.
**Düzeltme: timeout 20→45s + tek retry.** Sonuç: **tek-sorgu (prod senaryosu) 0/8 hata**;
marathon 24%→10% (kalan spike'lar 66-çağrı sürekli yükünde — prod'da tek sorgu güvenilir).

## Latency ve kapsam

~4-6s/görüntü (ara sıra 20s+ spike). Top-8 doğrulama ~30-45s. İnteraktif için sınırda →
koşullu tetik (yalnız renk/negasyon) + (ileride) async rafine. LLM sorgu-ayrıştırıcı
ERTELENDİ (4B+2B+CLIP aynı VRAM'e sığmaz + ölçülmüş parse-arızası yok).

## Sonraki adım adayları

1. Kar/hava false-accept: prompt sıkılaştırma ya da hava-özel doğrulama.
2. β kalibrasyonu (öznitelik rerank ağırlığı) — şu an renk rerank çalışıyor, ince ayar.
3. Async VLM rafine (interaktif latency) — CLIP sonucu anında, VLM arkadan düzeltir.
4. Küçük-özne over-reject: recrop'a daha fazla bağlam payı ya da kare-düzeyi doğrulama.
