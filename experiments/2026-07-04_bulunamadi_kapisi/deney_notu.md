# Deney: Bulunamadı kapısı (yolo_class sertlik kapısı)

**Tarih:** 4 Temmuz 2026
**Amaç:** Eval'in en kritik açık bulgusunu hedefli hafifletmek — sistemin güvenilir
bir "bulunamadı" sinyali yoktu (min-pozitif 0.282 < max-negatif 0.36; CLIP kosinüsü
sorgular arası kalibre değil, tek eşik ayırmıyor).

## Fikir

Skor eşiği yerine **YOLO tespitine** dayan: sorgu, tespit edebildiğimiz bir nesne
sınıfı istiyorsa (Türkçe eş anlamlı + çekim ekiyle) ve o sınıf korpusta **hiç yoksa**,
CLIP'e hiç sormadan boş dön. Eşik değil, envanter kontrolü — kalibrasyon gerektirmez.

## Uygulama

- `query.py:extract_object_intent` — görsel metni ASCII-fold edip (çekim ekine
  dayanıklı: arabalar→araba, insanı→insan) sınıf köklerine prefix eşler. Belirli sınıf
  (`required`) ile genel taşıt (`generic_vehicle`, "araç geçti") ayrılır.
- `store.py:available_object_classes` — korpustaki kırpık `yolo_class`'larını bir kez
  tarayıp önbelleğe alır (korpus büyüdükçe otomatik güncellenir).
- `search.py` — **yalnız prod hattında** (source=None): `required − available` boş
  değilse `not_found_reason` ile boş dön. Ablation koşuları (frame/crop) kapıyı
  atlar → ham retrieval ölçümü bozulmaz.
- `cli.py`/`viewer.py` — "BULUNAMADI: Korpusta 'X' tespit edilmedi" mesajı.

## Sonuç (aynı dondurulmuş eval seti, eşleştirilmiş)

- **Manşet METRİKLER DEĞİŞMEDİ:** skorlanabilir R@1=0.917, R@5=1.0, MRR=0.958.
  → Kapı hiçbir pozitifi bozmadı (**yanlış kapı = 0**, runner denetliyor).
- **Kapı yakalama oranı: 0.25** (4 negatiften 1'i). `neg_bisiklet` ("kırmızı bisiklet
  süren çocuk") artık boş dönüyor — bisiklet YOLO sınıfı, korpusta 0 kırpık.
- Uçtan uca doğrulandı: `search "kırmızı bisiklet süren çocuk"` →
  "BULUNAMADI: Korpusta 'bisiklet' tespit edilmedi — bu nesne kayıtlarda yok."

## Kapının kapsamı ve dürüst sınırı

Kapı **yalnızca "tespit edilebilir sınıf korpusta yok"** vakasını çözer. Kalan 3 negatif
neden geçti:
- `neg_kopek` ("köpek gezdiren adam") — **köpek bizim 6 YOLO sınıfımızda yok**, "adam"→insan
  var. Kapı köpeği bilmiyor. (Kolay genişletme: COCO'da köpek/kedi var; tespit
  sınıflarına eklenirse kapı bunu da yakalar.)
- `neg_yagmur`, `neg_kar` — öznitelik/hava durumu; nesne sınıfı değil, kapı görmez.

Bu üçü için kalan ayrım marjı hâlâ -0.078 (örtüşme) — **Faz 2 VLM doğrulayıcının
gerekçesi.** Kapı ucuz ve kesin olan parçayı hallediyor; belirsiz kısım VLM'e kalıyor.

## Sonraki adım

1. Tespit sınıflarını genişlet (köpek/kedi vb.) → kapı kapsamı büyür, ucuz kazanç.
2. Faz 2 VLM re-rank → öznitelik/hava/eylem doğrulaması (kalan örtüşme).
3. Kapı yakaladığında kullanıcıya "korpusta şu sınıflar var: ..." önerisi (UX).
