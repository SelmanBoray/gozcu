# Hard-negative eval kapsamı + ayrım ekseni — 21 Temmuz 2026

## Amaç
Bu oturumun asıl bug'ına regresyon koruması: **"kırmızı kıyafetli adam" araması KIRMIZI ARABA
getiriyordu** (kullanıcı iki kez şikâyet etti). Düzeltmeler (verifier: kişi→"is wearing X
clothing"; füzyon: `color_match=False` araç elenir) çalışıyordu ama **eval'de hiç test edilmiyordu**
— ileride bir prompt tweak'i bozarsa hiçbir şey yakalamazdı.

## Yapılan
Eval'e **ortogonal ayrım ekseni** eklendi (AI Engineer tasarımı):
- `queries_faz2.yaml`: 2 hard-negative sorgu, şema `expect: any` + `forbid: vehicles`.
- `faz2_eval.py`: outcome eksenine (not_found/has_results) DOKUNMADAN ikinci eksen — yasaklı YOLO
  sınıfı (araç) sonuçlara sızarsa FAIL. Geriye tam uyumlu → mevcut skor korunur.
- `forbid: vehicles` **sembolik** → `gozcu.query.VEHICLE_CLASSES` (Türkçe etiketler: araba/kamyon/…).
  İngilizce liste yazılsaydı depodaki Türkçe `yolo_class` ile hiç eşleşmez, test sahte-PASS ederdi
  (AI Engineer'ın yakaladığı kritik tuzak).

## Sonuç: 11/12 (%92) — baseline korundu, ayrım 2/2

| id | çıktı | ayrım | not |
|----|-------|-------|-----|
| `hn_kirmizi_kiyafet` | not_found (0) | **sızan-araç=0, VLM 6 araç adayını eledi ✓** | **KANITLI (non-vacuous)** |
| `hn_siyah_kiyafet` | has_results (5) | sızan-araç=0, ⚠ havuzda araç adayı yok | vacuous — latent canary |

**Kilit bulgu:** `hn_kirmizi_kiyafet` non-vacuous. CLIP tam da eski bug'daki gibi **6 kırmızı araba
adayı** yüzeye çıkardı; VLM doğrulayıcı **6'sını da eledi, 0 sızdı.** Yani düzeltme artık gerçek
(boşa geçmeyen) bir testle kilitli — retrieval hâlâ kırmızı arabaları aday yapıyor ama verify
katmanı temizliyor.

`hn_siyah_kiyafet` şu an vacuous: kişi sorgusu siyah aracı retrieval'da aday YAPMADI (embedding
insana yöneldi), forbid tetiklenecek şey bulamadı → boşa PASS. Dürüstçe ⚠ ile işaretlendi. Latent
canary olarak tutuluyor: korpus/embedding değişip siyah araç aday olursa test otomatik aktifleşir.

## Kalibrasyon: bilinçli YAPILMADI
`vlm_drop_below` / `vlm_beta` kalibrasyonu değerlendirildi, **gereksiz** bulundu: verifier
`confidence` şu an **binary** {0.0, 1.0} → (0,1) aralığındaki her eşik özdeş (no-op). Kalibrasyonun
anlamlı olması için önce sürekli-confidence (multi-frame consensus / token logprob) gerekir — o bir
feature, kalibrasyon değil ("yeni klip yok" kısıtı altında kapsam dışı). `config.py` yorumu bunu
belgeleyecek şekilde dürüstleştirildi (gelecekte boşa kalibrasyon çabasını önler).

## Ham sonuç
`faz2_eval_sonuc.json` (bu klasörde kopya) · üretici: `eval/faz2_eval.py` · VLM=qwen2.5vl:3b
