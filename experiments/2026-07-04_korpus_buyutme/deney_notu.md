# Deney: Korpus büyütme (MEVA) — eval'i şişmeden çıkarıp ayrımcı yapmak

**Tarih:** 4 Temmuz 2026
**Amaç:** Eval'in R@5=1.0'ı şişmişti — 4 sahne (garaj/gece/trafik/otopark) görsel
olarak çok ayrıydı, metrik "hangi sahne"yi ölçüyordu, "doğru sıralama"yı değil.
AI Engineer: kamera ailesini sabit CCTV tut, mevcut sahne tiplerinin örneğini artır.

## Ne eklendi

**MEVA KF1** (VIRAT halefi, S3 public `mevadata-public-01`, HTTP Range destekli —
Kitware'in no-Range/yavaş sorununu çözdü). ffmpeg ile URL'den yalnız ilk 60 sn çekilip
(tam klibin ~1/5'i, klip başına ~14 sn) mp4'e normalize edildi. 6 farklı sabit kamera:

| video_id | MEVA anahtarı (drops-123-r13/2018-03-05/...) | sahne | kare | kırpık |
|---|---|---|---|---|
| meva_hastane1 | 10/...10-20-00.10-25-00.hospital.G301 | bina + beyaz araba + **40 yürüyen insan** | 17 | 41 |
| meva_hastane2 | 10/...10-20-00.10-25-00.hospital.G436 | kampüs yolları + araç (VIRAT_kampus benzeri) | 114 | 56 |
| meva_okul1 | 09/...09-50-00.09-55-00.school.G336 | dönel kavşak + mavi SUV + otopark | 50 | 13 |
| meva_okul2 | 09/...09-50-00.09-55-00.school.G424 | araçlarla dolu açık otopark | 1 | 10 |
| meva_otobus | 10/...10-10-00.10-15-00.bus.G340 | açık lot + uzak araçlar | 1 | 7 |
| meva_idari | 10/...10-30-01.10-35-01.admin.G326 | iç mekan merdiven (çeşitlilik) | 1 | 0 |

Korpus: 7 → **13 video**, 4080 → **4391 vektör** (+54 yeni insan kırpığı yeni sahnelerde).
Not: statik sahneler (okul2/otobus/idari) 1 kare verdi — hareket kapısı doğru çalışıyor,
park etmiş araçlar kırpık olarak yakalandı ama sahne karesi tek (dürüst sınır).

## Sonuç 1 — v1 (kilitli) çeldiricilere rağmen SAĞLAM

Aynı dondurulmuş 22 sorgu, +6 MEVA çeldiriciyle yeniden koştu (`2026-07-04_eval_meva/`):
**R@1=0.917, R@5=1.0, MRR=0.958 — DEĞİŞMEDİ.** MEVA kırpık düzeyinde rekabet ediyor
("beyaz araba"→meva_hastane1 rank-4; "park etmiş arabalar"→meva_okul2 rank-4) ama doğru
cevap yine kazanıyor. **Yorum:** R@5=1.0 artık daha güvenilir — 6 görsel-benzer açık alan
çeldirici eklendiği halde ayrım korundu. Şişme "sahte" değilmiş; embedding gerçekten ayırıyor.

## Sonuç 2 — v2 (yeni, eklemeli) 2 GERÇEK olguyu ifşa etti

Pre-registered v2 (`queries_v2.yaml`, 2 gözle-doğrulanmış MEVA golden): R@1=0.333,
MRR=0.375 — v1'den çok düşük, çünkü artık **gerçekten zor** (`2026-07-04_eval_v2/`).

**Olgu A — 8sn kümeleme spesifik golden'ı yutabiliyor (retrieval hatası DEĞİL).**
`v2_gold_meva_yuruyen`: top-1 = meva_hastane1 (DOĞRU video), ama golden kare 225 kümelemeyle
kaçtı — kişi 75-225 kareleri boyunca yürüyor, 8sn penceresi tek kare tutuyor. `fail_attribution`
= "dedup/kümeleme" bunu doğru işaretledi. Kullanıcı için sonuç başarı (doğru sahne zirvede);
golden etiketim fazla spesifikti. AI Engineer'ın Boşluk 3 uyarısı tam da buydu.

**Olgu B — sahne sorgusu, nesne kırpıklarının seli altında gömülüyor (GERÇEK zaaf).**
`v2_gold_meva_otopark` ("araçlarla dolu açık otopark") → rank 8, top-1 = ucf_trafik. "Araç"
kelimesi araç kırpıklarını sele çeviriyor, doğru sahne karesini aşağı itiyor. **Faz 1.5 marjini
v2'de −0.333** (yalnız-kare R@5=0.667 > birleşik 0.333) bunu ölçtü: nesne sorgusunda YARDIM
eden kırpık, nesne-kelimeli sahne sorgusunda KÖSTEK. Skor-ofsetsiz birleşik indeksin sınırı.

## Çıkarım

Korpus büyütme amacına ulaştı: eval artık **ayrımcı**. v1 "regresyon yaptım mı?" (hayır,
sağlam), v2 "yeni içerik + zor sıralama çalışıyor mu?" (kısmen — Olgu B gerçek açık) sorularını
ayrı yanıtlıyor. Pre-registration korundu: v1 dokunulmadı, v2 sonuca bakmadan yazıldı.

## Sonraki adım adayları

1. **Olgu B'yi çöz (öncelikli):** sahne-niyetli sorgularda (yer adı: otopark/cadde/kampüs)
   kareyi kırpığa tercih et ya da niyet tespiti. Skor-ofset tasarımının yeniden değerlendirmesi
   — AI Engineer danışması gerekir.
2. **Gece/trafik çeşitliliği:** UA-DETRAC (gündüz+gece, Kaggle) — Kaggle token gerekiyor (yok).
   ucf_gece hâlâ tek gece kaynağı → "gece" sorgusu hâlâ 1:1 eşleşiyor.
3. **v2'yi büyüt:** daha fazla MEVA golden + statik lotlara daha uzun segment (kare zenginliği).
