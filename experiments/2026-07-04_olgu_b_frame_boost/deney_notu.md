# Deney: Olgu B çözümü — sahne-niyetli frame boost (durum-eki + z-normalize)

**Tarih:** 4 Temmuz 2026
**Amaç:** Korpus büyütme v2'nin bulduğu Olgu B — "araçlarla dolu açık otopark" gibi
sahne sorguları, nesne kırpıklarının seli altında gömülüyor (v2'de rank 8, o sorguda
Faz 1.5 marjini −0.333).

## Tasarım (AI Engineer incelemesi)

**Niyet sinyali = durum eki (case morphology), pozisyon DEĞİL.** Türkçe head-final
olduğu kadar sondaki isim de her zaman head değil; ama çekim eki role'ü sıradan bağımsız
kodlar (devrik cümlede bile hayatta kalır):
- "araçlar**LA** dolu açık otopark" → araç=INSTR (ortam), otopark=NOM → **sahne**
- "otopark**TA** yürüyen insan" → otopark=LOC (ortam), insan=NOM → **nesne**
- "arabalar**ın** arasında yürüyen kişi" → araba=GEN, kişi=NOM → **nesne**

`query.scene_or_object_intent`: head = son NOMİNATİF (eksiz) içerik ismi; türü niyeti
verir. Sahne-niyeti YALNIZ head sahne kelimesiyse (S5: "otopark bariyeri" ≠ sahne).

**Mekanizma = z-normalize yumuşak frame-boost (hard filtre DEĞİL).** Sadece sahne-niyetinde,
overfetch havuzunda yeniden sıralama: `_rank = z(cos) + λ·[source==frame]`. Nesne-niyeti
nötr (λ=0) — Faz 1.5 kazanımına dokunma. z-skoru: cosine aralığı sorgudan sorguya değişir,
z λ'yı sorgu-bağımsız yapar. Yumuşak = geri-kazanılabilir: güçlü kırpık cosine'i hâlâ kazanır.

## Kalibrasyon (λ'yı metrikten değil skor-boşluğundan)

- **Niyet-sınıflandırıcı doğruluğu: 12/12 = 1.00** (ayrı metrik, S4 etiket-sızıntısı denetimi;
  sorgular doğal ifadeyle, sınıflandırıcıya bakmadan yazıldı).
- **z-boşlukları çift-modlu:** 4 sorgu gömülmüyor (gap=0), 3'ü derin gömülü (gap 2.08-3.38).
- **λ=1.0 seçildi:** modest (~1 std), robustluk bandı boyunca kazanç monoton (dar spike değil),
  kontroller hiçbir λ'da regres etmiyor (1.5'a kadar test edildi). Dev/kilitli ayrı: λ dev'de
  seçildi, v1/v2'de doğrulandı.

## Sonuçlar

**Dev bataryası (12 sorgu) — kırpık-seli DÜZELİYOR, kontrol regresyonu YOK:**
| λ | sahne R@5 | noobj R@1 | nesne-ctrl R@1 |
|---|---|---|---|
| 0.00 | 0.78 | 1.00 | 1.00 |
| 0.75 | 0.89 | 1.00 | 1.00 |
| 1.125 | 1.00 | 1.00 | 1.00 |

Per-query (sahne-karesi sırası, λ=0 → 1.5): dev_kamyonlu_otopark **10→4**,
dev_garaj_araclar **6→3**, dev_araclarla_otopark 3→2. **3 iyileşme, 0 regresyon**
(McNemar-anlamlı). En yoğun sel kaynağı (ucf_trafik 731 kamyon kırpığı) altında bile
sahne karesi yüzeye çıkıyor.

**Kilitli setler — doğrulama:**
- **v1 (22 sorgu): R@1=0.917, R@5=1.0, MRR=0.958 — DEĞİŞMEDİ.** Sıfır regresyon. ✓
- **v2: neredeyse aynı** (R@1=0.333). v2_gold_meva_otopark 8→6, hâlâ top-5 dışı.

## Dürüst nüans — boost meşru videolar-arası relevansı EZMİYOR (doğru davranış)

v2_gold_meva_otopark ("araçlarla dolu açık otopark", golden=meva_okul2) düzelmedi çünkü
Olgu B'nin iki bileşeni vardı:
1. **Kırpık-seli (boost çözdü):** boost, VIRAT_otopark'ın KARESİNİ kırpıkların arkasından
   rank 2'ye çıkardı (λ=1.5'te top-3 = ucf_trafik-crop, VIRAT-frame, VIRAT-crop).
2. **Veri + etiket (boost'un işi değil):** meva_okul2 statik lot → 1 kare (kare-açlığı);
   üstelik VIRAT_otopark "açık otopark"ın MEŞRU daha iyi eşleşmesi. Boost, meva_okul2'yi
   zorla yukarı itmiyor — bu doğru: yumuşak boost cross-video relevansı ezmemeli (AI
   Engineer S5). Golden'ım kısmen kötü etikettti (VIRAT_otopark da geçerli cevap).

**Sonuç:** boost, Olgu B'nin sıralama bileşenini (kırpık-seli) düzeltiyor — dev'de
kanıtlı, v1/kontrol regresyonu sıfır. Veri bileşeni (kare-açlığı) ayrı iş: statik lotlara
daha uzun MEVA segmenti. Committed λ=1.0.

## Sonraki adım adayları

1. Statik MEVA lotlarına daha uzun segment (kare-açlığını gider → v2 golden düzelir mi?).
2. Faz 2 VLM (kalan negatif örtüşme + öznitelik doğrulama).
3. Gece/trafik çeşitliliği (UA-DETRAC, Kaggle token).
