# Gözcü 👁️

**Güvenlik kamerası arşivinde Türkçe doğal dil arama motoru.**
_Ask your security-camera archive in plain Turkish — Gözcü finds the moment in seconds. Fully local._

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Qdrant" src="https://img.shields.io/badge/Qdrant-vektör%20db-DC244C">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-VLM-000000">
  <img alt="Local-first" src="https://img.shields.io/badge/veri-%25100%20lokal%20(KVKK)-2ea44f">
</p>

---

> 🔍 **"dün gece otoparka giren siyah SUV"**
> 🔍 **"kırmızı kıyafetli adam"**
> 🔍 **"telefonla konuşan sürücü"**
> 🔍 **"araçlarla dolu açık otopark"**

Bir olay olduğunda saatlerce kayıt taramak yerine, **arşive soru sorarsın.** Gözcü ilgili
anları saniyeler içinde bulur, doğrudan o saniyeye atlatır — ve **hiçbir görüntü binadan çıkmaz.**

---

## Neden Gözcü?

- 🏢 Site yönetimleri, AVM'ler ve fabrikalar bir olay olduğunda **saatlerce video izliyor.**
- 💸 Yurt dışı muadili **Conntour**, Mart 2026'da General Catalyst + Y Combinator'dan **$7M** aldı — pazar doğrulanmış.
- 🇹🇷 Türkiye'de muadili **yok**; **Türkçe doğal dil** desteği kimsede yok — üstelik çeviri katmanı olmadan, doğrudan.
- 🔒 **KVKK avantajı:** her şey lokalde çalışır (orta seviye GPU + CPU fallback). Görüntü, embedding, sorgu — hiçbiri dışarı gitmez.

---

## Öne çıkan mühendislik

Gözcü bir "CLIP'e sor" demosu değil; her katmanı gerçek CCTV'nin zorluklarına göre ölçülüp tasarlandı.

- **🌐 Çeviri yok, doğrudan Türkçe embedding.** `jinaai/jina-clip-v2` (1024-boyut, çok dilli)
  Türkçe sorguyu doğrudan görsel uzayına gömer. "telefonla konuşan sürücü" → davranış düzeyinde eşleşir; çeviri kaybı yok, +0 gecikme.

- **🎯 Küçük özneler için YOLO-kırpık embedding.** 1080p geniş planda uzaktaki bir insan 40 pikseldir ve
  tam-kare embedding'de kaybolur. Gözcü `yolo11m@1280` ile insan/araç kırpıklarını **ayrı vektör** olarak indeksler → "uzakta yürüyen tek kişi" bile bulunur. (Eval'de +0.33 recall marjini.)

- **🧠 Türkçe durum-eki (case morphology) ile niyet ayrımı.** "araçlar**la** dolu açık otopark" → sahne;
  "otopark**ta** yürüyen insan" → nesne. Gözcü baş ismi (head) **çekim ekinden** çıkarır (locative/instrumental/genitive → adjunct), pozisyondan değil — kırpık selinin sahne sorgusunu gömmesini engeller.

- **🚪 Halüsinasyon yerine dürüst "bulunamadı".** Skor eşiğine güvenmek yerine, korpusun **YOLO sınıf envanterine**
  bakar: aranan nesne kayıtlarda hiç tespit edilmemişse, makul-ama-yanlış sonuç uydurmaz — "bu nesne arşivde yok" der.

- **✅ Retrieve-then-verify: küçük VLM'i doğru kullanmak.** CLIP adayları `qwen2.5vl:3b` ile
  tam çözünürlükte doğrulanır. Kritik ders: küçük VLM'ler karmaşık JSON şemasında **rubber-stamp'liyor**
  (kırmızı arabaya "mavi araba? evet" diyor), ama **atomik yes/no** sorularında rengi kusursuz ayırıyor (kırmızı 9/9 ↔ mavi 0/9). Gözcü sorguyu (nesne, renk) hedeflerine indirip yes/no sorar → CLIP'in yanlış getirdiği kırmızı arabayı "kırmızı kıyafetli adam" sorgusunda eler.

- **📏 Ölçüm disiplini.** 22 sorgu dondurulmuş (pre-registration), etiket tipine göre metrik
  (video-düzeyi Recall@k / doğrulanmış golden-frame / advisory), Wilson %95 GA, McNemar eşleştirmeli karşılaştırma. Her model/eşik değişikliği bu sabit sette test edilir.

- **⚡ İnteraktif per-item streaming.** CLIP sonucu ~0.1s'de gösterilir; VLM her adayı
  sırayla doğrulayıp o kartın rozetini **canlı** doldurur (⏳ → ✅ / 🚫). Kullanıcı beklemez.

---

## Mimari

```mermaid
flowchart TD
    subgraph Offline["📼 Offline indeksleme"]
        V[Video arşivi] --> S[Akıllı kare örnekleme<br/>hareket kapısı · dedup · OSD maskesi]
        S --> Y[YOLO11m@1280<br/>insan/araç kırpıkları]
        S --> E[jina-clip-v2<br/>kare + kırpık embedding]
        Y --> E
        E --> Q[(Qdrant<br/>lokal vektör db)]
    end

    subgraph Online["🔍 Sorgu anı"]
        T["Türkçe sorgu<br/>'dün gece giren siyah SUV'"] --> P[Ayrıştırıcı<br/>zaman ⟂ görsel · niyet · bulunamadı kapısı]
        P --> EM[Sorgu embedding]
        EM --> Q
        Q --> R[Top-k aday<br/>kare + zaman damgası]
        R --> VLM[qwen2.5vl:3b<br/>yes/no VQA doğrulama]
        VLM --> UI[Sonuç · canlı rozet · videoyu o saniyeden oynat]
    end
```

Her kararın gerekçesi: **[ARCHITECTURE.md](ARCHITECTURE.md)** · Proje günlüğü: **[PROGRESS.md](PROGRESS.md)**

---

## Teknoloji yığını

| Katman | Teknoloji | Not |
|--------|-----------|-----|
| Görsel/metin embedding | `jinaai/jina-clip-v2` (1024-d, çok dilli) | GPU fp16, Türkçe doğrudan |
| Nesne tespiti | `yolo11m` @ 1280px | küçük özne kırpık embedding |
| Vektör veritabanı | Qdrant (lokal mod) | zaman/kamera/kaynak filtreli arama |
| VLM doğrulayıcı | `qwen2.5vl:3b` (Ollama) | non-thinking, atomik yes/no VQA |
| Arayüz | Streamlit | per-item streaming, videoyu o saniyeden oynat |
| Sorgu ayrıştırma | kural-tabanlı (Türkçe morfoloji) | zaman ⟂ görsel, niyet, bulunamadı kapısı |

**Donanım hedefi:** RTX 4070 Laptop (8GB) — CLIP + VLM aynı kartta eşzamanlı sığar. Her şey lokal.

---

## Durum & yol haritası

| Faz | Kapsam | Durum |
|-----|--------|-------|
| **1 — MVP indeksleyici** | Video → aranabilir indeks → Türkçe sorgu → kare + zaman | ✅ Tamam |
| **1.5 — YOLO-kırpık embedding** | Küçük özneler için ayrı kırpık vektörleri | ✅ Tamam |
| **2 — VLM doğrulama** | Retrieve-then-verify, yes/no VQA, renk/negasyon | ✅ Tamam |
| **3 — Gerçek pilot** | Bir sitenin 1 haftalık arşivi, saha testi | ⏳ Sırada |
| **İleri** | Araç takibi + plaka ile "ne zaman girdi/çıktı" zaman çifti | 🔭 Planlı |

Güncel korpus: **13 kamera/video · ~4.400 vektör** (kare + YOLO kırpık) — VIRAT, MEVA ve UCF trafik/gece kayıtları.

---

## Kurulum & kullanım

```bash
# Bağımlılıklar (jina-clip-v2 için transformers<4.50 sabit)
pip install -r requirements.txt

# VLM doğrulayıcı (opsiyonel — renk/negasyon sorguları için)
ollama pull qwen2.5vl:3b

# Video klasörünü indeksle
python -m gozcu index <video_klasörü>

# Terminalden ara
python -m gozcu search "dün gece otoparka giren siyah SUV"

# Arayüzü aç
python -m gozcu ui        # → http://localhost:8501
```

---

## Gizlilik önce (KVKK)

Gözcü'nün en büyük satış argümanı aynı zamanda mimari ilkesi: **hiçbir veri dışarı çıkmaz.**
Görüntüler, embedding'ler, sorgular ve VLM çağrılarının tamamı lokalde çalışır — bulut yok, API yok, üçüncü taraf yok. Bu, güvenlik kamerası verisiyle çalışmanın tek etik ve yasal yolu.

---

<p align="center">
  <sub>Türkçe için, Türkçe ile. Bir Teknofest yan-ürünü olarak başladı, bağımsız bir ürüne dönüşüyor.</sub>
</p>
