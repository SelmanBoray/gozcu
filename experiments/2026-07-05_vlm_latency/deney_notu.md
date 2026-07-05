# VLM çağrı güvenilirliği — teşhis (5 Temmuz 2026)

**Sorun**: Faz 2 VLM doğrulaması ara sıra None döndürüyor (UI'da "doğrulanamadı").
Önce latency spike'ı (VRAM sayfalaması) sanıldı; asıl neden farklı çıktı.

**Donanım**: RTX 4070 Laptop 8GB, Windows 11. CLIP (jina-clip-v2 fp16 ~2GB) sürekli GPU'da.
**VLM**: qwen3-vl:2b (Ollama, Q4_K_M, 1.9GB). `format:json`, `temperature:0`, 384px recrop.

---

## Kesin teşhis: qwen3 thinking-modu kaçak üretimi (VRAM DEĞİL)

Bir renk sorgusunun ("kırmızı araba" → "red car") 9 CLIP adayı doğrulandı.
**Her görüntü 2× tekrar → birebir aynı sonuç. Tam deterministik, rastgelelik YOK.**

İki temiz rejim (`kacak_icerik.py`):

| Görüntü | ptok (vision) | gen token | Sonuç |
|---------|--------------|-----------|-------|
| 0,1,2,4,6,7 (6/9) | 1323–3910 | **21** | geçerli JSON |
| 3,5,8 (3/9) | 1191 | **2905** | `content=''` (BOŞ) |

Başarılı çağrılar yalnız **21 token** üretir → temiz JSON; latency prompt_eval (vision
token) ile ölçeklenir (ptok=1323 → 3.6s; ptok=3910 → 23s). Latency-vision ilişkisi ayrı
ve kabul edilebilir bir olgu.

Başarısız çağrılarda tam message nesnesi (`think_alani.py`):
```
done_reason = length          (n_ctx_slot=4096 doldu, truncate)
content     = ''              (nihai cevap HİÇ yazılmadı)
thinking    = 11863 karakter  ("is the main subject clearly visible? The image sh...")
```

→ Model belirsiz kırpıklarda **sonsuz düşünme döngüsüne** giriyor, JSON'u yazmadan
context'i doldurup truncate oluyor. Ollama thinking'i ayrı alana koyuyor → `content`
boş → `json.loads('')` patlar → None. `ollama show`: Capabilities → **thinking** (evet,
thinking modeli).

**~25s'lik "hatalar" timeout değil**: 22s'lik *generation* (2905 token), prefill değil.

---

## Elenen yazılım leverleri (hepsi kanıtlı başarısız)

| Lever | Sonuç | Kanıt |
|-------|-------|-------|
| `think:false` (API param) | ÇALIŞMIYOR — thinking sürüyor (11871 kr), `<think>` içeriğe sızıp normal görüntüleri de bozuyor (5/9→1/9) | `thinking_hipotezi.py`, `think_alani.py` |
| `num_predict:200` (emniyet) | token'ı keser ama 200'de model hâlâ düşünüyor → content yine boş | `think_alani.py` (C) |
| `/no_think` (qwen3 soft-switch) | GÜVENİLMEZ — 3 başarısızdan 2'sini düzeltti ama sağlam görüntüyü BOZDU (21→2902), latency oynak | `no_think_testi.py` |
| q8_0 KV cache (önceki oturum) | JSON bozdu + latency kötüleşti | (önceki) |
| num_ctx:2048 (önceki oturum) | yüksek-token görüntüleri kesti | (önceki) |
| flash attention | latency'ye fayda yok, zarar da yok → açık bırakıldı | server config |

Config şu an temiz: **flash on, default f16 KV, temperature 0, timeout 40s, tek retry.**
Kalıcı `OLLAMA_KV_CACHE_TYPE=q8_0` env var registry'den silindi (gelecekteki
başlatmalar temiz).

**Kök özet**: qwen3-vl:2b thinking-modu modeli; bu Ollama build'inde tek-atış JSON'u
garanti eden çalışan bir yazılım anahtarı yok. Belirsiz kırpıklarda %33 boş içerik.

---

## Karar (AI Engineer): thinking-modu modelini verify katmanından at

**B — non-thinking VLM + YES/NO VQA sözleşmesi.** Gerekçe: bir doğrulayıcının işi
sınırsız-süreli akıl yürütme değil, sınırlı-süreli deterministik sınıflandırma. Thinking
modeli yanlış araç; %33 boş içerik bir bug değil tasarımın doğal sonucu.

### Model: qwen2.5vl:3b (qwen3-vl:2b yerine)

Co-residency + doğruluk kapısı (`coresidency_kapisi.py`) — aynı 9 kırpık, CLIP resident:
- **9/9 geçerli JSON, done_reason hepsi `stop`** (hiç `length`/kaçak yok) → thinking-loop gitti.
- Ollama log: **`offloaded 37/37 layers to GPU`** — CPU offload YOK, CLIP ile sığıyor
  (VRAM 7467/8188 MiB). scheduler: "fits alongside existing models".

### Ama JSON şeması RUBBER-STAMP'liyor (kritik ara bulgu)

Ayırt-etme testi (`ayirt_etme_v2.py`) — `format:SCHEMA` ile:

| Sorgu | present | color_match=True |
|-------|---------|------------------|
| red car (pozitif) | 9/9 | 9/9 ✓ |
| **blue car (negatif)** | 9/9 | **9/9 ✗ FELAKET** |
| dog (negatif) | 4/9 ✗ | — |

Model kırmızı arabaya "mavi araba? evet, conf 1.0" diyor — renk sıfatını hiç dikkate
almıyor. Ama **düz yes/no**'da (`renk_gorebiliyor_mu.py`): "What color?" → **Red 9/9**,
"Is it blue?" → **No 9/9**. Model KÖR DEĞİL — **JSON/şema çerçevesi** rubber-stamp'e zorluyor.

### Çözüm: YES/NO VQA sözleşmesi (`yesno_sozlesme.py`)

Sorguyu (nesne, renk) hedeflerine indir; iki ayrı atomik yes/no sor:

| Sorgu | present | color=True | latency |
|-------|---------|-----------|---------|
| red car (pozitif) | 9/9 ✓ | **9/9** ✓ | 4.8s |
| blue car (negatif) | 9/9 | **0/9** ✓ | 4.8s |
| dog (negatif) | **0/9** ✓ | — | 2.3s |

**Her eksende kazanç**: reliability %100 (thinking-loop yok), ayrım kusursuz (kırmızı vs
mavi 9/9↔0/9, köpek 0/9), latency 4.8s/2.3s (JSON'un 16.8s'inden 3.5× hızlı), VRAM sığıyor.

İçgörü: **küçük VLM'ler karmaşık yapılandırılmış JSON'da rubber-stamp'liyor ama atomik
yes/no'da kusursuz ayırıyor** — verify katmanının zaten yalnız boolean'a ihtiyacı var.

### Entegrasyon (`entegrasyon_kontrol.py` — uçtan uca doğrulandı)

- `query.extract_vqa_targets(text)` → (nesne_en, renk_en); zor kavram (köpek) sınıf isminden önceliklidir.
- `verifier.verify_hit(hit, obj_en, color_en)` → yes/no VQA; `num_predict:4` kaçağı yapısal imkânsız kılar.
- Gerçek hat: "köpek gezdiren insan" → hedef `dog`, **4 yanlış-pozitif elendi**, 1 köpekli kaldı. ✓

**Latency notu**: renk sorgusu = kart başına 2 çağrı (~7s), top-8 için ~50s toplam
(cold-load dahil) — per-item streaming ilerlemeyi gösterir, kalite>hız ürününde kabul.
`vlm_timeout_s` 40→20s (atomik yes/no ~2-5s). Optimizasyon burada DURUR (AI Engineer).
