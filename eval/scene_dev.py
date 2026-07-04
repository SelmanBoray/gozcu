"""Olgu B — sahne-niyeti frame-boost kalibrasyonu ve doğrulaması.

Dev bataryası (queries_scene_dev.yaml) üzerinde λ süpürür:
  - Sahne-karesi recall@1/@5 (video ∈ relevant VE source==frame) — fix'in hedefi.
  - Nesne-kontrol recall@1 — regres ETMEMELİ (fix nesne-niyetine dokunmamalı).
  - z-skoru boşluğu (λ=0'da) — λ'yı metrikten değil skor-boşluğundan türetmek için.
  - Niyet-sınıflandırıcı doğruluğu (ayrı metrik, S4 etiket-sızıntısı denetimi).

Kullanım: .venv/Scripts/python.exe eval/scene_dev.py
"""

import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LAMBDAS = [0.0, 0.375, 0.75, 1.125]  # 0, λ/2, λ, 1.5λ (robustluk bandı)


def _pool(query: str, k: int) -> list[dict]:
    """Ham skorlu aday havuzu (dedup/boost YOK) — z-boşluğu ölçümü için."""
    from gozcu.query import parse_query
    from gozcu.search import get_embedder, get_store
    p = parse_query(query)
    vec = get_embedder().encode_text(p.visual_text or query)
    return get_store().search(vec, top_k=k)


def _zgap(query: str, relevant: set) -> float | None:
    """λ=0'da: en iyi relevant KARE ile onu gömen kırpıklar arası z-skor boşluğu."""
    pool = _pool(query, 40)
    if not pool:
        return None
    scores = [h["score"] for h in pool]
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5 or 1e-9
    z = lambda s: (s - mean) / std
    # en iyi relevant kare
    rel_frames = [h for h in pool if h["video_id"] in relevant and h.get("source") == "frame"]
    if not rel_frames:
        return None
    best_frame = max(rel_frames, key=lambda h: h["score"])
    # o kareden yüksek skorlu kırpıklar
    crops_above = [h for h in pool if h.get("source") == "crop" and h["score"] > best_frame["score"]]
    if not crops_above:
        return 0.0  # zaten gömülmüyor
    top_crop = max(crops_above, key=lambda h: h["score"])
    return z(top_crop["score"]) - z(best_frame["score"])


def _scene_frame_rank(results: list[dict], relevant: set) -> int | None:
    for i, h in enumerate(results, 1):
        if h["video_id"] in relevant and h.get("source") == "frame":
            return i
    return None


def _first_relevant_rank(results: list[dict], relevant: set) -> int | None:
    for i, h in enumerate(results, 1):
        if h["video_id"] in relevant:
            return i
    return None


def main() -> None:
    from gozcu.config import settings
    from gozcu.query import scene_or_object_intent
    from gozcu.search import parse_query, search

    entries = yaml.safe_load((_ROOT / "eval" / "queries_scene_dev.yaml").read_text("utf-8"))
    print(f"{len(entries)} dev sorgusu. Model ısıtılıyor...\n")

    # ── 1. Niyet-sınıflandırıcı doğruluğu (ayrı metrik) ──
    print("=== Niyet-sınıflandırıcı doğruluğu ===")
    ok = 0
    for e in entries:
        vt = parse_query(e["query"]).visual_text or e["query"]
        got = scene_or_object_intent(vt)
        match = got == e["intent"]
        ok += match
        if not match:
            print(f"  MİSMATCH: {e['id']} beklenen={e['intent']} bulundu={got}")
    print(f"  {ok}/{len(entries)} = {ok/len(entries):.2f}\n")

    # ── 2. z-boşluğu (λ kalibrasyon dayanağı) ──
    floods = [e for e in entries if e["kind"] == "scene_flood"]
    print("=== z-skoru boşluğu (λ=0, scene_flood) — λ bunun biraz üstü olmalı ===")
    gaps = []
    for e in floods:
        g = _zgap(e["query"], set(e["relevant_videos"]))
        if g is not None:
            gaps.append(g)
        print(f"  {e['id']:24} z-gap={g}")
    if gaps:
        gaps.sort()
        med = gaps[len(gaps) // 2]
        print(f"  → medyan gap={med:.3f}, maks={max(gaps):.3f}  (config λ={settings.scene_boost_lambda})\n")

    # ── 3. λ süpürmesi ──
    scenes = [e for e in entries if e["kind"] in ("scene_flood", "scene_noobj")]
    ctrls = [e for e in entries if e["kind"] == "object_ctrl"]
    print("=== λ süpürmesi ===")
    print(f"{'λ':>7} | {'sahne R@1':>9} {'sahne R@5':>9} | {'noobj R@1':>9} | {'ctrl R@1':>8}")
    orig = settings.scene_boost_lambda
    for lam in LAMBDAS:
        settings.scene_boost_lambda = lam
        sf1 = sf5 = 0
        no1 = 0
        c1 = 0
        n_no = 0
        for e in scenes:
            o = search(e["query"], top_k=10)
            r = _scene_frame_rank(o.results, set(e["relevant_videos"]))
            sf1 += bool(r and r <= 1)
            sf5 += bool(r and r <= 5)
            if e["kind"] == "scene_noobj":
                n_no += 1
                no1 += bool(r and r <= 1)
        for e in ctrls:
            o = search(e["query"], top_k=10)
            r = _first_relevant_rank(o.results, set(e["relevant_videos"]))
            c1 += bool(r and r <= 1)
        ns = len(scenes)
        print(f"{lam:>7.3f} | {sf1/ns:>9.2f} {sf5/ns:>9.2f} | {no1/max(n_no,1):>9.2f} | {c1/len(ctrls):>8.2f}")
    settings.scene_boost_lambda = orig


if __name__ == "__main__":
    main()
