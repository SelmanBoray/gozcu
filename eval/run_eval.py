"""Gözcü retrieval eval koşucusu (AI Engineer incelemesi sonrası tam sürüm).

Metriği etiket tipine eşler (gt_type):
  class/scene → video-düzeyi Recall@k (birleşik + kare + kırpık ayrı; Faz 1.5 marjini)
  golden      → gözle doğrulanmış tek kareye Recall@1/MRR + fail_attribution
  advisory    → yalnız top-1 gözle denetim (renk GT'si otomatik değil — recall'a girmez)
  zaman       → parse ayrıştırma + geri düşüş davranışı
  negatif     → ayrımcılık testi (pozitif vs negatif top-1 skor dağılımı; mutlak eşik yok)

Her agregata Wilson %95 CI basılır — n≈20'de sayılar GÖSTERGE, benchmark değil
(AI Engineer S5: R@5 CI yarı-genişliği ≈±0.15). Asıl değer sabit set üzerinde
eşleştirilmiş regresyon karşılaştırmasında.

Kullanım:
    .venv/Scripts/python.exe eval/run_eval.py
    .venv/Scripts/python.exe eval/run_eval.py --k 10 --queries eval/queries.yaml
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

# ── Paket kökünü yola ekle ──
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CUTS = (1, 5, 10)
SOURCES = (None, "frame", "crop")  # None = birleşik (prod hattı)

# ── Koşmadan ilan edilen provizyonel hedefler (post-hoc rasyonalizasyonu önler, S5) ──
TARGETS = {
    "video_scorable_recall@5": 0.80,
    "golden_recall@1": 0.66,        # 3 golden'ın en az 2'si rank-1
    "faz15_marginal_recall@5": 0.0, # kırpık kareyi bozmamalı (≥0), ideal >0
}


# ── İstatistik: Wilson skoru %95 güven aralığı ──

def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Bir oran için Wilson skoru aralığı (küçük n'de normal yaklaşımdan sağlam)."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ── Sıralama yardımcıları ──

def _first_relevant_rank(ranked: list[dict], relevant: set[str]) -> int | None:
    for i, h in enumerate(ranked, 1):
        if h["video_id"] in relevant:
            return i
    return None


def _golden_rank(ranked: list[dict], gv: str, gf: int) -> int | None:
    for i, h in enumerate(ranked, 1):
        if h["video_id"] == gv and h["frame_idx"] == gf:
            return i
    return None


def _raw_hits(query: str, fetch_k: int) -> list[dict]:
    """Prod hattının tekilleştirme ÖNCESI ham sonucu — golden fail_attribution için."""
    from gozcu.query import parse_query
    from gozcu.search import get_embedder, get_store

    parsed = parse_query(query)
    vec = get_embedder().encode_text(parsed.visual_text or query)
    hits = get_store().search(
        vec, top_k=fetch_k, ts_from=parsed.ts_from, ts_to=parsed.ts_to
    )
    if not hits and parsed.ts_from is not None:
        hits = get_store().search(vec, top_k=fetch_k)
    return hits


# ── Tek sorgu değerlendirmesi ──

def eval_query(entry: dict, k: int) -> dict:
    from gozcu.config import settings
    from gozcu.search import parse_query, search

    q = entry["query"]
    gt = entry["gt_type"]
    relevant = set(entry.get("relevant_videos") or [])

    # ── Üç kaynak varyantını koştur (kare/kırpık/birleşik) ──
    variants = {}
    outcome_combined = None
    for src in SOURCES:
        o = search(q, top_k=k, source=src)
        variants[src] = o.results
        if src is None:
            outcome_combined = o

    parsed = outcome_combined.parsed
    combined = variants[None]

    r: dict = {
        "id": entry["id"],
        "query": q,
        "gt_type": gt,
        "category": entry.get("category", "?"),
        "scorable_recall": entry.get("scorable_recall", False),
        "relevant_videos": sorted(relevant),
        # Boşluk 1 denetimi: ne embedlendi, ne parse edildi?
        "parsed_visual": parsed.visual_text,
        "time_phrase": parsed.time_phrase,
        "time_filter_dropped": outcome_combined.time_filter_dropped,
        "top1": {
            "video": combined[0]["video_id"] if combined else None,
            "source": combined[0].get("source") if combined else None,
            "yolo_class": combined[0].get("yolo_class") if combined else None,
            "score": round(combined[0]["score"], 3) if combined else None,
        } if combined else None,
        "top5": [
            {"rank": i, "video": h["video_id"], "source": h.get("source", "frame"),
             "yolo_class": h.get("yolo_class"), "score": round(h["score"], 3)}
            for i, h in enumerate(combined[:5], 1)
        ],
    }

    # ── Video-düzeyi sıralama (class/scene): üç varyant için de ──
    if gt in ("class", "scene"):
        ranks = {}
        for src in SOURCES:
            rank = _first_relevant_rank(variants[src], relevant)
            key = src or "combined"
            ranks[key] = rank
        r["video_ranks"] = ranks
        r["recall"] = {f"@{c}": bool(ranks["combined"] and ranks["combined"] <= c) for c in CUTS}
        r["rr"] = (1.0 / ranks["combined"]) if ranks["combined"] else 0.0

    # ── Golden-frame: gözle doğrulanmış tek kare + fail_attribution ──
    if gt == "golden":
        g = entry["golden_frame"]
        gv, gf = g["video_id"], g["frame_idx"]
        ranks = {}
        for src in SOURCES:
            ranks[src or "combined"] = _golden_rank(variants[src], gv, gf)
        r["golden_ranks"] = ranks
        r["recall"] = {f"@{c}": bool(ranks["combined"] and ranks["combined"] <= c) for c in CUTS}
        r["rr"] = (1.0 / ranks["combined"]) if ranks["combined"] else 0.0
        # Kaçırıldıysa: ham (dedup öncesi) sonuçta var mı? → dedup/kümeleme artefaktı mı?
        if ranks["combined"] is None:
            raw = _raw_hits(q, k * settings.search_overfetch)
            in_raw = _golden_rank(raw, gv, gf)
            r["fail_attribution"] = "dedup/kümeleme" if in_raw else "retrieval"
        else:
            r["fail_attribution"] = None

    # ── Negatif: ayrımcılık verisi (mutlak eşik YOK) ──
    if gt == "negatif":
        r["neg_type"] = entry.get("neg_type")
        # class_absent için: top-1 kırpığı güvenle yanlış sınıf mı sunuyor?
        r["top1_wrong_class"] = combined[0].get("yolo_class") if combined else None

    return r


# ── Agregasyon ──

def aggregate(results: list[dict]) -> dict:
    video = [r for r in results if r["gt_type"] in ("class", "scene") and r["scorable_recall"]]
    golden = [r for r in results if r["gt_type"] == "golden"]
    scorable = video + golden
    negs = [r for r in results if r["gt_type"] == "negatif"]

    def recall_block(rows: list[dict], src_key: str = "combined") -> dict:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        out = {"n": n}
        for c in CUTS:
            if src_key == "combined":
                hits = sum(1 for r in rows if r["recall"][f"@{c}"])
            else:
                key = "video_ranks" if rows[0]["gt_type"] in ("class", "scene") else "golden_ranks"
                hits = sum(1 for r in rows
                           if r.get(key, {}).get(src_key) and r[key][src_key] <= c)
            lo, hi = wilson(hits, n)
            out[f"recall@{c}"] = {"value": round(hits / n, 3),
                                  "ci95": [round(lo, 3), round(hi, 3)]}
        out["mrr"] = round(sum(r["rr"] for r in rows) / n, 3)
        return out

    # ── Makro-ortalama (kategori başına eşit ağırlık — baskınlığı kontrol eder, S2) ──
    by_cat = defaultdict(list)
    for r in scorable:
        by_cat[r["category"]].append(r)
    macro_r5 = None
    if by_cat:
        cat_r5 = [sum(1 for r in rows if r["recall"]["@5"]) / len(rows)
                  for rows in by_cat.values()]
        macro_r5 = round(sum(cat_r5) / len(cat_r5), 3)

    # ── Faz 1.5 marjinal değeri: R@5(birleşik) − R@5(yalnız-kare) ──
    def r5(rows, src):
        if not rows:
            return None
        key = None
        hits = 0
        for r in rows:
            key = "video_ranks" if r["gt_type"] in ("class", "scene") else "golden_ranks"
            rank = r.get(key, {}).get(src if src != "combined" else "combined")
            if rank and rank <= 5:
                hits += 1
        return hits / len(rows)

    faz15 = None
    if scorable:
        c5, f5 = r5(scorable, "combined"), r5(scorable, "frame")
        if c5 is not None and f5 is not None:
            faz15 = {"combined_r@5": round(c5, 3), "frame_only_r@5": round(f5, 3),
                     "crop_only_r@5": round(r5(scorable, "crop"), 3),
                     "marginal": round(c5 - f5, 3)}

    return {
        "video_level": {src or "combined": recall_block(video, src or "combined")
                        for src in SOURCES},
        "golden": {src or "combined": recall_block(golden, src or "combined")
                   for src in SOURCES},
        "scorable_all_combined": recall_block(scorable, "combined"),
        "macro_recall@5": macro_r5,
        "faz15_ablation": faz15,
        "discrimination": {
            "positive_top1_scores": sorted(
                r["top1"]["score"] for r in scorable if r.get("top1")),
            "negative_top1_scores": sorted(
                r["top1"]["score"] for r in negs if r.get("top1")),
            "negatives_detail": [
                {"id": r["id"], "neg_type": r.get("neg_type"),
                 "top1_score": r["top1"]["score"] if r.get("top1") else None,
                 "top1_wrong_class": r.get("top1_wrong_class")}
                for r in negs
            ],
        },
        "targets": TARGETS,
    }


# ── Markdown rapor ──

def write_markdown(agg: dict, results: list[dict], meta: dict) -> str:
    L: list[str] = []
    L.append(f"# Gözcü Eval — {meta['date']}")
    L.append("")
    L.append(f"- Sorgu: {meta['n_queries']} (skorlanabilir {meta['n_scorable']}, "
             f"advisory {meta['n_advisory']}, zaman {meta['n_zaman']}, negatif {meta['n_neg']})")
    L.append(f"- İndeks: {meta['index_count']} vektör · top_k={meta['k']} · model {meta['model']}")
    L.append("")
    L.append("> **n≈20 uyarısı (AI Engineer S5):** Bu sayılar GÖSTERGE + regresyon temeli, "
             "istatistiksel benchmark değil. R@5 için %95 CI yarı-genişliği ≈±0.15 — "
             "0.6 vs 0.9 ayırt edilir, 0.82 vs 0.88 EDİLMEZ. Asıl değer sabit sette "
             "eşleştirilmiş öncesi/sonrası karşılaştırmada.")
    L.append("")

    # ── Manşet ──
    sc = agg["scorable_all_combined"]
    L.append("## Manşet (skorlanabilir, birleşik hat)")
    L.append("")
    L.append("| Metrik | Değer | %95 CI | Hedef |")
    L.append("|---|---|---|---|")
    for c in CUTS:
        m = sc[f"recall@{c}"]
        L.append(f"| Recall@{c} | {m['value']:.3f} | {m['ci95']} | |")
    L.append(f"| MRR | {sc['mrr']:.3f} | | |")
    L.append(f"| Makro R@5 (kategori eşit ağırlık) | {agg['macro_recall@5']} | | "
             f"{TARGETS['video_scorable_recall@5']} |")
    L.append("")

    # ── Faz 1.5 ablation ──
    f = agg["faz15_ablation"]
    if f:
        L.append("## Faz 1.5 marjinal değeri (kırpık embedding)")
        L.append("")
        L.append(f"- Birleşik R@5: **{f['combined_r@5']}** · Yalnız-kare R@5: {f['frame_only_r@5']} "
                 f"· Yalnız-kırpık R@5: {f['crop_only_r@5']}")
        L.append(f"- **Marjinal katkı (birleşik − kare): {f['marginal']:+.3f}** "
                 f"(hedef ≥ {TARGETS['faz15_marginal_recall@5']}: "
                 f"{'✅' if f['marginal'] >= TARGETS['faz15_marginal_recall@5'] else '❌'})")
        L.append("")

    # ── Golden ──
    g = agg["golden"]["combined"]
    L.append("## Golden-frame (gözle doğrulanmış tek kare)")
    L.append("")
    if g.get("n"):
        L.append(f"- n={g['n']} · Recall@1={g['recall@1']['value']} "
                 f"(hedef {TARGETS['golden_recall@1']}) · Recall@5={g['recall@5']['value']} "
                 f"· MRR={g['mrr']}")
        for r in [x for x in results if x["gt_type"] == "golden"]:
            gc = r["golden_ranks"]["combined"]
            rank = gc if gc else f"KAÇTI ({r['fail_attribution']})"
            L.append(f"  - `{r['id']}` sıra={rank}")
    L.append("")

    # ── Negatif ayrımcılık ──
    d = agg["discrimination"]
    L.append("## Negatif ayrımcılık (S4 — mutlak eşik yok, göreli)")
    L.append("")
    L.append(f"- Pozitif top-1 skorları: {d['positive_top1_scores']}")
    L.append(f"- Negatif top-1 skorları: {d['negative_top1_scores']}")
    if d["positive_top1_scores"] and d["negative_top1_scores"]:
        pmin = min(d["positive_top1_scores"])
        nmax = max(d["negative_top1_scores"])
        L.append(f"- Ayrım marjı (min-pozitif − max-negatif): {pmin - nmax:+.3f} "
                 f"→ {'temiz ayrım' if pmin > nmax else 'ÖRTÜŞME (tek eşik ayırmaz)'}")
    L.append("")
    L.append("| negatif | tip | top-1 skor | top-1 sınıf |")
    L.append("|---|---|---|---|")
    for nd in d["negatives_detail"]:
        L.append(f"| {nd['id']} | {nd['neg_type']} | {nd['top1_score']} | {nd['top1_wrong_class']} |")
    L.append("")

    # ── Zaman ayrıştırma denetimi (Boşluk 1) ──
    L.append("## Zaman ayrıştırma (Boşluk 1 — 'gece' embed mi parse mi?)")
    L.append("")
    L.append("| id | görsel metin (embedlenen) | zaman ifadesi (parse) | filtre düştü |")
    L.append("|---|---|---|---|")
    for r in results:
        if r["gt_type"] in ("zaman", "scene") and ("gece" in r["query"] or r["gt_type"] == "zaman"):
            L.append(f"| {r['id']} | {r['parsed_visual']} | {r['time_phrase'] or '—'} | "
                     f"{'evet' if r['time_filter_dropped'] else 'hayır'} |")
    L.append("")

    # ── Tüm sorgu özeti ──
    L.append("## Tüm sorgular")
    L.append("")
    L.append("| id | tip | sorgu | top-1 | sıra |")
    L.append("|---|---|---|---|---|")
    for r in results:
        rank = "—"
        if "recall" in r:
            gc = (r.get("video_ranks") or r.get("golden_ranks") or {}).get("combined")
            rank = gc if gc else "❌"
        t1 = r["top1"]["video"] if r.get("top1") else "—"
        L.append(f"| {r['id']} | {r['gt_type']} | {r['query']} | {t1} | {rank} |")
    return "\n".join(L) + "\n"


# ── Ana akış ──

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(_ROOT / "eval" / "queries.yaml"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from gozcu.config import settings
    from gozcu.search import get_store

    entries = yaml.safe_load(Path(args.queries).read_text(encoding="utf-8"))
    print(f"{len(entries)} sorgu yüklendi. İndeks + model ısıtılıyor...")
    index_count = get_store().count()

    results = []
    for e in entries:
        r = eval_query(e, args.k)
        results.append(r)
        gc = (r.get("video_ranks") or r.get("golden_ranks") or {}).get("combined")
        mark = "∅" if e["gt_type"] == "negatif" else (gc or "—")
        print(f"  [{str(mark):>3}] {e['id']:<22} {e['query']}")

    agg = aggregate(results)

    counts = {t: sum(1 for e in entries if e["gt_type"] == t)
              for t in ("class", "scene", "golden", "advisory", "zaman", "negatif")}
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_queries": len(entries),
        "n_scorable": counts["class"] + counts["scene"] + counts["golden"],
        "n_advisory": counts["advisory"], "n_zaman": counts["zaman"], "n_neg": counts["negatif"],
        "index_count": index_count, "k": args.k, "model": settings.model_id,
    }

    date_slug = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(args.out) if args.out else (_ROOT / "experiments" / f"{date_slug}_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sonuc.json").write_text(
        json.dumps({"meta": meta, "aggregate": agg, "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "rapor.md").write_text(write_markdown(agg, results, meta), encoding="utf-8")

    sc = agg["scorable_all_combined"]
    f = agg["faz15_ablation"]
    print(f"\n=== MANŞET (skorlanabilir, birleşik) ===")
    print(f"  R@1={sc['recall@1']['value']}  R@5={sc['recall@5']['value']}  "
          f"R@10={sc['recall@10']['value']}  MRR={sc['mrr']}")
    print(f"  Makro R@5={agg['macro_recall@5']}")
    if f:
        print(f"  Faz 1.5 marjini (birleşik−kare) R@5: {f['marginal']:+.3f} "
              f"(kare {f['frame_only_r@5']} → birleşik {f['combined_r@5']})")
    print(f"Rapor: {out_dir}")


if __name__ == "__main__":
    main()
