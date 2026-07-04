"""Faz 2 VLM analiz/eval — CLIP top-N'i VLM ile doğrula, verdict + görüntü denetimi.

Her sorgu için CLIP top-N adayı alınır, her biri VLM ile doğrulanır; verdict'ler
basılır ve görüntüler görsel denetim için kaydedilir (VLM renk precision'ı GÖZLE
doğrulanmalı — AI Engineer: VLM'ler mavi/cyan'da zayıf, ölçmeden güvenme).

Kovalar: negation (VLM reddetmeli), attribute (renk doğrula), positive_ctrl (reddetme).

Kullanım: .venv/Scripts/python.exe eval/faz2_eval.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

N = 6  # doğrulanacak aday sayısı (denetim için)


def main() -> None:
    from gozcu.config import settings
    from gozcu.query import has_color, parse_query, translate_visual
    from gozcu.search import search
    from gozcu.verifier import is_available, verify_hit
    from gozcu.recrop import vlm_image_for_hit

    if not is_available():
        print(f"HATA: Ollama/{settings.vlm_model} erişilemez. `ollama serve` + `ollama pull` gerekli.")
        raise SystemExit(1)

    entries = yaml.safe_load((_ROOT / "eval" / "queries_faz2.yaml").read_text("utf-8"))
    out_dir = _ROOT / "experiments" / "2026-07-04_faz2_vlm"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    results = []
    by_bucket = defaultdict(list)
    print(f"{len(entries)} sorgu · VLM={settings.vlm_model} · top-{N} doğrulama\n")

    for e in entries:
        q = e["query"]
        vt = parse_query(q).visual_text or q
        en = translate_visual(vt)
        ask_color = has_color(vt)
        # ── CLIP-only top-N (VLM'siz) ──
        clip_hits = search(q, top_k=N, use_vlm=False).results
        print(f"=== [{e['bucket']}] {e['id']}: \"{q}\"  → EN: \"{en}\" (renk_sor={ask_color}) ===")
        rows = []
        for i, h in enumerate(clip_hits, 1):
            v = verify_hit(h, en, ask_color)
            # ── görüntüyü denetim için kaydet ──
            try:
                img = vlm_image_for_hit(h, out_size=384)
                img.save(out_dir / "images" / f"{e['id']}_{i}_{h['video_id']}_{h.get('source')}.jpg")
            except Exception:
                pass
            vs = (f"obj={v['object_present']} color={v['color_match']} conf={v['confidence']:.2f}"
                  if v else "VLM-HATA")
            print(f"  {i}. {h['video_id']:14} {h.get('source'):5} cls={str(h.get('yolo_class')):8} "
                  f"cos={h['score']:.3f} | {vs}")
            rows.append({"rank": i, "video": h["video_id"], "source": h.get("source"),
                         "yolo_class": h.get("yolo_class"), "cos": round(h["score"], 3),
                         "vlm": v})
        rec = {"id": e["id"], "bucket": e["bucket"], "query": q, "english": en,
               "ask_color": ask_color, "candidates": rows}
        results.append(rec)
        by_bucket[e["bucket"]].append(rec)
        print()

    # ── Özet: kova bazlı sinyaller ──
    print("=== ÖZET ===")
    for bucket, recs in by_bucket.items():
        print(f"\n[{bucket}] {len(recs)} sorgu")
        for r in recs:
            top = r["candidates"][0] if r["candidates"] else None
            if not top or not top["vlm"]:
                print(f"  {r['id']}: (veri yok)"); continue
            v = top["vlm"]
            # negation: obj_present=False iyi (reddetme). attribute: color_match=True iyi.
            if bucket == "negation":
                accepted = sum(1 for c in r["candidates"] if c["vlm"] and c["vlm"]["object_present"])
                print(f"  {r['id']}: {accepted}/{len(r['candidates'])} aday VLM'ce KABUL "
                      f"(düşük=iyi; false-accept)")
            elif bucket == "attribute":
                cmatch = sum(1 for c in r["candidates"] if c["vlm"] and c["vlm"]["color_match"])
                print(f"  {r['id']}: {cmatch}/{len(r['candidates'])} aday renk-eşleşti "
                      f"(görüntülerden GÖZLE doğrula)")
            else:  # positive_ctrl
                acc = sum(1 for c in r["candidates"] if c["vlm"] and c["vlm"]["object_present"])
                print(f"  {r['id']}: {acc}/{len(r['candidates'])} aday KABUL "
                      f"(yüksek=iyi; false-reject yok)")

    (out_dir / "sonuc.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nGörüntüler + sonuc.json: {out_dir}")


if __name__ == "__main__":
    main()
