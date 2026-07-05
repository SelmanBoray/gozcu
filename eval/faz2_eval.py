"""Faz 2 VLM eval — OBJEKTİF SKOR (otonom loop'un ölçtüğü metrik).

Her sorgu uçtan uca çalıştırılır (search use_vlm=True = CLIP + VLM doğrulama + füzyon),
çıktı beklentiyle (queries_faz2.yaml `expect`) eşleştirilir → pass/fail. Kova + genel
geçme oranı basılır. AI Engineer: "loop'un objektif skoru olmalı, yoksa his'le sürüklenir."

expect: not_found (bulunamadı/0 sonuç dönmeli) | has_results (≥1 sonuç dönmeli)

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


def _outcome_label(outcome) -> str:
    """Pipeline çıktısını sınıfla: 'not_found' | 'has_results'."""
    if outcome.not_found_reason or not outcome.results:
        return "not_found"
    return "has_results"


def main() -> None:
    from gozcu.config import settings
    from gozcu.search import search, vlm_available

    if not vlm_available():
        print(f"HATA: Ollama/{settings.vlm_model} erişilemez. `ollama serve` + model gerekli.")
        raise SystemExit(1)

    entries = yaml.safe_load((_ROOT / "eval" / "queries_faz2.yaml").read_text("utf-8"))
    print(f"{len(entries)} sorgu · VLM={settings.vlm_model} · uçtan uca (CLIP+VLM)\n")

    by_bucket: dict[str, list] = defaultdict(list)
    records = []
    for e in entries:
        outcome = search(e["query"], use_vlm=True)
        got = _outcome_label(outcome)
        expect = e["expect"]
        ok = got == expect
        n = len(outcome.results)
        by_bucket[e["bucket"]].append(ok)
        mark = "✅" if ok else "❌"
        print(f"{mark} [{e['bucket']:13}] {e['id']:22} \"{e['query']}\"")
        print(f"      beklenen={expect:11} → çıktı={got:11} ({n} sonuç, "
              f"elenen={len(outcome.vlm_filtered)})")
        records.append({"id": e["id"], "bucket": e["bucket"], "query": e["query"],
                        "expect": expect, "got": got, "pass": ok, "n_results": n})

    # ── Skor ──
    print("\n" + "=" * 55)
    total_ok = sum(1 for r in records if r["pass"])
    for bucket, oks in by_bucket.items():
        print(f"  [{bucket:13}] {sum(oks)}/{len(oks)} geçti")
    print(f"\n  GENEL SKOR: {total_ok}/{len(records)} "
          f"({100 * total_ok / len(records):.0f}%)")

    out = _ROOT / "experiments" / "2026-07-05_vlm_latency" / "faz2_eval_sonuc.json"
    out.write_text(json.dumps({"score": f"{total_ok}/{len(records)}", "records": records},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
