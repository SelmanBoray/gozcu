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


# ── Ayrım ekseni (hard-negative): yasaklı YOLO sınıfı sonuçlara sızmasın ──
def _forbid_set(entry) -> set:
    """queries yaml `forbid` alanını gerçek YOLO etiket setine çevir.

    'vehicles' → gozcu.query.VEHICLE_CLASSES (TEK KAYNAK). Depodaki yolo_class TÜRKÇE
    (araba/kamyon/otobüs) — İngilizce liste elle yazılırsa eşleşme hiç olmaz, kontrol
    sessizce boşa geçer ve sahte-PASS üretir (AI Engineer teşhisi). Sembolik değer bunu önler.
    """
    from gozcu.query import VEHICLE_CLASSES
    forbid = entry.get("forbid")
    if forbid == "vehicles":
        return set(VEHICLE_CLASSES)
    return set(entry.get("forbid_classes", []))  # opsiyonel açık kaçış (Türkçe etiketler)


def _forbidden_hits(hits, forbid_set) -> list:
    """Yasaklı sınıfa sahip crop-kaynaklı hit'ler. Frame hit'te yolo_class YOK → güvenle atlanır
    (doğru bir insan sonucu frame-source dönebilir; onu yanlış-fail etmeyiz)."""
    return [h for h in hits
            if h.get("source") == "crop" and h.get("yolo_class") in forbid_set]


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
        n = len(outcome.results)

        # ── Eksen 1: outcome (mevcut) — 'any' → iddia edilmez, hep geçer ──
        outcome_ok = expect == "any" or got == expect

        # ── Eksen 2: ayrım (yalnız `forbid` olan sorgu) — outcome'dan ORTOGONAL ──
        # sızan = yasaklı sınıf SONUÇLARA girmiş (kötü); elenen = VLM'in yakalayıp attığı (iyi).
        forbid_set = _forbid_set(e)
        discrim_ok = None
        leaked = caught = 0
        if forbid_set and not outcome.vlm_unavailable:  # VLM çökükse ayrım fair asserte edilemez
            leaked = len(_forbidden_hits(outcome.results, forbid_set))
            caught = len(_forbidden_hits(outcome.vlm_filtered, forbid_set))
            discrim_ok = leaked == 0

        ok = outcome_ok and discrim_ok is not False
        by_bucket[e["bucket"]].append(ok)
        if discrim_ok is not None:
            by_bucket["_ayrım"].append(discrim_ok)  # çapraz-kesit özet satırı

        mark = "✅" if ok else "❌"
        print(f"{mark} [{e['bucket']:13}] {e['id']:22} \"{e['query']}\"")
        print(f"      beklenen={expect:11} → çıktı={got:11} ({n} sonuç, "
              f"elenen={len(outcome.vlm_filtered)})")
        if forbid_set:
            if discrim_ok is None:
                print("      ayrım: VLM erişilemez — atlandı")
            elif leaked == 0 and caught == 0:
                # Yasaklı aday hiç gelmemiş → forbid tetiklenmedi (vacuous PASS — sahte güven değil,
                # ama düzeltmeyi de kanıtlamadı). Dürüstçe işaretle.
                print("      ayrım: sızan-araç=0  ⚠ havuzda araç adayı yok (vacuous — fix tetiklenmedi)")
            else:
                print(f"      ayrım: sızan-araç={leaked}  (VLM {caught} araç adayını eledi ✓)")
        records.append({"id": e["id"], "bucket": e["bucket"], "query": e["query"],
                        "expect": expect, "got": got, "pass": ok, "n_results": n,
                        "discrim_ok": discrim_ok, "leaked": leaked, "caught": caught})

    # ── Skor ──
    print("\n" + "=" * 55)
    total_ok = sum(1 for r in records if r["pass"])
    for bucket, oks in by_bucket.items():
        if bucket == "_ayrım":
            continue  # çapraz-kesit — aşağıda ayrı basılır
        print(f"  [{bucket:13}] {sum(oks)}/{len(oks)} geçti")
    discrim = by_bucket.get("_ayrım", [])
    if discrim:
        print(f"  [AYRIM]         {sum(discrim)}/{len(discrim)} — araç sızmadı (hard-negative ekseni)")
    print(f"\n  GENEL SKOR: {total_ok}/{len(records)} "
          f"({100 * total_ok / len(records):.0f}%)")

    out = _ROOT / "experiments" / "2026-07-05_vlm_latency" / "faz2_eval_sonuc.json"
    out.write_text(json.dumps({
        "score": f"{total_ok}/{len(records)}",
        "discrimination": f"{sum(discrim)}/{len(discrim)}" if discrim else "n/a",
        "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
