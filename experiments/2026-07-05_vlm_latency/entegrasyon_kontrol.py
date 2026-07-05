"""Uçtan uca entegrasyon: gerçek search() hattı + yeni yes/no VQA verifier.

Renk sorgusu ve zor-kavram sorgusunda VLM verdict'leri + fusion doğru mu?
extract_vqa_targets → verify_hit → _fuse_verdicts zinciri çalışıyor mu?
"""

import time

from gozcu.query import extract_vqa_targets
from gozcu.search import search as run_search


def show(query: str) -> None:
    obj, color = extract_vqa_targets(query)
    print(f'\n=== "{query}"  → VQA hedef: nesne={obj!r} renk={color!r} ===')
    t0 = time.perf_counter()
    out = run_search(query, top_k=8, use_vlm=True)
    dt = time.perf_counter() - t0
    print(f"vlm_applied={out.vlm_applied}  sonuç={len(out.results)}  "
          f"elenen={len(out.vlm_filtered)}  ({dt:.1f}s)")
    if out.not_found_reason:
        print(f"  BULUNAMADI: {out.not_found_reason}")
    for h in out.results[:8]:
        v = h.get("_vlm")
        vs = "—" if v is None else (f"present={v['object_present']} "
                                    f"color={v['color_match']} conf={v['confidence']}")
        print(f"  {h.get('yolo_class','frame'):8} score={h['score']:.3f}  VLM[{vs}]")


def main() -> None:
    for q in ["kırmızı araba", "köpek gezdiren insan"]:
        show(q)


if __name__ == "__main__":
    main()
