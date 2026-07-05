"""Selman'ın bildirdiği iki sorgu — gerçek hat + verdict teşhisi.
'köpek gezdiren adam' ve 'kırmızı kıyafetli adam' neden yanlış gösterdi?
"""

from gozcu.query import extract_vqa_targets
from gozcu.search import search as run_search, vlm_available


def show(query: str) -> None:
    obj, color = extract_vqa_targets(query)
    print(f'\n=== "{query}"  → hedef nesne={obj!r} renk={color!r}  '
          f'| VLM hazır={vlm_available()} ===')
    out = run_search(query, top_k=8, use_vlm=True)
    print(f"vlm_applied={out.vlm_applied}  gösterilen={len(out.results)}  "
          f"elenen={len(out.vlm_filtered)}")
    if out.not_found_reason:
        print(f"  BULUNAMADI: {out.not_found_reason}")
    for h in out.results[:8]:
        v = h.get("_vlm")
        vs = "—" if v is None else (f"present={v['object_present']} "
                                    f"color={v['color_match']} conf={v['confidence']}")
        print(f"  {h.get('yolo_class', 'frame'):8} {h.get('video_id','?'):16} "
              f"score={h['score']:.3f}  VLM[{vs}]")
    if out.vlm_filtered:
        print(f"  --- elenenler ({len(out.vlm_filtered)}) ---")
        for h in out.vlm_filtered[:8]:
            print(f"  {h.get('yolo_class','frame'):8} {h.get('video_id','?'):16} "
                  f"score={h['score']:.3f}")


def main() -> None:
    for q in ["köpek gezdiren adam", "kırmızı kıyafetli adam"]:
        show(q)


if __name__ == "__main__":
    main()
