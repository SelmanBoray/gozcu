"""siyah SUV adaylarının TAM-KARE+kutu doğrulama görüntülerini + verdict'lerini kaydet.
7→2 eleme doğru mu (gerçekten siyah SUV değiller mi) yoksa fazla-eleme mi? Gözle bak.
"""

from pathlib import Path

from gozcu.query import extract_vqa_targets
from gozcu.recrop import vlm_frame_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import verify_hit

OUT = Path(r"C:\Users\Selman\AppData\Local\Temp\claude\C--\246f84c1-3962-4f56-8c68-6fbcb5b797c9\scratchpad\suv_frames")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    obj, color = extract_vqa_targets("siyah SUV araç")
    out = run_search("siyah SUV araç", top_k=8, use_vlm=False)
    for i, h in enumerate(out.results[:8]):
        v = verify_hit(h, obj, color)
        img = vlm_frame_for_hit(h, draw_box=True)
        verdict = "ERR" if v is None else f"pres{int(v['object_present'])}_col{v.get('color_match')}"
        fn = OUT / f"{i}_{h.get('yolo_class','?')}_{verdict}.jpg"
        img.save(fn, quality=88)
        print(f"[{i}] {h.get('yolo_class','?'):7} score={h['score']:.3f} → {verdict}  {fn.name}")


if __name__ == "__main__":
    main()
