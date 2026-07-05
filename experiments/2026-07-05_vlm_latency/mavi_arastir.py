"""Eval flag'i: "mavi kamyonet" 5 sonuç döndü (beklenen bulunamadı). Bu araçlar gerçekten
mavi mi (ground-truth yanlış) yoksa VLM mavi-yanlış-kabulü mü? VLM'in gördüğü tight-kırpığı
+ verdict'i kaydet, gözle bak. AI Engineer: VLM mavi/cyan'da zayıf (~%56 F1).
"""

from pathlib import Path

from gozcu.query import extract_vqa_targets
from gozcu.recrop import vlm_image_for_hit
from gozcu.search import search as run_search
from gozcu.verifier import verify_hit

OUT = Path(r"C:\Users\Selman\AppData\Local\Temp\claude\C--\246f84c1-3962-4f56-8c68-6fbcb5b797c9\scratchpad\mavi")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    obj, color = extract_vqa_targets("mavi kamyonet")
    print(f"hedef nesne={obj!r} renk={color!r}\n")
    out = run_search("mavi kamyonet", top_k=8, use_vlm=False)
    for i, h in enumerate(out.results[:8]):
        v = verify_hit(h, obj, color)
        vs = "ERR" if v is None else f"pres{int(v['object_present'])}_col{v.get('color_match')}"
        img = vlm_image_for_hit(h, out_size=384)  # renk yolu = tight-kırpık (VLM'in gördüğü)
        fn = OUT / f"{i}_{h.get('yolo_class','?')}_{vs}.jpg"
        img.save(fn, quality=90)
        print(f"[{i}] {h.get('yolo_class','?'):8} cos={h['score']:.3f} → {vs}  {fn.name}")


if __name__ == "__main__":
    main()
