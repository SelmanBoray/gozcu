"""Füzyon değişikliği (present=False her iki modda düşer) iyi renk vakasını bozdu mu?
Kırmızı arabalar hâlâ kalmalı (present=True → düşmez)."""

from gozcu.search import search


def main() -> None:
    for q in ["kırmızı araba", "siyah SUV araç"]:
        o = search(q, top_k=8, use_vlm=True)
        print(f"{q!r}: gösterilen={len(o.results)} elenen={len(o.vlm_filtered)} "
              f"nf={o.not_found_reason}")
        for h in o.results[:5]:
            v = h.get("_vlm")
            vs = "-" if v is None else f"pres={v['object_present']} col={v['color_match']}"
            print(f"   {h.get('yolo_class', 'frame'):7} {h['score']:.3f} [{vs}]")
        print()


if __name__ == "__main__":
    main()
