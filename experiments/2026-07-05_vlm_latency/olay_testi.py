"""cluster_events testi — sonuçları olaylara kümeleme (zaman grounding)."""

from datetime import datetime

from gozcu.search import cluster_events, search


def main() -> None:
    for q in ["kırmızı araba", "yürüyen insan"]:
        o = search(q, top_k=12, use_vlm=False)
        ev = cluster_events(o.results)
        print(f"\n{q!r}: {len(o.results)} sonuç → {len(ev)} olay")
        for e in ev[:8]:
            f = datetime.fromtimestamp(e["first_ts"])
            l = datetime.fromtimestamp(e["last_ts"])
            span = "tek an" if e["count"] == 1 else f"~{int(e['last_ts'] - e['first_ts'])}s"
            print(f"  {e['video_id']:16} ilk={f:%H:%M:%S} son={l:%H:%M:%S} "
                  f"kare={e['count']} ({span})")


if __name__ == "__main__":
    main()
