"""Komut satırı arayüzü.

Kullanım:
    python -m gozcu index C:/videolar/
    python -m gozcu search "dün gece giren beyaz Transit"
    python -m gozcu stats
    python -m gozcu ui
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import typer

from gozcu.config import settings

# ── Windows konsolunda Türkçe karakter güvencesi ──
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = typer.Typer(help="Vel'Koz — kamera arşivinde Türkçe doğal dil arama", no_args_is_help=True)

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".dav"}


@app.command()
def index(
    folder: Path,
    camera_id: str = typer.Option(None, help="Kamera kimliği (boşsa video adı kullanılır)"),
) -> None:
    """Klasördeki tüm videoları örnekle, embedle ve indeksle."""
    # ── Ağır importlar komut içinde: 'search --help' model yüklemesin ──
    from gozcu.detector import Detection, Detector, cap_crops, crop_image, suppress_static
    from gozcu.embedder import Embedder
    from gozcu.sampler import FrameRecord, sample_video, video_start_ts
    from gozcu.store import FrameStore
    from gozcu.thumbs import write_crop_thumb, write_thumb

    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        typer.echo(f"HATA: {folder} içinde video bulunamadı ({', '.join(sorted(VIDEO_EXTS))})")
        raise typer.Exit(1)

    typer.echo(f"Cihaz seçiliyor, modeller yükleniyor ({settings.model_id} + {settings.yolo_model})...")
    embedder = Embedder(device=settings.device)
    detector = Detector(device=settings.device)
    store = FrameStore()
    typer.echo(f"Modeller hazır (cihaz: {embedder.device}). {len(videos)} video indekslenecek.\n")

    total_kept = 0
    total_crops = 0
    grand_start = time.perf_counter()

    for video in videos:
        # ── Video başına: örnekle → embedle → tespit et → kırpıkları embedle → upsert ──
        cam = camera_id or video.stem
        base_ts = video_start_ts(video)
        video_start = time.perf_counter()
        kept = 0
        crops = 0
        batch: list[FrameRecord] = []
        prev_dets: list[Detection] = []  # statik nesne bastırma referansı

        def flush(batch: list[FrameRecord]) -> None:
            nonlocal prev_dets, crops
            # ── Kare vektörleri ──
            vectors = embedder.encode_images([r.image for r in batch])
            thumbs = [str(write_thumb(r.image, r.video_id, r.frame_idx)) for r in batch]
            store.upsert(batch, vectors, thumbs)

            # ── YOLO kırpıkları: tespit → statik bastırma → kırp → embedle ──
            items, crop_imgs, parent_thumbs = [], [], []
            for record, fthumb in zip(batch, thumbs):
                detections = detector.detect(record.image)
                fresh = cap_crops(suppress_static(detections, prev_dets))
                prev_dets = detections  # referans HAM tespitler — statik nesne hep bastırılır
                for k, det in enumerate(fresh):
                    img = crop_image(record.image, det)
                    if img.size == 0:
                        continue
                    items.append((record, det, k))
                    crop_imgs.append(img)
                    parent_thumbs.append(fthumb)
            if items:
                crop_vecs = embedder.encode_images(crop_imgs)
                crop_thumbs = [
                    str(write_crop_thumb(img, rec.video_id, rec.frame_idx, k))
                    for img, (rec, _, k) in zip(crop_imgs, items)
                ]
                store.upsert_crops(items, crop_vecs, parent_thumbs, crop_thumbs)
                crops += len(items)

        for record in sample_video(video, cam, base_ts):
            batch.append(record)
            kept += 1
            if len(batch) >= embedder.batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

        elapsed = time.perf_counter() - video_start
        total_kept += kept
        total_crops += crops
        start_str = datetime.fromtimestamp(base_ts).strftime("%d.%m.%Y %H:%M:%S")
        typer.echo(
            f"  {video.name}: {kept} kare + {crops} kırpık indekslendi ({elapsed:.1f} sn) "
            f"— tahmini başlangıç: {start_str}"
        )

    grand_elapsed = time.perf_counter() - grand_start
    typer.echo(
        f"\nBitti: {total_kept} kare + {total_crops} kırpık, toplam süre {grand_elapsed:.1f} sn. "
        f"İndeksteki toplam vektör: {store.count()}"
    )


@app.command()
def search(
    query: str,
    top_k: int = typer.Option(settings.default_top_k, help="Kaç sonuç gösterilsin"),
) -> None:
    """Türkçe sorguyla indekste ara, sonuçları tabloda göster."""
    from gozcu.search import search as run_search

    outcome = run_search(query, top_k=top_k)
    parsed = outcome.parsed

    # ── Ayrıştırma özeti: ne embedlendi, hangi zaman aralığı filtrelendi ──
    typer.echo(f'Görsel sorgu : "{parsed.visual_text or query}"')
    if parsed.ts_from is not None:
        fmt = "%d.%m.%Y %H:%M"
        span = (
            f"{datetime.fromtimestamp(parsed.ts_from):{fmt}} — "
            f"{datetime.fromtimestamp(parsed.ts_to):{fmt}}"
        )
        typer.echo(f'Zaman filtresi: "{parsed.time_phrase}" → {span}')
    if outcome.time_filter_dropped:
        typer.echo("UYARI: Zaman aralığında sonuç yok — filtre kaldırılıp tüm arşivde arandı.")
    typer.echo("")

    # ── Bulunamadı kapısı: sorgulanan nesne sınıfı korpusta yok ──
    if outcome.not_found_reason:
        typer.echo(f"BULUNAMADI: {outcome.not_found_reason}")
        raise typer.Exit(0)

    if not outcome.results:
        typer.echo("Sonuç bulunamadı. Önce `python -m gozcu index <klasör>` çalıştırıldı mı?")
        raise typer.Exit(1)

    # ── Sonuç tablosu ──
    typer.echo(f"{'#':>2}  {'skor':>6}  {'tür':<12} {'video':<14} {'zaman':<19} {'offset':>8}")
    for i, hit in enumerate(outcome.results, 1):
        when = datetime.fromtimestamp(hit["ts"]).strftime("%d.%m.%Y %H:%M:%S")
        offset = f"{int(hit['offset_s'] // 60):02d}:{hit['offset_s'] % 60:04.1f}"
        kind = f"🎯 {hit['yolo_class']}" if hit.get("source") == "crop" else "kare"
        typer.echo(
            f"{i:>2}  {hit['score']:>6.3f}  {kind:<12} {hit['video_id']:<14} {when:<19} "
            f"{offset:>8}"
        )


@app.command()
def stats() -> None:
    """İndeks istatistiklerini göster."""
    from gozcu.store import FrameStore

    store = FrameStore()
    typer.echo(f"İndeksteki toplam kare: {store.count()}")
    typer.echo(f"Qdrant yolu           : {settings.qdrant_path}")
    typer.echo(f"Küçük resim klasörü   : {settings.thumbs_dir}")


@app.command()
def ui() -> None:
    """Streamlit arayüzünü başlat."""
    viewer_path = Path(__file__).resolve().parent / "viewer.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(viewer_path)], check=False)


if __name__ == "__main__":
    app()
