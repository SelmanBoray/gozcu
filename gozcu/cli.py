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

app = typer.Typer(help="Gözcü — kamera arşivinde Türkçe doğal dil arama", no_args_is_help=True)

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".dav"}


@app.command()
def index(
    folder: Path,
    camera_id: str = typer.Option(None, help="Kamera kimliği (boşsa video adı kullanılır)"),
) -> None:
    """Klasördeki tüm videoları örnekle, embedle ve indeksle."""
    # ── Ağır importlar komut içinde: 'search --help' model yüklemesin ──
    from gozcu.embedder import Embedder
    from gozcu.sampler import FrameRecord, sample_video, video_start_ts
    from gozcu.store import FrameStore
    from gozcu.thumbs import write_thumb

    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        typer.echo(f"HATA: {folder} içinde video bulunamadı ({', '.join(sorted(VIDEO_EXTS))})")
        raise typer.Exit(1)

    typer.echo(f"Cihaz seçiliyor ve model yükleniyor ({settings.model_id})...")
    embedder = Embedder(device=settings.device)
    store = FrameStore()
    typer.echo(f"Model hazır (cihaz: {embedder.device}). {len(videos)} video indekslenecek.\n")

    total_kept = 0
    grand_start = time.perf_counter()

    for video in videos:
        # ── Video başına: örnekle → toplu embedle → thumb yaz → upsert ──
        cam = camera_id or video.stem
        base_ts = video_start_ts(video)
        video_start = time.perf_counter()
        kept = 0
        batch: list[FrameRecord] = []

        def flush(batch: list[FrameRecord]) -> None:
            vectors = embedder.encode_images([r.image for r in batch])
            thumbs = [str(write_thumb(r.image, r.video_id, r.frame_idx)) for r in batch]
            store.upsert(batch, vectors, thumbs)

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
        start_str = datetime.fromtimestamp(base_ts).strftime("%d.%m.%Y %H:%M:%S")
        typer.echo(
            f"  {video.name}: {kept} kare indekslendi ({elapsed:.1f} sn, "
            f"{kept / elapsed:.1f} kare/sn) — tahmini başlangıç: {start_str}"
        )

    grand_elapsed = time.perf_counter() - grand_start
    typer.echo(
        f"\nBitti: {total_kept} yeni kare, toplam süre {grand_elapsed:.1f} sn. "
        f"İndeksteki toplam kare: {store.count()}"
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

    if not outcome.results:
        typer.echo("Sonuç bulunamadı. Önce `python -m gozcu index <klasör>` çalıştırıldı mı?")
        raise typer.Exit(1)

    # ── Sonuç tablosu ──
    typer.echo(f"{'#':>2}  {'skor':>6}  {'video':<12} {'zaman':<19} {'offset':>8}  thumb")
    for i, hit in enumerate(outcome.results, 1):
        when = datetime.fromtimestamp(hit["ts"]).strftime("%d.%m.%Y %H:%M:%S")
        offset = f"{int(hit['offset_s'] // 60):02d}:{hit['offset_s'] % 60:04.1f}"
        typer.echo(
            f"{i:>2}  {hit['score']:>6.3f}  {hit['video_id']:<12} {when:<19} "
            f"{offset:>8}  {hit['thumb_path']}"
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
