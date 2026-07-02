"""Komut satırı arayüzü.

Kullanım:
    python -m gozcu index data/videolar/
    python -m gozcu search "dün gece giren beyaz Transit"
"""

from pathlib import Path

import typer

app = typer.Typer(help="Gözcü — kamera arşivinde Türkçe doğal dil arama")


@app.command()
def index(folder: Path) -> None:
    """Klasördeki tüm videoları örnekle, embedle ve indeksle."""
    # ── Klasörü tara → sampler → embedder (batch) → thumbs → store.upsert ──
    raise NotImplementedError("Faz 1 — sıradaki implementasyon adımı")


@app.command()
def search(query: str, top_k: int = 12) -> None:
    """Türkçe sorguyla indekste ara, sonuçları tabloda göster."""
    # ── search.search → terminale tablo bas (video, zaman, skor, thumb yolu) ──
    raise NotImplementedError("Faz 1 — sıradaki implementasyon adımı")


if __name__ == "__main__":
    app()
