"""Türkçe sorgu ayrıştırıcı — zamansal ifadeyi görsel ifadeden ayırır.

"dün gece siteye giren beyaz Transit" → görsel: "beyaz Transit siteye giriyor",
zaman filtresi: [dün 20:00, bugün 06:00].
Kural: zaman kelimeleri ASLA embedlenmez (en büyük kalite kaldıracı — ARCHITECTURE.md §4).
MVP: regex + dateparser(tr). Faz 2: lokal LLM ayrıştırıcı (Qwen3-4B, Ollama).
"""

from dataclasses import dataclass


@dataclass
class ParsedQuery:
    """Ayrıştırılmış sorgu: embedlenecek görsel metin + Qdrant filtreleri."""

    visual_text: str          # embedlenecek kısım ("beyaz Transit")
    ts_from: float | None     # epoch UTC
    ts_to: float | None
    camera_id: str | None


def parse_query(raw_query: str) -> ParsedQuery:
    """Ham Türkçe sorguyu görsel metin + zaman aralığına ayırır."""
    # ── 1. Regex ile zamansal kalıpları yakala ("dün gece", "bu sabah 8'den sonra") ──
    # ── 2. dateparser(languages=['tr']) ile epoch aralığına çevir ──
    # ── 3. Kalan metni görsel sorgu olarak döndür ──
    raise NotImplementedError("Faz 1 — sıradaki implementasyon adımı")
