"""Hibrit LLM parser testi: kural-yolu (LLM'e uğramaz) vs OOV-yolu (LLM doldurur)."""

import time

from gozcu.query import augment_intent, extract_vqa_targets

CASES = [
    "kırmızı araba",        # kural → car/red (LLM YOK)
    "köpek gezdiren adam",  # kural → dog (hard-concept, LLM YOK)
    "siyah SUV",            # kural → SUV/black (LLM YOK)
    "kırmızı forklift",     # OOV → kural None → LLM: forklift/red
    "sarı vinç",            # OOV → LLM: crane/yellow
    "turuncu traktör",      # OOV → LLM: tractor/orange
]


def main() -> None:
    for q in CASES:
        rule = extract_vqa_targets(q)
        t = time.perf_counter()
        aug = augment_intent(q)
        dt = (time.perf_counter() - t) * 1000
        yol = "KURAL" if rule[0] is not None else "LLM"
        print(f"{q:22} kural={str(rule):24} → augment={str(aug):24} [{yol}, {dt:.0f}ms]")


if __name__ == "__main__":
    main()
