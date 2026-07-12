"""Execute every gallery page's code blocks -- the doc-rot gate for the example gallery.

The gallery markdown is the single source of truth (no separate example scripts to drift).
Each page's ```python blocks are concatenated and executed in a fresh namespace; a page that
declares `<!-- requires-file: <path> -->` is skipped when that file is absent (the Telco CSV
is user-downloaded, never vendored).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GALLERY = Path(__file__).resolve().parents[1] / "docsite" / "gallery"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
REQUIRES = re.compile(r"<!--\s*requires-file:\s*(\S+)\s*-->")

PAGES = sorted(p for p in GALLERY.glob("*.md") if p.name != "index.md")


def test_gallery_pages_exist():
    assert PAGES, "the gallery has no example pages"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_gallery_page_code_runs(page):
    text = page.read_text(encoding="utf-8")
    missing = [m for m in REQUIRES.findall(text) if not Path(m).exists()]
    if missing:
        pytest.skip(f"requires local file(s): {missing}")
    blocks = BLOCK.findall(text)
    assert blocks, f"{page.name} has no python blocks"
    code = "\n\n".join(blocks)
    plt.close("all")
    try:
        exec(compile(code, str(page), "exec"), {})  # noqa: S102 - executing our own docs
    finally:
        plt.close("all")
