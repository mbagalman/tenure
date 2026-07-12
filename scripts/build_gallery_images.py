"""Regenerate the committed gallery images from the gallery pages' own code blocks.

The markdown pages under docsite/gallery/ are the single source of truth: this script extracts
their ```python blocks, executes each page in a fresh namespace (headless), and saves every
figure the page created as docsite/gallery/img/<page>-fig<N>.png. Run it from the repo root
after changing a gallery page; the images are deterministic (fixed seeds), so a no-op run
produces identical files.

Pages carrying a `<!-- requires-file: <path> -->` marker are skipped when the file is absent
(e.g. the Telco CSV, which is never vendored).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GALLERY = Path(__file__).resolve().parents[1] / "docsite" / "gallery"
IMG = GALLERY / "img"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
REQUIRES = re.compile(r"<!--\s*requires-file:\s*(\S+)\s*-->")


def main() -> int:
    IMG.mkdir(exist_ok=True)
    for page in sorted(GALLERY.glob("*.md")):
        if page.name == "index.md":
            continue
        text = page.read_text(encoding="utf-8")
        missing = [m for m in REQUIRES.findall(text) if not Path(m).exists()]
        if missing:
            print(f"skip {page.name}: missing {missing}")
            continue
        code = "\n\n".join(BLOCK.findall(text))
        plt.close("all")
        namespace: dict = {}
        exec(compile(code, str(page), "exec"), namespace)  # noqa: S102 - our own docs code
        for i, num in enumerate(plt.get_fignums(), start=1):
            out = IMG / f"{page.stem}-fig{i}.png"
            plt.figure(num).savefig(out, dpi=110, bbox_inches="tight")
            print(f"wrote {out.relative_to(GALLERY.parent.parent)}")
        plt.close("all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
