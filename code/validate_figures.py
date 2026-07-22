#!/usr/bin/env python3
"""Validate the four manuscript-facing figure PDFs and their citations."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

EXPECTED = [f"Figure_{i}.pdf" for i in range(1, 5)]


def _command(name: str, path: Path) -> subprocess.CompletedProcess[str] | None:
    exe = shutil.which(name)
    if exe is None:
        return None
    return subprocess.run([exe, str(path)], capture_output=True, text=True, check=False)


def _layout(root: Path) -> tuple[Path, Path]:
    if (root / "manuscript.tex").is_file():
        return root / "manuscript.tex", root / "figures"
    return root / "manuscript" / "manuscript.tex", root / "manuscript" / "figures"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.root.resolve()
    manuscript, figures = _layout(root)
    errors: list[str] = []

    if not manuscript.is_file():
        errors.append(f"missing manuscript source: {manuscript}")
        cited: set[str] = set()
    else:
        text = manuscript.read_text(encoding="utf-8")
        cited = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text))
        if cited != set(EXPECTED):
            errors.append(f"manuscript figure citations differ: {sorted(cited)}")

    for name in EXPECTED:
        path = figures / name
        if not path.is_file() or path.stat().st_size < 5_000:
            errors.append(f"missing or implausibly small figure: {name}")
            continue
        if not path.read_bytes().startswith(b"%PDF-"):
            errors.append(f"invalid PDF signature: {name}")
            continue

        info = _command("pdfinfo", path)
        if info is not None:
            if info.returncode != 0:
                errors.append(f"pdfinfo failed for {name}")
            else:
                match = re.search(r"^Pages:\s+(\d+)", info.stdout, re.MULTILINE)
                if not match or int(match.group(1)) != 1:
                    errors.append(f"figure is not a one-page PDF: {name}")

        fonts = _command("pdffonts", path)
        if fonts is not None:
            if fonts.returncode != 0:
                errors.append(f"pdffonts failed for {name}")
            else:
                rows = [line.split() for line in fonts.stdout.splitlines()[2:] if line.strip()]
                for row in rows:
                    if len(row) >= 6 and row[4].lower() == "no":
                        errors.append(f"unembedded font in {name}: {' '.join(row[:2])}")
                        break

    if errors:
        print("FIGURE VALIDATION FAILED")
        for error in errors:
            print(f" - {error}")
        return 1

    print("FIGURE VALIDATION PASSED")
    print("Four cited, one-page manuscript figure PDFs are present; available font checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
