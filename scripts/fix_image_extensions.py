#!/usr/bin/env python3
"""
fix_image_extensions.py
Recursively scans a folder, detects the real format of image files,
and renames them to the correct extension.

Usage:
    python3 fix_image_extensions.py /path/to/your/godot/assets
"""

import os
import sys
from pathlib import Path

# Map of file magic bytes -> correct extension
# Each entry: (bytes_to_check, offset, correct_extension)
SIGNATURES = [
    (b"\x89PNG\r\n\x1a\n",   0, ".png"),
    (b"\xff\xd8\xff",         0, ".jpg"),
    (b"RIFF",                 0, ".webp"),   # WebP: RIFF....WEBP
    (b"DDS ",                 0, ".dds"),
    (b"\xabKTX",              0, ".ktx"),
    (b"BM",                   0, ".bmp"),
    (b"GIF87a",               0, ".gif"),
    (b"GIF89a",               0, ".gif"),
    (b"\x00\x00\x01\x00",     0, ".ico"),
    (b"#?RADIANCE",           0, ".hdr"),
    (b"PF",                   0, ".hdr"),  # Portable FloatMap
]

# Extensions we consider "candidate image files" to inspect
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".dds", ".ktx",
                    ".bmp", ".gif", ".tga", ".hdr", ".exr", ".ico", ".tiff", ".tif"}


def detect_format(filepath: Path) -> str | None:
    """Read the first bytes of a file and return the correct extension, or None."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(16)
    except OSError:
        return None

    for magic, offset, ext in SIGNATURES:
        if header[offset:offset + len(magic)] == magic:
            # Extra check for WebP: bytes 8-11 must be b"WEBP"
            if ext == ".webp" and header[8:12] != b"WEBP":
                continue
            return ext

    # TGA has no reliable magic — treat any unmatched file with a .tga extension as-is
    if filepath.suffix.lower() == ".tga":
        return ".tga"

    return None  # Unknown / not an image


def fix_extensions(root: str):
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        print(f"ERROR: '{root}' is not a valid directory.")
        sys.exit(1)

    print(f"Scanning: {root_path}\n")

    renamed = 0
    skipped = 0
    unknown = 0

    for filepath in sorted(root_path.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        correct_ext = detect_format(filepath)

        if correct_ext is None:
            print(f"  [UNKNOWN]  {filepath.relative_to(root_path)}")
            unknown += 1
            continue

        if filepath.suffix.lower() == correct_ext:
            skipped += 1
            continue  # Already correct, no need to print

        new_path = filepath.with_suffix(correct_ext)

        # Avoid overwriting an existing file
        if new_path.exists():
            print(f"  [CONFLICT] {filepath.relative_to(root_path)} -> {new_path.name} already exists, skipping.")
            skipped += 1
            continue

        filepath.rename(new_path)
        print(f"  [RENAMED]  {filepath.relative_to(root_path)}  ->  {new_path.name}")
        renamed += 1

    print(f"\nDone.  Renamed: {renamed}  |  Skipped: {skipped}  |  Unknown: {unknown}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 fix_image_extensions.py <root_folder>")
        sys.exit(1)

    fix_extensions(sys.argv[1])
