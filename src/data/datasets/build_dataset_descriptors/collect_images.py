#!/usr/bin/env python3
"""
Utility script written by ChatGPT:

Collect images referenced by an qa_pairs JSON file and rewrite their paths to be
portable (relative to the new JSON file).

Usage
-----
python collect_images.py path/to/qa_pairs.json \
    --dest_dir packaged_dataset \
    --images_subdir images \
    --output_json new_qa_pairs.json

Arguments
---------
qa_pairs (positional): Path to the source JSON file containing qa_pairs.
--dest_dir / -d        : Directory where the images and rewritten JSON will be
                         written. (default: ./portable_package)
--images_subdir / -i   : Sub‑directory within *dest_dir* to place copied images.
                         (default: images)
--output_json / -o     : Name (or full path) of the rewritten JSON file. If
                         omitted, it defaults to <qa_pairs_stem>_portable.json
                         inside *dest_dir*.

The script will:
1. Parse the JSON list and collect every unique "image_path".
2. Copy each image into *dest_dir/images_subdir* (creating directories as needed).
   If two different source paths share the same filename but differ in content,
   the latter copy gets a unique suffix derived from an MD5 hash to avoid
   overwriting.
3. Modify each annotation's "image_path" so it points to the copied file using a
   *relative* path (e.g. "images/<filename>.jpg").
4. Write the updated JSON next to the images.

Missing images are reported on STDERR but do not abort the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def files_identical(p1: Path, p2: Path) -> bool:
    """Return True if *p1* and *p2* refer to files with identical size & hash."""
    if p1.stat().st_size != p2.stat().st_size:
        return False
    return hashlib.md5(p1.read_bytes()).digest() == hashlib.md5(p2.read_bytes()).digest()


def copy_image(src: Path, dest_dir: Path) -> str:
    """Copy *src* into *dest_dir* (which is created if absent).

    Returns the *basename* of the copied file (possibly renamed to avoid a clash).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src.name

    if dest_path.exists():
        # If a clash but the files differ, append an 8‑char hash to make unique
        if not files_identical(src, dest_path):
            dest_path = dest_dir / f"{src.stem}_{hashlib.md5(str(src).encode()).hexdigest()[:8]}{src.suffix}"

    shutil.copy2(src, dest_path)
    return dest_path.name  # we only need the filename for relative paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect images referenced in a qa_pairs JSON and rewrite paths to be portable.")
    parser.add_argument("qa_pairs", type=Path, help="Path to the qa_pairs JSON file")
    parser.add_argument("--dest_dir", "-d", type=Path, default=Path.cwd() / "portable_package",
                        help="Destination directory for the package (default: ./portable_package)")
    parser.add_argument("--images_subdir", "-i", type=str, default="images",
                        help="Sub‑folder inside dest_dir for copied images (default: images)")
    parser.add_argument("--output_json", "-o", type=Path, default=None,
                        help="Filename for the rewritten JSON (default: <qa_pairs_stem>_portable.json in dest_dir)")

    args = parser.parse_args()

    # Load JSON (expects list of dicts)
    with args.qa_pairs.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images_dir = args.dest_dir / args.images_subdir
    copied = missing = 0

    for item in data:
        src_path = Path(item.get("image_path", ""))
        if not src_path.is_file():
            print(f"Warning: {src_path} not found — skipping", file=sys.stderr)
            missing += 1
            continue
        filename = copy_image(src_path, images_dir)
        # Replace with a *relative* path so the package is self‑contained
        item["image_path"] = str(Path(args.images_subdir) / filename)
        copied += 1

    # Ensure package directory exists
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    out_json = args.output_json or (args.dest_dir / f"{args.qa_pairs.stem}_portable.json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Copied {copied} images → {images_dir}")
    if missing:
        print(f"⚠️  {missing} image(s) listed in the JSON were not found.", file=sys.stderr)
    print(f"🔄 Wrote updated qa_pairs JSON to {out_json}")


if __name__ == "__main__":
    main()
