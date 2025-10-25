#!/usr/bin/env python3
"""
batch_binarize.py

batch convert images in a folder to binary (black/white) images

example:
    python batch_binarize.py \
        --input "./raw_images" \
        --output "./binarized" \
        --mode otsu \
        --ext png \
        --recursive
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(root: Path, recursive: bool) -> list[Path]:
    """collect image paths under root"""
    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.glob("*")

    images = [p for p in candidates if p.suffix.lower() in VALID_EXTS and p.is_file()]
    return images


def read_image_unicode(path: Path) -> np.ndarray | None:
    """
    read image using cv2 in a way that supports unicode paths;
    returns np.ndarray (BGR) or None if unreadable
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    convert to 8-bit grayscale;
    if already single-channel, just ensure dtype is uint8
    """
    if img is None:
        raise ValueError("Input image is None.")

    if len(img.shape) == 2:
        gray = img
    elif len(img.shape) == 3:
        ## handle BGR/RGBA/etc.
        if img.shape[2] == 4:
            ## has alpha; drop alpha first
            bgr = img[:, :, :3]
        else:
            bgr = img
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image shape: %s" % (img.shape,))

    ## ensure uint8
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray


def binarize(gray: np.ndarray, mode: str, thresh: int) -> np.ndarray:
    """
    produce binary image (0 or 255); 
    mode = 'fixed' or 'otsu'
    """
    if mode == "fixed":
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    elif mode == "otsu":
        ## otsu chooses threshold automatically
        _, bw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        raise ValueError("Unknown mode: %s" % mode)
    return bw


def write_image_unicode(path: Path, img: np.ndarray, out_ext: str) -> bool:
    """
    save img to path with the requested extension;
    encode first, then write raw bytes to handle unicode
    """
    out_ext = out_ext.lower()
    if not out_ext.startswith("."):
        out_ext = "." + out_ext

    ## pick codec based on ext
    ## default to PNG if extension unknown
    ext_for_cv = out_ext if out_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"] else ".png"

    try:
        success, buf = cv2.imencode(ext_for_cv, img)
        if not success:
            return False
        buf.tofile(str(path.with_suffix(out_ext)))
        return True
    except Exception:
        return False


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def process_folder(
    src_dir: Path,
    dst_dir: Path,
    mode: str,
    thresh: int,
    out_ext: str,
    recursive: bool,
) -> dict:
    """
    walk source dir, binarize all valid images, and save into dst_dir
    mirroring relative structure

    returns stats
    """
    images = find_images(src_dir, recursive)
    total = len(images)
    converted = 0
    failed = []

    for img_path in tqdm(images, desc="Binarizing", unit="img"):
        ## figure out save path
        rel = img_path.relative_to(src_dir)
        out_subdir = (dst_dir / rel.parent)
        ensure_dir(out_subdir)

        ## read
        img = read_image_unicode(img_path)
        if img is None:
            failed.append((img_path, "unreadable"))
            continue

        try:
            gray = to_grayscale(img)
            bw = binarize(gray, mode=mode, thresh=thresh)

            ok = write_image_unicode(out_subdir / rel.stem, bw, out_ext)
            if ok:
                converted += 1
            else:
                failed.append((img_path, "save_failed"))
        except Exception as e:
            failed.append((img_path, str(e)))

    return {
        "total": total,
        "converted": converted,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert images to black/white using fixed or Otsu thresholding."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Folder containing source images",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Folder to write binarized images",
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "otsu"],
        default="fixed",
        help="Thresholding mode: 'fixed' uses --thresh, 'otsu' auto-selects",
    )
    parser.add_argument(
        "--thresh",
        type=int,
        default=175,
        help="Fixed threshold [0-255] (used only if --mode fixed)",
    )
    parser.add_argument(
        "--ext",
        default="jpg",
        help="Output file extension: jpg, png, tif, ... (default: jpg)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process images inside subfolders",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists() or not args.input.is_dir():
        print(f"[!! ERROR] Input folder '{args.input}' not found or not a directory.", file=sys.stderr)
        sys.exit(1)

    ensure_dir(args.output)

    stats = process_folder(
        src_dir=args.input,
        dst_dir=args.output,
        mode=args.mode,
        thresh=args.thresh,
        out_ext=args.ext,
        recursive=args.recursive,
    )

    print("\n=== summary ===")
    print(f"total images found    : {stats['total']}")
    print(f"successfully saved    : {stats['converted']}")
    print(f"failed                : {len(stats['failed'])}")

    if stats["failed"]:
        print("\nFailures:")
        for p, reason in stats["failed"]:
            print(f" - {p} -> {reason}")


if __name__ == "__main__":
    main()