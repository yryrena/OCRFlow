#!/usr/bin/env python3
"""
split_columns.py 
ulti-column splitter with smarter gutter handling  

supports:
- light deskew
- adaptive binarization
- fallback to vertical rule lines (dictionary style)
- `--equal-columns K` (newspaper, 3-col, etc.)
- `--force-split-x X` manual cut

- instead of assuming there's ONE split x, we can now detect a WHOLE
  "blank corridor" between two columns: [gutter_left, gutter_right]; 
  then we crop:
      left  = page[:, :gutter_left-pad]
      right = page[:, gutter_right+pad:]
  this fixes cases where the middle whitespace is actually a BAND,
  not a razor-thin line (your textbook-style scan)
"""

import sys
import math
import argparse
from pathlib import Path

import cv2
import numpy as np

# -----------------------
# I/O HELPER
# -----------------------

def load_image_any_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def save_image_any_unicode(path: Path, img, ext=".png"):
    if not ext.startswith("."):
        ext = "." + ext
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# -----------------------
# PREPROCESS (binarize, deskew)
# -----------------------

def to_binary_document(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blur,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=35,
        C=15,
    )
    return bw     ## 0=ink, 255=bg


def estimate_rotation(bw_img):
    edges = cv2.Canny(bw_img, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
    if lines is None:
        return 0.0

    angles = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        deg = theta * 180.0 / math.pi
        if deg > 90:
            deg -= 180
        if abs(deg) < 20:   ## near-horizontal
            angles.append(deg)

    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_image_keep_bounds(img, angle_deg):
    if abs(angle_deg) < 0.5:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int((h * sin_a) + (w * cos_a))
    new_h = int((h * cos_a) + (w * sin_a))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )
    return rotated


# -----------------------
# INK PROFILE UTILITIES 
# -----------------------

def vertical_ink_profile(bw_img):
    """
    For each x, count how much black ink is there.
    Smaller number = whiter column.
    """
    ink_mask = (255 - bw_img) // 255    ## 1 where black, 0 where white
    col_sum = np.sum(ink_mask, axis=0)
    return col_sum.astype(np.int32)


def smooth_1d(arr, ksize=31):
    """
    Simple moving average smoothing.
    """
    k = np.ones(ksize, dtype=np.float32) / float(ksize)
    return np.convolve(arr.astype(np.float32), k, mode="same")


# -----------------------
# STRATEGY A: find a *band* of whitespace between two columns
# -----------------------

def find_two_column_band(bw_img, middle_frac=(0.15, 0.85), min_band_px=20):
    """
    goal:
      on clean 2-column academic pages, the split is not a hairline;
      it's a whole "gutter band" ~20-60 px wide with almost no text;
      we want both edges of that band

    steps:
      - build vertical ink density profile
      - keep only the central horizontal range (to avoid margins)
      - look for the WIDEST low-ink run, not just lowest single point

    returns:
      (gutter_left, gutter_right) as ints if we find a plausible band,
      else (None, None).

    heuristics:
      - we measure "ink density" after smoothing
      - we find all contiguous runs where density < (global_mean * 0.5)
        then pick the widest run
    """
    h, w = bw_img.shape[:2]
    prof = vertical_ink_profile(bw_img)
    prof_s = smooth_1d(prof, ksize=31)

    ## focus roughly on middle_frac of width, so we ignore outer page margins
    x_lo = int(w * middle_frac[0])
    x_hi = int(w * middle_frac[1])
    if x_hi <= x_lo:
        x_lo, x_hi = 0, w

    mid_slice = prof_s[x_lo:x_hi]
    if mid_slice.size == 0:
        return (None, None)

    ## threshold for "blank-enough"
    thr = np.mean(mid_slice) * 0.5    ## pretty white

    ## scan for longest run where prof_s[x] < thr
    best_len = -1
    best_pair = (None, None)

    run_start = None
    for i, val in enumerate(mid_slice):
        if val < thr:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_end = i - 1
                run_len = run_end - run_start + 1
                if run_len > best_len:
                    best_len = run_len
                    best_pair = (x_lo + run_start, x_lo + run_end)
                run_start = None
    ## tail run
    if run_start is not None:
        run_end = len(mid_slice) - 1
        run_len = run_end - run_start + 1
        if run_len > best_len:
            best_len = run_len
            best_pair = (x_lo + run_start, x_lo + run_end)

    if best_len < min_band_px:
        return (None, None)

    gutter_left, gutter_right = best_pair

    ## sanity: the band should be somewhere near page center,
    ## not hugging far-left or far-right margin.
    ## e.g. reject if it's basically the left margin whitespace.
    center_x = w * 0.5
    if gutter_right < center_x * 0.5:
        ## way too left, probs margin
        return (None, None)
    if gutter_left > w - center_x * 0.5:
        ## way too right, probs margin
        return (None, None)

    return (int(gutter_left), int(gutter_right))


# -----------------------
# STRATEGY B: vertical rule lines (for dictionary scans)
# -----------------------

def _gather_vertical_segments(bw_img, x_left, x_right):
    h, w = bw_img.shape[:2]
    edges = cv2.Canny(bw_img, 50, 150, apertureSize=3)

    mask = np.zeros_like(edges)
    mask[:, x_left:x_right] = 255
    masked_edges = cv2.bitwise_and(edges, mask)

    segs = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=int(h * 0.15),
        maxLineGap=25,
    )

    if segs is None:
        return []

    out = []
    for (x1, y1, x2, y2) in segs[:, 0, :]:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        angle_from_vertical = abs(90.0 - angle_deg)
        if angle_from_vertical > 10:
            continue

        x_mid = 0.5 * (x1 + x2)
        out.append(
            {
                "x_mid": x_mid,
                "length": length,
                "angle_from_vertical": angle_from_vertical,
            }
        )
    return out


def _dense_x_cluster(xs, window_px=20):
    if not xs:
        return None
    xs_sorted = np.sort(xs)
    best_count = -1
    best_center = None
    left = 0
    for right in range(len(xs_sorted)):
        while xs_sorted[right] - xs_sorted[left] > window_px:
            left += 1
        count = right - left + 1
        if count > best_count:
            best_count = count
            best_center = np.mean(xs_sorted[left:right + 1])
    return float(best_center) if best_center is not None else None


def estimate_single_split_by_lines(bw_img):
    """
    old behavior: guess ONE divider x from vertical line segments
    """
    h, w = bw_img.shape[:2]
    cx = w // 2
    band_half = 150
    x_left = max(cx - band_half, 0)
    x_right = min(cx + band_half, w)

    segs = _gather_vertical_segments(bw_img, x_left, x_right)
    if not segs:
        return None

    xs = []
    for s in segs:
        repeats = max(1, int(round(s["length"] / (0.1 * h))))
        xs.extend([s["x_mid"]] * repeats)

    divider_x = _dense_x_cluster(xs, window_px=20)
    if divider_x is None:
        return None
    return int(round(divider_x))


# -----------------------
# Cropping helpers
# -----------------------

def crop_from_band(bw_img, band_left, band_right, pad=6):
    """
    detected a gutter band [band_left, band_right]; 
    return two column crops with that band removed

    left  = [:, :band_left-pad]
    right = [:, band_right+pad:]
    """
    h, w = bw_img.shape[:2]
    L_end = max(band_left - pad, 0)
    R_start = min(band_right + pad, w)

    left_img = bw_img[:, :L_end]
    right_img = bw_img[:, R_start:]
    return [left_img, right_img]


def crop_two_columns_point(bw_img, x_split, pad=6):
    """
    fallback version: just have a single split x
    """
    h, w = bw_img.shape[:2]
    left_end = max(x_split - pad, 0)
    right_start = min(x_split + pad, w)
    left_img = bw_img[:, :left_end]
    right_img = bw_img[:, right_start:]
    return [left_img, right_img]


def crop_equal_columns(bw_img, k_cols):
    h, w = bw_img.shape[:2]
    cols = []
    for i in range(k_cols):
        x0 = int(round(i * w / k_cols))
        x1 = int(round((i + 1) * w / k_cols))
        cols.append(bw_img[:, x0:x1])
    return cols


# -----------------------
# Per-page pipeline
# -----------------------

def process_page(
    img_path: Path,
    out_dir: Path,
    fmt: str,
    equal_cols: int = None,
    force_x: int = None,
    pad: int = 6,
):
    color = load_image_any_unicode(img_path)
    if color is None:
        return False, f"{img_path.name}: cannot read image", 0

    bw0 = to_binary_document(color)
    skew_deg = estimate_rotation(bw0)
    fixed_color = rotate_image_keep_bounds(color, -skew_deg)
    bw = to_binary_document(fixed_color)

    h, w = bw.shape[:2]
    stem = img_path.stem

    ## explicit multi-col override (e.g. newspaper)
    if equal_cols and equal_cols > 1:
        crops = crop_equal_columns(bw, equal_cols)
        saved = 0
        for i, cimg in enumerate(crops, start=1):
            out_path = out_dir / f"{stem}_col{i}.{fmt}"
            ok = save_image_any_unicode(out_path, cimg, ext="." + fmt)
            if ok:
                saved += 1
        msg = (f"{img_path.name}: {equal_cols} equal columns "
               f"(override), skew={skew_deg:.2f}°")
        return True, msg, saved

    ## manual x override
    if force_x is not None:
        xs = int(force_x)
        xs = max(1, min(w - 2, xs))
        crops = crop_two_columns_point(bw, xs, pad=pad)
        saved = 0
        for i, cimg in enumerate(crops, start=1):
            out_path = out_dir / f"{stem}_col{i}.{fmt}"
            ok = save_image_any_unicode(out_path, cimg, ext="." + fmt)
            if ok:
                saved += 1
        msg = (f"{img_path.name}: forced split at x={xs}, "
               f"skew={skew_deg:.2f}°")
        return True, msg, saved

    ## AUTO: first try wide blank band
    band_left, band_right = find_two_column_band(
        bw,
        middle_frac=(0.15, 0.85),   ## ignore extreme margins
        min_band_px=20,             ## must be at least 20px wide
    )

    if band_left is not None and band_right is not None:
        crops = crop_from_band(bw, band_left, band_right, pad=pad)
        method = f"blank-band[{band_left}-{band_right}]"
    else:
        ## fallback: look for a single divider line cluster
        xs_line = estimate_single_split_by_lines(bw)
        if xs_line is not None:
            crops = crop_two_columns_point(bw, xs_line, pad=pad)
            method = f"vertical-lines x={xs_line}"
        else:
            ## ultimate fallback: just split at page center
            xs_mid = w // 2
            crops = crop_two_columns_point(bw, xs_mid, pad=pad)
            method = f"center-fallback x={xs_mid}"

    ## save crops
    saved = 0
    for i, cimg in enumerate(crops, start=1):
        out_path = out_dir / f"{stem}_col{i}.{fmt}"
        ok = save_image_any_unicode(out_path, cimg, ext="." + fmt)
        if ok:
            saved += 1

    msg = (f"{img_path.name}: {method}, skew={skew_deg:.2f}°")
    return True, msg, saved


# -----------------------
# HELPERS
# -----------------------

def collect_images(folder: Path):
    valid_exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    }
    out = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in valid_exts:
            out.append(p)
    out.sort()
    return out


# -----------------------
# main()
# -----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Split scanned multi-column pages into per-column images."
    )
    ap.add_argument("--input", "-i", required=True,
                    help="folder of scanned full pages")
    ap.add_argument("--output", "-o", required=True,
                    help="folder to save column crops")
    ap.add_argument("--format", "-f", default="png",
                    help="output image extension (png/jpg/...)")

    ap.add_argument("--equal-columns", type=int, default=None,
                    help="Force K equal-width vertical slices (newspaper mode).")
    ap.add_argument("--force-split-x", type=int, default=None,
                    help="Force 2-column split at this x pixel.")
    ap.add_argument("--pad", type=int, default=6,
                    help="Gap (px) to trim around gutter/band to avoid cutting glyphs.")

    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_fmt = args.format.lower().strip(".")
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[!! ERROR] {in_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    pages = collect_images(in_dir)
    if not pages:
        print("[!! WARN] no input images found")
        sys.exit(0)

    print(f"[INFO] found {len(pages)} page(s)\n")

    total_ok = 0
    total_cols = 0

    for idx, img_path in enumerate(pages, start=1):
        ok, msg, ncols = process_page(
            img_path,
            out_dir,
            out_fmt,
            equal_cols=args.equal_columns,
            force_x=args.force_split_x,
            pad=args.pad,
        )
        status = "OK" if ok else "FAIL"
        print(f"[{idx}/{len(pages)}] {status} {msg}")
        if ok:
            total_ok += 1
            total_cols += ncols

    print(f"\n[SUMMARY] {total_ok}/{len(pages)} pages processed, "
          f"{total_cols} column images saved.")


if __name__ == "__main__":
    main()