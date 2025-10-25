#!/usr/bin/env python3
"""
worker_ocr_one.py
Run OCR on a single image and save output.
This script is meant to be called by the batch driver, not by humans.
It isolates PaddleOCR so the parent process never crashes.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from paddleocr import PaddleOCR


def flatten_ocr_result(result):
    out = []
    for page in result:
        page_data = []
        for entry in page:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                box = np.array(entry[0]).tolist()
                text_info = entry[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = str(text_info[0])
                    score = float(text_info[1])
                elif isinstance(text_info, dict):
                    text = str(text_info.get("text", ""))
                    score = float(text_info.get("score", 0.0))
                else:
                    text = str(text_info)
                    score = 0.0
                page_data.append({"box": box, "text": text, "score": score})
            elif isinstance(entry, dict):
                box = np.array(entry.get("box", [])).tolist()
                text = str(entry.get("text", ""))
                score = float(entry.get("score", 0.0))
                page_data.append({"box": box, "text": text, "score": score})
        out.append(page_data)
    return out


def pick_model_dirs(models_root: Path):
    det_dir = rec_dir = cls_dir = None
    for sub in models_root.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        if det_dir is None and "det" in name:
            det_dir = sub
        elif rec_dir is None and "rec" in name:
            rec_dir = sub
        elif cls_dir is None and any(k in name for k in ("cls", "textline", "doc")):
            cls_dir = sub
    if det_dir is None or rec_dir is None:
        raise RuntimeError(
            f"Missing det/rec under {models_root}. Need folders containing 'det' and 'rec'."
        )
    return det_dir, rec_dir, cls_dir


def run_single_image(img_path: Path, out_dir: Path, models_root: Path, visualize: bool):
    ## init OCR inside the worker, so each worker has its own Paddle runtime
    det_dir, rec_dir, cls_dir = pick_model_dirs(models_root)
    print(f"[WORKER] init OCR for {img_path.name}")
    ocr = PaddleOCR(
        text_detection_model_dir=str(det_dir),
        text_recognition_model_dir=str(rec_dir),
        textline_orientation_model_dir=str(cls_dir) if cls_dir else None,
        use_textline_orientation=True,
        lang="ch",
    )

    result = ocr.ocr(str(img_path))
    flat = flatten_ocr_result(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = out_dir / f"{img_path.stem}_ocr_{timestamp}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    if visualize:
        try_visualize_safe(img_path, flat, out_dir, timestamp)
 
    ## hard exit the worker so Paddle's C++ teardown can't segfault 
    os._exit(0)


def try_visualize_safe(img_path: Path, flat_all_pages, out_dir: Path, timestamp: str):
    ## flat_all_pages is [page0, ...]; we just draw page0
    if not flat_all_pages or not flat_all_pages[0]:
        print(f"[WORKER] no regions to visualize for {img_path.name}")
        return

    page0 = flat_all_pages[0]

    try:
        import cv2
    except Exception as e:
        print(f"[WORKER] no cv2 ({e}), skip viz")
        return

    draw_ocr = None
    try:
        from paddleocr.tools.infer.utility import draw_ocr as draw_ocr
    except Exception:
        try:
            from paddleocr.ppocr.utils.visual import draw_ocr as draw_ocr
        except Exception:
            draw_ocr = None
    if draw_ocr is None:
        print(f"[WORKER] draw_ocr not available, skip viz for {img_path.name}")
        return

    img = cv2.imread(str(img_path))
    if img is None or img.size == 0:
        print(f"[WORKER] cannot read image for viz {img_path.name}")
        return

    boxes = [item["box"] for item in page0]
    texts = [item["text"] for item in page0]
    scores = [item["score"] for item in page0]
    font_path = "/System/Library/Fonts/STHeiti Light.ttc"

    try:
        vis_img = draw_ocr(img, boxes, texts, scores, font_path=font_path)
    except Exception as e:
        print(f"[WORKER] draw_ocr failed for {img_path.name}: {e}")
        return

    out_file = out_dir / f"{img_path.stem}_vis_{timestamp}.jpg"
    cv2.imwrite(str(out_file), vis_img)


def main():
    ## argv from parent:
    ## sys.argv[1] = image path
    ## sys.argv[2] = output dir
    ## sys.argv[3] = models dir
    ## sys.argv[4] = "1" or "0" for visualize
    if len(sys.argv) < 5:
        print("[WORKER] usage: worker_ocr_one.py IMG OUT MODELS VISFLAG", file=sys.stderr)
        os._exit(1)

    img_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    models_root = Path(sys.argv[3])
    visualize_flag = sys.argv[4] == "1"

    out_dir.mkdir(parents=True, exist_ok=True)

    run_single_image(img_path, out_dir, models_root, visualize_flag)


if __name__ == "__main__":
    main()