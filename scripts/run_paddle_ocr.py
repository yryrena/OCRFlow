#!/usr/bin/env python3
"""
run_paddle_ocr.py
enhanced PaddleOCR runner
- outputs both JSON and CSV per image
- automatically detects model folders
- optional visualization
"""

import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
from paddleocr import PaddleOCR
import numpy as np

# -----------------------
# FLATTEN OCR RESULTS
# -----------------------

def flatten_ocr_result(result):
    out = []
    for page in result:
        page_data = []
        for entry in page:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                box = np.array(entry[0]).tolist()
                text_info = entry[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text, score = str(text_info[0]), float(text_info[1])
                elif isinstance(text_info, dict):
                    text, score = str(text_info.get("text", "")), float(text_info.get("score", 0.0))
                else:
                    text, score = str(text_info), 0.0
                page_data.append({"box": box, "text": text, "score": score})
            elif isinstance(entry, dict):
                page_data.append({
                    "box": np.array(entry.get("box", [])).tolist(),
                    "text": str(entry.get("text", "")),
                    "score": float(entry.get("score", 0.0)),
                })
        out.append(page_data)
    return out

# -----------------------
# MODEL DETECTION  
# ----------------------- 

def pick_model_dirs(models_root: Path):
    det_dir = rec_dir = cls_dir = None
    for sub in models_root.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        if "det" in name:
            det_dir = sub
        elif "rec" in name:
            rec_dir = sub
        elif any(k in name for k in ("cls", "textline", "doc")):
            cls_dir = sub
    if det_dir is None or rec_dir is None:
        raise RuntimeError(f"Missing det/rec model under {models_root}")
    return det_dir, rec_dir, cls_dir

# -----------------------
# INITIALIZE OCR  
# ----------------------- 
 
def init_ocr(det_dir, rec_dir, cls_dir=None, lang="ch"):
    print("[INFO] Initializing OCR with:")
    print(f"  det: {det_dir}")
    print(f"  rec: {rec_dir}")
    print(f"  cls: {cls_dir or '(none)'}")
    return PaddleOCR(
        text_detection_model_dir=str(det_dir),
        text_recognition_model_dir=str(rec_dir),
        textline_orientation_model_dir=str(cls_dir) if cls_dir else None,
        use_textline_orientation=True,
        lang=lang,
    )

# -----------------------
# OCR ONE IMAGE    
# ----------------------- 
 
def run_single_page(ocr, img_path: Path, out_dir: Path, export_csv: bool):
    result = ocr.ocr(str(img_path))
    flattened = flatten_ocr_result(result)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ### JSON output
    json_out = out_dir / f"{img_path.stem}_ocr_{timestamp}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(flattened, f, ensure_ascii=False, indent=2)

    ## CSV output  
    if export_csv:
        csv_out = out_dir / f"{img_path.stem}_ocr_{timestamp}.csv"
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["page", "text", "score", "box"])
            for page_idx, page in enumerate(flattened, start=1):
                for item in page:
                    writer.writerow([page_idx, item["text"], item["score"], item["box"]])
    return len(flattened[0]) if flattened and flattened[0] else 0

# -----------------------
# BATCH RUNNER
# ----------------------- 
 
def run_batch(input_dir: Path, output_dir: Path, models_root: Path, export_csv: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    det_dir, rec_dir, cls_dir = pick_model_dirs(models_root)
    ocr = init_ocr(det_dir, rec_dir, cls_dir)

    images = [p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]
    if not images:
        print("[WARN] No images found.")
        return

    print(f"[INFO] Found {len(images)} image(s). Starting OCR...\n")
    for i, img_path in enumerate(images, start=1):
        try:
            n = run_single_page(ocr, img_path, output_dir, export_csv)
            print(f"[{i}/{len(images)}] OK {img_path.name}: {n} regions")
        except Exception as e:
            print(f"[{i}/{len(images)}] FAIL {img_path.name}: {e}")

# -----------------------
# CLI
# ----------------------- 
  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--models", "-m", default="./models")
    parser.add_argument("--export_csv", action="store_true", help="Also export results as CSV")
    args = parser.parse_args()

    run_batch(Path(args.input), Path(args.output), Path(args.models), export_csv=args.export_csv)

if __name__ == "__main__":
    main()