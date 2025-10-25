# run_all.sh
# bash run_all.sh

python scripts/batch_binarize.py \
  --input ./data/1_raw_scans \
  --output ./data/2_binarized --mode otsu

python scripts/split_columns.py \
  --input ./data/3_split_pages_full \
  --output ./data/4_split_pages_cols \
  --format png

python scripts/run_paddle_ocr.py \
  --input ./data/5_for_ocr \
  --output ./data/6_ocr_results \
  --models ./models \
  --export_csv