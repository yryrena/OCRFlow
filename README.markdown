# OCRFlow

An end-to-end pipeline for **document digitization**, combining image preprocessing, column segmentation, and OCR (Optical Character Recognition) — all automated and configurable.

```{mermaid}
flowchart LR
    A[1_raw_scans - Raw Scans] --> B[2_binarized - Binarization]
    B --> C[3_split_pages_full - Full Pages]
    C --> D[4_split_pages_cols - Column Crops]
    D --> E[5_for_ocr - OCR Input]
    E --> F[6_ocr_results - OCR Output CSV / JSON]

    style A fill:#f6e0b5,stroke:#444,stroke-width:1px
    style B fill:#eea990,stroke:#444,stroke-width:1px
    style C fill:#cebfb6,stroke:#444,stroke-width:1px
    style D fill:#aec8ce,stroke:#444,stroke-width:1px
    style E fill:#a4b8ac,stroke:#444,stroke-width:1px
    style F fill:#d6c7c7,stroke:#444,stroke-width:1px
```

## Project Structure

```text
OCRFlow/
├── scripts/
│   ├── batch_binarize.py      ## image binarization
│   ├── split_columns.py       ## automatic column segmentation
│   ├── run_paddle_ocr.py      ## batch OCR execution
│   └── worker_ocr_one.py      ## single-image OCR tool (optional)
│
├── data/
│   ├── 1_raw_scans/           ## original scanned pages
│   ├── 2_binarized/           ## binarized (cleaned) images
│   ├── 3_split_pages_full/    ## pre-split full pages
│   ├── 4_split_pages_cols/    ## column-cropped images
│   ├── 5_for_ocr/             ## final images for OCR
│   └── 6_ocr_results/         ## OCR outputs (CSV / JSON)
│
├── models/                    ## PaddleOCR model weights
│   ├── PP-OCRv5_server_det/
│   ├── PP-OCRv5_server_rec/
│   └── ...
│
├── run_all.sh                 ## full pipeline automation script
├── requirements.txt           ## python dependencies
└── README.md                   
```



## Environment Setup

Create a dedicated environment using **conda** or **miniforge**:

```bash
conda create -n paddle310 python=3.10
conda activate paddle310

pip install -r requirements.txt
```

For macOS (M1/M2), ensure you use `paddlepaddle>=2.6.1`.



## One-Click Full Pipeline

To process your entire dataset from raw scans to OCR results, simply run:

```bash
bash run_all.sh
```

This script executes the three main stages automatically:

| Stage        | Input Folder         | Output Folder        | Script              |
| ------------ | -------------------- | -------------------- | ------------------- |
| Binarization | `1_raw_scans`        | `2_binarized`        | `batch_binarize.py` |
| Column Split | `3_split_pages_full` | `4_split_pages_cols` | `split_columns.py`  |
| OCR          | `5_for_ocr`          | `6_ocr_results`      | `run_paddle_ocr.py` |



## Stage Descriptions

### Step 1. Image Binarization

Enhance contrast and remove noise from scanned documents.

```bash
python scripts/batch_binarize.py \
  --input ./data/1_raw_scans \
  --output ./data/2_binarized \
  --mode otsu \
  --ext png \
  --recursive
```

#### Example

*Input:*

<img src="./data/1_raw_scans/binarize_example_image1.png" alt="binarize_example_image1" width="400"> 

*Output:*

<img src="./data/2_binarized/binarize_example_image1.png" alt="binarize_example_image1"  width="400"> 

  

### Step 2. Column Splitting

Supports multiple column segmentation strategies — automatically detects whether a page should be divided by **white space**, **vertical line**, or **equal-width slices**. 

#### **Equal-width Split** (`--equal-columns K`)

Use when you know your pages have a fixed layout (e.g., academic papers, newspapers).

- Cuts each page into **K equal-width vertical slices**.
- Ignores text and whitespace detection.
- Perfect for consistent two- or three-column layouts.

```bash
python scripts/split_columns.py \
  --input ./data/3_split_pages_full \
  --output ./data/4_split_pages_cols \
  --format png \
  --equal-columns 2
```

#### **Example 1**

*Input:*  

<img src="./data/3_split_pages_full/split_example_image1.png" alt="split_example_image1" width="300"> 

*Output:* 

<img src="./data/4_split_pages_cols/split_example_image1_col1.png" alt="split_example_image1_col1"  width="130"  style="margin-right:20px;">             <img src="./data/4_split_pages_cols/split_example_image1_col2.png" alt="split_example_image1_col2"   width="130">     



#### Example 2

*Input:* 

<img src="./data/3_split_pages_full/split_example_image3.png" alt="split_example_image3" width="300"> 

*Output:*  

<img src="./data/4_split_pages_cols/split_example_image3_col1.png" alt="split_example_image3_col1"  width="80"  style="margin-right:20px;">    <img src="./data/4_split_pages_cols/split_example_image3_col1.png" alt="split_example_image3_col2"   width="80"  style="margin-right:20px;">   <img src="./data/4_split_pages_cols/split_example_image3_col1.png" alt="split_example_image3_col3"  width="80">



#### **Manual Split** (`--force-split-x X`)

Use when you know the exact pixel coordinate to split (for debugging or fixed layouts).

```bash
python scripts/split_columns.py \
  --input ./data/3_split_pages_full \
  --output ./data/4_split_pages_cols \
  --force-split-x 800
```



### Step 3. OCR Recognition  

Perform text extraction on prepared images using PaddleOCR.

```bash
python scripts/run_paddle_ocr.py \
  --input ./data/5_for_ocr \
  --output ./data/6_ocr_results \
  --models ./models \
  --export_csv
```

#### Example 1

*Input:* 

<img src="./data/5_for_ocr/ocr_example_image1.png" alt="ocr_example_image2"  width="400"> 

*Output:* 

<img src="./data/6_ocr_results/ocr_example_image1_ocr_20251026_040005.png" alt="ocr_example_image2_ocr_20251026_040005"  width="400"> 



#### Example 2

*Input:* 

<img src="./data/5_for_ocr/ocr_example_image2.png" alt="ocr_example_image2"  width="300"> 

*Output:* 

- `.json` file

- `.csv` file

<img src="./data/6_ocr_results/ocr_example_image2_ocr_20251026_040005.png" alt="ocr_example_image2_ocr_20251026_040005"  width="400"> 



## Model Information

PaddleOCR models used:

- Detection: `PP-OCRv5_server_det`
- Recognition: `PP-OCRv5_server_rec`

All model folders are stored under `/models` and automatically loaded by the scripts.



## Optional: Single-Image OCR

Run OCR on one image for quick testing:

```bash
python scripts/worker_ocr_one.py --image ./data/5_for_ocr/test.png
```
