# cutnmerge

专利/实验参数表截图提取：**云端 PaddleOCR** 识别文本 + **TableStructureRec** 原始 `pred_html` 输出表格结构。默认写出 **HTML** + **Markdown**（由 HTML 转换）+ 可视化图 + OCR 产物。

---

# 中文版

## 流程

```
加载图片 → 旋转校正 → 云端 OCR（缓存 + data/ocr 产物）
         → TableStructureRec（ocr_result → pred_html）
         → HTML / MD / 可视化图
```

- OCR：百度 AI Studio PaddleOCR API（本地短边放大、长边压缩后上传）
- 结构：RapidAI TableStructureRec（`table_cls` 有线/无线分流），**不使用**内置 RapidOCR
- HTML：直接来自 TSR `pred_html`，无自定义拓扑修复

## 安装

```bash
cd cutnmerge
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # 填写 PADDLEOCR_ACCESS_TOKEN
```

首次运行会下载 TableStructureRec 模型，需可访问外网。

## 用法

```bash
venv\Scripts\python main.py
venv\Scripts\python main.py --image data/input/demo.png
venv\Scripts\python main.py --format html
venv\Scripts\python main.py --tsr-kind wired
venv\Scripts\python main.py --orientation auto
venv\Scripts\python main.py --no-cache
venv\Scripts\python main.py --refresh-cache
venv\Scripts\python main.py --no-vis
```

| 参数 | 说明 |
|------|------|
| `--image` | 单张图片；默认批量处理 `data/input/` |
| `--format` | `both`（默认）/ `html` / `md` |
| `--tsr-kind` | `auto` / `wired` / `lineless` |
| `--orientation` | `auto` / `none` / `0` / `90` / `180` / `270` |
| `--no-cache` | 禁用 OCR 本地缓存 |
| `--refresh-cache` | 强制重新 OCR 并覆盖缓存 |
| `--no-vis` | 不写表格可视化图 |

## 输出目录

| 路径 | 内容 |
|------|------|
| `data/output/html/` | HTML |
| `data/output/md/` | Markdown |
| `data/output/images/` | `*_table_vis.png`、`*_table_vis_logic.png` |
| `data/ocr/` | OCR json/csv/标注图 |
| `data/cache/` | OCR 缓存 |

## 配置

见 `.env.example`。必填：`PADDLEOCR_ACCESS_TOKEN`。

OCR 缓存按图片内容 hash + 旋转角度键控。调 OCR/预处理参数时用 `--refresh-cache`；只调结构参数时复用缓存即可。

## 模块

| 文件 | 职责 |
|------|------|
| `main.py` | CLI |
| `src/core/pipeline.py` | 主流程 |
| `src/core/models.py` | 云端 OCR 封装 |
| `src/preprocess/orient.py` | 旋转校正 |
| `src/ocr/cloud_ocr.py` | HTTP OCR + 高清预处理 |
| `src/structure/tsr.py` | TableStructureRec + pred_html |
| `src/output/html2md.py` | HTML → MD |

---

# English Version

## Overview

Extract tables from patent/experiment screenshot tables: **cloud PaddleOCR** for text + **TableStructureRec** native `pred_html` for structure. Outputs HTML, Markdown (from HTML), vis images, and OCR artifacts.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env       # set PADDLEOCR_ACCESS_TOKEN
```

## Usage

```bash
python main.py
python main.py --image data/input/demo.png
python main.py --format both
python main.py --tsr-kind auto
python main.py --refresh-cache
```

## Pipeline

1. Load (unicode-safe paths)
2. Orientation correction
3. Cloud PaddleOCR (cached; artifacts under `data/ocr/`)
4. TableStructureRec with external `ocr_result` → `pred_html`
5. HTML → Markdown via `html2md`
6. VisTable visualization under `data/output/images/`

## Configuration

See `.env.example`. `PADDLEOCR_ACCESS_TOKEN` is required (Baidu AI Studio).

OCR cache is keyed by image hash + orientation. Use `--refresh-cache` when OCR/preprocess settings change.
