"""
表格提取 Pipeline 入口。

用法:
    python main.py
        # 批量处理 data/input/ 下全部图片，写出 html/md/images
    python main.py --image data/input/demo.png
    python main.py --format html
    python main.py --no-vis
    python main.py --no-cache
    python main.py --refresh-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from src.core.config import load_env

load_env()

from src.core.pipeline import extract_table_output

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "data" / "input"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "output"
OUTPUT_HTML_DIR = DEFAULT_OUTPUT_DIR / "html"
OUTPUT_MD_DIR = DEFAULT_OUTPUT_DIR / "md"
OUTPUT_IMAGE_DIR = DEFAULT_OUTPUT_DIR / "images"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_input_images(input_dir: Path = DEFAULT_INPUT_DIR) -> List[Path]:
    """扫描 data/input/，返回全部图片路径（按文件名排序）。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"默认输入目录不存在: {input_dir}")

    images = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise FileNotFoundError(
            f"未在 {input_dir} 找到测试图片，请放入 png/jpg 等表格截图后重试。"
        )
    return images


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="云端 PaddleOCR + TableStructureRec → HTML/MD + 可视化图",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=f"输入表格图片路径；默认扫描 {DEFAULT_INPUT_DIR} 下全部图片并批量处理",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出路径（单图模式）；扩展名可省略，将按 --format 补全。"
             "批量模式默认写入 data/output/html|md|images 子目录",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["html", "md", "both"],
        help="输出格式：both（默认，html+md）/ html / md；md 由 html2md 从 html 转换",
    )
    parser.add_argument(
        "--tsr-kind",
        type=str,
        default="auto",
        choices=["auto", "wired", "lineless"],
        help="TableStructureRec 引擎：auto（默认）/ wired / lineless",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="auto",
        choices=["auto", "none", "0", "90", "180", "270"],
        help="方向归正：auto（默认，投影定轴+OCR 消歧180）/ none / 强制角度",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用 OCR 本地缓存，每次都请求云端",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="强制重新请求云端 OCR 并覆盖本地缓存",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="关闭表格划线可视化（默认会写出 <stem>_table_vis.png / _table_vis_logic.png）",
    )
    return parser.parse_args(argv)


def _resolve_out_paths(
    image_path: Path,
    output_arg: str | None,
    fmt: str,
    *,
    single: bool,
) -> list[tuple[str, Path]]:
    """返回 [(format_key, path), ...]。"""
    stem = image_path.stem
    if single and output_arg:
        out = Path(output_arg)
        if out.suffix.lower() in {".html", ".htm", ".md", ".markdown"}:
            key = "html" if out.suffix.lower() in {".html", ".htm"} else "md"
            if fmt == "both":
                other = "md" if key == "html" else "html"
                other_path = out.with_suffix(".md" if other == "md" else ".html")
                return [(key, out), (other, other_path)]
            return [(key if fmt == key else fmt, out if fmt == key else out.with_suffix(
                ".html" if fmt == "html" else ".md"
            ))]
        base = out
        if fmt == "html":
            return [("html", base.with_suffix(".html") if base.suffix == "" else Path(str(base) + ".html"))]
        if fmt == "md":
            return [("md", base.with_suffix(".md") if base.suffix == "" else Path(str(base) + ".md"))]
        return [
            ("html", Path(str(base) + ".html") if base.suffix == "" else base.with_suffix(".html")),
            ("md", Path(str(base) + ".md") if base.suffix == "" else base.with_suffix(".md")),
        ]

    if fmt == "html":
        return [("html", OUTPUT_HTML_DIR / f"{stem}.html")]
    if fmt == "md":
        return [("md", OUTPUT_MD_DIR / f"{stem}.md")]
    return [
        ("html", OUTPUT_HTML_DIR / f"{stem}.html"),
        ("md", OUTPUT_MD_DIR / f"{stem}.md"),
    ]


def process_one(
    image_path: Path,
    out_specs: list[tuple[str, Path]],
    *,
    use_cache: bool,
    refresh_cache: bool,
    orientation: str,
    tsr_kind: str = "auto",
    save_vis: bool = True,
    vis_dir: Path | None = None,
    ocr_engine=None,
) -> None:
    """处理单张图片并按 format 写入。"""
    print(f"[info] 处理图像: {image_path}")
    vis_dir = vis_dir if vis_dir is not None else OUTPUT_IMAGE_DIR
    result = extract_table_output(
        str(image_path),
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        orientation=orientation,
        tsr_kind=tsr_kind,
        save_vis=save_vis,
        vis_dir=vis_dir,
        debug_stem=image_path.stem,
    )

    for key, out_path in out_specs:
        content = result.get(key) or ""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        try:
            print(content)
        except UnicodeEncodeError:
            print(content.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8", errors="replace"
            ))
        print(f"[info] 已写入: {out_path}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            alt = ROOT / args.image
            if alt.is_file():
                image_path = alt
            else:
                print(f"错误: 找不到图片 {args.image}", file=sys.stderr)
                return 1
        images = [image_path]
        single = True
    else:
        try:
            images = list_input_images()
        except FileNotFoundError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1
        single = False

    print(
        f"[info] 将处理 {len(images)} 张图片 "
        f"(tsr_kind={args.tsr_kind}, format={args.format})"
    )

    from src.core.models import load_ocr
    from src.structure.tsr import load_tsr_models

    ocr = load_ocr()
    print("[info] 预加载 TableStructureRec…")
    load_tsr_models()

    vis_dir = (
        Path(args.output).parent if single and args.output else OUTPUT_IMAGE_DIR
    )
    ok, fail = 0, 0
    for i, image_path in enumerate(images, start=1):
        out_specs = _resolve_out_paths(
            image_path, args.output, args.format, single=single and len(images) == 1
        )
        print(f"[info] ({i}/{len(images)}) {image_path.name}")
        try:
            process_one(
                image_path,
                out_specs,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                orientation=args.orientation,
                tsr_kind=args.tsr_kind,
                save_vis=not args.no_vis,
                vis_dir=vis_dir,
                ocr_engine=ocr,
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[error] 处理失败 {image_path.name}: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    print(f"[info] 完成: 成功 {ok}，失败 {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
