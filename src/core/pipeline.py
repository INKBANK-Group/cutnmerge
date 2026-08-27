"""表格提取主流程：旋转 → 云端 PaddleOCR → TableStructureRec pred_html。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .config import ROOT
from .models import load_ocr, predict_texts
from ..ocr.ocr_post import postprocess_text_boxes
from ..ocr.glyph_fix import fix_thin_glyphs
from ..output.html2md import html_to_markdown
from ..preprocess.orient import (
    apply_orientation_axis,
    ensure_upright_axis,
    maybe_flip_180_by_ocr,
)
from ..structure.tsr import (
    predict_table_html,
    save_tsr_vis,
    wrap_pred_html,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = ROOT / "data" / "output" / "images"


def _imread_unicode(path: str) -> np.ndarray:
    """Unicode 安全读图。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"无法读取图像: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法解码图像: {path}")
    return img


def _load_image(image_path: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(image_path, np.ndarray):
        if image_path.ndim == 2:
            return cv2.cvtColor(image_path, cv2.COLOR_GRAY2BGR)
        return image_path.copy()
    return _imread_unicode(str(image_path))


def extract_table_output(
    image_path: Union[str, np.ndarray],
    ocr_engine=None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    orientation: Union[str, int] = "auto",
    tsr_kind: Optional[str] = None,
    save_vis: bool = True,
    vis_dir: Optional[Union[str, Path]] = None,
    debug_stem: Optional[str] = None,
) -> Dict[str, Any]:
    """
    极简表格提取 Pipeline。

    Returns:
        {"html": str, "md": str, "orientation": int, "vis_paths": list[str]}
    """
    image = _load_image(image_path)
    stem = debug_stem
    if stem is None:
        if isinstance(image_path, str):
            stem = Path(image_path).stem
        else:
            stem = "image"

    ocr = ocr_engine or load_ocr()
    image, axis_angle, orient_kind = apply_orientation_axis(
        image, mode=orientation
    )

    provisional_orient = int(axis_angle)
    cache_extra = f"orient={provisional_orient}"
    text_boxes = predict_texts(
        image,
        ocr_engine=ocr,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=cache_extra,
        artifact_stem=stem,
    )

    orient_angle = provisional_orient
    if orient_kind == "auto":
        image, axis_delta, text_boxes = ensure_upright_axis(
            image,
            text_boxes,
            ocr_engine=ocr,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_extra_base=cache_extra,
        )
        orient_angle = (provisional_orient + axis_delta) % 360
        if axis_delta:
            cache_extra = f"orient={orient_angle}"
        image, flip, text_boxes = maybe_flip_180_by_ocr(
            image,
            text_boxes,
            ocr_engine=ocr,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_extra_base=cache_extra,
            force_check=bool(axis_delta),
        )
        orient_angle = (orient_angle + flip) % 360
        if axis_delta or flip:
            try:
                from ..ocr.ocr_cache import save_ocr_cache

                save_ocr_cache(
                    image,
                    text_boxes,
                    extra=f"orient={orient_angle}",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("写入 OCR 缓存失败: %s", exc)

    text_boxes = postprocess_text_boxes(text_boxes)
    text_boxes = fix_thin_glyphs(image, text_boxes)

    force = None if not tsr_kind or (tsr_kind or "").lower() == "auto" else tsr_kind
    tsr_result, _ = predict_table_html(
        image,
        text_boxes,
        force_kind=force,
    )

    html = wrap_pred_html(getattr(tsr_result, "pred_html", None))
    md = html_to_markdown(html) if html else ""

    vis_paths: List[str] = []
    if save_vis:
        out_dir = Path(vis_dir) if vis_dir else DEFAULT_OUTPUT_DIR
        vis_paths = save_tsr_vis(image, tsr_result, stem, out_dir)

    return {
        "html": html,
        "md": md,
        "orientation": orient_angle,
        "vis_paths": vis_paths,
    }


def extract_table_markdown(
    image_path: Union[str, np.ndarray],
    **kwargs: Any,
) -> str:
    """兼容旧接口：返回 Markdown 字符串。"""
    return extract_table_output(image_path, **kwargs).get("md", "")
