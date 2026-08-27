"""OCR 文本后处理：符号规范化与 NMS 去重。"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# 各类破折号 / 否定符 → ASCII '-'
_DASH_MAP = str.maketrans(
    {
        "\u2014": "-",
        "\u2013": "-",
        "\u2212": "-",
        "\u00ac": "-",
        "\u30fc": "-",
        "\u2015": "-",
        "\u2500": "-",
        "\uff0d": "-",
    }
)

_CM_TILDE_RE = re.compile(r"cm\s*[~～]\s*1", re.IGNORECASE)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_ocr_text(text: str) -> str:
    """单段文本规范化。"""
    if not text:
        return text
    t = _CTRL_RE.sub("", text)
    t = t.translate(_DASH_MAP)
    t = _CM_TILDE_RE.sub("cm-1", t)
    if re.fullmatch(r"[\-\s]+", t):
        return "-"
    return t


def _tb_xyxy(tb: Dict[str, Any]) -> tuple[float, float, float, float]:
    poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
    if poly.size < 4:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 1e-9:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = max(area_a + area_b - inter, 1e-9)
    return float(inter / denom)


def _nms_text_boxes(
    text_boxes: List[Dict[str, Any]],
    *,
    iou_thresh: float = 0.55,
) -> List[Dict[str, Any]]:
    """按 IoU 去重，保留置信度更高或文本更长的框。"""
    if len(text_boxes) <= 1:
        return list(text_boxes)

    indexed = list(enumerate(text_boxes))
    indexed.sort(
        key=lambda item: (
            float(item[1].get("score", 1.0) or 1.0),
            len(str(item[1].get("text") or "")),
        ),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    kept_boxes: List[tuple[float, float, float, float]] = []
    for _idx, tb in indexed:
        rect = _tb_xyxy(tb)
        if rect == (0.0, 0.0, 0.0, 0.0):
            kept.append(tb)
            continue
        if any(_iou_xyxy(rect, k) >= iou_thresh for k in kept_boxes):
            continue
        kept.append(tb)
        kept_boxes.append(rect)
    return kept


def postprocess_text_boxes(
    text_boxes: List[Dict[str, Any]],
    *,
    binary: Any = None,  # noqa: ARG001 — 保留签名兼容，不再使用
) -> List[Dict[str, Any]]:
    """NMS 去重 + 文本规范化。"""
    out = _nms_text_boxes(text_boxes)
    for tb in out:
        raw = str(tb.get("text") or "")
        tb["text"] = normalize_ocr_text(raw)
    return out
