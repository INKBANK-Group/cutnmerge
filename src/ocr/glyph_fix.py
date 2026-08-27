"""细笔画字形纠偏：用墨迹几何判别被 OCR 高置信误读的单字符。

背景：表格里单独成格的横杠 ``-`` 常被 PP-OCR 读成 ``1`` 或 ``一``（置信度 0.98~1.00，
纯调阈值改不动）。实测大量样本证明这两种形状可 100% 区分：

- 真字符（“5”“B”“A” 等）墨迹高 13~15px、占框高 0.52~0.68、宽高比 0.50~1.00
- 横杠 ``-``（含被误读成 ``1``/``一`` 的）墨迹高仅 2~3px、占框高 0.11~0.25、宽高比 1.67~3.00

因此只需量墨迹宽高比 AR 与墨迹高/框高 ratio，即可把误读的单字符改回 ``-``，且不会误伤真 ``1``
（真 ``1`` 同 “5”/``8`` 字号同基线，墨迹高必 ~14px，二个特征都不符合横杠）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 单字符且可能被误读成横杠/数字 1 的文本集合（_DASH_MAP 已把 —–−－ 等归一成 ASCII '-'）
_AMBIGUOUS = {"-", "1", "一", "l", "I", "|", "丨", "_"}

# 横杠判定阈值（见模块注释；真字符 AR<=1.0，横杠 AR>=1.67，取 1.3 留余量）
_DASH_AR = 1.3
# 墨迹高 / 框高：真字符 >=0.45，横杠 <=0.35，取 0.40 为界
_HEIGHT_RATIO_DASH = 0.35
_HEIGHT_RATIO_REAL = 0.45


def _aabb(poly: Any) -> Optional[tuple[float, float, float, float]]:
    """取 polygon 的轴对齐框 (x0,y0,x1,y1)；非四点返回 None。"""
    arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 4:
        return None
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
    )


def _measure_ink(
    gray: np.ndarray,
    box: tuple[float, float, float, float],
) -> Optional[tuple[float, float]]:
    """
    量裁剪区内最大墨迹连通域的 (宽高比, 墨迹高/框高)。

    剔除贴裁剪边的连通域（来自表格线/邻字），只取剩余最大者。返回 None 表示无有效墨迹。
    """
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = gray.shape[:2]
    # 外扩裁剪框：OCR 框常被 unclip 撑得比横杠还窄，导致墨迹贴到裁剪边、
    # 被下方「贴边剔除」当成表格线丢掉的误判。外扩后横杠落在内部即可保留。
    pad = max(2, int((box[2] - box[0]) * 0.15))
    x0 = max(0, min(x0 - pad, w - 1))
    x1 = max(x0 + 1, min(x1 + pad, w))
    y0 = max(0, min(y0 - pad, h - 1))
    y1 = max(y0 + 1, min(y1 + pad, h))
    crop = gray[y0:y1, x0:x1]
    if crop.size < 9:
        return None

    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ch, cw = binary.shape
    num, _labels, stats, _cent = cv2.connectedComponentsWithStats(binary, 8)

    best: Optional[np.ndarray] = None
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < 4:
            continue
        # 贴边 → 表格线或邻字残留，丢弃
        if x <= 0 or y <= 0 or x + ww >= cw or y + hh >= ch:
            continue
        # 过宽 → 不是单字符横杠
        if ww > 0.55 * cw:
            continue
        if best is None or area > best[4]:
            best = stats[i]

    if best is None:
        return None
    iw = float(best[2])
    ih = float(best[3])
    if ih <= 0:
        return None
    return (iw / ih, ih / (y1 - y0))


def _resolve(text: str, ink: Optional[tuple[float, float]]) -> Optional[str]:
    """根据墨迹几何修正单字符文本；无法判断返回 None。"""
    if ink is None:
        return None
    ar, rel = ink
    if ar >= _DASH_AR and rel <= _HEIGHT_RATIO_DASH:
        return "-"
    if ar <= 0.6 and rel >= _HEIGHT_RATIO_REAL:
        return "1"
    return None


def fix_thin_glyphs(
    image: np.ndarray,
    text_boxes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """修正单字符薄笔画的误读（主要 ``1``/``一`` → ``-``）。

    只处理文本为单字符且落在 ``_AMBIGUOUS`` 集合内的框；多字符 token 一律不碰。
    """
    if image is None or getattr(image, "size", 0) == 0:
        return text_boxes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    fixed = 0
    for tb in text_boxes:
        text = str(tb.get("text") or "").strip()
        if len(text) != 1 or text not in _AMBIGUOUS:
            continue
        box = _aabb(tb.get("polygon"))
        if box is None:
            continue
        ink = _measure_ink(gray, box)
        new = _resolve(text, ink)
        if new is not None and new != text:
            logger.debug(
                "字形纠偏: %s → %s (AR=%.2f rel=%.2f)",
                text,
                new,
                ink[0],
                ink[1],
            )
            tb["text"] = new
            tb["score"] = 0.99  # 几何确证，提分避免下游按分数丢弃
            fixed += 1

    if fixed:
        logger.info("字形纠偏: 修正 %d 个单字符", fixed)
    return text_boxes
