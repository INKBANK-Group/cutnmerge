"""TableStructureRec：云端 PaddleOCR 作 ocr_result，输出 pred_html。"""

from __future__ import annotations

import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 文档建议：wired v2 对约 1500px 内效果最好；超过此长边等比缩小再推理
_TSR_MAX_LONG_SIDE = 1500

# 当 wired 网格被拓扑判定为非法（同一物理列被多个逻辑列占用，过切列）时，
# 是否改用 OCR 几何重建表格。默认开；若批量回归发现误伤，可用环境变量关闭：
#   CUTNMERGE_TSR_OCR_GRID_FALLBACK=0
_OCR_GRID_FALLBACK = os.getenv("CUTNMERGE_TSR_OCR_GRID_FALLBACK", "1") not in {
    "0",
    "false",
    "no",
}

_table_cls = None
_wired_engine = None
_lineless_engine = None


def load_tsr_models() -> Tuple[Any, Any, Any]:
    """懒加载 table_cls / wired / lineless 单例。"""
    global _table_cls, _wired_engine, _lineless_engine
    if _table_cls is not None and _wired_engine is not None and _lineless_engine is not None:
        return _table_cls, _wired_engine, _lineless_engine

    from table_cls import TableCls
    from wired_table_rec.main import WiredTableInput, WiredTableRecognition
    from lineless_table_rec.main import LinelessTableInput, LinelessTableRecognition

    logger.info("加载 TableStructureRec（table_cls + wired/unet + lineless/lore）…")
    _table_cls = TableCls()
    _wired_engine = WiredTableRecognition(WiredTableInput(model_type="unet"))
    _lineless_engine = LinelessTableRecognition(LinelessTableInput())
    return _table_cls, _wired_engine, _lineless_engine


def _maybe_downscale(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """长边超过阈值则缩小；返回 (推理图, 相对原图的缩放比)。"""
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= _TSR_MAX_LONG_SIDE:
        return image, 1.0
    scale = _TSR_MAX_LONG_SIDE / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.info(
        "TSR 输入缩放: %dx%d -> %dx%d (scale=%.4f)",
        w,
        h,
        new_w,
        new_h,
        scale,
    )
    return resized, scale


def _classify_table(table_cls: Any, infer_image: np.ndarray) -> str:
    """返回 'wired' 或 'lineless'。"""
    cls_out, _elapse = table_cls(infer_image)
    label = str(cls_out).strip().lower()
    if label in {"wired", "wire", "有线"}:
        return "wired"
    return "lineless"


def _cluster_centers_bands(
    values,
    *,
    gap_factor: float = 1.5,
    floor: float = 12.0,
) -> List[float]:
    """把一维中心值按自适应间距聚成若干带，返回各带中心。

    间距阈值由相邻中心差的中位数导出（几何自适应，非内容魔法常数）；
    floor 只用于压制亚像素噪声，避免把同一物理列的轻微抖动拆成两列。
    """
    xs = np.sort(np.asarray(list(values), dtype=np.float64))
    if xs.size == 0:
        return []
    diffs = np.diff(xs)
    med = float(np.median(diffs)) if diffs.size else 1.0
    gap = max(floor, gap_factor * med)
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= gap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [float(np.mean(g)) for g in groups]


def _band_boundaries(centers: List[float]) -> List[float]:
    """带中心 -> 相邻带中点作为分界（首尾用 ±inf）。"""
    if not centers:
        return []
    bnd = [float("-inf")]
    for a, b in zip(centers[:-1], centers[1:]):
        bnd.append((a + b) / 2.0)
    bnd.append(float("inf"))
    return bnd


def _band_index(v: float, boundaries: List[float]) -> int:
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= v < boundaries[i + 1]:
            return i
    return len(boundaries) - 2


def _physical_grid_is_valid(
    cell_bboxes: np.ndarray,
    logic_points: np.ndarray,
    *,
    gap_factor: float = 1.5,
    floor: float = 12.0,
) -> bool:
    """纯几何判定 wired 网格是否拓扑合法。

    同一物理列带内出现 ≥2 个不同逻辑列号 → 一张表不可能让两个逻辑列占用同一个
    物理位置，故此网格自相矛盾（过切列/翻倍），视为非法，需走 OCR 几何重建。
    正常表每个物理列带只对应一个逻辑列号，返回 True（零改动）。
    """
    from collections import defaultdict

    cb = np.asarray(cell_bboxes, dtype=np.float64)
    lp = np.asarray(logic_points, dtype=np.float64)
    if cb.ndim == 2 and cb.shape[1] == 8:  # 引擎把 polygon 展平为 (N,8)
        cb = cb.reshape(-1, 4, 2)
    if cb.size == 0 or lp.size == 0 or cb.ndim != 3 or cb.shape[1] != 4:
        return True
    x_center = (cb[:, :, 0].min(1) + cb[:, :, 0].max(1)) / 2.0
    centers = _cluster_centers_bands(x_center, gap_factor=gap_factor, floor=floor)
    if len(centers) < 2:
        return True
    boundaries = _band_boundaries(centers)
    per_band: Dict[int, set] = defaultdict(set)
    col_start = lp[:, 2]  # 每格 col_start；用中心落入的带归组
    for i, c in enumerate(x_center):
        per_band[_band_index(float(c), boundaries)].add(int(col_start[i]))
    for _, cols in per_band.items():
        if len(cols) >= 2:
            return False
    return True


def _rebuild_grid_from_ocr(
    text_boxes: List[Dict[str, Any]],
    *,
    gap_factor: float = 1.5,
    floor: float = 12.0,
) -> str:
    """用 OCR 文本位置重建行列并填字（无损：所用文字均来自 OCR，逐 token 含中心）。

    仅用于 wired 网格被拓扑判定为非法的少数表；正常表不走此路径。
    返回 <html><body><table>…</table></body></html>，由 wrap_pred_html 统一注入边框样式。
    """
    boxes = text_boxes or []
    if not boxes:
        return ""

    def _center(tb: Dict[str, Any]) -> Tuple[float, float]:
        p = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
        if p.shape[0] < 4:
            return float("nan"), float("nan")
        return float(p[:, 0].mean()), float(p[:, 1].mean())

    centers = [_center(tb) for tb in boxes]
    xcs = [c[0] for c in centers if not np.isnan(c[0])]
    ycs = [c[1] for c in centers if not np.isnan(c[1])]
    xb = _cluster_centers_bands(xcs, gap_factor=gap_factor, floor=floor)
    yb = _cluster_centers_bands(ycs, gap_factor=gap_factor, floor=floor)
    if not xb or not yb:
        return ""
    xbnd = _band_boundaries(xb)
    ybnd = _band_boundaries(yb)
    R, C = len(yb), len(xb)
    grid = [[""] * C for _ in range(R)]
    for tb, (cx, cy) in zip(boxes, centers):
        if np.isnan(cx) or np.isnan(cy):
            continue
        r = _band_index(float(cy), ybnd)
        c = _band_index(float(cx), xbnd)
        t = str(tb.get("text") or "").strip().replace("\n", " ")
        if t:
            grid[r][c] = (grid[r][c] + " " + t).strip()

    parts = ["<html><body><table>"]
    for r in range(R):
        tds = []
        for c in range(C):
            cell = grid[r][c]
            tds.append(f"<td>{cell}</td>" if cell else "<td></td>")
        parts.append("<tr>" + "".join(tds) + "</tr>")
    parts.append("</table></body></html>")
    return "".join(parts)


def _normalize_tsr_quad(poly: np.ndarray) -> List[List[float]]:
    """
    转为 wired_table_rec match_ocr_cell 期望的四点序：左上、右上、右下、左下。

    match_ocr_cell 用 index 0 与 2 取 xmin/ymin/xmax/ymax，角点顺序错误会导致
    OCR 框与单元格无法匹配，HTML 行列错位（散架）。
    """
    arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    xmin, ymin = arr.min(axis=0)
    xmax, ymax = arr.max(axis=0)
    return [
        [float(xmin), float(ymin)],
        [float(xmax), float(ymin)],
        [float(xmax), float(ymax)],
        [float(xmin), float(ymax)],
    ]


def text_boxes_to_ocr_result(
    text_boxes: List[Dict[str, Any]],
    *,
    scale: float = 1.0,
) -> List[Tuple[Any, str, float]]:
    """
    将云端 PaddleOCR text_boxes 转为 TableStructureRec 的 ocr_result 格式。

    ocr_result: [(box, text, score), ...]，box 为 4 点坐标列表。
    scale: 将原图坐标缩放到 TSR 推理图坐标（_maybe_downscale 的 scale）。
    """
    out: List[Tuple[Any, str, float]] = []
    for tb in text_boxes or []:
        poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
        if poly.shape[0] < 4:
            continue
        if abs(scale - 1.0) > 1e-9:
            poly = poly * float(scale)
        box = _normalize_tsr_quad(poly)
        text = str(tb.get("text") or "")
        score = float(tb.get("score", 1.0) or 1.0)
        out.append((box, text, score))
    return out


def _assign_text_boxes_to_cells(
    text_boxes: List[Dict[str, Any]],
    cell_polygons: np.ndarray,
    *,
    scale: float = 1.0,
) -> List[Tuple[Any, str, float]]:
    """
    按 OCR 中心点落入的物理单元格分组；ocr_result 的 box 用单元格四角。

    云端 Paddle 检测框与 wired match_ocr_cell 的 IoU 匹配易错位，
    导致文字进错逻辑格、HTML 散架；吸附到结构检测格可对齐原型 RapidOCR 行为。
    """
    cells = np.asarray(cell_polygons, dtype=np.float64).reshape(-1, 4, 2)
    if cells.size == 0:
        return []
    if abs(scale - 1.0) > 1e-9:
        cells = cells * float(scale)

    cell_boxes = [_normalize_tsr_quad(c) for c in cells]
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    def _cell_center(cbox: List[List[float]]) -> Tuple[float, float]:
        xmin, ymin = cbox[0]
        xmax, ymax = cbox[2]
        return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    for tb in text_boxes or []:
        poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
        if poly.shape[0] < 4:
            continue
        if abs(scale - 1.0) > 1e-9:
            poly = poly * float(scale)
        cx = float(poly[:, 0].mean())
        cy = float(poly[:, 1].mean())
        best_i = -1
        best_score: Optional[float] = None
        for i, cbox in enumerate(cell_boxes):
            xmin, ymin = cbox[0]
            xmax, ymax = cbox[2]
            pad = 4.0
            if (
                xmin - pad > cx
                or xmax + pad < cx
                or ymin - pad > cy
                or ymax + pad < cy
            ):
                continue
            ccx, ccy = _cell_center(cbox)
            dist2 = (cx - ccx) ** 2 + (cy - ccy) ** 2
            area = max((xmax - xmin) * (ymax - ymin), 1.0)
            score = dist2 / area
            if best_score is None or score < best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            # 竖排表头等：中心点不在格内时，取最近单元格
            best_dist = float("inf")
            for i, cbox in enumerate(cell_boxes):
                ccx, ccy = _cell_center(cbox)
                dist2 = (cx - ccx) ** 2 + (cy - ccy) ** 2
                if dist2 < best_dist:
                    best_dist = dist2
                    best_i = i
        if best_i >= 0:
            groups[best_i].append(tb)

    def _sort_key(tb: Dict[str, Any]) -> Tuple[float, float]:
        tl = tb.get("top_left")
        if isinstance(tl, dict):
            return (float(tl.get("y", 0)), float(tl.get("x", 0)))
        poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
        return (float(poly[:, 1].min()), float(poly[:, 0].min()))

    out: List[Tuple[Any, str, float]] = []
    for i, cbox in enumerate(cell_boxes):
        items = groups.get(i)
        if not items:
            continue
        items.sort(key=_sort_key)
        for tb in items:
            text = str(tb.get("text") or "").strip()
            if not text:
                continue
            out.append((cbox, text, float(tb.get("score", 1.0) or 1.0)))
    return out


def _wired_ocr_result_from_cells(
    wired_engine: Any,
    infer_img: np.ndarray,
    text_boxes: List[Dict[str, Any]],
    *,
    scale: float = 1.0,
) -> List[Tuple[Any, str, float]]:
    """有线路径：结构检测后把云端 OCR 吸附到物理单元格再交给 wired。"""
    table_structure = getattr(wired_engine, "table_structure", None)
    if table_structure is None:
        return text_boxes_to_ocr_result(text_boxes, scale=scale)
    polygons, _rot = table_structure(infer_img)
    if polygons is None or getattr(polygons, "size", 0) == 0:
        return text_boxes_to_ocr_result(text_boxes, scale=scale)
    polygons = np.asarray(polygons).reshape(-1, 4, 2).copy()
    # 与 WiredTableRecognition.__call__ 中 match_ocr_cell 使用的 polygons 对齐
    polygons[:, 1, :], polygons[:, 3, :] = (
        polygons[:, 3, :].copy(),
        polygons[:, 1, :].copy(),
    )
    snapped = _assign_text_boxes_to_cells(text_boxes, polygons, scale=scale)
    if snapped:
        logger.info(
            "wired OCR 吸附: raw=%d snapped=%d cells=%d",
            len(text_boxes or []),
            len(snapped),
            len(polygons),
        )
        return snapped
    return text_boxes_to_ocr_result(text_boxes, scale=scale)


def _run_engine_with_ocr(
    engine: Any,
    image: np.ndarray,
    ocr_result: List[Tuple[Any, str, float]],
) -> Any:
    """调用 TSR 引擎，传入外部 OCR 结果（不使用 RapidOCR）。"""
    # need_ocr=False 会跳过 OCR 填格并返回空 pred_html，不可作为回退
    attempts = [
        {"ocr_result": ocr_result, "need_ocr": True},
        {"ocr_result": ocr_result},
    ]
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            return engine(image, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return engine(image, ocr_result=ocr_result, need_ocr=True)


def predict_table_html(
    image: np.ndarray,
    text_boxes: List[Dict[str, Any]],
    *,
    table_cls=None,
    wired_engine=None,
    lineless_engine=None,
    force_kind: Optional[str] = None,
) -> Tuple[Any, float]:
    """
    用 TableStructureRec 预测表格 HTML（pred_html）。

    Returns:
        (WiredTableOutput | LinelessTableOutput, infer_scale)
    """
    cls_eng, wired, lineless = load_tsr_models()
    table_cls = table_cls or cls_eng
    wired_engine = wired_engine or wired
    lineless_engine = lineless_engine or lineless

    infer_img, scale = _maybe_downscale(image)
    kind = (force_kind or _classify_table(table_cls, infer_img)).strip().lower()
    if kind not in {"wired", "lineless"}:
        kind = "lineless"

    engine = wired_engine if kind == "wired" else lineless_engine
    if kind == "wired":
        ocr_result = _wired_ocr_result_from_cells(
            wired_engine, infer_img, text_boxes, scale=scale
        )
    else:
        ocr_result = text_boxes_to_ocr_result(text_boxes, scale=scale)
    logger.info("TSR 路径: %s，ocr_boxes=%d", kind, len(ocr_result))

    result = _run_engine_with_ocr(engine, infer_img, ocr_result)

    if kind == "wired" and _OCR_GRID_FALLBACK:
        cb = getattr(result, "cell_bboxes", None)
        lp = getattr(result, "logic_points", None)
        if cb is not None and lp is not None:
            try:
                if not _physical_grid_is_valid(cb, lp):
                    logger.warning(
                        "wired 网格拓扑非法（同一物理列被多个逻辑列占用），改用 OCR 几何重建"
                    )
                    rebuilt = _rebuild_grid_from_ocr(text_boxes)
                    if rebuilt:
                        result.pred_html = rebuilt
            except Exception as exc:  # noqa: BLE001
                logger.warning("网格拓扑校验/重建失败，保留原 wired 结果: %s", exc)

    elapse = getattr(result, "elapse", None)
    n_cells = 0
    cell_bboxes = getattr(result, "cell_bboxes", None)
    if cell_bboxes is not None:
        n_cells = len(cell_bboxes)
    if elapse is not None:
        logger.info("TSR %s 耗时 %.3fs，cells=%d", kind, float(elapse), n_cells)
    else:
        logger.info("TSR %s cells=%d", kind, n_cells)
    return result, scale


_TABLE_BORDER_STYLE = """<meta charset="UTF-8">
<style>
table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    border: 1px solid #000;
    padding: 4px 8px;
    text-align: center;
    vertical-align: middle;
}
</style>"""


def _inject_table_border_style(html: str) -> str:
    """为 pred_html 注入表格边框样式（对齐 VisTable.insert_border_style）。"""
    if not html or "border-collapse" in html:
        return html
    import re

    if re.search(r"<head\b", html, re.I):
        return re.sub(
            r"(<head[^>]*>)",
            rf"\1\n{_TABLE_BORDER_STYLE}\n",
            html,
            count=1,
            flags=re.I,
        )
    if re.search(r"<body\b", html, re.I):
        return re.sub(
            r"(<body[^>]*>)",
            rf"\1\n{_TABLE_BORDER_STYLE}\n",
            html,
            count=1,
            flags=re.I,
        )
    return f"{_TABLE_BORDER_STYLE}\n{html}"


def wrap_pred_html(html: Optional[str]) -> str:
    """补全 pred_html 文档壳，并注入表格边框 CSS。"""
    if not html:
        return ""
    s = str(html).strip()
    if not s:
        return ""
    lower = s.lower()
    if "<html" in lower:
        if "<head" not in lower and "<body" in lower:
            s = s.replace("<html>", "<html>\n<head>\n</head>\n", 1)
            s = s.replace("<HTML>", "<html>\n<head>\n</head>\n", 1)
        s = _inject_table_border_style(s)
        if "<!doctype" not in lower:
            return f"<!DOCTYPE html>\n{s}"
        return s
    if "<body" in lower:
        s = _inject_table_border_style(s)
        return f"<!DOCTYPE html>\n<html>\n{s}\n</html>"
    s = _inject_table_border_style(s)
    return (
        f"<!DOCTYPE html>\n<html>\n<head>\n{_TABLE_BORDER_STYLE}\n</head>\n"
        f"<body>\n{s}\n</body>\n</html>"
    )


def _iter_cell_bboxes(tsr_result: Any) -> List[Any]:
    """安全取出 cell_bboxes（避免 numpy 数组做布尔判断）。"""
    cell_bboxes = getattr(tsr_result, "cell_bboxes", None)
    if cell_bboxes is None:
        return []
    arr = np.asarray(cell_bboxes)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        return [arr]
    return [arr[i] for i in range(len(arr))]


def _render_vis_fallback(
    image: np.ndarray,
    tsr_result: Any,
    *,
    colorful: bool = True,
) -> np.ndarray:
    """VisTable 不可用时的简单框线可视化。"""
    import random

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    for i, box in enumerate(_iter_cell_bboxes(tsr_result)):
        arr = np.asarray(box, dtype=np.float64).reshape(-1)
        if arr.size == 4:
            x1, y1, x2, y2 = arr.tolist()
            poly_i = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.int32,
            )
        elif arr.size >= 8:
            poly_i = np.round(arr[:8].reshape(4, 2)).astype(np.int32)
        else:
            continue
        if colorful:
            rng = random.Random(i * 9973 + 17)
            color = (rng.randint(32, 255), rng.randint(32, 255), rng.randint(32, 255))
        else:
            color = (255, 0, 0)
        cv2.polylines(canvas, [poly_i], True, color, 2)
    return canvas


def _render_vis_logic_fallback(
    image: np.ndarray,
    tsr_result: Any,
) -> np.ndarray:
    """VisTable 不可用时的逻辑行列标注图。"""
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
    canvas = cv2.copyMakeBorder(
        canvas, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    logic_points = getattr(tsr_result, "logic_points", None)
    logics: List[Any] = []
    if logic_points is not None:
        lp = np.asarray(logic_points)
        if lp.size > 0:
            logics = [lp[i] for i in range(len(lp))]

    for i, box in enumerate(_iter_cell_bboxes(tsr_result)):
        arr = np.asarray(box, dtype=np.float64).reshape(-1)
        if arr.size == 4:
            x0, y0, x1, y1 = arr.tolist()
        elif arr.size >= 8:
            pts = arr[:8].reshape(4, 2)
            x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        else:
            continue
        x0_i, y0_i, x1_i, y1_i = map(int, map(round, (x0, y0, x1, y1)))
        cv2.rectangle(canvas, (x0_i, y0_i), (x1_i, y1_i), (0, 0, 255), 1)
        if i < len(logics):
            logi = np.asarray(logics[i], dtype=np.float64).reshape(-1)
            if logi.size >= 4:
                rs, re, cs, ce = [int(round(float(x))) for x in logi[:4]]
                cv2.putText(
                    canvas,
                    f"row: {rs}-{re}",
                    (x0_i + 3, y0_i + 12),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.9,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"col: {cs}-{ce}",
                    (x0_i + 3, y0_i + 24),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.9,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
    return canvas


def save_tsr_vis(
    image: np.ndarray,
    tsr_result: Any,
    stem: str,
    out_dir: Path | str,
) -> List[str]:
    """写出表格可视化图（优先 VisTable，失败则回退 OpenCV 描框）。"""
    from ..utils.image_io import imwrite_unicode

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_paths: List[str] = []

    draw_path = out_dir / f"{stem}_table_vis.png"
    logic_path = out_dir / f"{stem}_table_vis_logic.png"

    try:
        from wired_table_rec.utils.utils import VisTable

        # VisTable 内部用 cv2.imread/imwrite，临时路径必须是纯 ASCII
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img_path = tmp_path / "input.png"
            html_path = tmp_path / "table.html"
            draw_tmp = tmp_path / "table_vis.png"
            logic_tmp = tmp_path / "table_vis_logic.png"
            if not imwrite_unicode(str(img_path), image):
                raise RuntimeError("无法写入临时图像")
            viser = VisTable()
            viser(
                str(img_path),
                tsr_result,
                str(html_path),
                str(draw_tmp),
                str(logic_tmp),
            )
            if draw_tmp.is_file():
                draw_path.write_bytes(draw_tmp.read_bytes())
                vis_paths.append(str(draw_path))
            if logic_tmp.is_file():
                logic_path.write_bytes(logic_tmp.read_bytes())
                vis_paths.append(str(logic_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("VisTable 失败，回退 OpenCV 描框: %s", exc)
        vis_img = _render_vis_fallback(image, tsr_result, colorful=True)
        if imwrite_unicode(str(draw_path), vis_img):
            vis_paths.append(str(draw_path))
        logic_img = _render_vis_logic_fallback(image, tsr_result)
        if imwrite_unicode(str(logic_path), logic_img):
            vis_paths.append(str(logic_path))

    for p in vis_paths:
        logger.info("表格可视化已写入: %s", p)
        print(f"[info] 表格划线图: {p}")
    return vis_paths
