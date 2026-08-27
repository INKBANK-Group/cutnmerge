"""TableStructureRec：云端 PaddleOCR 作 ocr_result，输出 pred_html。"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 文档建议：wired v2 对约 1500px 内效果最好；超过此长边等比缩小再推理
_TSR_MAX_LONG_SIDE = 1500

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


def _classify_table(table_cls: Any, image: np.ndarray) -> str:
    """返回 'wired' 或 'lineless'。"""
    cls_out, _elapse = table_cls(image)
    label = str(cls_out).strip().lower()
    if label in {"wired", "wire", "有线"}:
        return "wired"
    return "lineless"


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
        box = poly[:4].tolist()
        text = str(tb.get("text") or "")
        score = float(tb.get("score", 1.0) or 1.0)
        out.append((box, text, score))
    return out


def _run_engine_with_ocr(
    engine: Any,
    image: np.ndarray,
    ocr_result: List[Tuple[Any, str, float]],
) -> Any:
    """调用 TSR 引擎，传入外部 OCR 结果（不使用 RapidOCR）。"""
    attempts = [
        {"ocr_result": ocr_result, "need_ocr": True},
        {"ocr_result": ocr_result},
        {"ocr_result": ocr_result, "need_ocr": False},
    ]
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            return engine(image, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return engine(image, ocr_result=ocr_result)


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
    ocr_result = text_boxes_to_ocr_result(text_boxes, scale=scale)
    logger.info("TSR 路径: %s，ocr_boxes=%d", kind, len(ocr_result))

    result = _run_engine_with_ocr(engine, infer_img, ocr_result)
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
