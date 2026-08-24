"""表头带假横切/竖切修复：按列检测横/竖线墨迹，局部恢复 rowspan/colspan。

- rowspan：仅处理「首个表体行之前」的行界；合并条件为横线弱 + 碎片/sliver/OCR 跨界/同列折行。
- colspan：仅处理表头行内相邻原子格；表头竖线弱且表体竖线强时合并（表体竖线穿入表头）。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .tsr_refine import (
    _cell_bbox,
    dedupe_overlapping_cells,
    _derive_seps,
    _rebuild_polygon,
    _refresh_spans,
)

logger = logging.getLogger(__name__)

_DATA_ROW_RE = re.compile(
    r"(合成例|实施例|実施例|比較例|比较例|对照例|参考例)"
)
_PARENT_HDR_RE = re.compile(r"(单体\s*[\[［]|聚合物)")
_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]"
)
_SUBHEADER_TAIL_RE = re.compile(
    r"(衍生物|化合物|单体|封端剂|硅烷|当量|比率|低聚物|\)|）)$"
)
# 短而完整的独立列头（P98「烯基」等）；不含「二羟基」这类可拼回前缀
_STANDALONE_SHORT_HDR_RE = re.compile(
    r"^(硝基|烯基|烷基|芳基|酰基|氨基|羟基|羧基|巯基|氰基|"
    r"醛基|酮基|醚基|酯基|卤素|粘度|酸值)$"
)


def _tb_bbox(tb: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(tb["polygon"], dtype=np.float64).reshape(-1, 2)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def _median_text_h(text_boxes: Sequence[Dict[str, Any]]) -> float:
    hs: List[float] = []
    for tb in text_boxes:
        _, y1, _, y2 = _tb_bbox(tb)
        h = max(0.0, y2 - y1)
        if h > 1e-3:
            hs.append(h)
    return float(np.median(hs)) if hs else 12.0


def _compact(text: str) -> str:
    return "".join((text or "").split())


def _is_fragment_text(text: str) -> bool:
    t = _compact(text)
    return (not t) or len(t) <= 2


def _hline_coverage_ratio(
    binary: np.ndarray,
    y: float,
    x1: float,
    x2: float,
    *,
    tol: float = 4.0,
) -> float:
    """
    本列区间内、y 附近的「横线」覆盖率（经水平形态学，抑制笔画误判）。
    返回 [0,1]；失败时偏保守返回 1.0（视为有线，不合并）。
    """
    if binary is None or binary.size == 0:
        return 1.0
    h, w = binary.shape[:2]
    ya = max(0, int(round(y - tol)))
    yb = min(h, int(round(y + tol + 1)))
    xa = max(0, int(round(x1)))
    xb = min(w, int(round(x2)))
    if yb <= ya or xb <= xa + 2:
        return 1.0
    band = binary[ya:yb, xa:xb]
    bw = int(band.shape[1])
    kw = max(9, min(bw // 3, bw))
    if kw < 3:
        return float(np.mean(band > 0))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    hmask = cv2.morphologyEx(band, cv2.MORPH_OPEN, ker, iterations=1)
    col_hit = np.any(hmask > 0, axis=0)
    if col_hit.size == 0:
        return 1.0
    return float(np.mean(col_hit))


def _vline_coverage_ratio(
    binary: np.ndarray,
    x: float,
    y1: float,
    y2: float,
    *,
    tol: float = 4.0,
) -> float:
    """
    y1..y2 区间内、x 附近的「竖线」覆盖率（经竖直形态学）。
    失败时偏保守返回 1.0（视为有线，不合并）。
    """
    if binary is None or binary.size == 0:
        return 1.0
    h, w = binary.shape[:2]
    xa = max(0, int(round(x - tol)))
    xb = min(w, int(round(x + tol + 1)))
    ya = max(0, int(round(y1)))
    yb = min(h, int(round(y2)))
    if yb <= ya or xb <= xa + 2:
        return 1.0
    band = binary[ya:yb, xa:xb]
    bh = int(band.shape[0])
    kw = max(9, min(bh // 3, bh))
    if kw < 3:
        return float(np.mean(band > 0))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kw))
    vmask = cv2.morphologyEx(band, cv2.MORPH_OPEN, ker, iterations=1)
    row_hit = np.any(vmask > 0, axis=1)
    if row_hit.size == 0:
        return 1.0
    return float(np.mean(row_hit))


def _substantive_cjk_label(text: str) -> bool:
    """表头子标签是否像完整的一列标题（而非折行碎片）。"""
    t = _compact(text)
    if len(t) < 5 or not _CJK_RE.search(t):
        return False
    if len(t) >= 12:
        return True
    return bool(_SUBHEADER_TAIL_RE.search(t))


def _is_standalone_short_header(text: str) -> bool:
    """短独立列头（如「烯基」），不可当碎片吞进邻列。"""
    t = _compact(text)
    return bool(t and _STANDALONE_SHORT_HDR_RE.fullmatch(t))


def _looks_like_two_independent_subheaders(left_text: str, right_text: str) -> bool:
    """两侧 OCR 都像独立子表头时，禁止 colspan 合并。"""
    if _substantive_cjk_label(left_text) and _substantive_cjk_label(right_text):
        return True
    l, r = _compact(left_text), _compact(right_text)
    # 一侧是短完整列头、另一侧仍有标签文本 → 禁止合并（P98 卤素列 vs 烯基）
    if _is_standalone_short_header(l) and (
        _substantive_cjk_label(r) or (_CJK_RE.search(r) and len(r) >= 2)
    ):
        return True
    if _is_standalone_short_header(r) and (
        _substantive_cjk_label(l) or (_CJK_RE.search(l) and len(l) >= 2)
    ):
        return True
    return False


def _same_column_wrap(
    upper: Dict[str, Any],
    lower: Dict[str, Any],
    *,
    min_ratio: float = 0.75,
) -> bool:
    ux1, _uy1, ux2, _uy2 = _cell_bbox(upper)
    lx1, _ly1, lx2, _ly2 = _cell_bbox(lower)
    overlap = min(ux2, lx2) - max(ux1, lx1)
    uw = max(ux2 - ux1, 1.0)
    lw = max(lx2 - lx1, 1.0)
    return overlap / min(uw, lw) >= min_ratio


def _ocr_texts_in_bbox(
    text_boxes: Sequence[Dict[str, Any]],
    bbox: Tuple[float, float, float, float],
) -> List[str]:
    x1, y1, x2, y2 = bbox
    out: List[str] = []
    for tb in text_boxes:
        bx1, by1, bx2, by2 = _tb_bbox(tb)
        cx = 0.5 * (bx1 + bx2)
        cy = 0.5 * (by1 + by2)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            out.append(str(tb.get("text") or "").strip())
    return out


def _ocr_crosses_y_in_x(
    text_boxes: Sequence[Dict[str, Any]],
    y: float,
    x1: float,
    x2: float,
    *,
    strip_half: float,
) -> bool:
    lo, hi = y - strip_half, y + strip_half
    for tb in text_boxes:
        bx1, by1, bx2, by2 = _tb_bbox(tb)
        cx = 0.5 * (bx1 + bx2)
        if not (x1 <= cx <= x2):
            continue
        if by1 < lo and by2 > hi:
            return True
    return False


def _first_body_row_index(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Sequence[Dict[str, Any]],
    row_seps: Sequence[float],
) -> int:
    """返回首个表体逻辑行索引；找不到则退化为 min(3, n_rows)。"""
    n_rows = max(0, len(row_seps) - 1)
    if n_rows <= 1:
        return n_rows

    for r in range(n_rows):
        y0, y1 = float(row_seps[r]), float(row_seps[r + 1])
        for tb in text_boxes:
            text = str(tb.get("text") or "")
            if not _DATA_ROW_RE.search(text):
                continue
            _x1, by1, _x2, by2 = _tb_bbox(tb)
            cy = 0.5 * (by1 + by2)
            if y0 <= cy < y1:
                return r

    for c in cells:
        text = str(c.get("text") or "")
        if _DATA_ROW_RE.search(text):
            return int(c["row_start"])

    return min(3, n_rows)


def _same_col_key(cell: Dict[str, Any]) -> Tuple[int, int]:
    return int(cell["col_start"]), int(cell["col_end"])


def _merge_pair(
    upper: Dict[str, Any],
    lower: Dict[str, Any],
) -> Dict[str, Any]:
    ux1, uy1, ux2, uy2 = _cell_bbox(upper)
    lx1, ly1, lx2, ly2 = _cell_bbox(lower)
    x1, y1 = min(ux1, lx1), min(uy1, ly1)
    x2, y2 = max(ux2, lx2), max(uy2, ly2)
    merged = dict(upper)
    merged["row_start"] = min(int(upper["row_start"]), int(lower["row_start"]))
    merged["row_end"] = max(int(upper["row_end"]), int(lower["row_end"]))
    merged["col_start"] = min(int(upper["col_start"]), int(lower["col_start"]))
    merged["col_end"] = max(int(upper["col_end"]), int(lower["col_end"]))
    _refresh_spans(merged)
    merged["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
    return merged


def _merge_horizontal_pair(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, Any]:
    lx1, ly1, lx2, ly2 = _cell_bbox(left)
    rx1, ry1, rx2, ry2 = _cell_bbox(right)
    x1, y1 = min(lx1, rx1), min(ly1, ry1)
    x2, y2 = max(lx2, rx2), max(ly2, ry2)
    merged = dict(left)
    merged["row_start"] = int(left["row_start"])
    merged["row_end"] = int(left["row_end"])
    merged["col_start"] = min(int(left["col_start"]), int(right["col_start"]))
    merged["col_end"] = max(int(left["col_end"]), int(right["col_end"]))
    _refresh_spans(merged)
    merged["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
    return merged


def repair_rowspans_by_hline_gaps(
    cells: List[Dict[str, Any]],
    binary: np.ndarray,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    tol: float = 4.0,
    min_ink_ratio: float = 0.40,
    sliver_ratio: float = 0.40,
    max_header_boundaries: int = 3,
) -> List[Dict[str, Any]]:
    """
    表头带内：若某列在行界处横线墨迹弱，且有碎片/sliver/OCR 跨界证据，
    则把该列上下原子格合并为 rowspan。不改 col_*。
    """
    if not cells or binary is None or binary.size == 0:
        return cells

    text_boxes = list(text_boxes or [])
    row_seps, _col_seps = _derive_seps(cells)
    logger.debug(
        "hline_repair: n_cells=%d row_seps=%d",
        len(cells), len(row_seps),
    )
    if len(row_seps) < 3:
        return cells

    n_rows = len(row_seps) - 1
    body_row = _first_body_row_index(cells, text_boxes, row_seps)
    n_header_boundaries = max(0, body_row - 1)
    max_boundary = min(n_header_boundaries, max_header_boundaries, n_rows - 1)
    logger.debug(
        "hline_repair: body_row=%d n_rows=%d max_boundary=%d",
        body_row, n_rows, max_boundary,
    )
    if max_boundary <= 0:
        logger.info(
            "hline_repair: 无可修表头行界 (body_row=%d n_rows=%d)",
            body_row,
            n_rows,
        )
        return cells

    median_h = _median_text_h(text_boxes)
    strip_half = max(3.0, 0.20 * median_h)
    sliver_h = max(4.0, sliver_ratio * median_h)

    work = [dict(c) for c in cells]
    merges = 0

    for boundary in range(max_boundary - 1, -1, -1):
        y = float(row_seps[boundary + 1])
        upper_row = boundary
        lower_row = boundary + 1

        uppers: Dict[Tuple[int, int], List[int]] = {}
        lowers: Dict[Tuple[int, int], List[int]] = {}
        for idx, cell in enumerate(work):
            rs, re = int(cell["row_start"]), int(cell["row_end"])
            key = _same_col_key(cell)
            if rs == re == upper_row:
                uppers.setdefault(key, []).append(idx)
            elif rs == re == lower_row:
                lowers.setdefault(key, []).append(idx)
        logger.debug(
            "hline_repair: boundary=%d upper_row=%d lower_row=%d "
            "upper_keys=%s lower_keys=%s",
            boundary, upper_row, lower_row,
            list(uppers.keys()), list(lowers.keys()),
        )

        to_drop: set = set()
        to_add: List[Dict[str, Any]] = []

        for key, u_idxs in uppers.items():
            l_idxs = lowers.get(key) or []
            if len(u_idxs) != 1 or len(l_idxs) != 1:
                continue
            ui, li = u_idxs[0], l_idxs[0]
            if ui in to_drop or li in to_drop:
                continue
            upper, lower = work[ui], work[li]
            x1, y1, x2, y2 = _cell_bbox(upper)
            lx1, ly1, lx2, ly2 = _cell_bbox(lower)
            xa, xb = min(x1, lx1), max(x2, lx2)

            ink = _hline_coverage_ratio(binary, y, xa, xb, tol=tol)

            u_texts = _ocr_texts_in_bbox(text_boxes, (x1, y1, x2, y2))
            l_texts = _ocr_texts_in_bbox(text_boxes, (lx1, ly1, lx2, ly2))
            u_joined = " ".join(u_texts)
            l_joined = " ".join(l_texts)

            # 短 CJK 标签对：两行都是 3-4 字 CJK，视为同一格折行。
            # 横线可能存在（全表贯穿线经过该格），故不受 ink 门槛限制。
            short_cjk_merge = False
            u_compact = _compact(u_joined)
            l_compact = _compact(l_joined)
            if (
                3 <= len(u_compact) <= 4
                and 3 <= len(l_compact) <= 4
                and _CJK_RE.search(u_compact)
                and _CJK_RE.search(l_compact)
                and _same_column_wrap(upper, lower)
                and not _SUBHEADER_TAIL_RE.search(u_compact)
                and not _SUBHEADER_TAIL_RE.search(l_compact)
            ):
                short_cjk_merge = True

            if ink >= min_ink_ratio and not short_cjk_merge:
                continue

            u_h = max(0.0, y2 - y1)
            l_h = max(0.0, ly2 - ly1)
            u_frag = _is_fragment_text(u_joined)
            l_frag = _is_fragment_text(l_joined)
            u_sliver = u_h <= sliver_h and u_frag
            l_sliver = l_h <= sliver_h and l_frag
            crosses = _ocr_crosses_y_in_x(
                text_boxes, y, xa, xb, strip_half=strip_half
            )

            wrap_merge = False
            if (
                not u_frag
                and not l_frag
                and _substantive_cjk_label(u_joined)
                and _substantive_cjk_label(l_joined)
                and _same_column_wrap(upper, lower)
                and not _PARENT_HDR_RE.search(u_joined)
                and not _PARENT_HDR_RE.search(l_joined)
            ):
                wrap_merge = True

            evidence = (
                (u_frag and not l_frag)
                or (l_frag and not u_frag)
                or (u_frag and l_frag)
                or u_sliver
                or l_sliver
                or crosses
                or wrap_merge
                or short_cjk_merge
            )
            if not evidence:
                continue

            merged = _merge_pair(upper, lower)
            to_drop.add(ui)
            to_drop.add(li)
            to_add.append(merged)
            merges += 1
            logger.debug(
                "hline_repair: merge col=%s rows=%d+%d ink=%.2f frag=(%s,%s) cross=%s wrap=%s",
                key,
                upper_row,
                lower_row,
                ink,
                u_frag,
                l_frag,
                crosses,
                wrap_merge,
            )

        if to_drop:
            work = [c for i, c in enumerate(work) if i not in to_drop]
            work.extend(to_add)

    if merges:
        logger.info(
            "hline_repair: 合并 %d 对表头格 (body_row=%d boundaries=0..%d)",
            merges,
            body_row,
            max_boundary - 1,
        )
    return work


def repair_colspans_by_vline_gaps(
    cells: List[Dict[str, Any]],
    binary: np.ndarray,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    tol: float = 4.0,
    header_max_vline: float = 0.15,
    body_min_vline: float = 0.40,
) -> List[Dict[str, Any]]:
    """
    表头带内：若列界在表头 Y 带竖线弱、在表体 Y 带竖线强，则合并相邻表头格为 colspan。

    只改表头 col_* 与 polygon；表体列结构不动。
    """
    if not cells or binary is None or binary.size == 0:
        return cells

    text_boxes = list(text_boxes or [])
    row_seps, col_seps = _derive_seps(cells)
    if len(row_seps) < 3 or len(col_seps) < 3:
        return cells

    n_rows = len(row_seps) - 1
    body_row = _first_body_row_index(cells, text_boxes, row_seps)
    body_row = min(max(body_row, 1), n_rows - 1)
    if body_row <= 0:
        return cells

    body_y1 = float(row_seps[body_row])
    body_y2 = float(row_seps[min(body_row + 2, n_rows)])

    work = [dict(c) for c in cells]
    merges = 0

    for header_row in range(body_row):
        for col_boundary in range(len(col_seps) - 2, 0, -1):
            left_col = col_boundary - 1
            right_col = col_boundary
            left_idx = None
            right_idx = None
            for idx, cell in enumerate(work):
                rs, re = int(cell["row_start"]), int(cell["row_end"])
                cs, ce = int(cell["col_start"]), int(cell["col_end"])
                if rs != re or rs != header_row:
                    continue
                if max(int(cell.get("col_span") or 1), 1) != 1:
                    continue
                if cs == ce == left_col:
                    left_idx = idx
                elif cs == ce == right_col:
                    right_idx = idx
            if left_idx is None or right_idx is None:
                continue

            left, right = work[left_idx], work[right_idx]
            lx1, ly1, lx2, ly2 = _cell_bbox(left)
            rx1, ry1, rx2, ry2 = _cell_bbox(right)
            x = float(col_seps[col_boundary])

            header_v = _vline_coverage_ratio(binary, x, ly1, ly2, tol=tol)
            if header_v >= header_max_vline:
                continue

            body_v = _vline_coverage_ratio(binary, x, body_y1, body_y2, tol=tol)
            if body_v < body_min_vline:
                continue

            l_texts = _ocr_texts_in_bbox(text_boxes, (lx1, ly1, lx2, ly2))
            r_texts = _ocr_texts_in_bbox(text_boxes, (rx1, ry1, rx2, ry2))
            if _looks_like_two_independent_subheaders(
                " ".join(l_texts), " ".join(r_texts)
            ):
                continue

            merged = _merge_horizontal_pair(left, right)
            work[left_idx] = merged
            work.pop(right_idx)
            merges += 1
            logger.debug(
                "vline_repair: merge header row=%d cols %d+%d hv=%.2f bv=%.2f",
                header_row,
                left_col,
                right_col,
                header_v,
                body_v,
            )

    # --- Second pass: absorb empty header cells into the neighboring labeled cell ---
    # 不限原子格、不限左右方向。P123 这类「左空块 + 右有字组表头」靠此收敛；
    # 两侧都空或都有字时跳过，避免误吞独立子表头。
    changed = True
    while changed:
        changed = False
        for header_row in range(body_row):
            row_cells = [
                (idx, c)
                for idx, c in enumerate(work)
                if int(c["row_start"]) == int(c["row_end"]) == header_row
            ]
            row_cells.sort(key=lambda t: int(t[1]["col_start"]))
            for ci in range(len(row_cells) - 1):
                l_idx, left = row_cells[ci]
                r_idx, right = row_cells[ci + 1]
                if int(left["col_end"]) + 1 != int(right["col_start"]):
                    continue
                l_texts = _ocr_texts_in_bbox(text_boxes, _cell_bbox(left))
                r_texts = _ocr_texts_in_bbox(text_boxes, _cell_bbox(right))
                l_has_text = any(t.strip() for t in l_texts)
                r_has_text = any(t.strip() for t in r_texts)
                if l_has_text == r_has_text:
                    continue
                col_boundary = int(right["col_start"])
                if col_boundary < 1 or col_boundary >= len(col_seps):
                    continue
                x = float(col_seps[col_boundary])
                lx1, ly1, lx2, ly2 = _cell_bbox(left)
                header_v = _vline_coverage_ratio(
                    binary, x, ly1, ly2, tol=tol
                )
                if header_v >= header_max_vline:
                    continue
                body_v = _vline_coverage_ratio(
                    binary, x, body_y1, body_y2, tol=tol
                )
                if body_v < body_min_vline:
                    continue
                # 以有字一侧为底，避免双向吸收时丢掉标签文本
                if l_has_text:
                    merged = _merge_horizontal_pair(left, right)
                else:
                    merged = _merge_horizontal_pair(right, left)
                work[l_idx] = merged
                work.pop(r_idx)
                merges += 1
                changed = True
                logger.debug(
                    "vline_repair(pass2): absorb empty neighbor "
                    "row=%d cols %d..%d+%d..%d hv=%.2f bv=%.2f",
                    header_row,
                    int(left["col_start"]),
                    int(left["col_end"]),
                    int(right["col_start"]),
                    int(right["col_end"]),
                    header_v,
                    body_v,
                )
                break
            if changed:
                break

    if merges:
        logger.info(
            "vline_repair: 合并 %d 对表头 colspan (body_row=%d)",
            merges,
            body_row,
        )
        work = dedupe_overlapping_cells(work)
    return work


# 常见子表头：种类 / 量 / pKa 等（非父表头折行碎片）
_SUBHEADER_TYPE_RE = re.compile(
    r"^(种类|类型|名称|组分|品种|项目|序号|牌号)$"
)
_SUBHEADER_AMOUNT_RE = re.compile(
    r"^(量|添加量|含量|用量|摩尔|质量|份数)"
)
_SUBHEADER_METRIC_RE = re.compile(
    r"(pKa|保护率|反应率|浓度|比率|比例|率[\(%）]|值[\(%）])",
    re.IGNORECASE,
)


def _cell_header_text(
    cell: Dict[str, Any],
    text_boxes: Sequence[Dict[str, Any]],
) -> str:
    own = str(cell.get("text") or "").strip()
    if own:
        return _compact(own)
    texts = _ocr_texts_in_bbox(text_boxes, _cell_bbox(cell))
    return _compact("".join(texts))


def _is_header_shard(text: str) -> bool:
    """≤3 字 CJK 且不像独立子表头 → 父表头横向折行碎片（如 酸催 / 化剂）。"""
    t = _compact(text)
    if not t or len(t) > 3 or not _CJK_RE.search(t):
        return False
    if _SUBHEADER_TYPE_RE.match(t):
        return False
    if _SUBHEADER_AMOUNT_RE.match(t):
        return False
    if _SUBHEADER_METRIC_RE.search(t):
        return False
    return True


def _atomic_cell_at_row_col(
    cells: Sequence[Dict[str, Any]],
    row: int,
    col: int,
) -> Optional[Dict[str, Any]]:
    for c in cells:
        if (
            int(c["row_start"]) == int(c["row_end"]) == row
            and int(c["col_start"]) == int(c["col_end"]) == col
        ):
            return c
    return None


def _cell_covering_row_col(
    cells: Sequence[Dict[str, Any]],
    row: int,
    col: int,
) -> Optional[Dict[str, Any]]:
    """覆盖 (row,col) 的格；同行多格时取 span 最小者。"""
    best: Optional[Dict[str, Any]] = None
    best_span = 10**9
    for c in cells:
        if int(c["row_start"]) <= row <= int(c["row_end"]) and int(c["col_start"]) <= col <= int(
            c["col_end"]
        ):
            span = (int(c["col_end"]) - int(c["col_start"]) + 1) * (
                int(c["row_end"]) - int(c["row_start"]) + 1
            )
            if span < best_span:
                best = c
                best_span = span
    return best


_CAPTION_ROW_RE = re.compile(r"^[\[［【\(（]?表\s*\d")


def _row_is_full_width_caption(
    cells: Sequence[Dict[str, Any]],
    row: int,
    max_col: int,
) -> bool:
    """整行仅一宽格且像表题 → 跳过，不作为父表头行。"""
    row_cells = [
        c
        for c in cells
        if int(c["row_start"]) <= row <= int(c["row_end"])
    ]
    if len(row_cells) != 1:
        return False
    c = row_cells[0]
    width = int(c["col_end"]) - int(c["col_start"]) + 1
    if width < max(max_col, 1) - 1:
        return False
    text = _compact(str(c.get("text") or ""))
    if not text:
        return False
    return bool(_CAPTION_ROW_RE.search(text) or len(text) <= 12 and "表" in text)


def _nested_header_parent_sub_rows(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Sequence[Dict[str, Any]],
    row_seps: Sequence[float],
) -> Tuple[int, int]:
    """
    多级列组表头的 (父行, 子行)。

    有题注行时父行在子行之上（P38：row1 父 / row2 子 / row3 表体），
    无题注时仍为 row0 父 / row1 子（P97 类）。
    """
    body_row = _first_body_row_index(cells, text_boxes, row_seps)
    if body_row < 2:
        return 0, max(0, body_row - 1)
    sub_row = body_row - 1
    parent_row = sub_row - 1
    max_col = max(int(c["col_end"]) for c in cells)
    while parent_row >= 0 and _row_is_full_width_caption(cells, parent_row, max_col):
        parent_row -= 1
    if parent_row < 0:
        parent_row = 0
    return parent_row, sub_row


def _subheader_cols_in_range(
    cells: Sequence[Dict[str, Any]],
    sub_row: int,
    cs: int,
    max_col: int,
    text_boxes: Sequence[Dict[str, Any]],
    *,
    parent_row: Optional[int] = None,
) -> List[int]:
    """从 cs 起子表头行连续列（含空槽但有表体证据）。"""
    cols: List[int] = []
    col = cs
    while col <= max_col:
        if col > cs and parent_row is not None:
            # 邻组父表头起始列 → 停止（如 酸催化剂|IPVE）
            pc = _cell_covering_row_col(cells, parent_row, col)
            if pc is not None and int(pc["col_start"]) == col:
                pt = _cell_header_text(pc, text_boxes)
                if pt and not _is_header_shard(pt):
                    break
            atom = _atomic_cell_at_row_col(cells, parent_row, col)
            if atom is not None:
                at = _cell_header_text(atom, text_boxes)
                if at and not _is_header_shard(at):
                    break
        label = _labels_in_row_range(cells, sub_row, col, col, text_boxes)[0]
        if label and not _is_header_shard(label):
            cols.append(col)
            col += 1
            continue
        if cols:
            nxt = (
                _labels_in_row_range(cells, sub_row, col + 1, col + 1, text_boxes)[0]
                if col + 1 <= max_col
                else ""
            )
            if nxt and not _is_header_shard(nxt):
                cols.append(col)
                col += 1
                continue
            break
        col += 1
    return cols


def should_preserve_header_colspan(
    cell: Dict[str, Any],
    cells: Sequence[Dict[str, Any]],
    text_boxes: Sequence[Dict[str, Any]],
) -> bool:
    """归属阶段：勿将多级父表头 colspan 按 col_seps 拆成原子列。"""
    if max(int(cell.get("col_span") or 1), 1) <= 1:
        return False
    row_seps, _ = _derive_seps(cells)
    if len(row_seps) < 3:
        return False
    parent_row, sub_row = _nested_header_parent_sub_rows(cells, text_boxes, row_seps)
    rs = int(cell["row_start"])
    if not (parent_row <= rs <= sub_row):
        return False
    cs, ce = int(cell["col_start"]), int(cell["col_end"])
    max_col = max(int(c["col_end"]) for c in cells)
    sub_cols = _subheader_cols_in_range(
        cells, sub_row, cs, max_col, text_boxes, parent_row=parent_row
    )
    if len(sub_cols) < 2:
        return False
    top_labels = _labels_in_row_range(cells, parent_row, cs, ce, text_boxes)
    bottom_labels = [_labels_in_row_range(cells, sub_row, c, c, text_boxes)[0] for c in sub_cols]
    return _should_merge_parent_shard_run(top_labels, bottom_labels)


def _labels_in_row_range(
    cells: Sequence[Dict[str, Any]],
    row: int,
    cs: int,
    ce: int,
    text_boxes: Sequence[Dict[str, Any]],
) -> List[str]:
    out: List[str] = []
    for col in range(cs, ce + 1):
        cell = _atomic_cell_at_row_col(cells, row, col)
        out.append(_cell_header_text(cell, text_boxes) if cell else "")
    return out


def _is_paired_dual_column_group(
    top_labels: Sequence[str],
    bottom_labels: Sequence[str],
) -> bool:
    """两列组：顶行两个独立语义词 + 底行「种类|量」型配对 → 保持横向拆分（如 基础|聚合物）。"""
    tops = [t for t in top_labels if t]
    bots = [t for t in bottom_labels if t]
    if len(tops) != 2 or len(bottom_labels) != 2:
        return False
    t0, t1 = tops[0], tops[1]
    b0, b1 = bottom_labels[0], bottom_labels[1]
    if _is_header_shard(t0) or _is_header_shard(t1):
        return False
    type_first = bool(
        _SUBHEADER_TYPE_RE.match(b0) or "种类" in b0 or "类型" in b0
    )
    amt_second = bool(
        _SUBHEADER_AMOUNT_RE.match(b1) or re.search(r"量", b1 or "")
    )
    return type_first and amt_second


def _should_merge_parent_shard_run(
    top_labels: Sequence[str],
    bottom_labels: Sequence[str],
) -> bool:
    """父表头行短碎片应合并为 colspan 父头，而非与下方子列 1:1 对齐。"""
    top_texts = [t for t in top_labels if t]
    bottom_texts = [t for t in bottom_labels if t]
    shard_count = sum(1 for t in top_texts if _is_header_shard(t))
    if shard_count < 2:
        return False
    if len(bottom_texts) >= 3 and len(top_texts) < len(bottom_texts):
        return True
    if len(bottom_labels) == 2 and len(top_texts) == 2:
        return not _is_paired_dual_column_group(top_labels, bottom_labels)
    return False


def repair_parent_header_shard_colspans(
    cells: List[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    父表头行：相邻短 CJK 碎片合并为 colspan 父头。

    典型：TSR 将「酸催化剂」拆成「酸催|化剂|空」三格，而子表头行有 种类|pKa|量 三列。
    不合并「基础|聚合物 + 种类|量(g)」这类双列配对布局。

    须在 OCR 归属之后调用（或至少 cells 上已有父表头碎片文本），以便从
    assign 拆开的原子格重新合并；自动跳过整行表题行。
    """
    if not cells:
        return cells

    text_boxes = list(text_boxes or [])
    row_seps, _col_seps = _derive_seps(cells)
    if len(row_seps) < 3:
        return cells

    body_row = _first_body_row_index(cells, text_boxes, row_seps)
    if body_row < 2:
        return cells

    parent_row, sub_row = _nested_header_parent_sub_rows(cells, text_boxes, row_seps)
    max_col = max(int(c["col_end"]) for c in cells)

    work = [dict(c) for c in cells]
    merges = 0
    col = 0
    while col <= max_col:
        first = _atomic_cell_at_row_col(work, parent_row, col)
        if first is None:
            # 已被 colspan 覆盖
            cov = _cell_covering_row_col(work, parent_row, col)
            if cov is not None and int(cov.get("col_span") or 1) > 1:
                col = int(cov["col_end"]) + 1
                continue
            col += 1
            continue

        run_cols: List[int] = [col]
        c = col
        while c + 1 <= max_col:
            nxt = _atomic_cell_at_row_col(work, parent_row, c + 1)
            if nxt is None:
                break
            run_cols.append(c + 1)
            c += 1

        cs, ce = run_cols[0], run_cols[-1]
        top_labels = _labels_in_row_range(work, parent_row, cs, ce, text_boxes)
        sub_cols = _subheader_cols_in_range(
            work, sub_row, cs, max_col, text_boxes, parent_row=parent_row
        )
        bottom_labels = [
            _labels_in_row_range(work, sub_row, sc, sc, text_boxes)[0] for sc in sub_cols
        ] if sub_cols else _labels_in_row_range(work, sub_row, cs, ce, text_boxes)

        if len(run_cols) >= 2 and _should_merge_parent_shard_run(
            top_labels, bottom_labels
        ):
            idxs = [
                i
                for i, cell in enumerate(work)
                if int(cell["row_start"]) == int(cell["row_end"]) == parent_row
                and int(cell["col_start"]) == int(cell["col_end"])
                and cs <= int(cell["col_start"]) <= ce
            ]
            idxs.sort(key=lambda i: int(work[i]["col_start"]))
            if len(idxs) >= 2:
                cells_to_merge = [work[i] for i in idxs]
                merged = dict(cells_to_merge[0])
                for mc in cells_to_merge[1:]:
                    merged = _merge_horizontal_pair(merged, mc)
                combined = "".join(
                    _cell_header_text(c, text_boxes) for c in cells_to_merge
                )
                if combined:
                    merged["text"] = combined
                # 子表头列数 > 父碎片列数 → 扩展 colspan（如 酸催|化剂 + 种类|pKa|量）
                if sub_cols and sub_cols[-1] > int(merged["col_end"]):
                    merged["col_end"] = sub_cols[-1]
                    _refresh_spans(merged)
                    lx1, ly1, lx2, ly2 = _cell_bbox(merged)
                    rx = _cell_bbox(cells_to_merge[-1])[2]
                    merged["polygon"] = _rebuild_polygon(lx1, ly1, rx, ly2)
                drop_idx = set(idxs)
                new_work: List[Dict[str, Any]] = []
                for i, c in enumerate(work):
                    if i in drop_idx:
                        if i == idxs[0]:
                            new_work.append(merged)
                        continue
                    new_work.append(c)
                work = new_work
                merges += 1
                logger.debug(
                    "shard_colspan: merge cols %d..%d → %r (span=%d)",
                    cs,
                    ce,
                    combined[:16],
                    int(merged.get("col_span") or 1),
                )
        elif len(run_cols) == 1 and sub_cols and sub_cols[-1] > ce:
            # 单格父头已存在但 colspan 不足（assign 前 TSR 宽格未被拆的情况兜底）
            idx = next(
                (
                    i
                    for i, cell in enumerate(work)
                    if int(cell["row_start"]) == int(cell["row_end"]) == parent_row
                    and int(cell["col_start"]) == cs
                ),
                None,
            )
            if idx is not None:
                top_labels_single = top_labels
                if _should_merge_parent_shard_run(top_labels_single, bottom_labels):
                    cell = work[idx]
                    if int(cell["col_end"]) < sub_cols[-1]:
                        cell = dict(cell)
                        cell["col_end"] = sub_cols[-1]
                        _refresh_spans(cell)
                        x1, y1, x2, y2 = _cell_bbox(cell)
                        cell["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
                        work[idx] = cell
                        merges += 1

        col = ce + 1

    if merges:
        logger.info(
            "shard_colspan: 合并 %d 组父表头碎片 (parent_row=%d sub_row=%d body_row=%d)",
            merges,
            parent_row,
            sub_row,
            body_row,
        )
        work = dedupe_overlapping_cells(work)
    return work
