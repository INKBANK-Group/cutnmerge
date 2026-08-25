# -*- coding: utf-8 -*-
"""多级表头拓扑修复：分组父头扩跨 + 泄漏碎片回收。

从 hline_repair 拆出，避免与其它表头补丁互相覆盖。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from .hline_repair import (
    _cell_covering_row_col,
    _cell_header_text,
    _compact,
    _first_body_row_index,
    _header_origin_at,
    _is_fragment_text,
    _is_header_shard,
    repair_parent_header_shard_colspans,
)
from .tsr_refine import (
    _cell_bbox,
    _derive_seps,
    _rebuild_polygon,
    _refresh_spans,
)

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]"
)
_GROUP_PARENT_HINT_RE = re.compile(r"(mol\s*比|\[mol|组成|组合物)")
_LETTERED_STUB_SHORT_RE = re.compile(
    r"^([（(][A-Za-zΑ-Ωα-ω][)）])(.{0,6})$"
)
_LEAF_UNIT_HDR_TAIL_RE = re.compile(r"(单元|结构|含量|闭环率|比率)$")
_LEAF_REST_START_RE = re.compile(r"^(聚|酰|苯|特|通|式)")


def _looks_like_group_parent_label(text: str) -> bool:
    """分组父头（mol比 / 组成）；排除「…结构单元及结构」侧栏。"""
    t = _compact(text)
    if len(t) < 8 or not _CJK_RE.search(t):
        return False
    if "及结构" in t:
        return False
    return bool(_GROUP_PARENT_HINT_RE.search(t))


def looks_like_leaf_unit_header(text: str) -> bool:
    """叶子单元列头（…单元 / …结构）；允许「…前」截断。对外供 html_formatter 使用。"""
    t = _compact(text)
    if len(t) < 3:
        return False
    if _looks_like_group_parent_label(t):
        return False
    if _LEAF_UNIT_HDR_TAIL_RE.search(t):
        return True
    if t.endswith("前") and "聚" in t and len(t) >= 5:
        return True
    return False


def _header_slot_absorbable(
    cells: Sequence[Dict[str, Any]],
    row: int,
    col: int,
    text_boxes: Sequence[Dict[str, Any]],
) -> bool:
    cov = _cell_covering_row_col(cells, row, col)
    if cov is None:
        return True
    if int(cov["row_start"]) < row and str(cov.get("text") or "").strip():
        return False
    if int(cov["row_end"]) > row and str(cov.get("text") or "").strip():
        if int(cov["row_start"]) == row and int(cov["col_start"]) == col:
            return False
    t = _cell_header_text(cov, text_boxes)
    if not t:
        return True
    if _looks_like_group_parent_label(t):
        return False
    if looks_like_leaf_unit_header(t):
        return False
    return _is_fragment_text(t) or _is_header_shard(t)


def _leaf_unit_cols_in_header(
    cells: Sequence[Dict[str, Any]],
    header_end: int,
    text_boxes: Sequence[Dict[str, Any]],
) -> List[int]:
    cols: List[int] = []
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        if rs > header_end or rs < 1:
            continue
        # 整高侧栏（特定元素 / 闭环率 / 结构描述）
        if rs <= 1 and re >= header_end and int(c.get("col_span") or 1) == 1:
            continue
        t = _cell_header_text(c, text_boxes)
        if not t or not looks_like_leaf_unit_header(t):
            continue
        cols.append(int(c["col_start"]))
    return sorted(set(cols))


def expand_group_parent_over_empty_slots(
    cells: List[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """分组父头向两侧吸收空槽，使 colspan 覆盖下方叶子列。"""
    if not cells:
        return cells
    text_boxes = list(text_boxes or [])
    row_seps, col_seps = _derive_seps(cells)
    if len(row_seps) < 3:
        return cells
    body_row = _first_body_row_index(cells, text_boxes, row_seps)
    if body_row < 2:
        return cells
    header_end = body_row - 1
    max_col = max(int(c["col_end"]) for c in cells)
    leaf_cols = _leaf_unit_cols_in_header(cells, header_end, text_boxes)
    if len(leaf_cols) < 2:
        return cells

    work = [dict(c) for c in cells]
    parent_idxs = [
        i
        for i, c in enumerate(work)
        if int(c["row_start"]) <= header_end
        and _looks_like_group_parent_label(_cell_header_text(c, text_boxes) or "")
    ]
    expanded = 0
    for pi in parent_idxs:
        if pi >= len(work):
            continue
        cell = work[pi]
        t = _cell_header_text(cell, text_boxes)
        if not t:
            continue
        rs = int(cell["row_start"])
        if int(cell["row_end"]) > rs and _LETTERED_STUB_SHORT_RE.match(
            _compact(t) or ""
        ):
            continue
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        nearby = [c for c in leaf_cols if cs - 6 <= c <= ce + 2]
        if len(nearby) < 2:
            continue
        new_cs, new_ce = cs, ce
        while new_cs > 0:
            left = new_cs - 1
            if not _header_slot_absorbable(work, rs, left, text_boxes):
                break
            if not any(c <= left for c in nearby):
                break
            new_cs = left
        while new_ce < max_col:
            right = new_ce + 1
            if not _header_slot_absorbable(work, rs, right, text_boxes):
                break
            if not any(c >= right for c in nearby):
                break
            new_ce = right
        leaf_lo, leaf_hi = min(nearby), max(nearby)
        new_cs = max(new_cs, leaf_lo)
        new_ce = min(new_ce, leaf_hi)
        if new_cs >= cs and new_ce <= ce:
            continue
        if new_ce - new_cs + 1 < 2:
            continue
        drop_ids = [
            i
            for i, o in enumerate(work)
            if i != pi
            and int(o["row_start"]) <= rs <= int(o["row_end"])
            and not (int(o["col_end"]) < new_cs or int(o["col_start"]) > new_ce)
            and not str(o.get("text") or "").strip()
        ]
        for i in reversed(drop_ids):
            work.pop(i)
            if i < pi:
                pi -= 1
        cell = work[pi]
        cell["col_start"] = new_cs
        cell["col_end"] = new_ce
        _refresh_spans(cell)
        if col_seps and len(col_seps) > new_ce + 1:
            x1 = float(col_seps[new_cs])
            x2 = float(col_seps[new_ce + 1])
            y1, _, _, y2 = _cell_bbox(cell)
            cell["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
        expanded += 1
        logger.info("group_parent_expand: %r → c%d-%d", t[:20], new_cs, new_ce)
    if expanded:
        return work
    return cells


def _try_reclaim_stub_prefix(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    lc = _compact(str(left.get("text") or ""))
    rc = _compact(str(right.get("text") or ""))
    if not lc or not rc:
        return False
    if not _LETTERED_STUB_SHORT_RE.match(lc):
        return False
    # 已是完整侧栏标签（…树脂/剂），勿再吞右侧叶子单元
    if re.search(r"(树脂|剂|物)$", lc) and len(lc) >= 6:
        return False
    if looks_like_leaf_unit_header(lc):
        return False
    best_n = None
    for n in range(1, min(8, len(rc)) + 1):
        peeled, rest = rc[:n], rc[n:]
        # 必须留下非空叶子列头，禁止整格吞并「聚酰亚胺单元」
        if not rest or not _LEAF_REST_START_RE.match(rest):
            continue
        if not (looks_like_leaf_unit_header(rest) or rest.endswith("前")):
            continue
        combined = lc + peeled
        if not (6 <= len(combined) <= 12):
            continue
        if looks_like_leaf_unit_header(combined):
            continue
        best_n = n
        break
    if best_n is None:
        return False
    peeled, rest = rc[:best_n], rc[best_n:]
    left["text"] = lc + peeled
    right["text"] = rest
    logger.info(
        "reclaim_stub: %r ← %r | rest=%r", left["text"][:16], peeled, rest[:16]
    )
    return True


def reclaim_leaked_header_fragments(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """回收（A）粘←合剂树脂… 与 前←体 泄漏。"""
    if not cells:
        return cells
    work = [dict(c) for c in cells]
    row_seps, _ = _derive_seps(work)
    if len(row_seps) < 2:
        return cells
    body_row = _first_body_row_index(work, [], row_seps)
    header_end = max(0, body_row - 1)
    max_col = max(int(c["col_end"]) for c in work)
    changed = 0

    def _reclaim_ti(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        lc = _compact(str(left.get("text") or ""))
        rc = _compact(str(right.get("text") or ""))
        if not rc.startswith("体") or "前" not in lc or "前体" in lc:
            return False
        raw = str(left.get("text") or "")
        left["text"] = re.sub(r"前(?!体)", "前体", raw, count=1)
        right["text"] = rc[1:]
        logger.info(
            "reclaim_char: %r ← 体 | rest=%r",
            _compact(left["text"])[:16],
            str(right["text"])[:16],
        )
        return True

    for row in range(0, header_end + 1):
        origins: List[Dict[str, Any]] = []
        for col in range(0, max_col + 1):
            o = _header_origin_at(work, row, col)
            if o is not None:
                origins.append(o)
        for i, left in enumerate(origins):
            if i + 1 >= len(origins):
                break
            right = origins[i + 1]
            if int(right["col_start"]) > int(left["col_end"]) + 1:
                continue
            if _try_reclaim_stub_prefix(left, right):
                changed += 1
            if _reclaim_ti(left, right):
                changed += 1

    for left in work:
        if int(left["row_start"]) > header_end:
            continue
        if int(left["row_end"]) <= int(left["row_start"]):
            continue
        lc = _compact(str(left.get("text") or ""))
        if not _LETTERED_STUB_SHORT_RE.match(lc):
            continue
        lce = int(left["col_end"])
        candidates = [
            c
            for c in work
            if c is not left
            and int(c["row_start"]) <= header_end
            and int(c["row_start"]) > int(left["row_start"])
            and int(c["col_start"]) > lce
            and _compact(str(c.get("text") or ""))
        ]
        candidates.sort(key=lambda c: (int(c["col_start"]), int(c["row_start"])))
        for right in candidates:
            if _try_reclaim_stub_prefix(left, right):
                changed += 1
                break

    return work if changed else cells


def repair_parent_header_topology(
    cells: List[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """碎片合并 → 泄漏回收 → 分组父头扩跨。"""
    text_boxes = list(text_boxes or [])
    work = repair_parent_header_shard_colspans(cells, text_boxes)
    work = reclaim_leaked_header_fragments(work)
    work = expand_group_parent_over_empty_slots(work, text_boxes)
    return work
