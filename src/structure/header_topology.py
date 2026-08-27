# -*- coding: utf-8 -*-
"""多级表头拓扑修复：分组父头扩跨 + 泄漏碎片回收。

从 hline_repair 拆出，避免与其它表头补丁互相覆盖。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .hline_repair import (
    _cell_covering_row_col,
    _cell_header_text,
    _compact,
    _first_body_row_index,
    _header_origin_at,
    _is_fragment_text,
    _is_header_shard,
    _is_numeric_subcol_label,
    _labels_in_row_range,
    _nested_header_parent_sub_rows,
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
    # 其它字母分类父头占坑时不可吞（避免 光聚合引发剂(B) 吞掉 聚合物(A)）
    if _looks_like_lettered_category_parent(t):
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


_LETTERED_CATEGORY_PARENT_RE = re.compile(
    r"^([（(][A-Za-z][)）](-\d+)?|"
    r".{0,12}(聚合物|引发剂).{0,4}[（(][A-Za-z][)）])$"
)
_LEAF_KIND_RE = re.compile(r"^(种类|種類|类型|類型|名称|名稱|组分|組分|品种|品種)$")
_LEAF_AMOUNT_RE = re.compile(r"^(配混量|量|添加量|含量|用量|分子量)")
_NUMERIC_METRIC_LEAF_RE = re.compile(r"^\d{2,5}$")


def _looks_like_numeric_metric_leaf(text: str) -> bool:
    """底行纯数字列头（200/300/… 或曝光宽容度刻度）。"""
    t = _compact(text)
    return bool(t) and bool(_NUMERIC_METRIC_LEAF_RE.match(t))


def _looks_like_lettered_category_parent(text: str) -> bool:
    """(D) / (C)-1 / 聚合物(A) / 光聚合引发剂(B) 等分类父头。"""
    t = _compact(text)
    if not t or len(t) > 24:
        return False
    if _LETTERED_CATEGORY_PARENT_RE.match(t):
        return True
    if re.search(r"[（(][A-Za-z][)）]", t) and not _LEAF_KIND_RE.match(t):
        if not _LEAF_AMOUNT_RE.match(t):
            return True
    return False


def _looks_like_kind_or_amount_leaf(text: str) -> bool:
    t = _compact(text)
    if not t:
        return False
    # 叶子里夹了父头标签（聚合物(A)分子量）→ 不当纯叶子
    if _looks_like_lettered_category_parent(t) and not (
        _LEAF_KIND_RE.match(t) or _LEAF_AMOUNT_RE.match(t)
    ):
        if re.search(r"(聚合物|引发剂).{0,8}[（(][A-Za-z][)）]", t):
            return False
    if _LEAF_KIND_RE.match(t) or _LEAF_AMOUNT_RE.match(t):
        return True
    if t.startswith("配混量") or t.startswith("分子量"):
        return True
    return False


def _parent_label_at_col(
    cells: Sequence[Dict[str, Any]],
    row: int,
    col: int,
    text_boxes: Sequence[Dict[str, Any]],
) -> str:
    for c in cells:
        if int(c["row_start"]) > row or int(c["row_end"]) < row:
            continue
        if int(c["col_start"]) <= col <= int(c["col_end"]):
            return _cell_header_text(c, text_boxes) or ""
    return ""


def _col_for_x(col_seps: Sequence[float], x: float) -> int:
    n = len(col_seps) - 1
    for c in range(n):
        if col_seps[c] <= x < col_seps[c + 1]:
            return c
    if n >= 1 and x >= col_seps[-2]:
        return n - 1
    return 0


def _leaf_text_at_col(
    cells: Sequence[Dict[str, Any]],
    leaf_row: int,
    col: int,
    text_boxes: Sequence[Dict[str, Any]],
) -> str:
    for c in cells:
        if int(c["row_start"]) != leaf_row or int(c["row_end"]) != leaf_row:
            continue
        if int(c["col_start"]) <= col <= int(c["col_end"]):
            return _cell_header_text(c, text_boxes) or ""
    return ""


def _split_embedded_parent_leaf(text: str) -> tuple[str, str]:
    """聚合物(A)分子量(Mw) → (聚合物(A), 分子量(Mw))。"""
    t = str(text or "").strip()
    if not t:
        return "", ""
    compact = _compact(t)
    m = re.search(
        r"((?:聚合物|引发剂|光聚合引发剂).{0,6}[（(][A-Za-z][)）])",
        compact,
    )
    if not m:
        if _looks_like_lettered_category_parent(compact):
            return compact, ""
        return "", t
    parent = m.group(1)
    # 用原串大致切分残留
    raw_m = re.search(
        r"((?:聚合物|引发剂|光聚合引发剂).{0,6}[（(][A-Za-z][)）])",
        t,
    )
    if raw_m:
        rest = (t[: raw_m.start()] + t[raw_m.end() :]).strip(" \n\r\t|/｜")
    else:
        rest = compact[m.end() :]
    rest = re.sub(r"^(分子量|配混量)", r"\1", rest)
    if rest and not _looks_like_kind_or_amount_leaf(rest):
        if "分子量" in rest or "Mw" in rest:
            rest = "分子量(Mw)" if "Mw" in rest or "mw" in rest.lower() else "分子量"
        elif not rest:
            rest = ""
    return parent, rest


def reanchor_lettered_parents_from_ocr(
    cells: List[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    用顶带 OCR 把字母分类父头钉回父行，并按叶子组（种类|…|配混量）划定 colspan。

    修：聚合物(A) OCR 在顶、结构却把字塞进叶子、再被 (B) 扩跨吞掉。
    """
    if not cells or not text_boxes:
        return cells
    text_boxes = list(text_boxes)
    row_seps, col_seps = _derive_seps(cells)
    if len(row_seps) < 3 or len(col_seps) < 3:
        return cells
    body_row = _first_body_row_index(cells, text_boxes, row_seps)
    if body_row < 2:
        return cells
    leaf_row = body_row - 1
    parent_row = 0
    header_bottom = float(row_seps[leaf_row])
    parent_band_y = float(row_seps[0]) + 0.55 * (
        header_bottom - float(row_seps[0])
    )

    # 顶带 OCR 父头（按 x 排序）
    ocr_parents: List[tuple[str, float]] = []
    for tb in text_boxes:
        t = _compact(str(tb.get("text") or ""))
        if not t or not _looks_like_lettered_category_parent(t):
            continue
        if _looks_like_kind_or_amount_leaf(t) and len(t) <= 4:
            continue
        try:
            poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
            cx = float(poly[:, 0].mean())
            cy = float(poly[:, 1].mean())
        except Exception:
            continue
        if cy > parent_band_y:
            continue
        ocr_parents.append((t, cx))
    if not ocr_parents:
        return cells
    ocr_parents.sort(key=lambda x: x[1])

    # 叶子原子列 → 分组：种类 开头；已有配混量后再来配混量则新开组（(C)-1）
    leaf_atoms: List[tuple[int, str]] = []
    for c in cells:
        if int(c["row_start"]) != leaf_row or int(c["row_end"]) != leaf_row:
            continue
        if int(c.get("col_span") or 1) != 1:
            continue
        # 跳过整高侧栏（耐化学等 rowspan 盖住叶子行）
        if any(
            int(o["row_start"]) < leaf_row
            and int(o["row_end"]) >= leaf_row
            and int(o["col_start"]) <= int(c["col_start"]) <= int(o["col_end"])
            and str(o.get("text") or "").strip()
            for o in cells
        ):
            continue
        t = _cell_header_text(c, text_boxes) or str(c.get("text") or "")
        leaf_atoms.append((int(c["col_start"]), t))
    leaf_atoms.sort(key=lambda x: x[0])

    groups: List[List[int]] = []
    cur: List[int] = []
    cur_has_amount = False
    for col, t in leaf_atoms:
        tc = _compact(t)
        is_kind = bool(_LEAF_KIND_RE.match(tc))
        is_amt = bool(
            _LEAF_AMOUNT_RE.match(tc)
            or tc.startswith("配混量")
            or tc.startswith("分子量")
        )
        if _looks_like_numeric_metric_leaf(t):
            if cur:
                groups.append(cur)
                cur = []
                cur_has_amount = False
            groups.append([col])
            continue
        if is_kind:
            if cur:
                groups.append(cur)
            cur = [col]
            cur_has_amount = False
            continue
        if is_amt and "配混" in tc and cur_has_amount:
            groups.append(cur)
            cur = [col]
            cur_has_amount = True
            continue
        if not cur:
            cur = [col]
        else:
            cur.append(col)
        if is_amt and "配混" in tc:
            cur_has_amount = True
        if is_amt and tc.startswith("分子量"):
            pass
    if cur:
        groups.append(cur)
    if not groups:
        return cells

    # 父 OCR ↔ 叶子组：数量相等则按 x 对齐，否则就近
    group_cx = []
    for g in groups:
        xs = []
        for col in g:
            if 0 <= col < len(col_seps) - 1:
                xs.append(0.5 * (float(col_seps[col]) + float(col_seps[col + 1])))
        group_cx.append(float(np.mean(xs)) if xs else 0.0)

    bounds: List[tuple[str, int, int]] = []
    used_g: set = set()
    for label, cx in ocr_parents:
        best_i, best_d = None, 1e18
        for i, gcx in enumerate(group_cx):
            if i in used_g:
                continue
            d = abs(gcx - cx)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None:
            continue
        used_g.add(best_i)
        g = groups[best_i]
        bounds.append((label, min(g), max(g)))

    if not bounds:
        return cells

    work = [dict(c) for c in cells]
    changed = 0

    # 先把叶子里夹带的父头字抬回（分子量残留留下）
    for c in work:
        if int(c["row_start"]) != leaf_row or int(c["row_end"]) != leaf_row:
            continue
        raw = str(c.get("text") or "")
        parent, rest = _split_embedded_parent_leaf(raw)
        if not parent:
            continue
        if parent == _compact(raw) and not rest:
            c["text"] = ""
            c["texts"] = []
            changed += 1
            continue
        if rest and rest != raw:
            c["text"] = rest
            c["texts"] = []
            changed += 1

    for label, cs, ce in bounds:
        drop: List[int] = []
        for i, c in enumerate(work):
            if int(c["row_start"]) > parent_row or int(c["row_end"]) < parent_row:
                continue
            # 不要动跨叶的 rowspan 侧栏
            if int(c["row_end"]) > parent_row and str(c.get("text") or "").strip():
                continue
            c0, c1 = int(c["col_start"]), int(c["col_end"])
            if c1 < cs or c0 > ce:
                continue
            ct = _compact(_cell_header_text(c, text_boxes) or str(c.get("text") or ""))
            if ct and ct != _compact(label) and _looks_like_lettered_category_parent(ct):
                if c0 >= cs and c1 <= ce:
                    drop.append(i)
                elif c0 < cs <= c1:
                    c["col_end"] = cs - 1
                    _refresh_spans(c)
                    if col_seps and c["col_end"] >= c["col_start"]:
                        x1 = float(col_seps[int(c["col_start"])])
                        x2 = float(col_seps[int(c["col_end"]) + 1])
                        y1, _, _, y2 = _cell_bbox(c)
                        c["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
                    changed += 1
                elif c0 <= ce < c1:
                    c["col_start"] = ce + 1
                    _refresh_spans(c)
                    if col_seps and c["col_end"] >= c["col_start"]:
                        x1 = float(col_seps[int(c["col_start"])])
                        x2 = float(col_seps[int(c["col_end"]) + 1])
                        y1, _, _, y2 = _cell_bbox(c)
                        c["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
                    changed += 1
            elif not ct or ct == _compact(label):
                drop.append(i)
        for i in reversed(sorted(set(drop))):
            work.pop(i)
            changed += 1

        y1 = float(row_seps[parent_row])
        y2 = float(row_seps[min(parent_row + 1, len(row_seps) - 1)])
        x1 = float(col_seps[cs])
        x2 = float(col_seps[min(ce + 1, len(col_seps) - 1)])
        work.append(
            {
                "polygon": _rebuild_polygon(x1, y1, x2, y2),
                "row_start": parent_row,
                "row_end": parent_row,
                "col_start": cs,
                "col_end": ce,
                "row_span": 1,
                "col_span": ce - cs + 1,
                "texts": [],
                "text": label,
            }
        )
        changed += 1
        logger.info(
            "reanchor_lettered_parent: %r → c%d-%d", label[:20], cs, ce
        )

    if changed:
        logger.info("reanchor_lettered_parents_from_ocr: 调整 %d 处", changed)
        return work
    return cells


def expand_lettered_category_parents(
    cells: List[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    字母编号分类父头向空槽扩跨，使 colspan 覆盖底行「种类|配混量」叶子列。

    与 expand_group_parent_over_empty_slots（组成/mol比）并列，互不替代。
    """
    if not cells:
        return cells
    text_boxes = list(text_boxes or [])
    row_seps, col_seps = _derive_seps(cells)
    if len(row_seps) < 3 or len(col_seps) < 3:
        return cells
    body_row = _first_body_row_index(cells, text_boxes, row_seps)
    if body_row < 2:
        return cells
    leaf_row = body_row - 1
    parent_row = 0
    max_col = max(int(c["col_end"]) for c in cells)

    leaf_cols: List[int] = []
    for c in cells:
        if int(c["row_start"]) != leaf_row or int(c["row_end"]) != leaf_row:
            continue
        if int(c.get("col_span") or 1) != 1:
            continue
        t = _cell_header_text(c, text_boxes)
        if _looks_like_kind_or_amount_leaf(t or ""):
            leaf_cols.append(int(c["col_start"]))
    leaf_cols = sorted(set(leaf_cols))
    if not leaf_cols:
        return cells

    work = [dict(c) for c in cells]
    parent_idxs = [
        i
        for i, c in enumerate(work)
        if int(c["row_start"]) == parent_row
        and int(c["row_end"]) == parent_row
        and _looks_like_lettered_category_parent(
            _cell_header_text(c, text_boxes) or ""
        )
    ]
    expanded = 0
    for pi in list(parent_idxs):
        if pi >= len(work):
            continue
        cell = work[pi]
        t = _cell_header_text(cell, text_boxes) or ""
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        new_cs, new_ce = cs, ce

        # 向右吞空槽，直到撞上另一父头标签或叶子组结束
        while new_ce < max_col:
            nxt = new_ce + 1
            other = _parent_label_at_col(work, parent_row, nxt, text_boxes)
            if other and _looks_like_lettered_category_parent(other):
                break
            if not _header_slot_absorbable(work, parent_row, nxt, text_boxes):
                break
            # 底行若是「聚合物(A)分子量」一类夹带父头的叶子，禁止继续吞
            leaf_under = _leaf_text_at_col(work, leaf_row, nxt, text_boxes)
            if leaf_under and _looks_like_numeric_metric_leaf(leaf_under):
                break
            if leaf_under and not _looks_like_kind_or_amount_leaf(leaf_under):
                if _looks_like_lettered_category_parent(
                    leaf_under
                ) or "聚合物" in _compact(leaf_under):
                    break
            # 已完成一组 種類|添加量，下一列不是本组叶子 → 停止（避免吞掉数字区段）
            pair_leaves = [c for c in leaf_cols if new_cs <= c <= new_ce]
            if nxt not in leaf_cols and len(pair_leaves) >= 2:
                break
            new_ce = nxt
            leaves_here = [c for c in leaf_cols if new_cs <= c <= new_ce]
            # 已盖住 ≥2 叶子且当前列是叶子 → 停（一组 种类|量 结束）
            if len(leaves_here) >= 2 and new_ce in leaf_cols:
                nxt2 = new_ce + 1
                if nxt2 > max_col:
                    break
                # 下一列若是新叶子「种类」则本组结束
                nxt_leaf_txt = ""
                for c in work:
                    if (
                        int(c["row_start"]) == leaf_row
                        and int(c["col_start"]) == nxt2
                        and int(c["col_end"]) == nxt2
                    ):
                        nxt_leaf_txt = _cell_header_text(c, text_boxes) or ""
                        break
                if _LEAF_KIND_RE.match(_compact(nxt_leaf_txt) or ""):
                    break
                # 下一列是另一父头
                o2 = _parent_label_at_col(work, parent_row, nxt2, text_boxes)
                if o2 and _looks_like_lettered_category_parent(o2):
                    break
                # 聚合物三列：种类|分子量|配混量 — 继续到配混量
                if len(leaves_here) < 3 and new_ce + 1 in leaf_cols:
                    continue
                break

        while new_cs > 0:
            left = new_cs - 1
            other = _parent_label_at_col(work, parent_row, left, text_boxes)
            if other and _looks_like_lettered_category_parent(other):
                break
            if not _header_slot_absorbable(work, parent_row, left, text_boxes):
                break
            if left not in leaf_cols and left < min(leaf_cols):
                break
            # 仅当左侧叶子属于本组（紧邻）时吸收
            if left in leaf_cols or _header_slot_absorbable(
                work, parent_row, left, text_boxes
            ):
                leaves_after = [c for c in leaf_cols if left <= c <= new_ce]
                if not leaves_after:
                    break
                new_cs = left
            else:
                break

        if new_cs == cs and new_ce == ce:
            continue
        if new_ce < new_cs:
            continue

        drop_ids = [
            i
            for i, o in enumerate(work)
            if i != pi
            and int(o["row_start"]) <= parent_row <= int(o["row_end"])
            and not (int(o["col_end"]) < new_cs or int(o["col_start"]) > new_ce)
            and not str(o.get("text") or "").strip()
        ]
        for i in reversed(sorted(drop_ids)):
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
        logger.info(
            "lettered_category_expand: %r → c%d-%d", t[:20], new_cs, new_ce
        )

    if expanded:
        logger.info("lettered_category_expand: %d 处", expanded)
        return work
    return cells


def drop_intr_span_header_shards(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """删除 lettered 父头 colspan 内、非起点列上的 OCR 泄漏单列格（如 カル光重合）。"""
    if not cells:
        return cells
    row_seps, _ = _derive_seps(cells)
    if len(row_seps) < 2:
        return cells
    body_row = _first_body_row_index(cells, [], row_seps)
    header_end = max(0, body_row - 1)
    dropped = 0
    out: List[Dict[str, Any]] = []
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if rs > header_end or rs != re or cs != ce:
            out.append(c)
            continue
        shard = False
        for pc in cells:
            if pc is c:
                continue
            prs, pre = int(pc["row_start"]), int(pc["row_end"])
            if prs > rs or pre < rs:
                continue
            pcs, pce = int(pc["col_start"]), int(pc["col_end"])
            if pce <= pcs or cs < pcs or cs > pce:
                continue
            if not _looks_like_lettered_category_parent(str(pc.get("text") or "")):
                continue
            if cs == pcs:
                break
            shard = True
            break
        if shard:
            dropped += 1
            logger.info(
                "drop_intr_span_shard: r%d c%d %r",
                rs,
                cs,
                str(c.get("text") or "")[:20],
            )
            continue
        out.append(c)
    if dropped:
        logger.info("drop_intr_span_header_shards: %d 处", dropped)
        return out
    return cells


def logic_body_start_row(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> int:
    """表体起始逻辑行：有 OCR 时用行标签/数值带信号，否则回退 _body_header_bounds。"""
    from .hline_repair import _first_body_row_index
    from .tsr_refine import _body_header_bounds, _derive_seps

    if text_boxes:
        row_seps, _col_seps = _derive_seps(cells)
        if len(row_seps) >= 3:
            return int(_first_body_row_index(cells, text_boxes, row_seps))
    _he, body_start, _body_h = _body_header_bounds(cells)
    return body_start


def needs_lettered_header_escalation(
    cells: Sequence[Dict[str, Any]],
    text_boxes: Sequence[Dict[str, Any]],
) -> bool:
    """轻量 TSR 表头碎成大量空 1×1 格，但 OCR 已检出 (A)~(J) 分类父头 → 应升级融合。

    几何信号：字母分类 OCR 框 ≥3，表头带空单元格过多且父头格过少。
    非 P67 专用——凡「成分列 (A)(B)… + 種類/重量部」双层表头均适用。
    """
    from .hline_repair import _looks_like_lettered_category_header

    if not cells or not text_boxes:
        return False
    lettered_ocr = sum(
        1
        for tb in text_boxes
        if _looks_like_lettered_category_header(str(tb.get("text") or ""))
    )
    if lettered_ocr < 1:
        return False
    body_start = logic_body_start_row(cells, text_boxes)
    top_row = min(int(c["row_start"]) for c in cells)
    scan_end = body_start if body_start >= 2 else min(
        top_row + 8, max(int(c["row_end"]) for c in cells) + 1
    )
    header = [c for c in cells if int(c["row_end"]) < scan_end]
    if len(header) < 2:
        return False
    empty_unit = 0
    lettered_cells = 0
    max_top_colspan = 1
    ghost_header_rows = 0
    for r in range(top_row + 1, scan_end):
        row_cells = [c for c in cells if int(c["row_start"]) <= r <= int(c["row_end"])]
        if not row_cells:
            continue
        if all(
            not str(c.get("text") or "").strip()
            and int(c.get("row_span") or 1) == 1
            and int(c.get("col_span") or 1) == 1
            for c in row_cells
        ):
            ghost_header_rows += 1
    for c in header:
        t = str(c.get("text") or "").strip()
        rs = int(c.get("row_span") or (int(c["row_end"]) - int(c["row_start"]) + 1))
        cs = int(c.get("col_span") or (int(c["col_end"]) - int(c["col_start"]) + 1))
        if not t and rs == 1 and cs == 1:
            empty_unit += 1
        if _looks_like_lettered_category_header(t):
            lettered_cells += 1
        if int(c["row_start"]) == top_row:
            max_top_colspan = max(max_top_colspan, cs)
    # 信号 A：表头碎成大量空 1×1，OCR 父头未落入拓扑
    if empty_unit >= 4 and lettered_cells < max(lettered_ocr, 2):
        return True
    # 信号 B：顶行超宽合并，但 OCR 检出多个 (A)(B)… 父头（典型成分表）
    if lettered_ocr >= 2 and max_top_colspan >= 8 and lettered_cells < lettered_ocr:
        return True
    # 信号 C：顶行有字母父头 + 下方多行全空幽灵行（轻量 TSR 典型碎表头）
    if (
        lettered_ocr >= 1
        and max_top_colspan >= 12
        and ghost_header_rows >= 2
        and lettered_cells <= lettered_ocr
    ):
        return True
    # 信号 D：OCR 已检出多个 (A)(B)… 父头，但拓扑表头区无一落入（轻量 TSR 典型）
    if lettered_ocr >= 2 and lettered_cells == 0:
        return True
    return False


def expand_parent_over_numeric_band(
    cells: List[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """数字子列带上方父格扩 colspan（Example 1..17 等，语言无关）。"""
    text_boxes = list(text_boxes or [])
    if not cells:
        return cells
    row_seps, col_seps = _derive_seps(cells)
    if len(row_seps) < 3 or len(col_seps) < 3:
        return cells

    parent_row, sub_row = _nested_header_parent_sub_rows(cells, text_boxes, row_seps)
    if sub_row <= parent_row:
        return cells

    max_col = max(int(c["col_end"]) for c in cells)
    changed = 0
    col = 0
    while col <= max_col:
        band: List[int] = []
        scan = col
        while scan <= max_col:
            label = _labels_in_row_range(cells, sub_row, scan, scan, text_boxes)[0]
            if _is_numeric_subcol_label(label):
                band.append(scan)
                scan += 1
            elif band:
                break
            else:
                scan += 1
        if len(band) < 3:
            col = scan + 1 if scan > col else col + 1
            continue

        cs_band, ce_band = band[0], band[-1]
        parent = _header_origin_at(cells, parent_row, cs_band)
        if parent is None:
            parent = _cell_covering_row_col(cells, parent_row, cs_band)
        if parent is None:
            col = ce_band + 1
            continue

        pcs, pce = int(parent["col_start"]), int(parent["col_end"])
        new_cs = min(pcs, cs_band)
        new_ce = max(pce, ce_band)
        if new_cs == pcs and new_ce == pce:
            col = ce_band + 1
            continue

        x1 = float(col_seps[new_cs])
        x2 = float(col_seps[new_ce + 1])
        y1 = float(row_seps[int(parent["row_start"])])
        y2 = float(row_seps[int(parent["row_end"]) + 1])
        parent["col_start"] = new_cs
        parent["col_end"] = new_ce
        _refresh_spans(parent)
        parent["polygon"] = _rebuild_polygon(x1, y1, x2, y2)
        changed += 1
        logger.info(
            "numeric_band_parent: r%d c%d-%d → colspan %d",
            parent_row,
            new_cs,
            new_ce,
            new_ce - new_cs + 1,
        )
        col = ce_band + 1

    if changed:
        logger.info("expand_parent_over_numeric_band: %d 处", changed)
    return cells


def repair_parent_header_topology(
    cells: List[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """碎片合并 → 泄漏回收 → OCR 父头回钉 → 分组父头扩跨 → 字母分类父头扩跨。"""
    text_boxes = list(text_boxes or [])
    work = repair_parent_header_shard_colspans(cells, text_boxes)
    work = reclaim_leaked_header_fragments(work)
    work = drop_intr_span_header_shards(work)
    work = reanchor_lettered_parents_from_ocr(work, text_boxes)
    work = expand_group_parent_over_empty_slots(work, text_boxes)
    work = expand_lettered_category_parents(work, text_boxes)
    work = expand_parent_over_numeric_band(work, text_boxes)
    return work
