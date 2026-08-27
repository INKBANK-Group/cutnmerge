"""基于逻辑拓扑的 HTML 表格生成（保留 rowspan/colspan）。

与 Markdown unroll 不同：合并单元格在起始格输出带 span 的 <td>，
被覆盖的子格跳过，不展开为空格子。
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .formatter import (
    _caption_norm_key,
    extract_caption_row,
    format_free_texts,
    split_cells_into_subtables,
)
from ..structure.row_header import is_physically_right_child
from ..utils.segments import _HEADER_CAPTION_RE
from ..utils.label_patterns import (
    is_index_column,
    parse_clean_example_label,
    repair_glued_example_label_index,
    split_example_local_and_composition_id,
)

logger = logging.getLogger(__name__)

# 噪声行：整行仅剩孤立字母/符号（如末行 "L"）
_NOISE_CELL_RE = re.compile(r"^[A-Za-z]$|^[·•\.\,;:]$")
# 身列数据值：纯数字、比率、缺测横线、单字母等级代号（A/B/C…）。
# 不含 jER-604 / 实施例1 / 4,4'-DAE。
# 身列数据值：纯数字、比率、缺测横线、单字母等级代号（A/B/C…）。
# 不含 jER-604 / 实施例1 / 4,4'-DAE。
_DATA_VALUE_RE = re.compile(
    r"^(?:[-–—−~～]|[-–—−]?\d+(?:\.\d+)?(?:\s*[\/／]\s*\d+(?:\.\d+)?)?|[A-Za-z][+＋]?|[A-Z]{2,3}|[A-Za-z]\d{1,2})$"
)
_LETTER_DATA_RE = re.compile(r"^(?:[A-Za-z][+＋]?|[A-Z]{2,3}|[A-Za-z]\d{1,2})$")
_EXAMPLE_COL_HEADER_RE = re.compile(
    r"(合成例|实施例|実施例|比較例|比较例|对照例|参考例)"
)
_MONOMER_PARENT_LABEL_RE = re.compile(r"单体\s*[\[［]")
# 左锚「聚合物」孤列：表体侧常见树脂/溶液名（迁到左侧有字无头列）
_LEFT_ANCHOR_HEADER_RE = re.compile(r"聚合物")
_POLYMER_BODY_NAME_RE = re.compile(
    r"(树脂|溶液|聚酰|聚硅|聚酰胺|聚苯|噁唑|酯|预聚|"
    r"PI-|PBO-|PS-|AC-|CR-|AE-|Bk-)"
)
_BODY_INDEX_LABEL_RE = re.compile(
    r"(合成例|实施例|実施例|比較例|比较例|对照例|参考例)"
)
# 组合物父表头 vs 颜料分散液：禁止逻辑重叠时互相拼字
_COMPOSITION_SPAN_HEADER_RE = re.compile(r"组成\s*[\[［]?\s*质量")
_COMPOSITION_PARENT_HDR_RE = re.compile(r"^组合物$")
# OCR 把侧栏「组合物」与顶栏「组成[…」粘在同一格
_GLUED_COMP_PARENT_SPAN_RE = re.compile(r"^组合物\s*组成")
_MASS_PARTS_LEADING_RE = re.compile(r"^质量\s*份")
_ACID_CONTENT_HDR_RE = re.compile(r"酸\s*的?\s*含量")
_TALL_METRIC_NMP_RE = re.compile(r"NMP")
_FORMULA_HDR_TOKEN_RE = re.compile(r"通\s*式")
_ROOM_STORAGE_MID_RE = re.compile(r"^室温\s*放置\s*(\d+)\s*日?")
_ROOM_STORAGE_FULL_RE = re.compile(r"^室温\s*放置\s*\d+\s*日\s*后")
_DAY_AFTER_FRAG_RE = re.compile(r"^日?\s*后$")
_EVAL_FILM_LEAF_RE = re.compile(r"显影时的?\s*膜\s*减少\s*量")
_EVAL_SENS_LEAF_RE = re.compile(r"^(感度变化|感度|化)$")
_TIMES_SHARD_RE = re.compile(r"^[×xX]$")
_TIMES_UNIT_LINE_RE = re.compile(
    r"^[（(]?\s*[×xX]\s*10\s*[-−–]?\s*\d*", re.I
)


def _has_formula_header_token(text: str) -> bool:
    return bool(_FORMULA_HDR_TOKEN_RE.search(text or ""))
_PIGMENT_DISPERSION_HEADER_RE = re.compile(r"颜料|分散\s*液")
_RECIPE_CODE_BODY_RE = re.compile(
    r"^(?:Bk|PI|PBO|PS|AC|CR|AE|MBA|PGMEA|NMP|GBL|DPHA|DPCA)[-－]?",
    re.IGNORECASE,
)
# 侧栏子行代号：A-1 / (A)-7 / B-12 等，与折行长标签碎片区分
_SIDE_COMPONENT_INDEX_RE = re.compile(
    r"^\(?[A-Za-z]{1,3}\)?\s*[-\u2013－]?\s*\d{1,3}$"
)


def _cell_key(cell: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return (
        int(cell["row_start"]),
        int(cell["row_end"]),
        int(cell["col_start"]),
        int(cell["col_end"]),
    )


def _occupied_columns(cells: List[Dict[str, Any]]) -> Set[int]:
    """逻辑上被任意 cell 覆盖的列索引。"""
    cols: Set[int] = set()
    for c in cells:
        for col in range(int(c["col_start"]), int(c["col_end"]) + 1):
            cols.add(col)
    return cols


def _occupied_rows(cells: List[Dict[str, Any]]) -> Set[int]:
    rows: Set[int] = set()
    for c in cells:
        for row in range(int(c["row_start"]), int(c["row_end"]) + 1):
            rows.add(row)
    return rows


def compress_empty_logic_columns(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    删除「没有任何 cell 覆盖」的幽灵逻辑列，并重映射 col_*。

    与 MD 的整列文本为空压缩不同：这里按拓扑占用判断，避免误删
    仅作 rowspan/colspan 占位、文本在起始格的合并列。
    """
    if not cells:
        return cells
    occupied = sorted(_occupied_columns(cells))
    if not occupied:
        return cells
    max_col = max(int(c["col_end"]) for c in cells)
    all_cols = list(range(max_col + 1))
    if occupied == all_cols:
        return cells

    remap = {old: new for new, old in enumerate(occupied)}
    out: List[Dict[str, Any]] = []
    for c in cells:
        nc = dict(c)
        cs, ce = int(c["col_start"]), int(c["col_end"])
        # 覆盖区间映射到压缩后连续区间
        new_idxs = [remap[i] for i in range(cs, ce + 1) if i in remap]
        if not new_idxs:
            continue
        nc["col_start"] = min(new_idxs)
        nc["col_end"] = max(new_idxs)
        nc["col_span"] = nc["col_end"] - nc["col_start"] + 1
        out.append(nc)
    return out


def compress_empty_logic_rows(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """删除没有任何 cell 覆盖的幽灵逻辑行。"""
    if not cells:
        return cells
    occupied = sorted(_occupied_rows(cells))
    if not occupied:
        return cells
    max_row = max(int(c["row_end"]) for c in cells)
    if occupied == list(range(max_row + 1)):
        return cells

    remap = {old: new for new, old in enumerate(occupied)}
    out: List[Dict[str, Any]] = []
    for c in cells:
        nc = dict(c)
        rs, re = int(c["row_start"]), int(c["row_end"])
        new_idxs = [remap[i] for i in range(rs, re + 1) if i in remap]
        if not new_idxs:
            continue
        nc["row_start"] = min(new_idxs)
        nc["row_end"] = max(new_idxs)
        nc["row_span"] = nc["row_end"] - nc["row_start"] + 1
        out.append(nc)
    return out


def _logic_area(cell: Dict[str, Any]) -> int:
    return (
        (int(cell["row_end"]) - int(cell["row_start"]) + 1)
        * (int(cell["col_end"]) - int(cell["col_start"]) + 1)
    )


def _unique_cell_refs(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[int] = set()
    out: List[Dict[str, Any]] = []
    for cell in cells:
        cid = id(cell)
        if cid not in seen:
            seen.add(cid)
            out.append(cell)
    return out


def _evict_occupants(
    occupants: List[Dict[str, Any]],
    occupancy: Dict[Tuple[int, int], Dict[str, Any]],
    out: List[Dict[str, Any]],
) -> None:
    ids = {id(o) for o in occupants}
    for owner in occupants:
        rs, re = int(owner["row_start"]), int(owner["row_end"])
        cs, ce = int(owner["col_start"]), int(owner["col_end"])
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                if occupancy.get((r, c)) is owner:
                    del occupancy[(r, c)]
    out[:] = [c for c in out if id(c) not in ids]


def _try_place_rect_remainder(
    cell: Dict[str, Any],
    free_positions: List[Tuple[int, int]],
    occupancy: Dict[Tuple[int, int], Dict[str, Any]],
    out: List[Dict[str, Any]],
    *,
    clear_text: bool,
) -> None:
    """若剩余空位能拼成完整矩形则占位；L 形不拆成额外空格。"""
    if not free_positions:
        return
    frs = min(p[0] for p in free_positions)
    fre = max(p[0] for p in free_positions)
    fcs = min(p[1] for p in free_positions)
    fce = max(p[1] for p in free_positions)
    needed = {
        (r, c)
        for r in range(frs, fre + 1)
        for c in range(fcs, fce + 1)
    }
    if not needed.issubset(set(free_positions)):
        return
    nc = dict(cell)
    nc["row_start"], nc["row_end"] = frs, fre
    nc["col_start"], nc["col_end"] = fcs, fce
    nc["row_span"] = fre - frs + 1
    nc["col_span"] = fce - fcs + 1
    if clear_text:
        nc["text"] = ""
    out.append(nc)
    for r, c in needed:
        occupancy[(r, c)] = nc


def _place_cell_rect(
    cell: Dict[str, Any],
    occupancy: Dict[Tuple[int, int], Dict[str, Any]],
    out: List[Dict[str, Any]],
) -> None:
    nc = dict(cell)
    out.append(nc)
    rs, re = int(nc["row_start"]), int(nc["row_end"])
    cs, ce = int(nc["col_start"]), int(nc["col_end"])
    for r in range(rs, re + 1):
        for c in range(cs, ce + 1):
            occupancy[(r, c)] = nc


def _monomer_subheader_merge_forbidden(a: str, b: str) -> bool:
    """「单体[…]」父格与中段子表头不得互相拼字（P97）；共聚子头异类也不拼（P47）。"""
    mon = re.compile(r"单体\s*[\[［]")
    poly = re.compile(r"聚合物")
    # 聚合物 ↔ 单体：禁止拼成「聚合物 单体[mol比]」
    if (poly.search(a) and mon.search(b)) or (poly.search(b) and mon.search(a)):
        return True
    # 侧栏（分子量/当量）↔ 单体：禁止拼字
    side = re.compile(
        r"(重量?平均分子量|重均分子量|数均分子量|平均分子量|酸当量|双键当量)"
    )
    if (side.search(a) and mon.search(b)) or (side.search(b) and mon.search(a)):
        return True
    sub = re.compile(
        r"(二羧酸|双氨基|封端剂|三官能|四官能|四羧酸|二胺及其|二羟基|二甲酰|有机硅烷)"
    )
    if (mon.search(a) and sub.search(b)) or (mon.search(b) and sub.search(a)):
        return True
    # 酸性/芳香/脂环/烯键 互不拼
    cats = []
    for t in (a, b):
        if re.search(r"烯键式|烯属", t):
            cats.append("ene")
        elif re.search(r"脂环式", t):
            cats.append("alicyclic")
        elif re.search(r"芳香族", t):
            cats.append("aromatic")
        elif re.search(r"酸性基团", t):
            cats.append("acidic")
        else:
            cats.append("other")
    if cats[0] != cats[1] and "other" not in cats:
        return True
    return False


def _composition_header_merge_forbidden(a: str, b: str) -> bool:
    """「组成[质量份]」与「颜料分散液」不得因逻辑重叠拼成一格。"""
    a_comp = bool(_COMPOSITION_SPAN_HEADER_RE.search(a or ""))
    b_comp = bool(_COMPOSITION_SPAN_HEADER_RE.search(b or ""))
    a_pig = bool(_PIGMENT_DISPERSION_HEADER_RE.search(a or ""))
    b_pig = bool(_PIGMENT_DISPERSION_HEADER_RE.search(b or ""))
    return (a_comp and b_pig) or (b_comp and a_pig)


def _tiered_column_header_merge_forbidden(a: str, b: str) -> bool:
    """表头带内不同父列/区段标签禁止因逻辑重叠拼进同一格。"""
    from ..structure.header_topology import (
        _looks_like_kind_or_amount_leaf,
        _looks_like_lettered_category_parent,
    )

    ca = "".join((a or "").split())
    cb = "".join((b or "").split())
    if not ca or not cb:
        return False
    if _looks_like_kind_or_amount_leaf(a) and _looks_like_independent_header_label(b):
        return True
    if _looks_like_kind_or_amount_leaf(b) and _looks_like_independent_header_label(a):
        return True
    if _looks_like_kind_or_amount_leaf(a) or _looks_like_kind_or_amount_leaf(b):
        return False
    la = _lettered_parent_letter(a)
    lb = _lettered_parent_letter(b)
    if la and lb and la.upper() != lb.upper():
        return True
    if _looks_like_lettered_category_parent(a) and _looks_like_lettered_category_parent(b):
        return ca != cb
    if _looks_like_lettered_category_parent(a) ^ _looks_like_lettered_category_parent(b):
        if len(ca) >= 4 and len(cb) >= 4 and not _is_cell_frag(a) and not _is_cell_frag(b):
            return True
    if (
        not la
        and not lb
        and len(ca) >= 3
        and len(cb) >= 3
        and not _is_cell_frag(a)
        and not _is_cell_frag(b)
        and _looks_like_independent_header_label(a)
        and _looks_like_independent_header_label(b)
    ):
        return True
    return False


def _clip_monomer_parent_row_overlap(
    cell: Dict[str, Any],
    unique_owners: List[Dict[str, Any]],
    occupancy: Dict[Tuple[int, int], Dict[str, Any]],
    out: List[Dict[str, Any]],
) -> bool:
    """单体父格 rowspan 盖住子表头行时：裁父格到单行，让子头独立占位。"""
    mon = re.compile(r"单体\s*[\[［]")
    sub = re.compile(
        r"(二羧酸|双氨基|封端剂|三官能|四官能|四羧酸|二胺及其|二羟基|二甲酰|有机硅烷)"
    )
    ct = str(cell.get("text") or "")
    # incoming 是子表头、owner 是跨行单体父
    for o in unique_owners:
        ot = str(o.get("text") or "")
        if sub.search(ct) and mon.search(ot) and int(o["row_end"]) > int(o["row_start"]):
            ors, ore = int(o["row_start"]), int(o["row_end"])
            ocs, oce = int(o["col_start"]), int(o["col_end"])
            for r in range(ors + 1, ore + 1):
                for c in range(ocs, oce + 1):
                    if occupancy.get((r, c)) is o:
                        del occupancy[(r, c)]
            o["row_end"] = ors
            o["row_span"] = 1
            # 再试放置 incoming
            rs, re_ = int(cell["row_start"]), int(cell["row_end"])
            cs, ce = int(cell["col_start"]), int(cell["col_end"])
            if all(occupancy.get((r, c)) is None for r in range(rs, re_ + 1) for c in range(cs, ce + 1)):
                _place_cell_rect(cell, occupancy, out)
                return True
        # incoming 是单体父、owner 已是子表头：裁 incoming 到单行
        if mon.search(ct) and sub.search(ot) and int(cell["row_end"]) > int(cell["row_start"]):
            cell["row_end"] = int(cell["row_start"])
            cell["row_span"] = 1
            rs, re_ = int(cell["row_start"]), int(cell["row_end"])
            cs, ce = int(cell["col_start"]), int(cell["col_end"])
            if all(occupancy.get((r, c)) is None for r in range(rs, re_ + 1) for c in range(cs, ce + 1)):
                _place_cell_rect(cell, occupancy, out)
                return True
    return False


def _clip_row_header_logic_overlap(
    cell: Dict[str, Any],
    unique_owners: List[Dict[str, Any]],
    occupancy: Dict[Tuple[int, int], Dict[str, Any]],
    out: List[Dict[str, Any]],
    rs: int,
    re: int,
    cs: int,
    ce: int,
) -> bool:
    """左右相邻的行头父子格：裁过宽父格的 col_end，而不是合并文本。

    成功把 incoming 放到无冲突矩形时返回 True。
    """
    if not unique_owners:
        return False

    right_owners = [
        o for o in unique_owners if is_physically_right_child(cell, o)
    ]
    if right_owners:
        clip_at = min(int(o["col_start"]) for o in right_owners) - 1
        new_end = max(cs, min(ce, clip_at))
        if new_end < ce:
            cell["col_end"] = new_end
            cell["col_span"] = new_end - cs + 1
            ce = new_end
            still = False
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    owner = occupancy.get((r, c))
                    if owner is not None and owner is not cell:
                        still = True
                        break
                if still:
                    break
            if not still:
                _place_cell_rect(cell, occupancy, out)
                return True

    left_parents = [
        o for o in unique_owners if is_physically_right_child(o, cell)
    ]
    if left_parents:
        for parent in left_parents:
            clip_at = int(cell["col_start"]) - 1
            pcs = int(parent["col_start"])
            old_ce = int(parent["col_end"])
            if clip_at < pcs or old_ce <= clip_at:
                continue
            for r in range(int(parent["row_start"]), int(parent["row_end"]) + 1):
                for c in range(clip_at + 1, old_ce + 1):
                    if occupancy.get((r, c)) is parent:
                        del occupancy[(r, c)]
            parent["col_end"] = clip_at
            parent["col_span"] = clip_at - pcs + 1
        still = False
        for r in range(rs, re + 1):
            for c in range(int(cell["col_start"]), int(cell["col_end"]) + 1):
                owner = occupancy.get((r, c))
                if owner is not None and owner is not cell:
                    still = True
                    break
            if still:
                break
        if not still:
            _place_cell_rect(cell, occupancy, out)
            return True

    return False


def _resolve_logic_overlaps(
    cells: List[Dict[str, Any]],
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    渲染前消解逻辑矩形重叠：后到的非空文本并入占位格，并裁掉冲突覆盖。

    空/碎片占位者遇到更大有字格时被赶走，避免大格剩余 L 形空位渲染成幽灵线。

    表头带不参与：多级表头父子格在逻辑格上重叠，拼字/裁切会破坏 (A)~(J) 拓扑。
    """
    if not cells:
        return cells

    from ..structure.header_topology import logic_body_start_row

    body_start = logic_body_start_row(cells, text_boxes)
    header_cells = [c for c in cells if int(c["row_start"]) < body_start]
    body_cells = [c for c in cells if int(c["row_start"]) >= body_start]

    out: List[Dict[str, Any]] = [dict(c) for c in header_cells]
    occupancy: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for cell in out:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                occupancy[(r, c)] = cell

    if not body_cells:
        return out

    # 按面积升序：先处理小格（更具体）——仅表体
    ordered = sorted(body_cells, key=lambda c: (
        _logic_area(c),
        int(c["row_start"]),
        int(c["col_start"]),
    ))

    for cell in ordered:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        text = str(cell.get("text") or "").strip()
        conflict_owners: List[Dict[str, Any]] = []
        free_positions: List[Tuple[int, int]] = []
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                owner = occupancy.get((r, c))
                if owner is None:
                    free_positions.append((r, c))
                elif owner is not cell:
                    conflict_owners.append(owner)

        if not conflict_owners:
            nc = dict(cell)
            out.append(nc)
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    occupancy[(r, c)] = nc
            continue

        unique_owners = _unique_cell_refs(conflict_owners)
        incoming_area = _logic_area(cell)
        if text and not _is_cell_frag(text):
            frag_ids = {
                id(o)
                for o in unique_owners
                if _is_cell_frag(str(o.get("text") or ""))
                and incoming_area > _logic_area(o)
            }
            if frag_ids:
                _evict_occupants(
                    [o for o in unique_owners if id(o) in frag_ids],
                    occupancy,
                    out,
                )
                unique_owners = [o for o in unique_owners if id(o) not in frag_ids]
                conflict_owners = unique_owners
                free_positions = [
                    (r, c)
                    for r in range(rs, re + 1)
                    for c in range(cs, ce + 1)
                    if occupancy.get((r, c)) is None
                ]
                if not conflict_owners:
                    nc = dict(cell)
                    out.append(nc)
                    for r in range(rs, re + 1):
                        for c in range(cs, ce + 1):
                            occupancy[(r, c)] = nc
                    continue

        # 左行头父格盖住右侧子格：裁父格列，而不是把子标签拼进父格。
        placed = _clip_row_header_logic_overlap(
            cell, unique_owners, occupancy, out, rs, re, cs, ce
        )
        if placed:
            continue
        # P97：单体父 rowspan 盖中段子表头 → 裁父行，禁止拼字
        if _clip_monomer_parent_row_overlap(cell, unique_owners, occupancy, out):
            continue
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        conflict_owners = []
        free_positions = []
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                owner = occupancy.get((r, c))
                if owner is None:
                    free_positions.append((r, c))
                elif owner is not cell:
                    conflict_owners.append(owner)
        if not conflict_owners:
            nc = dict(cell)
            out.append(nc)
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    occupancy[(r, c)] = nc
            continue
        unique_owners = _unique_cell_refs(conflict_owners)

        if text:
            phys_row_header = any(
                is_physically_right_child(cell, o) or is_physically_right_child(o, cell)
                for o in unique_owners
            )
            if phys_row_header:
                _try_place_rect_remainder(
                    cell, free_positions, occupancy, out, clear_text=False
                )
                continue
            owner = conflict_owners[0]
            prev = str(owner.get("text") or "").strip()
            if _monomer_subheader_merge_forbidden(prev, text):
                # 仍冲突：保留双方文本，尝试 L 形剩余，绝不拼进单体父格
                _try_place_rect_remainder(
                    cell, free_positions, occupancy, out, clear_text=False
                )
                continue
            if _composition_header_merge_forbidden(prev, text):
                # 「组成[质量份]」落入「颜料分散液」等左格时禁止拼字，保留双方
                _try_place_rect_remainder(
                    cell, free_positions, occupancy, out, clear_text=False
                )
                continue
            if _tiered_column_header_merge_forbidden(prev, text):
                _try_place_rect_remainder(
                    cell, free_positions, occupancy, out, clear_text=False
                )
                continue
            if text and text not in prev:
                owner["text"] = (prev + " " + text).strip() if prev else text
                logger.warning(
                    "逻辑格重叠，文本并入占位格 (%s,%s): %r",
                    owner.get("row_start"),
                    owner.get("col_start"),
                    text[:40],
                )
            # 有字冲突：不把 L 形剩余拆成额外空 <td>
            _try_place_rect_remainder(
                cell, free_positions, occupancy, out, clear_text=True
            )
            continue

        _try_place_rect_remainder(
            cell, free_positions, occupancy, out, clear_text=False
        )

    return out


def _looks_like_data_value(text: str) -> bool:
    t = "".join((text or "").split())
    return bool(t) and bool(_DATA_VALUE_RE.fullmatch(t))


def _column_has_data_values(
    cells: Sequence[Dict[str, Any]],
    col: int,
    *,
    skip_rows: Optional[Set[int]] = None,
) -> bool:
    """其他行在该列是否有原子数据格（数字 / 缺测横线）。"""
    skip = skip_rows or set()
    for cell in cells:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        if skip and not skip.isdisjoint(range(rs, re + 1)):
            continue
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if cs != ce or cs != col:
            continue
        if _looks_like_data_value(str(cell.get("text") or "")):
            return True
    return False


def _colspan_sibling_has_body_data(
    cells: Sequence[Dict[str, Any]], col: int, header_end: int
) -> bool:
    """表头 colspan 带内除 col 外是否另有列承担表体数据（幽灵列典型模式）。"""
    for cell in cells:
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if ce <= cs or int(cell["row_start"]) > header_end:
            continue
        if not (cs <= col <= ce):
            continue
        for sibling in range(cs, ce + 1):
            if sibling != col and (
                _column_has_data_values(cells, sibling)
                or _column_has_body_content(cells, sibling, header_end)
            ):
                return True
    return False


def _column_has_body_content(
    cells: Sequence[Dict[str, Any]],
    col: int,
    header_end: int,
    *,
    skip_rows: Optional[Set[int]] = None,
) -> bool:
    """表体行该列是否有非碎片文本（含 溶解/不溶 等分类值）。"""
    skip = skip_rows or set()
    for cell in cells:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        if rs <= header_end:
            continue
        if skip and not skip.isdisjoint(range(rs, re + 1)):
            continue
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if cs != ce or cs != col:
            continue
        t = str(cell.get("text") or "").strip()
        if t and not _is_cell_frag(t):
            return True
    return False


def _header_colspan_spans(cells: Sequence[Dict[str, Any]]) -> Set[Tuple[int, int]]:
    header_end = _effective_header_end(cells)
    spans: Set[Tuple[int, int]] = set()
    if header_end < 0:
        return spans
    for cell in cells:
        if int(cell["row_end"]) > header_end:
            continue
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if ce > cs:
            spans.add((cs, ce))
    return spans


def _empty_atomic_from_bbox(
    src: Dict[str, Any],
    col: int,
    orig_cs: int,
    orig_ce: int,
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """按原宽格比例切出一列空占位，避免幽灵列清理把空数据列删掉。"""
    x1, y1, x2, y2 = bbox
    span = max(orig_ce - orig_cs + 1, 1)
    w = (x2 - x1) / float(span)
    off = col - orig_cs
    nx1 = x1 + off * w
    nx2 = x1 + (off + 1) * w
    return {
        "row_start": int(src["row_start"]),
        "row_end": int(src["row_end"]),
        "col_start": col,
        "col_end": col,
        "row_span": int(src["row_end"]) - int(src["row_start"]) + 1,
        "col_span": 1,
        "text": "",
        "texts": [],
        "polygon": [[nx1, y1], [nx2, y1], [nx2, y2], [nx1, y2]],
    }


def _first_example_data_column(cells: Sequence[Dict[str, Any]]) -> Optional[int]:
    """表头带内实施例/比较例/参考例列头的最小 col_start（须 ≥1）。

    用于区分左侧行头区与数据区。列头落在第 0 列（行索引表）时返回 None，
    避免把空数据格当行头空隙吞掉。
    """
    if not cells:
        return None
    header_end = _effective_header_end(cells)
    first: Optional[int] = None
    for cell in cells:
        if int(cell["row_start"]) > header_end:
            continue
        cs = int(cell["col_start"])
        if cs < 1:
            continue
        if not _HEADER_HAS_EXAMPLE_RE.search(str(cell.get("text") or "")):
            continue
        if first is None or cs < first:
            first = cs
    return first


def _is_short_row_label_for_colspan_clip(text: str) -> bool:
    """仅短行头标签才允许按身列数据拆 colspan；中段换栏长描述/说明宽格不拆。"""
    t = str(text or "").strip()
    if not t:
        return False
    compact = "".join(t.split())
    if len(compact) > 16:
        return False
    if re.search(r"[；;。、]", t):
        return False
    if "说明" in compact or "结构单元" in compact:
        return False
    if compact.endswith("结构"):
        return False
    return True


def _extend_sparse_section_trailing_colspans(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    中段换栏：连续身行呈「左标签 + 已跨多列的右侧宽格」，且本段尾列全空时，
    将右侧宽格扩到表宽（修 TSR 宽格少盖末列 / 误拆后空尾列）。
    """
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    max_col = max(int(c["col_end"]) for c in cells)
    if max_col < 2:
        return cells

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for c in cells:
        if c.get("_drop_render"):
            continue
        rs, re = int(c["row_start"]), int(c["row_end"])
        if rs != re or rs <= header_end:
            continue
        by_row.setdefault(rs, []).append(c)
    if len(by_row) < 2:
        return cells

    def _sparse_wide_pair(
        row_cells: List[Dict[str, Any]],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        filled = [c for c in row_cells if str(c.get("text") or "").strip()]
        if len(filled) != 2:
            return None
        left, right = sorted(filled, key=lambda x: int(x["col_start"]))
        if int(left["col_start"]) > 0:
            return None
        if int(right["col_start"]) < 1:
            return None
        if int(right["col_end"]) - int(right["col_start"]) < 2:
            return None
        for c in row_cells:
            if c is left or c is right:
                continue
            if str(c.get("text") or "").strip():
                return None
        return left, right

    body_rows = sorted(by_row.keys())
    sparse_flags = {r: _sparse_wide_pair(by_row[r]) is not None for r in body_rows}
    runs: List[Tuple[int, int]] = []
    run_lo: Optional[int] = None
    run_hi = -1
    for r in body_rows:
        if sparse_flags[r]:
            if run_lo is None:
                run_lo = r
            run_hi = r
        elif run_lo is not None:
            if run_hi > run_lo:
                runs.append((run_lo, run_hi))
            run_lo = None
    if run_lo is not None and run_hi > run_lo:
        runs.append((run_lo, run_hi))

    changed = 0
    for r0, r1 in runs:
        for r in range(r0, r1 + 1):
            pair = _sparse_wide_pair(by_row[r])
            if pair is None:
                continue
            _left, right = pair
            old_ce = int(right["col_end"])
            if old_ce >= max_col:
                continue
            trail_busy = False
            for rr in range(r0, r1 + 1):
                for c in by_row[rr]:
                    if int(c["col_start"]) <= old_ce:
                        continue
                    if str(c.get("text") or "").strip():
                        trail_busy = True
                        break
                if trail_busy:
                    break
            if trail_busy:
                continue
            right["col_end"] = max_col
            right["col_span"] = max_col - int(right["col_start"]) + 1
            for c in by_row[r]:
                if c is right:
                    continue
                if int(c["col_start"]) > old_ce and not str(c.get("text") or "").strip():
                    c["_drop_render"] = True
            changed += 1
    if changed:
        logger.info("中段换栏尾列扩 colspan: %d 格 → col_end=%d", changed, max_col)
    return cells


def _split_label_over_data_columns(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    身行标签 colspan 盖住其他行已有数字的数据列时，收到左缘并补空格。

    保留与表头同区间的真合并（P98 NC-7000L）；不拆表头带。
    若能定位实施例列头，则仅当 colspan 伸进该列及以右时才裁切，
    避免把行头区的 A/B/C 子列当成数据列。
    """
    if len(cells) < 2:
        return cells
    header_end = _effective_header_end(cells)
    aligned = _header_colspan_spans(cells)
    first_data = _first_example_data_column(cells)
    extras: List[Dict[str, Any]] = []
    changed = 0
    for cell in cells:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if re != rs or ce <= cs:
            continue
        if rs <= header_end:
            continue
        if (cs, ce) in aligned:
            continue
        text = str(cell.get("text") or "").strip()
        if not text:
            continue
        if not _is_short_row_label_for_colspan_clip(text):
            continue
        clip_end = cs
        hit_data = False
        for col in range(cs + 1, ce + 1):
            if first_data is not None and col < first_data:
                clip_end = col
                continue
            if _column_has_data_values(cells, col, skip_rows={rs}):
                hit_data = True
                break
            clip_end = col
        if not hit_data or clip_end >= ce:
            continue
        orig_cs, orig_ce = cs, ce
        bbox = _cell_physical_bbox_x_y(cell)
        x1, y1, x2, y2 = bbox
        span = max(orig_ce - orig_cs + 1, 1)
        w = (x2 - x1) / float(span)
        nx2 = x1 + (clip_end - orig_cs + 1) * w
        cell["col_end"] = clip_end
        cell["col_span"] = clip_end - cs + 1
        cell["polygon"] = [[x1, y1], [nx2, y1], [nx2, y2], [x1, y2]]
        for col in range(clip_end + 1, orig_ce + 1):
            extras.append(_empty_atomic_from_bbox(cell, col, orig_cs, orig_ce, bbox))
        changed += 1
        logger.info(
            "身列空格拆 colspan: text=%r col %s-%s → %s-%s",
            str(cell.get("text") or "")[:24],
            orig_cs,
            orig_ce,
            cs,
            clip_end,
        )
    if changed:
        logger.info("身列空格拆 colspan 完成: %d 格", changed)
        cells.extend(extras)
    return cells


def _drop_leading_header_only_columns(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去掉仅被表头空格占用、表体从未覆盖的第 0 列（P26X194 左上角幽灵列）。
    """
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    max_col = max(int(c["col_end"]) for c in cells)
    drop_leading = 0
    for col in range(max_col + 1):
        covers = [
            c
            for c in cells
            if int(c["col_start"]) <= col <= int(c["col_end"])
        ]
        if not covers:
            drop_leading = col + 1
            continue
        body_hits = [
            c
            for c in covers
            if int(c["row_start"]) > header_end
            and str(c.get("text") or "").strip()
            and not _is_cell_frag(str(c.get("text") or ""))
        ]
        if body_hits:
            break
        header_only = all(int(c["row_end"]) <= header_end for c in covers)
        if header_only and all(not str(c.get("text") or "").strip() for c in covers):
            drop_leading = col + 1
            continue
        break
    if drop_leading <= 0:
        return cells
    out: List[Dict[str, Any]] = []
    for c in cells:
        nc = dict(c)
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if ce < drop_leading:
            continue
        nc["col_start"] = max(0, cs - drop_leading)
        nc["col_end"] = ce - drop_leading
        nc["col_span"] = int(nc["col_end"]) - int(nc["col_start"]) + 1
        out.append(nc)
    logger.info("去掉表头幽灵列: drop_leading=%d", drop_leading)
    return out


def _left_cols_have_rowspan_body_labels(
    cells: Sequence[Dict[str, Any]],
    max_col: int,
    header_end: int,
) -> bool:
    """表体左侧列存在跨多行的行头大格（如聚酰亚胺组成），区别于 P26 索引列。"""
    for cell in cells:
        if int(cell["row_start"]) <= header_end:
            continue
        if int(cell["col_start"]) >= max_col:
            continue
        rsp = int(
            cell.get("row_span")
            or (int(cell["row_end"]) - int(cell["row_start"]) + 1)
        )
        if rsp < 2:
            continue
        t = str(cell.get("text") or "").strip()
        if t and not _is_cell_frag(t):
            return True
    return False


def _collapse_header_empty_corners(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    表头行左侧纯空角格并入右侧首个表头标签（P24/P25：项目前有 3 个空 td）。

    仅当左侧列在表体已被 rowspan 大行头占用时触发，避免 P26 索引列空角被吞。
    """
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    _CJK_RE = re.compile(r"[\u4e00-\u9fff]")
    min_label_col: int | None = None
    for cell in cells:
        rs, re_ = int(cell["row_start"]), int(cell["row_end"])
        if rs > header_end or re_ != rs:
            continue
        t = str(cell.get("text") or "").strip()
        if t and not _is_cell_frag(t) and _CJK_RE.search(t):
            cs = int(cell["col_start"])
            if min_label_col is None or cs < min_label_col:
                min_label_col = cs
    if min_label_col is None or min_label_col <= 0:
        return cells
    if not _left_cols_have_rowspan_body_labels(cells, min_label_col, header_end):
        return cells

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for cell in cells:
        rs, re_ = int(cell["row_start"]), int(cell["row_end"])
        if rs > header_end or re_ != rs:
            continue
        by_row.setdefault(rs, []).append(cell)

    changed = 0
    for row_cells in by_row.values():
        row_cells.sort(key=lambda c: int(c["col_start"]))
        label_idx: int | None = None
        for i, cell in enumerate(row_cells):
            if int(cell["col_start"]) < min_label_col:
                continue
            t = str(cell.get("text") or "").strip()
            if t and not _is_cell_frag(t):
                label_idx = i
                break
        if label_idx is None:
            continue
        label = row_cells[label_idx]
        if _MONOMER_PARENT_LABEL_RE.search(str(label.get("text") or "")):
            continue
        # 实施例/比较例列头锚定数据列，不得左扩进侧栏 stub。
        # 否则会盖住表体 rowspan 行头，并把它们误判为「嵌套中层表头」从而把
        # 整段表体当成表头带（上下 rowspan 散架）。
        if _EXAMPLE_COL_HEADER_RE.search(str(label.get("text") or "")):
            continue
        orig_cs = int(label["col_start"])
        absorbed: List[Dict[str, Any]] = []
        for j in range(label_idx):
            cell = row_cells[j]
            if int(cell.get("col_span") or 1) != 1:
                break
            if str(cell.get("text") or "").strip():
                break
            col = int(cell["col_start"])
            if col >= min_label_col:
                break
            if not _column_has_body_content(cells, col, header_end):
                break
            absorbed.append(cell)
        if absorbed:
            new_start = int(absorbed[0]["col_start"])
            label["col_start"] = new_start
            label["col_span"] = int(label["col_end"]) - new_start + 1
            for cell in absorbed:
                cell["_drop_render"] = True
            changed += 1
            continue
        # 表头行无占位 cell、但逻辑列 0..orig_cs-1 由表体 rowspan 占用 → 左扩标签 colspan
        rs = int(label["row_start"])
        has_left = any(
            int(c["col_start"]) < orig_cs
            and int(c["row_start"]) <= rs <= int(c["row_end"])
            for c in cells
            if not c.get("_drop_render")
        )
        if not has_left and orig_cs >= min_label_col:
            if _MONOMER_PARENT_LABEL_RE.search(str(label.get("text") or "")):
                continue
            label["col_start"] = 0
            label["col_span"] = int(label["col_end"]) + 1
            changed += 1

    if changed:
        logger.info("表头空角压缩: %d 行", changed)
    return [c for c in cells if not c.get("_drop_render")]


def _split_header_over_index_column(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """表头格跨第 0 列、而身列 0 为实施例索引时，拆成空角 + 独立标签列。"""
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    if header_end < 0:
        return cells
    if not _column_has_body_content(cells, 0, header_end):
        return cells
    # P24/P25：身列 0 为 rowspan 大类行头，不是实施例索引列，勿拆表头角格
    if _left_cols_have_rowspan_body_labels(cells, 1, header_end):
        return cells

    max_col = max(int(c["col_end"]) for c in cells)

    def _skip_index_split(cell: Dict[str, Any]) -> bool:
        """整行题注 / 全宽表头不得按索引列角拆分。"""
        text = str(cell.get("text") or "").strip()
        if text and _HEADER_CAPTION_RE.search(text):
            return True
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        width = ce - cs + 1
        if width >= max(max_col, 1) - 1:
            return True
        if not text and width >= max(max_col + 1, 4) // 2:
            return True
        return False

    extras: List[Dict[str, Any]] = []
    changed = 0
    for cell in cells:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        if rs > header_end:
            continue
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if cs > 0 or ce <= cs:
            continue
        if not str(cell.get("text") or "").strip():
            continue
        if _skip_index_split(cell):
            continue
        bbox = _cell_physical_bbox_x_y(cell)
        x1, y1, x2, y2 = bbox
        span = ce - cs + 1
        w = (x2 - x1) / float(span)
        label_col = ce
        cell["col_start"] = label_col
        cell["col_end"] = label_col
        cell["col_span"] = 1
        nx1 = x1 + (label_col - cs) * w
        nx2 = nx1 + w
        cell["polygon"] = [[nx1, y1], [nx2, y1], [nx2, y2], [nx1, y2]]
        for col in range(cs, label_col):
            ex = _empty_atomic_from_bbox(cell, col, cs, ce, bbox)
            ex["row_start"] = rs
            ex["row_end"] = re
            ex["row_span"] = re - rs + 1
            extras.append(ex)
        changed += 1
        logger.info(
            "表头索引列角拆分: text=%r col %s-%s → %s",
            str(cell.get("text") or "")[:24],
            cs,
            ce,
            label_col,
        )
    if changed:
        cells.extend(extras)
    return cells


def _body_anchors_between(
    cells: Sequence[Dict[str, Any]],
    lo: int,
    hi: int,
    header_end: int,
) -> bool:
    """表体是否有格子的 col_start 落在 [lo, hi) 内。"""
    for cell in cells:
        if int(cell["row_start"]) <= header_end:
            continue
        cs = int(cell["col_start"])
        if lo <= cs < hi and str(cell.get("text") or "").strip():
            return True
    return False


def _column_has_upper_header(
    cells: Sequence[Dict[str, Any]],
    row: int,
    col_start: int,
    col_end: int,
) -> bool:
    """单位格正上方同列是否有表头描述（如 显影膜损失量 在上、(μm) 在下）。"""
    if row < 1:
        return False
    upper_row = row - 1
    for c in cells:
        if c.get("_drop_render"):
            continue
        if int(c["row_start"]) > upper_row or int(c["row_end"]) < upper_row:
            continue
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if cs <= col_end and ce >= col_start:
            if str(c.get("text") or "").strip():
                return True
    return False


_METRIC_SUMMARY_LABEL_RE = re.compile(r"分辨率|耐热性|粘合强度|应力")


def _row_has_trailing_data_values(
    cells: Sequence[Dict[str, Any]], row: int, after_col: int
) -> bool:
    """标签区右侧同一行是否已有实施例数值（测量汇总行）。"""
    for c in cells:
        if int(c["row_start"]) != row or int(c["row_end"]) != row:
            continue
        if int(c["col_start"]) <= after_col:
            continue
        t = str(c.get("text") or "").strip()
        if t and _looks_like_data_value(t):
            return True
    return False


def _should_merge_label_with_unit_tail(
    cells: Sequence[Dict[str, Any]],
    label_cell: Dict[str, Any],
    unit_cell: Dict[str, Any],
) -> bool:
    """
    同行标签 + 括号单位是否应合成一格。

    - P24 测量汇总行（分辨率 + 同行数值）：合并为「分辨率(μm)」
    - P33 / 嵌套表头（单位列上方有独立列头、同行无数值）：保持分列
    """
    label_text = str(label_cell.get("text") or "").strip()
    unit_text = str(unit_cell.get("text") or "").strip()
    if not _is_header_unit_line(unit_text):
        return False
    row = int(label_cell["row_start"])
    unit_cs = int(unit_cell["col_start"])
    unit_ce = int(unit_cell["col_end"])
    if not _column_has_upper_header(cells, row, unit_cs, unit_ce):
        return True
    if not _METRIC_SUMMARY_LABEL_RE.search(label_text):
        return False
    first_data = _first_example_data_column(cells)
    if first_data is not None:
        return _row_has_trailing_data_values(cells, row, first_data - 1)
    return _row_has_trailing_data_values(cells, row, int(label_cell["col_end"]))


def _is_side_component_index_token(text: str) -> bool:
    """左侧子列是否像组分代号（A-1 / (B)-4），而非折行标签碎片。"""
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    if _SIDE_COMPONENT_INDEX_RE.fullmatch(compact):
        return True
    if _LETTER_DATA_RE.fullmatch(compact):
        return True
    return False


def _merge_split_body_side_labels(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """表体左侧行头区：把跨 col0|col1 被竖线误切的长标签拼回 colspan。

    几何规则：仅在行头区（``col < first_data``）合并相邻有字格；右侧片段若像
    ``A-1`` 等组分代号则保留分列（聚合物|A-1）。评价行「分辨|率评价」等不满足
    代号形态，合并为一格。
    """
    if not cells:
        return cells
    first_data = _first_example_data_column(cells)
    if first_data is None or first_data < 2:
        return cells
    header_end = _effective_header_end(cells)
    _CJK = re.compile(r"[\u4e00-\u9fff]")

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for cell in cells:
        if cell.get("_drop_render"):
            continue
        if int(cell["row_start"]) != int(cell["row_end"]):
            continue
        if int(cell["row_start"]) <= header_end:
            continue
        if int(cell["col_end"]) >= first_data:
            continue
        by_row.setdefault(int(cell["row_start"]), []).append(cell)

    for row_cells in by_row.values():
        row_cells.sort(key=lambda c: int(c["col_start"]))
        i = 0
        while i + 1 < len(row_cells):
            left = row_cells[i]
            right = row_cells[i + 1]
            if int(right["col_start"]) != int(left["col_end"]) + 1:
                i += 1
                continue
            if int(left.get("col_span") or 1) != 1 or int(right.get("col_span") or 1) != 1:
                i += 1
                continue
            lt = str(left.get("text") or "").strip()
            rt = str(right.get("text") or "").strip()
            if not lt or not rt:
                i += 1
                continue
            if not _CJK.search(lt) or not _CJK.search(rt):
                i += 1
                continue
            if _is_side_component_index_token(rt) or _is_side_component_index_token(lt):
                i += 1
                continue
            # 两侧皆为描述性 CJK → 同一行头被竖线切开
            left["text"] = f"{lt}{rt}"
            left["col_end"] = int(right["col_end"])
            left["col_span"] = int(left["col_end"]) - int(left["col_start"]) + 1
            try:
                lx1, ly1, lx2, ly2 = _cell_physical_bbox_x_y(left)
                rx1, ry1, rx2, ry2 = _cell_physical_bbox_x_y(right)
                x1, y1 = min(lx1, rx1), min(ly1, ry1)
                x2, y2 = max(lx2, rx2), max(ly2, ry2)
                left["polygon"] = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
                )
            except Exception:
                pass
            right["_drop_render"] = True
            row_cells.pop(i + 1)
            # 继续尝试向右吸收更多碎片（罕见）
            continue
    return [c for c in cells if not c.get("_drop_render")]


def _retract_value_rowspan_across_side_indexes(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """表体数值格不得 rowspan 盖住带侧栏代号（A-2 / B-1）的行。

    转置表常见：配混量 ``100`` 被 TSR 误扩到下一空行；侧栏已有 ``A-2`` 时
    应收回为单行，留给空数据格。
    """
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    first_data = _first_example_data_column(cells)
    if first_data is None or first_data < 1:
        return cells

    index_rows: Set[int] = set()
    for c in cells:
        # 侧栏代号行：不要用 header_end 过滤——转置表左侧分类 rowspan
        # 常把 he 误扩到身行，导致 A-2 被当成「表头」而漏检。
        if int(c["col_start"]) >= first_data:
            continue
        t = str(c.get("text") or "").strip()
        if not _is_side_component_index_token(t):
            continue
        for r in range(int(c["row_start"]), int(c["row_end"]) + 1):
            index_rows.add(r)
    if not index_rows:
        return cells

    for c in cells:
        rs, re_ = int(c["row_start"]), int(c["row_end"])
        if re_ <= rs:
            continue
        if int(c["col_start"]) < first_data:
            continue
        t = str(c.get("text") or "").strip()
        if not t:
            continue
        compact = re.sub(r"\s+", "", t)
        if not (
            _DATA_VALUE_RE.fullmatch(compact)
            or _LETTER_DATA_RE.fullmatch(compact)
            or re.fullmatch(r"\d+(?:\.\d+)?", compact)
        ):
            continue
        # 数值格起点若落在侧栏代号行上，或跨过后续代号行 → 收回
        new_end = rs
        for r in range(rs + 1, re_ + 1):
            if r in index_rows:
                break
            new_end = r
        # 起点本身是代号行（罕见）也钳回单行
        if rs in index_rows and re_ > rs:
            new_end = rs
        if new_end < re_:
            c["row_end"] = new_end
            c["row_span"] = new_end - rs + 1
    return cells


def _merge_leading_empty_into_label(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """行首空列并入右侧中文标签格；同行「分辨率」+(μm) 合成单格。"""
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    _CJK_RE_LOCAL = re.compile(r"[\u4e00-\u9fff]")
    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for cell in cells:
        if int(cell["row_start"]) != int(cell["row_end"]):
            continue
        by_row.setdefault(int(cell["row_start"]), []).append(cell)

    for row_cells in by_row.values():
        row_cells.sort(key=lambda c: int(c["col_start"]))
        if len(row_cells) < 2:
            continue
        label_idx: int | None = None
        for i, cell in enumerate(row_cells):
            t = str(cell.get("text") or "").strip()
            if t and _CJK_RE_LOCAL.search(t):
                label_idx = i
                break
        if label_idx is None or label_idx == 0:
            continue
        label = row_cells[label_idx]
        # 「单体[…]」父格左侧常是聚合物/空角，不可左扩吞列
        if _MONOMER_PARENT_LABEL_RE.search(str(label.get("text") or "")):
            continue
        orig_cs = int(label["col_start"])
        new_start = orig_cs
        absorbed: List[Dict[str, Any]] = []
        for j in range(label_idx):
            cell = row_cells[j]
            if int(cell.get("col_span") or 1) != 1:
                break
            if str(cell.get("text") or "").strip():
                break
            col = int(cell["col_start"])
            if _column_has_body_content(cells, col, header_end):
                break
            if _body_anchors_between(cells, col, orig_cs, header_end):
                break
            new_start = col
            absorbed.append(cell)
        if not absorbed:
            continue
        label["col_start"] = new_start
        label["col_span"] = int(label["col_end"]) - int(label["col_start"]) + 1
        for cell in absorbed:
            cell["_drop_render"] = True

    # 分辨率 + (μm) 等：按表格结构决定同行合并或保持分列
    for row_cells in by_row.values():
        row_cells.sort(key=lambda c: int(c["col_start"]))
        for i, cell in enumerate(row_cells):
            t = str(cell.get("text") or "").strip()
            if not t or not _CJK_RE_LOCAL.search(t):
                continue
            if i + 1 >= len(row_cells):
                continue
            nxt = row_cells[i + 1]
            if int(nxt["col_start"]) != int(cell["col_end"]) + 1:
                continue
            if not _should_merge_label_with_unit_tail(cells, cell, nxt):
                continue
            nt = str(nxt.get("text") or "").strip()
            cell["text"] = f"{t}{nt}"
            cell["col_end"] = int(nxt["col_end"])
            cell["col_span"] = int(cell["col_end"]) - int(cell["col_start"]) + 1
            nxt["_drop_render"] = True

    return [c for c in cells if not c.get("_drop_render")]


def _cell_physical_bbox_x_y(cell: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = np.asarray(cell["polygon"], dtype=np.float64).reshape(-1, 2)
    return float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())


def _label_polygon_covers_neighbor(
    label: Dict[str, Any],
    neighbor: Dict[str, Any],
    *,
    min_cover: float = 0.5,
) -> bool:
    """标签物理框水平覆盖邻格宽度达到阈值 → 视为物理欠合并，可吸收。"""
    lx1, _, lx2, _ = _cell_physical_bbox_x_y(label)
    nx1, _, nx2, _ = _cell_physical_bbox_x_y(neighbor)
    nw = nx2 - nx1
    if nw <= 1e-6:
        return False
    overlap = max(0.0, min(lx2, nx2) - max(lx1, nx1))
    return overlap >= min_cover * nw


def _repair_row_header_coverage(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """行头区覆盖修复：补未覆盖逻辑洞；仅当多边形已盖住且列无实质内容时吸收空原子格。

    不跨实施例数据列。真实空子列（他行同列有文本、或标签框未盖住）一律保留。
    """
    if not cells:
        return cells
    first_data = _first_example_data_column(cells)
    if first_data is None or first_data <= 1:
        return cells

    header_end = _effective_header_end(cells)
    work = [c for c in cells if not c.get("_drop_render")]
    occupancy: Dict[Tuple[int, int], Dict[str, Any]] = {}
    max_row = 0
    for cell in work:
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        max_row = max(max_row, re)
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                occupancy[(r, c)] = cell

    for row in range(header_end + 1, max_row + 1):
        col = 0
        while col < first_data:
            owner = occupancy.get((row, col))
            if (
                owner is None
                or owner.get("_drop_render")
                or int(owner["row_start"]) != row
                or int(owner["row_end"]) != row
                or int(owner["col_start"]) != col
                or not str(owner.get("text") or "").strip()
            ):
                col += 1
                continue
            end = int(owner["col_end"])
            absorbed: List[Dict[str, Any]] = []
            j = end + 1
            while j < first_data:
                nxt = occupancy.get((row, j))
                if nxt is not None and nxt.get("_drop_render"):
                    nxt = None
                if nxt is None:
                    # 纯逻辑洞：只扩 colspan，不删格
                    end = j
                    j += 1
                    continue
                if nxt is owner:
                    j += 1
                    continue
                if int(nxt["row_start"]) != row or int(nxt["row_end"]) != row:
                    break
                if str(nxt.get("text") or "").strip():
                    break
                if int(nxt.get("col_span") or 1) != 1:
                    break
                # 列他处有实质内容 → 真实空子列，保留
                if _column_has_data_values(work, j, skip_rows={row}):
                    break
                if _column_has_body_content(work, j, header_end, skip_rows={row}):
                    break
                # 标签物理框未盖住邻格 → 非欠合并，保留
                if not _label_polygon_covers_neighbor(owner, nxt):
                    break
                absorbed.append(nxt)
                end = int(nxt["col_end"])
                j = end + 1
            if end > int(owner["col_end"]):
                owner["col_end"] = end
                owner["col_span"] = end - int(owner["col_start"]) + 1
                for a in absorbed:
                    a["_drop_render"] = True
                for c in range(int(owner["col_start"]), end + 1):
                    occupancy[(row, c)] = owner
            col = int(owner["col_end"]) + 1
    return [c for c in work if not c.get("_drop_render")]


def _merge_leading_label_gaps(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """行头区右侧覆盖：逻辑补洞 + 物理欠合并吸收（见 _repair_row_header_coverage）。"""
    return _repair_row_header_coverage(cells)


_KNOWN_LEAF_SUBHEADER_RE = re.compile(
    r"^(种类|種類|类型|類型|名称|名稱|组分|組分|品种|品種|项目|序号|牌号|"
    r"量|添加量|含量|用量|摩尔|质量|份数)"
    r"(?:[\(（].*)?$"
)


def _is_cell_frag(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    compact = "".join(t.split())
    # 单字母等级代号是表体数据，不是空角碎片
    if _LETTER_DATA_RE.fullmatch(compact):
        return False
    if _is_known_leaf_subheader(t):
        return False
    if len(compact) <= 2:
        # 两汉字表体/表头词（溶解/不溶等）
        if re.fullmatch(r"[\u4e00-\u9fff]{2}", compact):
            return False
        return True
    return False


def _is_known_leaf_subheader(text: str) -> bool:
    """种类 / 量(g) / 含量 等短子表头，不可当碎片被同宽模板盖掉。"""
    compact = "".join((text or "").split())
    if not compact:
        return False
    if _KNOWN_LEAF_SUBHEADER_RE.match(compact):
        return True
    return bool(re.search(r"(pKa|保护率|反应率)", compact, re.I))


def _is_leaf_type_amount_subheader(text: str) -> bool:
    """叶子「种类 / 量 / 种类(量)」类子表头（可多行，如 种类\\n(量)）。"""
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    if not lines:
        return False
    joined = "".join("".join(ln.split()) for ln in lines)
    if not re.search(r"(种类|種類|类型|類型|名称|名稱|量|含量|用量|添加量)", joined):
        return False
    for ln in lines:
        compact = "".join(ln.split())
        if _KNOWN_LEAF_SUBHEADER_RE.match(compact):
            continue
        # 单独一行的 (量) / (g) 等括号量注
        if re.fullmatch(r"[\(（][^）\)]{0,12}[\)）]", compact):
            continue
        return False
    return True


def _is_lettered_parent_over_leaf_subheader(upper: str, lower: str) -> bool:
    """(B)光酸产生剂 与下方 种类(量) → 应保持上下两格，不是同一短语折行。"""
    if not _lettered_parent_letter(upper):
        return False
    return _is_leaf_type_amount_subheader(lower)


def _merge_text_into_cell(cell: Dict[str, Any], text: str) -> None:
    add = (text or "").strip()
    if not add:
        return
    prev = str(cell.get("text") or "").strip()
    if add not in prev:
        cell["text"] = (prev + " " + add).strip() if prev else add


def _salvage_dropped_cell_text(
    out: List[Dict[str, Any]],
    *,
    row: int,
    old_start: int,
    text: str,
    kept: List[int],
    remap: Dict[int, int],
    axis: str,
) -> None:
    """把整段逻辑范围都被删掉的 cell 文本并入最近保留邻格。"""
    add = (text or "").strip()
    if not add or not out or not kept:
        return
    neighbor = next((k for k in reversed(kept) if k < old_start), None)
    if neighbor is None:
        logger.warning("删%s后左侧无邻格，文本已丢: %r", axis, add[:40])
        return
    target = remap[neighbor]
    for oc in out:
        if axis == "col":
            if int(oc["row_start"]) <= row <= int(oc["row_end"]) and int(
                oc["col_start"]
            ) <= target <= int(oc["col_end"]):
                _merge_text_into_cell(oc, add)
                return
        else:
            if int(oc["col_start"]) <= row <= int(oc["col_end"]) and int(
                oc["row_start"]
            ) <= target <= int(oc["row_end"]):
                _merge_text_into_cell(oc, add)
                return
    logger.warning("删%s后无法并入邻格，文本已丢: %r", axis, add[:40])


def drop_evidenceless_columns(cells: List[Dict[str, Any]], *, narrow_ratio: float = 0.35) -> List[Dict[str, Any]]:
    """
    丢弃“证据不足”的幽灵列（零文本、且几何上非常窄、且不参与跨列 colspan）。
    窄列若仅有短碎片文本、且不参与 colspan，同样丢弃。

    非空 cell 的 col_start（含纯 colspan 原点）一律保留，避免 P98 合成例 15/16
    那种「表头+表体都是 colspan、没有原子格」的列把字连格删掉。

    注意：此实现无法直接判定“无框线/无墨迹”（html_formatter 不接收原图），
    因此使用可观测代理：窄宽度 + 不存在跨列占用 + 覆盖该列的文本为空/碎片。
    """
    if not cells:
        return cells

    # 物理宽度统计：用所有 col_span==1 的 cell 估计“标准列宽”
    atomic_widths: List[float] = []
    for c in cells:
        if int(c.get("col_span") or 1) == 1:
            x1, _y1, x2, _y2 = _cell_physical_bbox_x_y(c)
            atomic_widths.append(max(0.0, x2 - x1))
    if not atomic_widths:
        return cells
    median_w = float(np.median(atomic_widths))
    if median_w <= 1e-6:
        return cells

    max_col = max(int(c["col_end"]) for c in cells)
    header_end = _effective_header_end(cells)
    dropped: Set[int] = set()

    # 先统计每列被哪些 cell 覆盖，以及这些 cell 是否带文本/跨列
    col_to_cells: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(max_col + 1)}
    for c in cells:
        cs, ce = int(c["col_start"]), int(c["col_end"])
        for col in range(cs, ce + 1):
            col_to_cells[col].append(c)

    for col, covered_cells in col_to_cells.items():
        if not covered_cells:
            continue
        texts = [str(c.get("text") or "").strip() for c in covered_cells]
        any_text = any(texts)
        # 整列多为行序号（如 16–30）：窄但仍是真实列，保留
        if is_index_column(texts):
            continue
        # 表头锚点列：有非空标签即保留（含 m/X/s/t 等短化学符号列头）
        if any(
            int(c["col_start"]) == col
            and int(c["row_start"]) <= header_end
            and str(c.get("text") or "").strip()
            for c in covered_cells
        ):
            continue
        # colspan 原点列：格子从这里起步且有实质文本，不能当幽灵列删
        if any(
            int(c["col_start"]) == col
            and not _is_cell_frag(str(c.get("text") or ""))
            for c in covered_cells
        ):
            continue
        only_frag = (not any_text) or all(_is_cell_frag(t) for t in texts if t) or (
            any_text and all(_is_cell_frag(t) for t in texts)
        )
        spans_cross = any(int(c.get("col_span") or 1) > 1 for c in covered_cells)
        if spans_cross:
            atomic_texts = [
                str(c.get("text") or "").strip()
                for c in covered_cells
                if int(c.get("col_span") or 1) == 1
            ]
            if any(not _is_cell_frag(t) for t in atomic_texts):
                continue
            if not atomic_texts:
                dropped.add(col)
                continue

        # 该列的窄宽度判断：取其原子 cell 的宽度中位数
        widths: List[float] = []
        for c in covered_cells:
            if int(c.get("col_span") or 1) != 1:
                continue
            x1, _y1, x2, _y2 = _cell_physical_bbox_x_y(c)
            widths.append(max(0.0, x2 - x1))
        if not widths:
            # 无原子格宽度时，若整列只有碎片文本且无跨列，仍丢弃
            if only_frag and not spans_cross:
                dropped.add(col)
            continue
        narrow = float(np.median(widths)) < narrow_ratio * median_w
        # 窄列碎片，或非窄但整列仅孤立数字/单字且多数格为空
        empty_n = sum(1 for t in texts if not t)
        
        # 【修改】增加整列全空的绝对判断
        is_completely_empty = (empty_n == len(texts))
        
        frag_only_sparse = only_frag and empty_n >= max(1, len(texts) // 2)
        
        # 【修改】将 is_completely_empty 加入删除条件
        if is_completely_empty or (narrow and only_frag) or (frag_only_sparse and not spans_cross):
            # 碎片文本并入左侧邻列（若存在）
            if any_text and col > 0:
                frag_bits = [t for t in texts if t]
                for c in cells:
                    if int(c["col_start"]) <= col - 1 <= int(c["col_end"]) and int(
                        c.get("row_start", 0)
                    ) == min(int(x.get("row_start", 0)) for x in covered_cells):
                        _merge_text_into_cell(c, "".join(frag_bits))
                        break
            dropped.add(col)

    if not dropped:
        return cells

    kept_cols = [i for i in range(max_col + 1) if i not in dropped]
    if not kept_cols:
        return cells

    remap = {old: new for new, old in enumerate(kept_cols)}
    out: List[Dict[str, Any]] = []
    orphans: List[Tuple[int, int, str]] = []
    for c in cells:
        cs, ce = int(c["col_start"]), int(c["col_end"])
        new_idxs = [remap[i] for i in range(cs, ce + 1) if i in remap]
        if not new_idxs:
            text = str(c.get("text") or "").strip()
            if text and not _is_cell_frag(text):
                orphans.append((int(c["row_start"]), cs, text))
            continue
        nc = dict(c)
        nc["col_start"] = min(new_idxs)
        nc["col_end"] = max(new_idxs)
        nc["col_span"] = nc["col_end"] - nc["col_start"] + 1
        out.append(nc)
    for row, old_cs, text in orphans:
        _salvage_dropped_cell_text(
            out,
            row=row,
            old_start=old_cs,
            text=text,
            kept=kept_cols,
            remap=remap,
            axis="col",
        )
    return out


def drop_evidenceless_rows(cells: List[Dict[str, Any]], *, short_ratio: float = 0.35) -> List[Dict[str, Any]]:
    """与 `drop_evidenceless_columns` 对称：窄行/零文本/不参与跨行 rowspan。"""
    if not cells:
        return cells

    atomic_heights: List[float] = []
    for c in cells:
        if int(c.get("row_span") or 1) == 1:
            _x1, y1, _x2, y2 = _cell_physical_bbox_x_y(c)
            atomic_heights.append(max(0.0, y2 - y1))
    if not atomic_heights:
        return cells
    median_h = float(np.median(atomic_heights))
    if median_h <= 1e-6:
        return cells

    max_row = max(int(c["row_end"]) for c in cells)
    dropped: Set[int] = set()

    row_to_cells: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(max_row + 1)}
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        for row in range(rs, re + 1):
            row_to_cells[row].append(c)

    for row, covered_cells in row_to_cells.items():
        if not covered_cells:
            continue
        any_text = any(str(c.get("text") or "").strip() for c in covered_cells)
        if any_text:
            continue

        heights: List[float] = []
        for c in covered_cells:
            if int(c.get("row_span") or 1) != 1:
                continue
            _x1, y1, _x2, y2 = _cell_physical_bbox_x_y(c)
            heights.append(max(0.0, y2 - y1))
        if not heights:
            continue
        if float(np.median(heights)) < short_ratio * median_h:
            dropped.add(row)

    if not dropped:
        return cells

    kept_rows = [i for i in range(max_row + 1) if i not in dropped]
    if not kept_rows:
        return cells

    remap = {old: new for new, old in enumerate(kept_rows)}
    out: List[Dict[str, Any]] = []
    orphans: List[Tuple[int, int, str]] = []
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        new_idxs = [remap[i] for i in range(rs, re + 1) if i in remap]
        if not new_idxs:
            text = str(c.get("text") or "").strip()
            if text and not _is_cell_frag(text):
                orphans.append((int(c["col_start"]), rs, text))
            continue
        nc = dict(c)
        nc["row_start"] = min(new_idxs)
        nc["row_end"] = max(new_idxs)
        nc["row_span"] = nc["row_end"] - nc["row_start"] + 1
        out.append(nc)
    for col, old_rs, text in orphans:
        _salvage_dropped_cell_text(
            out,
            row=col,
            old_start=old_rs,
            text=text,
            kept=kept_rows,
            remap=remap,
            axis="row",
        )
    return out


def _column_has_header_label(
    cells: Sequence[Dict[str, Any]],
    col: int,
    header_end: int,
) -> bool:
    """该列在表头带是否有实质描述性标签（非单位、非碎片）。"""
    for c in cells:
        if int(c["row_start"]) > header_end:
            continue
        if int(c["col_start"]) != col:
            continue
        t = str(c.get("text") or "").strip()
        if not t or _is_cell_frag(t) or _is_header_unit_line(t):
            continue
        return True
    return False


def _migrate_orphan_header_to_body_column(
    cells: List[Dict[str, Any]],
    col: int,
    *,
    header_end: int,
    max_col: int,
    col_to_cells: Dict[int, List[Dict[str, Any]]],
    dropped: Set[int],
    look_ahead: int = 2,
) -> bool:
    """把无表体列上的表头迁到右侧近邻「有表体、无表头」列。成功返回 True。

    仅用于简单指标表（如 P29 曝光量/密合强度）。左侧有大 rowspan 行头的
    项目表（P24/P25）一律不迁，避免拆坏 stub。
    """
    # 复杂左侧 stub：有偏左大 rowspan → 不做孤列迁移
    if any(
        int(c["row_span"] if c.get("row_span") is not None else int(c["row_end"]) - int(c["row_start"]) + 1) >= 3
        and int(c["col_start"]) <= 1
        for c in cells
    ):
        return False

    # 仅迁移「指标类」表头，避免「项目」等 stub 角标被挪走
    orphan_texts = [
        str(c.get("text") or "").strip()
        for c in col_to_cells.get(col, [])
        if int(c["col_start"]) == col
        and int(c["row_start"]) <= header_end
        and str(c.get("text") or "").strip()
    ]
    if not orphan_texts or not any(_METRIC_ORPHAN_HEADER_RE.search(t) for t in orphan_texts):
        return False

    target: Optional[int] = None
    for right in range(col + 1, min(col + 1 + look_ahead, max_col + 1)):
        if right in dropped:
            continue
        if _column_has_header_label(cells, right, header_end):
            break
        if not (
            _column_has_data_values(cells, right)
            or _column_has_body_content(cells, right, header_end)
        ):
            continue
        target = right
        break
    if target is None:
        return False

    moved = False
    for c in list(col_to_cells.get(col, [])):
        if int(c["col_start"]) != col:
            continue
        if int(c["row_start"]) > header_end:
            continue
        t = str(c.get("text") or "").strip()
        if not t or _is_cell_frag(t):
            continue
        dest = None
        for rc in col_to_cells.get(target, []):
            if int(rc["col_start"]) != target:
                continue
            if int(rc["row_start"]) == int(c["row_start"]):
                dest = rc
                break
        if dest is not None:
            prev = str(dest.get("text") or "").strip()
            dest["text"] = (prev + "\n" + t) if prev else t
            c["text"] = ""
        else:
            c["col_start"] = target
            c["col_end"] = target
            c["col_span"] = 1
            col_to_cells.setdefault(target, []).append(c)
        moved = True
    if moved:
        logger.info("孤列表头迁入数据列: %d → %d", col, target)
    return moved


_METRIC_ORPHAN_HEADER_RE = re.compile(
    r"(曝光|强度|温度|分子量|分辨率|应力|耐热|粘度|透过|折射|模量|硬度|密合|粘合|"
    r"Eth|mJ|MPa|℃|°C)"
)


def _shift_logic_columns_from(
    cells: List[Dict[str, Any]],
    from_col: int,
    delta: int = 1,
) -> None:
    """将 col_start >= from_col 的格子右移；跨越插入点的 colspan 向右扩张。"""
    for c in cells:
        if c.get("_drop_render"):
            continue
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if cs >= from_col:
            c["col_start"] = cs + delta
            c["col_end"] = ce + delta
        elif ce >= from_col:
            c["col_end"] = ce + delta
        c["col_span"] = int(c["col_end"]) - int(c["col_start"]) + 1


def _looks_like_recipe_code_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    head = t.split("\n", 1)[0].strip()
    if _RECIPE_CODE_BODY_RE.match(head):
        return True
    # PI-1(11.6) / Bk-S0100CF(34.8)
    if re.match(r"^[A-Za-z]{1,8}[-－.]?[A-Za-z0-9:]*\s*\(", head):
        return True
    return False


def _peel_glued_example_index_columns(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    列一致性：标签列与右邻的串位 / 组合物号粘连纠正。

    A) 右邻是序号列（多数为 1–3 位数字）：
       - 实施例 32 32 | ∅ → 实施例 32 | 32
       - 实施例 2 3 | 23 → 实施例 23 | 23
    B) 右邻是配方码（Bk-*）或空、且标签粘连 local+组合物号：
       - 比较例186 | Bk-1 → 比较例 1 | 86 | Bk-1（必要时插入序号列）
       - 比较例 3 88 | Bk-2 → 比较例 3 | 88 | Bk-2
    """
    if not cells:
        return cells
    work = list(cells)
    header_end = _effective_header_end(work)
    max_col = max(int(c["col_end"]) for c in work)

    by_rc: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for c in work:
        if c.get("_drop_render"):
            continue
        rs, re_ = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if rs != re_ or cs != ce:
            continue
        if rs <= header_end:
            continue
        by_rc[(rs, cs)] = c

    if not by_rc:
        return work

    body_rows = sorted({r for r, _ in by_rc})
    changed = 0

    # ----- Path A: 右邻序号列（原逻辑）-----
    for label_col in range(max_col + 1):
        index_col = label_col + 1
        clean = 0
        candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        label_bearing = 0
        for row in body_rows:
            left = by_rc.get((row, label_col))
            right = by_rc.get((row, index_col))
            if left is None or right is None:
                continue
            lt = str(left.get("text") or "").strip()
            rt = str(right.get("text") or "").strip()
            if not lt:
                continue
            label_bearing += 1
            parsed = parse_clean_example_label(lt)
            if parsed and rt == parsed[1]:
                clean += 1
                continue
            if repair_glued_example_label_index(lt, rt) is not None:
                candidates.append((left, right))
        if clean < 2 or not candidates:
            continue
        if clean < len(candidates):
            continue
        if label_bearing and clean < max(2, (label_bearing + 1) // 2):
            continue
        for left, right in candidates:
            lt = str(left.get("text") or "").strip()
            rt = str(right.get("text") or "").strip()
            fixed = repair_glued_example_label_index(lt, rt)
            if fixed is None:
                continue
            new_lab, new_idx = fixed
            if new_lab == lt and new_idx == rt:
                continue
            left["text"] = new_lab
            right["text"] = new_idx
            changed += 1
            logger.info(
                "标签/序号列串位纠正: %r|%r → %r|%r",
                lt,
                rt,
                new_lab,
                new_idx,
            )

    # ----- Path B: 标签粘连组合物号，右邻配方码/空 -----
    # 重建 by_rc（Path A 可能改过文本）
    by_rc = {}
    for c in work:
        if c.get("_drop_render"):
            continue
        rs, re_ = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if rs != re_ or cs != ce or rs <= header_end:
            continue
        by_rc[(rs, cs)] = c
    body_rows = sorted({r for r, _ in by_rc})
    max_col = max(int(c["col_end"]) for c in work) if work else 0

    for label_col in range(max_col + 1):
        right_col = label_col + 1
        peel_rows: List[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]], str, str]] = []
        label_bearing = 0
        right_texts: List[str] = []
        recipe_hits = 0
        for row in body_rows:
            left = by_rc.get((row, label_col))
            if left is None:
                continue
            lt = str(left.get("text") or "").strip()
            if not lt or not _BODY_INDEX_LABEL_RE.search(lt):
                continue
            label_bearing += 1
            right = by_rc.get((row, right_col))
            rt = str(right.get("text") or "").strip() if right else ""
            if rt:
                right_texts.append(rt)
                if _looks_like_recipe_code_text(rt):
                    recipe_hits += 1
            peeled = split_example_local_and_composition_id(lt)
            if peeled is None:
                continue
            # 右邻已是该组合物号 → 只需 scrub 标签
            clean_lab, comp_id = peeled
            peel_rows.append((row, left, right, clean_lab, comp_id))

        if label_bearing < 2 or len(peel_rows) < 2:
            continue
        # 门控：可剥离行占标签行一半以上；右邻不是已有序号列
        if len(peel_rows) < max(2, (label_bearing + 1) // 2):
            continue
        if is_index_column(right_texts, min_nonempty=2, min_ratio=0.6):
            continue
        # 右邻多数为空，或多数为配方码，才插入/填序号
        nonempty_right = len(right_texts)
        empty_right = sum(1 for _r, _l, right, _a, _b in peel_rows if right is None or not str(right.get("text") or "").strip())
        if nonempty_right and recipe_hits < max(1, (nonempty_right + 1) // 2) and empty_right < len(peel_rows) // 2:
            # 右邻有字但既非序号也非配方 —— 只 scrub 标签，不挪号
            for _row, left, _right, clean_lab, _comp in peel_rows:
                if str(left.get("text") or "").strip() != clean_lab:
                    left["text"] = clean_lab
                    changed += 1
            continue

        # 右邻空：把组合物号写入右邻；否则在标签与配方之间插入一列
        need_insert = empty_right < len(peel_rows) // 2 and recipe_hits >= max(
            1, (nonempty_right + 1) // 2
        )
        index_col = right_col
        if need_insert:
            _shift_logic_columns_from(work, right_col, 1)
            # 表头补「组合物」占位（与 P117 同类表对齐）
            header_cell = {
                "row_start": 0,
                "row_end": max(0, header_end),
                "col_start": index_col,
                "col_end": index_col,
                "row_span": max(1, header_end + 1),
                "col_span": 1,
                "text": "组合物",
                "polygon": [
                    [index_col * 10, 0],
                    [index_col * 10 + 8, 0],
                    [index_col * 10 + 8, 8],
                    [index_col * 10, 8],
                ],
                "texts": [],
            }
            work.append(header_cell)
            by_rc = {}
            for c in work:
                if c.get("_drop_render"):
                    continue
                rs, re_ = int(c["row_start"]), int(c["row_end"])
                cs, ce = int(c["col_start"]), int(c["col_end"])
                if rs != re_ or cs != ce or rs <= header_end:
                    continue
                by_rc[(rs, cs)] = c
            logger.info(
                "标签粘连组合物号：在列 %d 插入序号列（%d 行）",
                index_col,
                len(peel_rows),
            )

        for row, left, right, clean_lab, comp_id in peel_rows:
            left["text"] = clean_lab
            if need_insert:
                # 右邻配方码已随列平移到 index_col+1
                existing = by_rc.get((row, index_col))
                if existing is None:
                    nc = {
                        "row_start": row,
                        "row_end": row,
                        "col_start": index_col,
                        "col_end": index_col,
                        "row_span": 1,
                        "col_span": 1,
                        "text": comp_id,
                        "polygon": [
                            [index_col * 10, row * 10],
                            [index_col * 10 + 8, row * 10],
                            [index_col * 10 + 8, row * 10 + 8],
                            [index_col * 10, row * 10 + 8],
                        ],
                        "texts": [],
                    }
                    work.append(nc)
                    by_rc[(row, index_col)] = nc
                else:
                    existing["text"] = comp_id
            else:
                # 填空右邻
                if right is None:
                    nc = {
                        "row_start": row,
                        "row_end": row,
                        "col_start": index_col,
                        "col_end": index_col,
                        "row_span": 1,
                        "col_span": 1,
                        "text": comp_id,
                        "polygon": [
                            [index_col * 10, row * 10],
                            [index_col * 10 + 8, row * 10],
                            [index_col * 10 + 8, row * 10 + 8],
                            [index_col * 10, row * 10 + 8],
                        ],
                        "texts": [],
                    }
                    work.append(nc)
                elif not str(right.get("text") or "").strip():
                    right["text"] = comp_id
                else:
                    # 右邻有配方码但未走 insert（不应到此）——只 scrub
                    pass
            changed += 1
            logger.info(
                "标签组合物号剥离: → %r | %r",
                clean_lab,
                comp_id,
            )
        # 一列处理完即停，避免连锁插入
        if peel_rows:
            break

    if changed:
        logger.info("标签/序号列纠正合计: %d 处", changed)
    return work


def _restore_composition_parent_header(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    若「组成[质量份]」被拼进左侧「颜料分散液」格，而右侧存在空的宽 colspan 父格，
    则把组成文案迁回空父格，左侧只留颜料分散液。
    """
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    if header_end < 0:
        return cells

    donors: List[Dict[str, Any]] = []
    empty_parents: List[Dict[str, Any]] = []
    for c in cells:
        if c.get("_drop_render"):
            continue
        if int(c["row_start"]) > header_end:
            continue
        t = str(c.get("text") or "").strip()
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if (
            _PIGMENT_DISPERSION_HEADER_RE.search(t)
            and _COMPOSITION_SPAN_HEADER_RE.search(t)
        ):
            donors.append(c)
        if (
            not t
            and ce > cs
            and ce - cs >= 2
            and int(c["row_start"]) <= int(c["row_end"])
        ):
            empty_parents.append(c)

    if not donors or not empty_parents:
        return cells

    for donor in donors:
        dcs, dce = int(donor["col_start"]), int(donor["col_end"])
        # 选紧邻右侧、列范围在 donor 以右的空父格
        candidates = [
            p
            for p in empty_parents
            if int(p["col_start"]) >= dce
            or (
                int(p["col_start"]) > dcs
                and int(p["col_end"]) > dce
            )
        ]
        if not candidates:
            candidates = [
                p
                for p in empty_parents
                if int(p["col_start"]) > dcs
            ]
        if not candidates:
            continue
        parent = min(
            candidates,
            key=lambda p: (int(p["col_start"]), -_logic_area(p)),
        )
        raw = str(donor.get("text") or "")
        # 拆出组成片段
        m = _COMPOSITION_SPAN_HEADER_RE.search(raw)
        if not m:
            continue
        # 组成通常从 match 起直到串尾或换行块
        comp_text = raw[m.start() :].strip()
        left_text = raw[: m.start()].strip()
        # 清理左侧重复的「组成」残留空白
        left_text = re.sub(r"\s+", " ", left_text).strip()
        if not left_text:
            left_text = "颜料分散液"
        if not parent.get("text"):
            parent["text"] = comp_text if "质量" in comp_text else "组成[质量份]"
            donor["text"] = left_text
            logger.info(
                "组成父表头归位: 左=%r 父=%r",
                left_text[:20],
                str(parent.get("text"))[:20],
            )
    return cells


def _unglue_composition_side_and_span(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """拆开表头 OCR 粘连的「组合物」+「组成[质量份]」。

    常见形态：一格 ``组合物组成[``，右邻 ``质量份]…`` → 侧栏「组合物」+
    「组成[质量份]」，右邻剩余行保留。
    """
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if header_end < 0:
        return cells
    work = [dict(c) for c in cells]
    changed = 0
    for cell in work:
        if cell.get("_drop_render"):
            continue
        if int(cell["row_start"]) > header_end:
            continue
        raw = str(cell.get("text") or "").strip()
        compact = "".join(raw.split())
        if not _GLUED_COMP_PARENT_SPAN_RE.match(compact):
            continue
        # 整段已在一格：组合物组成[质量份]
        if _COMPOSITION_SPAN_HEADER_RE.search(compact):
            cs, ce = int(cell["col_start"]), int(cell["col_end"])
            if ce > cs:
                # 左列留侧栏，右段作组成父头
                mid = cs
                left = dict(cell)
                left["col_end"] = mid
                left["col_span"] = mid - cs + 1
                left["text"] = "组合物"
                _refresh_cell_spans(left)
                cell["col_start"] = mid + 1
                cell["col_span"] = ce - (mid + 1) + 1
                cell["text"] = "组成[质量份]"
                _refresh_cell_spans(cell)
                work.append(left)
                changed += 1
            else:
                cell["text"] = "组合物"
                changed += 1
            continue
        # 组合物组成[ + 右邻质量份]
        cell["text"] = "组合物"
        changed += 1
        rs = int(cell["row_start"])
        ce = int(cell["col_end"])
        for nb in work:
            if nb is cell or nb.get("_drop_render"):
                continue
            if int(nb["row_start"]) != rs:
                continue
            if int(nb["col_start"]) > ce + 1:
                continue
            if int(nb["col_end"]) < ce:
                continue
            nlines = [ln.strip() for ln in str(nb.get("text") or "").split("\n") if ln.strip()]
            if not nlines:
                continue
            if not _MASS_PARTS_LEADING_RE.match("".join(nlines[0].split())):
                continue
            rest = nlines[1:]
            nb["text"] = "组成[质量份]" + (("\n" + "\n".join(rest)) if rest else "")
            logger.info(
                "组合物/组成粘连拆分: → 组合物 | %r",
                nb["text"].split("\n")[0],
            )
            break
    if changed:
        logger.info("组合物/组成粘连拆分: %d 处", changed)
    return [c for c in work if not c.get("_drop_render")]


def _merge_split_composition_span_fragments(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """合并被 OCR 切开的「组成[」+「质量份]」相邻表头格。"""
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if header_end < 0:
        return cells
    work = [dict(c) for c in cells]
    changed = 0
    for cell in work:
        if cell.get("_drop_render"):
            continue
        if int(cell["row_start"]) > header_end:
            continue
        t = "".join(str(cell.get("text") or "").split())
        # 「组成[」或缺「质量」的残段
        if not (
            re.match(r"^组成[\[［]?$", t)
            or (t.startswith("组成") and "质量" not in t and ("[" in t or "［" in t))
        ):
            continue
        rs = int(cell["row_start"])
        ce = int(cell["col_end"])
        for nb in work:
            if nb is cell or nb.get("_drop_render"):
                continue
            if int(nb["row_start"]) != rs:
                continue
            if int(nb["col_start"]) != ce + 1:
                continue
            nt = "".join(str(nb.get("text") or "").split())
            if not (nt.startswith("质量份") or re.match(r"^质量\s*份", str(nb.get("text") or "").strip())):
                continue
            cell["text"] = "组成[质量份]"
            cell["col_end"] = int(nb["col_end"])
            _refresh_cell_spans(cell)
            nb["_drop_render"] = True
            changed += 1
            logger.info(
                "组成[质量份]碎片合并: c%d-%d",
                int(cell["col_start"]),
                int(cell["col_end"]),
            )
            break
    if changed:
        return [c for c in work if not c.get("_drop_render")]
    return work


def _extend_composition_parent_colspan(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """「组合物」/「组成[质量份]」父头应覆盖 (A)(B)/(D)(E) 子列。

    侧栏竖排「组合物」(rowspan>1 且 colspan=1) 不是顶栏父头，禁止横向扩展。
    """
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if header_end < 1:
        return cells
    work = [dict(c) for c in cells]
    top_row = min(int(c["row_start"]) for c in work if not c.get("_drop_render"))
    for parent in work:
        if parent.get("_drop_render"):
            continue
        if int(parent["row_start"]) > header_end:
            continue
        t = str(parent.get("text") or "").strip()
        compact = "".join(t.split())
        is_span = bool(_COMPOSITION_SPAN_HEADER_RE.search(t))
        is_parent = bool(_COMPOSITION_PARENT_HDR_RE.match(compact))
        if not t or not (is_span or is_parent):
            continue
        pr = int(parent["row_start"])
        pc_s, pc_e = int(parent["col_start"]), int(parent["col_end"])
        # 侧栏标签：跨表头行、单列 → 勿当顶栏父头扩展
        if is_parent and not is_span and int(parent["row_end"]) > pr and pc_e == pc_s:
            continue
        # 剥开误粘字母叶子（组成[质量份] + （D）着色剂）
        if is_span:
            lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
            if len(lines) >= 2 and _lettered_parent_letter(lines[1]):
                parent["text"] = lines[0]
                t = parent["text"]
                parent["row_end"] = pr
                _refresh_cell_spans(parent)
        letter_cols: List[int] = []
        search_rows = {pr, pr + 1, pr - 1}
        search_rows = {r for r in search_rows if 0 <= r <= header_end}
        for c in work:
            if c.get("_drop_render") or c is parent:
                continue
            cr = int(c["row_start"])
            if cr not in search_rows:
                continue
            col = int(c["col_start"])
            if col < pc_s - 4 or col > pc_e + 4:
                continue
            if _lettered_parent_letter(str(c.get("text") or "")):
                letter_cols.append(col)
        # 组成[质量份] 顶行右侧空邻格纳入覆盖
        if is_span and int(parent["row_start"]) == pr == int(parent["row_end"]):
            for c in work:
                if c.get("_drop_render") or c is parent:
                    continue
                if int(c["row_start"]) != pr or int(c["row_end"]) != pr:
                    continue
                if str(c.get("text") or "").strip():
                    continue
                cs, ce = int(c["col_start"]), int(c["col_end"])
                if cs == pc_e + 1 or (cs <= pc_e + 1 and ce >= pc_e + 1):
                    letter_cols.extend(range(cs, ce + 1))
        letter_cols = sorted(set(letter_cols))
        if not letter_cols:
            continue
        if not is_span and len(letter_cols) < 2:
            continue
        # 组合物：字母列已由「组成[质量份]」覆盖 → 勿抢占
        if is_parent and not is_span:
            covered_by_compose = False
            for c in work:
                if c.get("_drop_render") or c is parent:
                    continue
                if int(c["row_start"]) > header_end:
                    continue
                ct = "".join(str(c.get("text") or "").split())
                if not _COMPOSITION_SPAN_HEADER_RE.search(ct):
                    continue
                ccs, cce = int(c["col_start"]), int(c["col_end"])
                if letter_cols[0] >= ccs and letter_cols[-1] <= cce:
                    covered_by_compose = True
                    break
            if covered_by_compose:
                continue
        new_cs = min(pc_s, letter_cols[0])
        new_ce = max(pc_e, letter_cols[-1])
        if new_cs == pc_s and new_ce == pc_e and (
            not is_span or int(parent["row_end"]) == pr
        ):
            if int(parent["row_start"]) <= top_row:
                continue
        # 勿扩进同行其它非空顶栏父头所占列
        blocked = False
        for c in work:
            if c.get("_drop_render") or c is parent:
                continue
            if int(c["row_start"]) != pr:
                continue
            ot = str(c.get("text") or "").strip()
            if not ot:
                continue
            ocs, oce = int(c["col_start"]), int(c["col_end"])
            if oce < new_cs or ocs > new_ce:
                continue
            if ocs > pc_e or oce < pc_s:
                blocked = True
                break
        if blocked:
            continue
        changed = False
        if new_cs != pc_s or new_ce != pc_e:
            parent["col_start"] = new_cs
            parent["col_end"] = new_ce
            changed = True
        if int(parent["row_start"]) > top_row:
            parent["row_start"] = top_row
            changed = True
        if is_span and int(parent["row_end"]) > pr:
            parent["row_end"] = pr
            changed = True
        if changed:
            _refresh_cell_spans(parent)
            for c in work:
                if c.get("_drop_render") or c is parent:
                    continue
                if int(c["row_start"]) != int(parent["row_start"]):
                    continue
                if str(c.get("text") or "").strip():
                    continue
                cs, ce = int(c["col_start"]), int(c["col_end"])
                if cs >= int(parent["col_start"]) and ce <= int(parent["col_end"]):
                    c["_drop_render"] = True
            logger.info(
                "组成/组合物父头扩展: r%d c%d-%d",
                int(parent["row_start"]),
                int(parent["col_start"]),
                int(parent["col_end"]),
            )
    return work

def _ensure_composition_parent_header(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """表头带存在 (A)(B)(C) 字母父头但缺「组合物」顶栏时补建（P42 同类）。"""
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if header_end < 1:
        return cells
    work = [dict(c) for c in cells]
    top_row = min(int(c["row_start"]) for c in work if not c.get("_drop_render"))

    for c in work:
        if c.get("_drop_render") or int(c["row_start"]) > header_end:
            continue
        if _COMPOSITION_PARENT_HDR_RE.match("".join(str(c.get("text") or "").split())):
            return work

    letter_cells: List[Dict[str, Any]] = []
    for c in work:
        if c.get("_drop_render") or int(c["row_start"]) > header_end:
            continue
        if _lettered_parent_letter(str(c.get("text") or "")) in ("a", "b", "c"):
            letter_cells.append(c)
    if len(letter_cells) < 2:
        return work

    cs = min(int(c["col_start"]) for c in letter_cells)
    ce = max(int(c["col_end"]) for c in letter_cells)

    for c in work:
        if c.get("_drop_render") or int(c["row_start"]) != top_row:
            continue
        if str(c.get("text") or "").strip():
            pcs, pce = int(c["col_start"]), int(c["col_end"])
            if not (pce < cs or pcs > ce):
                return work

    for c in work:
        if c.get("_drop_render"):
            continue
        if int(c["row_start"]) != top_row:
            continue
        if str(c.get("text") or "").strip():
            continue
        pcs, pce = int(c["col_start"]), int(c["col_end"])
        if pcs <= cs and pce >= ce:
            c["text"] = "组合物"
            _refresh_cell_spans(c)
            logger.info("组合物父头补写: r%d c%d-%d", top_row, cs, ce)
            return work

    for c in work:
        if c.get("_drop_render") or int(c["row_start"]) != top_row:
            continue
        if str(c.get("text") or "").strip():
            continue
        pcs, pce = int(c["col_start"]), int(c["col_end"])
        if pce >= cs and pcs <= ce:
            c["_drop_render"] = True

    ncol = ce - cs + 1
    work.append(
        {
            "row_start": top_row,
            "row_end": top_row,
            "col_start": cs,
            "col_end": ce,
            "row_span": 1,
            "col_span": ncol,
            "text": "组合物",
            "texts": [],
            "polygon": [
                [cs * 10, 0],
                [ce * 10 + 8, 0],
                [ce * 10 + 8, 8],
                [cs * 10, 8],
            ],
        }
    )
    logger.info("组合物父头新建: r%d c%d-%d", top_row, cs, ce)
    return work


def _promote_tall_header_rowspans(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """NMP/酸含量等竖表头若上方同列全空，则 row_start 提到顶行（P41 NMP 漏行）。"""
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if header_end < 1:
        return cells
    work = [dict(c) for c in cells]
    top_row = min(int(c["row_start"]) for c in work if not c.get("_drop_render"))

    for c in work:
        if c.get("_drop_render"):
            continue
        rs, re = int(c["row_start"]), int(c["row_end"])
        if rs <= top_row:
            continue
        cs, ce = int(c["col_start"]), int(c["col_end"])
        t = str(c.get("text") or "").strip()
        if not t:
            continue
        compact = "".join(t.split())
        is_nmp = bool(_TALL_METRIC_NMP_RE.search(compact)) and "浓度" in compact
        is_acid = bool(_ACID_CONTENT_HDR_RE.search(compact)) and "通式" not in compact
        if not is_nmp and not is_acid:
            continue
        blocked = False
        for r in range(top_row, rs):
            if _occupied_by_col_row(work, r, cs, ce):
                blocked = True
                break
        if blocked:
            continue
        # 去掉同列上方占位空格，避免与 rowspan 冲突
        for other in work:
            if other is c or other.get("_drop_render"):
                continue
            ors, ore = int(other["row_start"]), int(other["row_end"])
            ocs, oce = int(other["col_start"]), int(other["col_end"])
            if oce < cs or ocs > ce:
                continue
            if ors >= rs:
                continue
            if not str(other.get("text") or "").strip():
                other["_drop_render"] = True
        c["row_start"] = top_row
        c["row_end"] = header_end
        _refresh_cell_spans(c)
        logger.info("竖表头上提: %r r%d→%d", t[:24], rs, top_row)
    return work


def _pull_up_tall_metric_headers(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """NMP 浓度 / 酸的含量 等应竖跨整段表头带；子行前缀或单位行向上并入。"""
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if header_end < 2:
        return cells
    top_row = min(int(c["row_start"]) for c in cells if not c.get("_drop_render"))
    work = [dict(c) for c in cells]
    by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for c in work:
        if c.get("_drop_render"):
            continue
        rs, re = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if rs == re:
            by_key[(rs, cs, ce)] = c

    drop_ids: Set[int] = set()

    def _same_col_range(c: Dict[str, Any], cs: int, ce: int) -> bool:
        return int(c["col_start"]) == cs and int(c["col_end"]) == ce

    for anchor in work:
        if anchor.get("_drop_render") or id(anchor) in drop_ids:
            continue
        t = str(anchor.get("text") or "").strip()
        if not t:
            continue
        cs, ce = int(anchor["col_start"]), int(anchor["col_end"])
        rs, re = int(anchor["row_start"]), int(anchor["row_end"])
        if rs <= top_row:
            continue
        if ce > cs:
            continue
        if _lettered_parent_letter(t) or _is_leaf_type_amount_subheader(t):
            continue
        compact = "".join(t.split())
        if _has_formula_header_token(compact) or _COMPOSITION_PARENT_HDR_RE.match(compact):
            continue
        is_nmp = bool(_TALL_METRIC_NMP_RE.search(compact)) and "浓度" in compact
        is_acid = bool(_ACID_CONTENT_HDR_RE.search(compact)) and "通式" not in compact
        if not is_nmp and not is_acid:
            continue
        # 已是完整 NMP 竖表头（组合物中+NMP）→ 不再 pull
        if is_nmp and "组合物" in compact and "NMP" in compact:
            continue

        merged_text = t
        new_rs = top_row

        # 向上吸收同列前缀/空槽（组合物中的 → NMP…）
        for up_r in range(rs - 1, top_row - 1, -1):
            up = by_key.get((up_r, cs, ce))
            if up is None or id(up) in drop_ids:
                if not _occupied_by_col_row(work, up_r, cs, ce, skip=drop_ids):
                    new_rs = up_r
                    continue
                break
            up_t = str(up.get("text") or "").strip()
            if not up_t:
                drop_ids.add(id(up))
                up["_drop_render"] = True
                new_rs = up_r
                continue
            if _is_header_phrase_continuation(up_t, merged_text):
                merged_text = up_t + "\n" + merged_text if up_t else merged_text
                drop_ids.add(id(up))
                up["_drop_render"] = True
                new_rs = up_r
                continue
            break

        # 向下吸收单位/括号续行
        new_re = re
        for down_r in range(re + 1, header_end + 1):
            down = by_key.get((down_r, cs, ce))
            if down is None:
                break
            if id(down) in drop_ids:
                break
            down_t = str(down.get("text") or "").strip()
            if not down_t:
                drop_ids.add(id(down))
                down["_drop_render"] = True
                new_re = down_r
                continue
            if _is_header_unit_line(down_t) or _is_header_phrase_continuation(
                merged_text, down_t
            ):
                merged_text = merged_text + "\n" + down_t if merged_text else down_t
                drop_ids.add(id(down))
                down["_drop_render"] = True
                new_re = down_r
                continue
            break

        anchor["text"] = merged_text
        anchor["row_start"] = new_rs
        anchor["row_end"] = new_re
        _refresh_cell_spans(anchor)

    return [c for c in work if id(c) not in drop_ids]


def _is_room_storage_over_eval_leaf(upper: str, lower: str) -> bool:
    """「室温放置N…」与「显影时的膜减少量/感度」是中层/叶子，不是折行。"""
    u, l = _compact_hdr(upper), _compact_hdr(lower)
    if not u or not l:
        return False
    if not _ROOM_STORAGE_MID_RE.match(u) and "室温放置" not in u:
        return False
    return bool(
        _EVAL_FILM_LEAF_RE.search(l)
        or _EVAL_SENS_LEAF_RE.match(l)
        or l in {"显影时的", "膜减少量", "感度变"}
    )


def _peel_room_storage_and_leaf(text: str) -> Tuple[str, str]:
    """从误粘的中层格拆出「室温放置N日后」前缀与叶子残文。"""
    raw = (text or "").strip()
    if not raw:
        return "", ""
    compact = _compact_hdr(raw)
    m_full = _ROOM_STORAGE_FULL_RE.match(compact)
    if m_full:
        parent = "室温放置" + re.search(r"\d+", compact).group(0) + "日后"
        rest = compact[m_full.end() :]
        return parent, rest
    m = _ROOM_STORAGE_MID_RE.match(compact)
    if not m:
        return raw, ""
    day = m.group(1)
    rest = compact[m.end() :]
    if rest.startswith("后"):
        rest = rest[1:]
        parent = f"室温放置{day}日后"
    else:
        parent = f"室温放置{day}"
    return parent, rest


def _is_times_shard_text(text: str) -> bool:
    compact = _compact_hdr(text)
    return bool(compact) and bool(_TIMES_SHARD_RE.match(compact))


def _is_times_unit_line(text: str) -> bool:
    compact = _compact_hdr(text)
    if not compact:
        return False
    if _TIMES_UNIT_LINE_RE.match(compact):
        return True
    # 「(×10-6质量%)」整段单位
    if compact[0] in "(（" and "×" in compact and "质量" in compact:
        return True
    if compact[0] in "(（" and "x10" in compact.lower() and "质量" in compact:
        return True
    return False


def _drop_logic_columns(
    cells: List[Dict[str, Any]], cols: Set[int]
) -> List[Dict[str, Any]]:
    """删除指定逻辑列并重映射 col_*。"""
    if not cells or not cols:
        return cells
    max_col = max(int(c["col_end"]) for c in cells)
    kept = [i for i in range(max_col + 1) if i not in cols]
    if not kept:
        return cells
    remap = {old: new for new, old in enumerate(kept)}
    out: List[Dict[str, Any]] = []
    for c in cells:
        if c.get("_drop_render"):
            continue
        cs, ce = int(c["col_start"]), int(c["col_end"])
        new_idxs = [remap[i] for i in range(cs, ce + 1) if i in remap]
        if not new_idxs:
            continue
        nc = dict(c)
        nc["col_start"] = min(new_idxs)
        nc["col_end"] = max(new_idxs)
        _refresh_cell_spans(nc)
        out.append(nc)
    return out


def _repair_room_storage_eval_headers(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """修复「室温放置N」与「日后」横拆、以及中层误吞评价叶子（通用）。"""
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if header_end < 2:
        return cells
    work = [dict(c) for c in cells]
    leaf_row = header_end
    mid_rows = list(range(1, header_end))
    if not mid_rows:
        mid_rows = [1]

    changed = False
    for mid in mid_rows:
        col = 0
        max_col = max(int(c["col_end"]) for c in work)
        while col <= max_col:
            left = _header_cell_origin(work, mid, col)
            if left is None or left.get("_drop_render"):
                col += 1
                continue
            lt = str(left.get("text") or "").strip()
            if not lt or "室温" not in _compact_hdr(lt):
                col += 1
                continue
            if not _ROOM_STORAGE_MID_RE.match(_compact_hdr(lt).split("显影")[0].split("膜")[0]):
                # 宽松：含室温放置即可
                if "室温放置" not in _compact_hdr(lt):
                    col += 1
                    continue

            parent, glued_leaf = _peel_room_storage_and_leaf(lt)
            right = _header_cell_origin(work, mid, col + 1)
            rt = str(right.get("text") or "").strip() if right else ""
            merge_day = bool(right) and bool(_DAY_AFTER_FRAG_RE.match(_compact_hdr(rt)))
            if not parent.startswith("室温放置"):
                col += 1
                continue
            # 已是完整「…日后」且无粘叶、无邻「日后」→ 跳过
            if (
                _ROOM_STORAGE_FULL_RE.match(_compact_hdr(parent))
                and not glued_leaf
                and not merge_day
                and int(left["row_end"]) == mid
            ):
                col += 1
                continue

            if merge_day and not parent.endswith("日后"):
                parent = parent.rstrip("日") + "日后"
            elif not parent.endswith("日后") and "日" in parent:
                # 室温放置5日 → 补「后」若右侧是日后碎片或已 peel
                if merge_day:
                    parent = re.sub(r"日$", "日后", parent)

            # 右侧「日后」并入后丢弃该格
            ce = int(left["col_end"])
            if merge_day and right is not None:
                ce = max(ce, int(right["col_end"]))
                right["_drop_render"] = True
                # 右列叶子「化」→ 感度变化
                frag = _atomic_header_at(work, leaf_row, col + 1)
                if frag is not None:
                    ft = _compact_hdr(str(frag.get("text") or ""))
                    if ft in {"化", "感度", "感度变"} or not ft:
                        frag["text"] = "感度变化"

            left["text"] = parent
            left["row_start"] = mid
            left["row_end"] = mid
            left["col_end"] = ce
            _refresh_cell_spans(left)

            # 粘连的叶子：显影时的膜减少量
            leaf_text = glued_leaf
            if leaf_text:
                if "显影" in leaf_text or "膜" in leaf_text:
                    leaf_text = "显影时的膜减少量"
                slot = _atomic_header_at(work, leaf_row, col)
                if slot is None:
                    bbox = _cell_physical_bbox_x_y(left)
                    slot = _empty_atomic_from_bbox(left, col, col, col, bbox)
                    slot["row_start"] = leaf_row
                    slot["row_end"] = leaf_row
                    work.append(slot)
                if not str(slot.get("text") or "").strip() or _EVAL_FILM_LEAF_RE.search(
                    leaf_text
                ):
                    slot["text"] = "显影时的膜减少量"
            else:
                # 用对称的 0 日叶子模板补全
                slot = _atomic_header_at(work, leaf_row, col)
                if slot is None or not str(slot.get("text") or "").strip():
                    # 找兄弟「室温放置0日后」下的叶子
                    for sib in work:
                        if sib.get("_drop_render"):
                            continue
                        if int(sib["row_start"]) != mid:
                            continue
                        st = _compact_hdr(str(sib.get("text") or ""))
                        if _ROOM_STORAGE_FULL_RE.match(st) and "0" in st:
                            scs = int(sib["col_start"])
                            ref = _atomic_header_at(work, leaf_row, scs)
                            if ref and _EVAL_FILM_LEAF_RE.search(
                                str(ref.get("text") or "")
                            ):
                                if slot is None:
                                    bbox = _cell_physical_bbox_x_y(left)
                                    slot = _empty_atomic_from_bbox(
                                        left, col, col, col, bbox
                                    )
                                    slot["row_start"] = leaf_row
                                    slot["row_end"] = leaf_row
                                    work.append(slot)
                                slot["text"] = str(ref.get("text") or "").strip()
                            break

            changed = True
            col = ce + 1

    if changed:
        work = [c for c in work if not c.get("_drop_render")]
        logger.info("室温放置评价表头修复完成")
    return work


def _fold_acid_times_shard_column(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """酸的含量 旁孤立「×」列 + 单位行 → 并入酸含量竖表头并删幽灵列。"""
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if header_end < 1:
        return cells
    work = [dict(c) for c in cells]
    drop_cols: Set[int] = set()

    acid_cells = []
    for c in work:
        if c.get("_drop_render") or int(c["row_start"]) > header_end:
            continue
        t = str(c.get("text") or "").strip()
        if not t:
            continue
        compact = _compact_hdr(t)
        if _ACID_CONTENT_HDR_RE.search(compact) and "通式" not in compact:
            acid_cells.append(c)

    for acid in acid_cells:
        ac_s = int(acid["col_start"])
        for side_col in (ac_s - 1, int(acid["col_end"]) + 1):
            if side_col < 0 or side_col in drop_cols:
                continue
            side_hdr = [
                c
                for c in work
                if not c.get("_drop_render")
                and int(c["col_start"]) == side_col == int(c["col_end"])
                and int(c["row_start"]) <= header_end
            ]
            side_texts = [
                str(c.get("text") or "").strip()
                for c in side_hdr
                if str(c.get("text") or "").strip()
            ]
            # 允许空列：仅靠跨列单位行证明是 × 碎片列
            spanning_units = [
                c
                for c in work
                if not c.get("_drop_render")
                and int(c["row_start"]) <= header_end
                and int(c["col_start"]) <= side_col <= int(c["col_end"])
                and int(c["col_end"]) >= ac_s
                and int(c["col_start"]) <= int(acid["col_end"])
                and (
                    _is_times_unit_line(str(c.get("text") or ""))
                    or _is_header_unit_line(str(c.get("text") or ""))
                )
            ]
            if side_texts and not all(
                _is_times_shard_text(t) or _is_times_unit_line(t) or _is_header_unit_line(t)
                for t in side_texts
            ):
                continue
            if not side_texts and not spanning_units:
                continue
            # 表体：该列无独立非空原子数据（或仅与酸列共用 colspan）
            body_conflict = False
            for c in work:
                if c.get("_drop_render") or int(c["row_start"]) <= header_end:
                    continue
                cs, ce = int(c["col_start"]), int(c["col_end"])
                if not (cs <= side_col <= ce):
                    continue
                t = str(c.get("text") or "").strip()
                if not t:
                    continue
                if cs == ce == side_col:
                    body_conflict = True
                    break
            if body_conflict:
                continue

            acid_text = str(acid.get("text") or "").strip()
            for t in side_texts:
                if _is_times_shard_text(t):
                    continue  # × 会由单位行体现
                if t and t not in acid_text:
                    acid_text = (acid_text + "\n" + t).strip()
            for u in spanning_units:
                ut = str(u.get("text") or "").strip()
                if ut and ut not in acid_text and "酸" not in _compact_hdr(ut):
                    acid_text = (acid_text + "\n" + ut).strip()
                u["_drop_render"] = True
            acid["text"] = acid_text
            # 酸头竖跨整段表头
            top_row = min(
                int(c["row_start"]) for c in work if not c.get("_drop_render")
            )
            acid["row_start"] = top_row
            acid["row_end"] = header_end
            _refresh_cell_spans(acid)
            for c in side_hdr:
                c["_drop_render"] = True
            drop_cols.add(side_col)
            logger.info("酸含量并入 × 碎片列: col %d → %d", side_col, ac_s)
            break

    if not drop_cols:
        return work
    work = [c for c in work if not c.get("_drop_render")]
    return _drop_logic_columns(work, drop_cols)


def _merge_formula_parent_colspans(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """「通式(5)…|、通式(6)…」在「种类|含量」上 → 合并 colspan=2（通用，补 pipeline/shard 漏网）。"""
    from ..structure.hline_repair import (
        _are_formula_header_complements,
        _is_type_amount_leaf_pair,
    )

    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if header_end < 2:
        return cells
    work = [dict(c) for c in cells]
    max_col = max(int(c["col_end"]) for c in work)
    leaf_row = header_end
    drop_ids: Set[int] = set()

    for row in range(0, header_end):
        col = 0
        while col < max_col:
            left = _header_cell_origin(work, row, col)
            right = _header_cell_origin(work, row, col + 1)
            if left is None or right is None:
                col += 1
                continue
            lt = str(left.get("text") or "").strip()
            rt = str(right.get("text") or "").strip()
            if not lt or not rt or not _are_formula_header_complements(lt, rt):
                col += 1
                continue
            cs, ce = col, col + 1
            b0 = _atomic_header_at(work, leaf_row, cs)
            b1 = _atomic_header_at(work, leaf_row, ce)
            if b0 is None or b1 is None:
                # 叶子可能在 header_end 上一行（中层 ABC 占用一行）
                for lr in range(header_end, max(0, header_end - 2), -1):
                    b0 = b0 or _atomic_header_at(work, lr, cs)
                    b1 = b1 or _atomic_header_at(work, lr, ce)
            bl = str(b0.get("text") or "").strip() if b0 else ""
            br = str(b1.get("text") or "").strip() if b1 else ""
            if not _is_type_amount_leaf_pair([bl, br]):
                col += 1
                continue
            merged = dict(left)
            merged["col_end"] = int(right["col_end"])
            merged["row_end"] = max(int(left["row_end"]), int(right["row_end"]))
            if rt.startswith(("、", "，", ",")):
                merged["text"] = lt + rt
            else:
                merged["text"] = (lt + " " + rt).strip()
            _refresh_cell_spans(merged)
            replaced = False
            for i, c in enumerate(work):
                if id(c) in drop_ids:
                    continue
                if int(c["row_start"]) == row and int(c["col_start"]) == cs:
                    work[i] = merged
                    replaced = True
                    break
            if not replaced:
                work.append(merged)
            drop_ids.add(id(right))
            right["_drop_render"] = True
            col += 2
    out = [c for c in work if id(c) not in drop_ids and not c.get("_drop_render")]
    if len(out) != len(cells):
        logger.info("通式父头 colspan 合并: %d → %d cells", len(cells), len(out))
    return out


def _occupied_by_col_row(
    cells: Sequence[Dict[str, Any]],
    row: int,
    cs: int,
    ce: int,
    *,
    skip: Optional[Set[int]] = None,
) -> bool:
    skip = skip or set()
    for c in cells:
        if id(c) in skip or c.get("_drop_render"):
            continue
        if not (int(c["row_start"]) <= row <= int(c["row_end"])):
            continue
        ocs, oce = int(c["col_start"]), int(c["col_end"])
        if oce < cs or ocs > ce:
            continue
        if str(c.get("text") or "").strip():
            return True
    return False


def _newline_parts(text: str) -> List[str]:
    return [p.strip() for p in str(text or "").split("\n") if p.strip()]


def _stacked_example_body_labels(text: str) -> List[str]:
    """左侧格堆叠的比较例/实施例身行标签（≥2 行）；否则空。"""
    parts = _newline_parts(text)
    if len(parts) < 2:
        return []
    if all(_HEADER_HAS_EXAMPLE_RE.search(p) for p in parts):
        return parts
    return []


_BODYISH_PART_RE = re.compile(
    r"\d|"
    r"\([A-Za-z]\)\s*-|"
    r"^[x×✗ⅹX]$|"
    r"^[△▲Δ▽▼○◯〇◎●]$|"
    r"^[良可可差]$|"
    r"^[A-Da]$",
)


def _parts_look_bodyish(parts: Sequence[str]) -> bool:
    """片段是否像表体数据（数值/代号/评价符号），而非表头词。"""
    for p in parts:
        t = re.sub(r"\s+", "", p)
        if not t:
            continue
        if _HEADER_HAS_EXAMPLE_RE.search(t):
            return True
        if _looks_like_data_value(t) or _LETTER_DATA_RE.fullmatch(t):
            return True
        if _BODYISH_PART_RE.search(t):
            return True
    return False


def _split_merged_example_body_rows(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """拆开把多条比较例/实施例粘在同一逻辑行的情形。

    几何规则（不绑样本）：
    - 左侧格有 ≥2 条堆叠比较例/实施例标签 → 身行数 n；
    - 其它格若多出前缀行，视为粘进来的表头，先剥到上方表头行；
    - 身段足 n 行则按行切开；仅 1 个数据值则 rowspan 盖住 n 行；
    - 纯表头格（无数据片段）整段留在表头带，不拆进身行。
    """
    if not cells:
        return cells
    work = [dict(c) for c in cells]
    max_row = max(int(c["row_end"]) for c in work)

    row = max_row
    while row >= 0:
        origins = [
            c
            for c in work
            if int(c["row_start"]) == int(c["row_end"]) == row
        ]
        nonempty = [c for c in origins if str(c.get("text") or "").strip()]
        if len(nonempty) < 2:
            row -= 1
            continue
        leftmost = min(nonempty, key=lambda c: int(c["col_start"]))
        labels = _stacked_example_body_labels(str(leftmost.get("text") or ""))
        if len(labels) < 2:
            row -= 1
            continue
        n = len(labels)

        # 各格拆成 header_prefix + body_parts
        plan: List[Tuple[Dict[str, Any], List[str], List[str], str]] = []
        # (cell, header_parts, body_parts, mode)
        # mode: split | rowspan | header_only
        header_depth = 0
        for c in origins:
            parts = _newline_parts(str(c.get("text") or ""))
            if not parts:
                plan.append((c, [], [], "empty"))
                continue
            if c is leftmost or id(c) == id(leftmost):
                plan.append((c, [], parts, "split"))
                continue
            if not _parts_look_bodyish(parts):
                # 纯表头粘连
                header_depth = max(header_depth, len(parts))
                plan.append((c, parts, [], "header_only"))
                continue
            if len(parts) > n:
                hdr, body = parts[:-n], parts[-n:]
                header_depth = max(header_depth, len(hdr))
                plan.append((c, hdr, body, "split"))
            elif len(parts) == n:
                plan.append((c, [], parts, "split"))
            elif len(parts) == 1:
                plan.append((c, [], parts, "rowspan"))
            else:
                # 2..n-1：无法可靠对齐，跳过本行
                plan = []
                break
        if not plan:
            row -= 1
            continue
        # 左侧必须是 split 且恰好 n 条标签
        if any(
            (c is leftmost or id(c) == id(leftmost)) and mode != "split"
            for c, _h, _b, mode in plan
        ):
            row -= 1
            continue

        extra = header_depth + n - 1  # 本行变成 header_depth + n 行，净增
        if extra < 0:
            row -= 1
            continue

        # 下方行下移
        for c in work:
            rs, re_ = int(c["row_start"]), int(c["row_end"])
            if rs > row:
                c["row_start"] = rs + extra
                c["row_end"] = re_ + extra
                c["row_span"] = int(c["row_end"]) - int(c["row_start"]) + 1
            elif rs <= row < re_:
                c["row_end"] = re_ + extra
                c["row_span"] = int(c["row_end"]) - int(c["row_start"]) + 1

        drop_ids = {id(c) for c in origins}
        new_cells: List[Dict[str, Any]] = []
        body_start = row + header_depth

        for c, hdr, body, mode in plan:
            cs, ce = int(c["col_start"]), int(c["col_end"])
            # 表头前缀（有身段的格）：底对齐放进 header 带
            if mode == "split" and hdr and header_depth > 0:
                for hi, text_i in enumerate(hdr):
                    r_i = body_start - len(hdr) + hi
                    if r_i < row:
                        continue
                    nc = dict(c)
                    nc["row_start"] = r_i
                    nc["row_end"] = r_i
                    nc["row_span"] = 1
                    nc["col_start"] = cs
                    nc["col_end"] = ce
                    nc["col_span"] = ce - cs + 1
                    nc["text"] = text_i
                    nc["texts"] = [text_i] if text_i else []
                    new_cells.append(nc)
            if mode == "header_only":
                for hi, text_i in enumerate(hdr):
                    r_i = body_start - len(hdr) + hi
                    if r_i < row:
                        continue
                    nc = dict(c)
                    nc["row_start"] = r_i
                    nc["row_end"] = r_i
                    nc["row_span"] = 1
                    nc["col_start"] = cs
                    nc["col_end"] = ce
                    nc["col_span"] = ce - cs + 1
                    nc["text"] = text_i
                    nc["texts"] = [text_i] if text_i else []
                    new_cells.append(nc)
                continue
            if mode == "rowspan":
                text_i = body[0] if body else ""
                nc = dict(c)
                nc["row_start"] = body_start
                nc["row_end"] = body_start + n - 1
                nc["row_span"] = n
                nc["text"] = text_i
                nc["texts"] = [text_i] if text_i else []
                new_cells.append(nc)
                continue
            if mode == "empty":
                for i in range(n):
                    nc = dict(c)
                    nc["row_start"] = body_start + i
                    nc["row_end"] = body_start + i
                    nc["row_span"] = 1
                    nc["text"] = ""
                    nc["texts"] = []
                    new_cells.append(nc)
                continue
            # split 身段
            for i in range(n):
                text_i = body[i] if i < len(body) else ""
                nc = dict(c)
                nc["row_start"] = body_start + i
                nc["row_end"] = body_start + i
                nc["row_span"] = 1
                nc["text"] = text_i
                nc["texts"] = [text_i] if text_i else []
                new_cells.append(nc)

        work = [c for c in work if id(c) not in drop_ids] + new_cells
        logger.info(
            "拆粘连实施例身行: row=%d → header=%d body=%d",
            row,
            header_depth,
            n,
        )
        row -= 1

    return work


def _column_body_is_index_label(
    cells: Sequence[Dict[str, Any]],
    col: int,
    header_end: int,
) -> bool:
    """表体该列是否主要是合成例/实施例索引。"""
    for cell in cells:
        if int(cell["row_start"]) <= header_end:
            continue
        if int(cell["col_start"]) != col or int(cell["col_end"]) != col:
            continue
        t = str(cell.get("text") or "").strip()
        if t and _BODY_INDEX_LABEL_RE.search(t):
            return True
    return False


def _column_body_looks_like_polymer_name(
    cells: Sequence[Dict[str, Any]],
    col: int,
    header_end: int,
) -> bool:
    """表体该列是否像聚合物名（树脂溶液等），而非化学代号用量。"""
    for cell in cells:
        if int(cell["row_start"]) <= header_end:
            continue
        if int(cell["col_start"]) != col or int(cell["col_end"]) != col:
            continue
        t = re.sub(r"\s+", "", str(cell.get("text") or ""))
        if not t or _is_cell_frag(t):
            continue
        if _BODY_INDEX_LABEL_RE.search(t):
            continue
        if _POLYMER_BODY_NAME_RE.search(t):
            return True
        # 含中文且非纯数值/代号行，也视为聚合物名候选
        if re.search(r"[\u4e00-\u9fff]", t) and not _looks_like_data_value(t):
            return True
    return False


def _align_orphan_left_anchor_headers(
    cells: List[Dict[str, Any]],
    *,
    look_back: int = 2,
) -> List[Dict[str, Any]]:
    """
    左锚表头「聚合物」落在无表体列、左侧近邻有聚合物名且无表头时，把表头迁到左侧。

    与 `_migrate_orphan_header_to_body_column`（指标头迁右）对称：TSR 过分割时常把
    「聚合物」偏右一列，而「丙烯酸树脂溶液」等落在其左的空头列，导致 HTML 中
    树脂名不在聚合物正下方。
    """
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    if header_end < 0:
        return cells

    # 候选：表头带内以「聚合物」为文案、单列、且该列无表体
    anchors: List[Dict[str, Any]] = []
    seen_cols: Set[int] = set()
    for cell in cells:
        if cell.get("_drop_render"):
            continue
        if int(cell["row_start"]) > header_end:
            continue
        t = re.sub(r"\s+", "", str(cell.get("text") or ""))
        if not t or not _LEFT_ANCHOR_HEADER_RE.search(t):
            continue
        if _MONOMER_PARENT_LABEL_RE.search(t):
            continue
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if cs != ce:
            continue
        if cs in seen_cols:
            continue
        if _column_has_body_content(cells, cs, header_end):
            continue
        seen_cols.add(cs)
        anchors.append(cell)

    if not anchors:
        return cells

    changed = 0
    for anchor in anchors:
        cs = int(anchor["col_start"])
        target: Optional[int] = None
        for left in range(cs - 1, max(-1, cs - look_back - 1), -1):
            if left < 0:
                break
            if _column_has_header_label(cells, left, header_end):
                break
            if not _column_has_body_content(cells, left, header_end):
                continue
            if _column_body_is_index_label(cells, left, header_end):
                break
            if not _column_body_looks_like_polymer_name(cells, left, header_end):
                break
            target = left
            break
        if target is None:
            continue
        moved = 0
        for cell in cells:
            if cell.get("_drop_render"):
                continue
            if int(cell["row_start"]) > header_end:
                continue
            if int(cell["col_start"]) != cs or int(cell["col_end"]) != cs:
                continue
            ct = re.sub(r"\s+", "", str(cell.get("text") or ""))
            if ct and not _LEFT_ANCHOR_HEADER_RE.search(ct):
                continue
            cell["col_start"] = target
            cell["col_end"] = target
            cell["col_span"] = 1
            moved += 1
        if moved:
            changed += 1
            logger.info(
                "左锚孤列表头对齐: 聚合物 col %d → %d",
                cs,
                target,
            )
    if changed:
        logger.info("左锚孤列表头对齐: %d 处", changed)
    return cells


def _drop_body_empty_columns(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    删除表体（表头以下）所有原子格均为空/碎片的列。

    与 drop_evidenceless_columns 不同：这里不看几何宽度，只看表体内容。
    表头 colspan 文本不算该列的证据——colspan 会在列删除后自动收缩。
    仅在表头 anchor 是该列唯一非空来源时才删。
    """
    if not cells:
        return cells
    header_end = _effective_header_end(cells)
    max_col = max(int(c["col_end"]) for c in cells)

    col_to_cells: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(max_col + 1)}
    for c in cells:
        for col in range(int(c["col_start"]), int(c["col_end"]) + 1):
            col_to_cells[col].append(c)

    dropped: Set[int] = set()
    for col in range(max_col + 1):
        covered = col_to_cells.get(col, [])
        if not covered:
            continue
        # 该列有数据值（数字/缺测横线等）或分类表体文本 → 真实数据列
        if _column_has_data_values(cells, col) or _column_has_body_content(
            cells, col, header_end
        ):
            continue
        # 表头锚点列（含短化学符号列头）：表体可全为同格符，仍保留
        if any(
            int(c["col_start"]) == col
            and int(c["row_start"]) <= header_end
            and str(c.get("text") or "").strip()
            for c in covered
        ) and not _colspan_sibling_has_body_data(cells, col, header_end):
            continue
        # 表头带起点有实质标签 → 保留；但若同 colspan 带内另有列承担表体数据，则本列是幽灵列
        header_anchor = any(
            int(c["col_start"]) == col
            and int(c["row_start"]) <= header_end
            and not _is_cell_frag(str(c.get("text") or ""))
            for c in covered
        )
        if header_anchor and not _colspan_sibling_has_body_data(cells, col, header_end):
            # P29：表头落在无表体列，右侧近邻有「有表体、无表头」列 → 把表头迁过去再删本列
            if (
                not _column_has_data_values(cells, col)
                and not _column_has_body_content(cells, col, header_end)
                and _migrate_orphan_header_to_body_column(
                    cells,
                    col,
                    header_end=header_end,
                    max_col=max_col,
                    col_to_cells=col_to_cells,
                    dropped=dropped,
                )
            ):
                dropped.add(col)
                continue
            continue
        # 子列表头（比较例 1 / 实施例 2 等）不是 colspan 幽灵列
        if header_anchor and any(
            int(c["col_start"]) == col
            and int(c["row_start"]) <= header_end
            and _HEADER_HAS_EXAMPLE_RE.search(str(c.get("text") or ""))
            for c in covered
        ):
            continue
        # 分组父头下的叶子单元列（…单元 / …结构）：表体可全空/稀疏，不可当幽灵列删
        # （否则会把「聚苯并噁唑单元|前体|聚酰胺酰亚胺」并进邻列）
        if header_anchor:
            from ..structure.header_topology import looks_like_leaf_unit_header

            if any(
                int(c["col_start"]) == col
                and int(c["row_start"]) <= header_end
                and looks_like_leaf_unit_header(str(c.get("text") or ""))
                for c in covered
            ):
                continue
        # 仅含括号单位、右侧邻列有表体数据 → 幽灵列（单位文本并入右侧）
        unit_texts = [
            str(c.get("text") or "").strip()
            for c in covered
            if str(c.get("text") or "").strip() and _is_header_unit_line(str(c.get("text") or ""))
        ]
        if (
            unit_texts
            and not _column_has_data_values(cells, col)
            and col + 1 <= max_col
            and _column_has_data_values(cells, col + 1)
        ):
            for c in covered:
                if int(c["col_start"]) != col:
                    continue
                t = str(c.get("text") or "").strip()
                if not t or not _is_header_unit_line(t):
                    continue
                row = int(c["row_start"])
                target = None
                # 同行右侧格
                for rc in col_to_cells.get(col + 1, []):
                    if int(rc["col_start"]) == col + 1 and int(rc["row_start"]) == row:
                        target = rc
                        break
                # 表头叠放：并入上一行或同 span 的描述性表头
                if target is None and row > 0:
                    for rc in col_to_cells.get(col + 1, []):
                        if int(rc["col_start"]) != col + 1:
                            continue
                        if int(rc["row_start"]) <= header_end and int(rc["row_end"]) >= row - 1:
                            if str(rc.get("text") or "").strip() and not _is_header_unit_line(
                                str(rc.get("text") or "")
                            ):
                                target = rc
                                break
                if target is not None:
                    ut = str(target.get("text") or "").strip()
                    target["text"] = (ut + "\n" + t) if ut else t
                    c["text"] = ""
            dropped.add(col)
            continue
        # 该列是某 cell 的 col_start 且有行标签等实质非数据文本 → 保留
        # （如「实施例 1」「聚酰亚胺」等，但不含短单位 "(mJ/cm2)"）
        body_anchors = [
            c for c in covered
            if int(c["row_start"]) > header_end
            and int(c["col_start"]) == col
            and not _is_cell_frag(str(c.get("text") or ""))
        ]
        # 有多个 body anchor 带文本：很可能是行标签列，保留
        if len(body_anchors) >= 2:
            continue
        # 唯一的 body anchor：若为单位/注释文本 → 幽灵列，但需保留文本
        if body_anchors:
            t = str(body_anchors[0].get("text") or "").strip()
            # 纯单位/注释：必须带括号，如 (mJ/cm2)、(℃)。勿把 jER828 / C 当单位删列。
            if not re.fullmatch(r"[\(\（][A-Za-z%°℃²³/\s\.\d\-μ]+[\)\）]", t):
                continue
        # 幽灵列：无数据值，无实质行标签
        dropped.add(col)

    # 删除前：把整段范围全在 dropped 中的 cell 的文本迁移到右侧保留列
    if dropped:
        for col in sorted(dropped):
            covered = col_to_cells.get(col, [])
            for c in covered:
                if int(c["col_start"]) != col:
                    continue
                cs, ce = int(c["col_start"]), int(c["col_end"])
                # 只迁移整段都在 dropped 中的 cell（跨保留列的 cell 会自行缩列）
                if not all(i in dropped for i in range(cs, ce + 1)):
                    continue
                t = str(c.get("text") or "").strip()
                if not t or _is_cell_frag(t):
                    continue
                row = int(c["row_start"])
                right = next((k for k in range(ce + 1, max_col + 1) if k not in dropped), None)
                if right is None:
                    continue
                for rc in col_to_cells.get(right, []):
                    if int(rc["row_start"]) <= row <= int(rc["row_end"]):
                        prev = str(rc.get("text") or "").strip()
                        if not prev:
                            rc["text"] = t
                        elif t not in prev:
                            rc["text"] = t + "\n" + prev
                        c["text"] = ""
                        break

    if not dropped:
        return cells

    kept_cols = [i for i in range(max_col + 1) if i not in dropped]
    if not kept_cols:
        return cells

    remap = {old: new for new, old in enumerate(kept_cols)}
    out: List[Dict[str, Any]] = []
    orphans: List[Tuple[int, int, str]] = []
    for c in cells:
        cs, ce = int(c["col_start"]), int(c["col_end"])
        new_idxs = [remap[i] for i in range(cs, ce + 1) if i in remap]
        if not new_idxs:
            text = str(c.get("text") or "").strip()
            if text and not _is_cell_frag(text):
                orphans.append((int(c["row_start"]), cs, text))
            continue
        nc = dict(c)
        nc["col_start"] = min(new_idxs)
        nc["col_end"] = max(new_idxs)
        nc["col_span"] = nc["col_end"] - nc["col_start"] + 1
        out.append(nc)
    for row, old_cs, text in orphans:
        _salvage_dropped_cell_text(
            out,
            row=row,
            old_start=old_cs,
            text=text,
            kept=kept_cols,
            remap=remap,
            axis="col",
        )
    if dropped:
        logger.info("删除表体空列: %s", sorted(dropped))
    return out


def _drop_fully_empty_rows(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """删除该行所有起点格均无文本的逻辑行（幽灵表头/间隙行）。

    仅看 ``row_start == row`` 的起点格；纯 rowspan 延续行（无起点）保留。
    """
    if not cells:
        return cells
    max_row = max(int(c["row_end"]) for c in cells)
    dropped: Set[int] = set()
    for row in range(max_row + 1):
        origins = [c for c in cells if int(c["row_start"]) == row]
        if not origins:
            continue
        if all(not str(c.get("text") or "").strip() for c in origins):
            dropped.add(row)
    if not dropped:
        return cells
    kept_rows = [i for i in range(max_row + 1) if i not in dropped]
    if not kept_rows:
        return cells
    remap = {old: new for new, old in enumerate(kept_rows)}
    out: List[Dict[str, Any]] = []
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        new_idxs = [remap[i] for i in range(rs, re + 1) if i in remap]
        if not new_idxs:
            continue
        nc = dict(c)
        nc["row_start"] = min(new_idxs)
        nc["row_end"] = max(new_idxs)
        nc["row_span"] = nc["row_end"] - nc["row_start"] + 1
        out.append(nc)
    if dropped:
        logger.info("删除全空逻辑行: %s", sorted(dropped))
    return out


def drop_noise_rows(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    丢弃几乎全空、仅含孤立噪声字符的逻辑行（如表末幻觉 'L'）。

    不丢弃含数字、多字中文或跨行 rowspan 起点的行。
    """
    if not cells:
        return cells

    max_row = max(int(c["row_end"]) for c in cells)
    by_origin_row: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(max_row + 1)}
    for c in cells:
        by_origin_row[int(c["row_start"])].append(c)

    dropped: Set[int] = set()
    for row, row_cells in by_origin_row.items():
        if not row_cells:
            continue
        # 有跨多行的起点格 → 保留
        if any(int(c.get("row_span") or 1) > 1 for c in row_cells):
            continue
        texts = [str(c.get("text") or "").strip() for c in row_cells]
        non_empty = [t for t in texts if t]
        if not non_empty:
            # 全空行留给 compress_empty_logic_rows
            continue
        if len(non_empty) <= 2 and all(_NOISE_CELL_RE.fullmatch(t) for t in non_empty):
            dropped.add(row)

    if not dropped:
        return cells

    kept_rows = [i for i in range(max_row + 1) if i not in dropped]
    if not kept_rows:
        return cells

    remap = {old: new for new, old in enumerate(kept_rows)}
    out: List[Dict[str, Any]] = []
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        new_idxs = [remap[i] for i in range(rs, re + 1) if i in remap]
        if not new_idxs:
            continue
        nc = dict(c)
        nc["row_start"] = min(new_idxs)
        nc["row_end"] = max(new_idxs)
        nc["row_span"] = nc["row_end"] - nc["row_start"] + 1
        out.append(nc)
    logger.info("丢弃噪声行: %s", sorted(dropped))
    return out


_HEADER_HAS_EXAMPLE_RE = re.compile(r"实[施試]例|実[施試]例|比較例|比较例|参考例")
_HEADER_UNIT_RE = re.compile(
    r"^[\(（][\w%°℃²³/\s\.\-μ]+[\)）]$"
)
# 括号内须含典型单位特征，避免 (B-2)/(D) 等编号被当成 (mN)/(℃)
_HEADER_UNIT_INNER_RE = re.compile(
    r"[%°℃²³/μ]|"
    r"(?:mJ|cm|mm|μm|min|mol|wt|vol|hr|mN|kPa|MPa)|"
    r"^\d+$",
    re.I,
)


def _is_header_unit_line(text: str) -> bool:
    """括号包裹的纯单位文本，如 (mJ/cm2)、(mN)、(%)、(℃)。"""
    t = "".join((text or "").split())
    if not t or not _HEADER_UNIT_RE.fullmatch(t):
        return False
    inner = t[1:-1] if t[0] in "(（" else t
    if re.fullmatch(r"[A-Za-z]-?\d+", inner):
        return False
    if re.fullmatch(r"[A-Z]", inner):
        return False
    if _HEADER_UNIT_INNER_RE.search(inner):
        return True
    if re.fullmatch(r"[A-Za-z]{1,4}", inner) and not re.search(r"\d", inner):
        return True
    return False


def _row_is_unit_subheader(cells: Sequence[Dict[str, Any]], row: int) -> bool:
    """整行非空格均为括号单位 → 二级表头单位行（P28/P29）。"""
    row_cells = [
        c
        for c in cells
        if int(c["row_start"]) == row == int(c["row_end"])
    ]
    non_empty = [c for c in row_cells if str(c.get("text") or "").strip()]
    if not non_empty:
        return False
    return all(_is_header_unit_line(str(c.get("text") or "")) for c in non_empty)


def _row_has_descriptive_header(cells: Sequence[Dict[str, Any]], row: int) -> bool:
    """该行是否存在描述性表头（非单位/非表体数据/非行标签）。"""
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        if not (rs <= row <= re):
            continue
        t = str(c.get("text") or "").strip()
        if not t or _is_header_unit_line(t):
            continue
        if _looks_like_data_value(t):
            continue
        if _HEADER_HAS_EXAMPLE_RE.search(t):
            continue
        return True
    return False


def _extend_header_end_for_unit_rows(
    cells: Sequence[Dict[str, Any]],
    header_end: int,
    *,
    cap_row: int | None = None,
) -> int:
    """无 rowspan 时，把紧挨其下的「纯单位行」纳入顶表头带。"""
    if header_end < 0:
        header_end = 0
    max_row = max(int(c["row_end"]) for c in cells)
    stop = cap_row if cap_row is not None else max_row + 1
    he = header_end
    for row in range(he + 1, max_row + 1):
        if row >= stop:
            break
        if _row_is_unit_subheader(cells, row) and _row_has_descriptive_header(cells, row - 1):
            he = row
        else:
            break
    return he


_LONE_EXAMPLE_NUM_RE = re.compile(r"^\d+$")


def _example_col_header_cols_in_row(
    cells: Sequence[Dict[str, Any]], row: int
) -> Set[int]:
    """同行中「实施例/比较例」**列头**所占的列起点集合。

    必须落在不同列：左侧身行标签（单列堆叠比较例1/2/3）不算列头横串。
    """
    cols: Set[int] = set()
    for c in cells:
        if int(c["row_start"]) > row or int(c["row_end"]) < row:
            continue
        t = str(c.get("text") or "").strip()
        if not t or _LONE_EXAMPLE_NUM_RE.fullmatch(t):
            continue
        if not _HEADER_HAS_EXAMPLE_RE.search(t):
            continue
        # 单格内多行比较例标签 → 身行，不是顶栏列头横串
        parts = [p.strip() for p in t.split("\n") if p.strip()]
        if len(parts) >= 2 and all(_HEADER_HAS_EXAMPLE_RE.search(p) for p in parts):
            continue
        cols.add(int(c["col_start"]))
    return cols


def _example_header_row_end(cells: Sequence[Dict[str, Any]]) -> int:
    """含 ≥2 列实施例/比较例/参考例**列头**的最下行（无则 -1）。"""
    by_row: Dict[int, Set[int]] = {}
    max_r = max((int(c["row_end"]) for c in cells), default=-1)
    for r in range(max_r + 1):
        cols = _example_col_header_cols_in_row(cells, r)
        if len(cols) >= 2:
            by_row[r] = cols
    return max(by_row) if by_row else -1


def _effective_header_end(cells: Sequence[Dict[str, Any]]) -> int:
    """顶表头带末行；无 rowspan 线索时取首个有字身行之上。"""
    cell_list = list(cells)
    he = _top_header_band_end(cell_list)
    if he < 0:
        body_starts = [
            int(c["row_start"])
            for c in cell_list
            if str(c.get("text") or "").strip()
            and not _is_cell_frag(str(c.get("text") or ""))
            and not _is_header_unit_line(str(c.get("text") or ""))
            and not _HEADER_HAS_EXAMPLE_RE.search(str(c.get("text") or ""))
        ]
        he = max(0, min(body_starts) - 1) if body_starts else 0
    # P24/P25：空幽灵行把 he 压成 0 时，仍要把含多列实施例/比较例的行算进表头带，
    # 否则「项目」colspan 会被身列拆格逻辑误拆。
    he = max(he, _example_header_row_end(cell_list))
    first_body_stub: int | None = None
    for c in cell_list:
        if not _is_body_left_rowspan_stub(cell_list, c):
            continue
        rs = int(c["row_start"])
        first_body_stub = rs if first_body_stub is None else min(first_body_stub, rs)
    return _extend_header_end_for_unit_rows(
        cell_list, he, cap_row=first_body_stub
    )


def _covered_by_upper_colspan_parent(
    cells: Sequence[Dict[str, Any]], cell: Dict[str, Any]
) -> bool:
    """是否被正上方 colspan>1 父格覆盖（嵌套中层表头，如「树脂」子列）。

    不算嵌套父格的情况：
    - 空角 colspan（转置表左上空白占位）——下方「(A)成分」等仍是表体 stub；
    - 实施例/比较例列头即使 colspan>1——它们是数据列锚点。
    """
    rs = int(cell["row_start"])
    cs, ce = int(cell["col_start"]), int(cell["col_end"])
    for p in cells:
        pre = int(p["row_end"])
        if pre >= rs:
            continue
        pcs, pce = int(p["col_start"]), int(p["col_end"])
        if pcs <= cs and pce >= ce and pce > pcs:
            pt = str(p.get("text") or "").strip()
            # 空角 colspan 不是嵌套父头；否则表头带会被左侧成分 rowspan 撑到身行
            if not pt:
                continue
            if _EXAMPLE_COL_HEADER_RE.search(pt):
                continue
            return True
    return False


def _row_has_example_column_headers(
    cells: Sequence[Dict[str, Any]], row: int
) -> bool:
    """该逻辑行是否含 ≥2 列实施例/比较例/参考例列头（非身行堆叠标签）。"""
    return len(_example_col_header_cols_in_row(cells, row)) >= 2


def _is_body_left_rowspan_stub(
    cells: Sequence[Dict[str, Any]], cell: Dict[str, Any]
) -> bool:
    """表体左侧大行头 stub（比较例/实施例 rowspan），排除嵌套中层表头。

    转置表常见：row0=实施例列头，row1 起左侧「聚合物」等分类 rowspan。
    这类 stub 的 ``row_start`` 可为 1，不可再把表头带向下扩到数据行。
    """
    rs, re = int(cell["row_start"]), int(cell["row_end"])
    cs = int(cell["col_start"])
    if not (re > rs and rs >= 1 and cs <= 1):
        return False
    # P33：中层「树脂」rowspan 落在父「树脂」colspan 下，不是表体 stub
    if _covered_by_upper_colspan_parent(cells, cell):
        return False
    text = str(cell.get("text") or "").strip()
    if not text:
        # 空左侧 stub：rs>=2 仍截断；rs==1 仅当上一行已是实施例列头
        if rs >= 2:
            return True
        return bool(rs >= 1 and _row_has_example_column_headers(cells, rs - 1))
    if _HEADER_HAS_EXAMPLE_RE.search(text):
        return True
    # 紧挨实施例列头行之下的左侧分类 rowspan（聚合物/化合物…）= 表体 stub
    if rs >= 1 and _row_has_example_column_headers(cells, rs - 1):
        return True
    # 描述性中层表头（酸成分旁的「树脂」等）即使无父 colspan 也不截断表头带
    return False


def _top_header_band_end(cells: List[Dict[str, Any]]) -> int:
    """顶表头带最后一行（含）。表体左侧大行头的 rowspan 不计入。

    只有 ``row_start <= 1`` 的跨行格才能把表头带向下扩；若存在表体左侧
    大行头（``row_start >= 2`` 且偏左），表头带截止在其上一行，避免把
    ``比较例 1`` / 数据 ``100`` 错误向下合并进表体空格。
    嵌套中层表头（被上方 colspan 父格覆盖）不算表体 stub。
    """
    header_end = -1
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        if re > rs and rs <= 1:
            header_end = max(header_end, re)

    first_body_stub: int | None = None
    for c in cells:
        if not _is_body_left_rowspan_stub(cells, c):
            continue
        rs = int(c["row_start"])
        first_body_stub = rs if first_body_stub is None else min(first_body_stub, rs)
    if first_body_stub is not None:
        cap = first_body_stub - 1
        header_end = cap if header_end < 0 else min(header_end, cap)
    if header_end < 0:
        header_end = 0
    header_end = _extend_header_end_for_unit_rows(
        cells, header_end, cap_row=first_body_stub
    )
    # 左列命名行标签（分散液101）定表体起点，勿让 rowspan 把表头带扩进首行数据
    try:
        from ..structure.hline_repair import _first_body_row_index
        from ..structure.tsr_refine import _derive_seps

        row_seps, _ = _derive_seps(cells)
        if len(row_seps) >= 2:
            body_row = _first_body_row_index(cells, [], row_seps)
            if body_row >= 1:
                header_end = min(header_end, body_row - 1)
    except Exception:  # noqa: BLE001
        pass
    return header_end


def _header_cell_above(
    cells: Sequence[Dict[str, Any]], row: int, cs: int, ce: int
) -> Dict[str, Any] | None:
    """单位子表头正上方、列范围被覆盖的描述性表头格（支持 colspan 父格）。"""
    upper_row = row - 1
    best: Dict[str, Any] | None = None
    best_span = 10**9
    for c in cells:
        if int(c["row_start"]) > upper_row or int(c["row_end"]) < upper_row:
            continue
        pcs, pce = int(c["col_start"]), int(c["col_end"])
        if pcs > cs or pce < ce:
            continue
        t = str(c.get("text") or "").strip()
        if not t or _is_header_unit_line(t):
            continue
        span = pce - pcs
        if span < best_span:
            best = c
            best_span = span
    return best


def _split_mismerged_unit_rowspans(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """单位格被误扩 rowspan 时收回单行，避免 P29 单位行错位。"""
    header_end = _top_header_band_end(cells)
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        if re <= rs:
            continue
        # 仅表头带内处理；表体子标签 (B-2) 等 rowspan 不得拆
        if rs > header_end:
            continue
        if _is_header_unit_line(str(c.get("text") or "")):
            c["row_end"] = rs
            c["row_span"] = 1
    return cells


def _merge_header_unit_rows(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    表头中：若某行所有非空格都是括号单位文本且上方同列有描述性表头，
    则将单位行合并到上方格（文本拼接为换行）并删除单位行。

    典型场景：row0=「最小曝光量（Eth）|密合强度」, row1=「(mJ/cm2)|(mN)」
    """
    if not cells:
        return cells

    max_row = max(int(c["row_end"]) for c in cells)
    if max_row < 1:
        return cells

    by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if rs == re:
            by_key[(rs, cs, ce)] = c

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for c in cells:
        rs = int(c["row_start"])
        re = int(c["row_end"])
        if rs == re:
            by_row.setdefault(rs, []).append(c)

    # 找候选单位行：该行所有非空格都是括号单位
    unit_rows: Set[int] = set()
    for row, row_cells in by_row.items():
        if row < 1:
            continue
        non_empty = [c for c in row_cells if str(c.get("text") or "").strip()]
        if not non_empty:
            continue
        if all(_is_header_unit_line(str(c.get("text") or "").strip()) for c in non_empty):
            has_upper = False
            for c in non_empty:
                cs, ce = int(c["col_start"]), int(c["col_end"])
                upper = _header_cell_above(cells, row, cs, ce)
                if upper and str(upper.get("text") or "").strip():
                    has_upper = True
                    break
            if has_upper:
                unit_rows.add(row)

    if not unit_rows:
        return cells

    dropped_ids: Set[int] = set()
    for row in unit_rows:
        for c in by_row.get(row, []):
            t = str(c.get("text") or "").strip()
            cs, ce = int(c["col_start"]), int(c["col_end"])
            upper = _header_cell_above(cells, row, cs, ce)
            if upper and t:
                ut = str(upper.get("text") or "").strip()
                upper["text"] = ut + "\n" + t if ut else t
                upper["row_end"] = max(int(upper["row_end"]), int(c["row_end"]))
                upper["row_span"] = int(upper["row_end"]) - int(upper["row_start"]) + 1
            dropped_ids.add(id(c))

    if not dropped_ids:
        return cells
    return [c for c in cells if id(c) not in dropped_ids]


def _split_example_header_rowspans(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """实施例/比较例列头若被误扩成跨两行 rowspan，收回为单行。

    P24：下层是聚酰亚胺行的 A/B/C 数据，不是二级表头。
    """
    if not cells:
        return cells
    changed = 0
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        if re <= rs:
            continue
        if rs > 1:
            continue
        text = str(c.get("text") or "")
        if not _EXAMPLE_COL_HEADER_RE.search(text):
            continue
        # 仅拆「跨一行」的假二级表头（避免误伤更复杂多级）
        if re != rs + 1:
            continue
        c["row_end"] = rs
        c["row_span"] = 1
        changed += 1
    if changed:
        logger.info("实施例列头 rowspan 收回: %d 格", changed)
    return cells


_HEADER_SUBCOL_CODE_RE = re.compile(
    r"^(?:"
    r"[A-Za-zα-ωΑ-Ω][A-Za-zα-ωΑ-Ω0-9\-\+']*|"
    r"摩尔[%％]"
    r")$"
)
_LETTERED_PARENT_HDR_RE = re.compile(
    r"[\(（]\s*([a-zA-Zα-ωΑ-Ω])\s*[\)）]"
)
_LETTERED_CHILD_HDR_RE = re.compile(
    r"[\(（]\s*([a-zA-Zα-ωΑ-Ω])\s*\d"
)
_GENERIC_SUBHDR_RE = re.compile(r"^其他$")


def _refresh_cell_spans(cell: Dict[str, Any]) -> None:
    cell["row_span"] = int(cell["row_end"]) - int(cell["row_start"]) + 1
    cell["col_span"] = int(cell["col_end"]) - int(cell["col_start"]) + 1


def _lettered_parent_letter(text: str) -> Optional[str]:
    m = _LETTERED_PARENT_HDR_RE.search("".join((text or "").split()))
    return m.group(1).lower() if m else None


def _lettered_child_letter(text: str) -> Optional[str]:
    m = _LETTERED_CHILD_HDR_RE.search("".join((text or "").split()))
    return m.group(1).lower() if m else None


def _split_lettered_parent_child_text(text: str) -> Tuple[str, str]:
    """字母父头格若吞入同字母子头 (a1…)，拆成父/子两段文本。"""
    raw = (text or "").strip()
    if not raw:
        return raw, ""
    pl = _lettered_parent_letter(raw)
    if not pl:
        return raw, ""
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    if len(lines) >= 2:
        child_idx: Optional[int] = None
        for i, ln in enumerate(lines):
            if i == 0:
                continue
            cl = _lettered_child_letter(ln)
            if cl == pl:
                child_idx = i
                break
        if child_idx is not None:
            parent = "\n".join(lines[:child_idx]).strip()
            child = "\n".join(lines[child_idx:]).strip()
            if parent and child:
                return parent, child
    compact = "".join(raw.split())
    m_parent = _LETTERED_PARENT_HDR_RE.search(compact)
    m_child = _LETTERED_CHILD_HDR_RE.search(compact)
    if (
        m_parent
        and m_child
        and m_child.start() > m_parent.start()
        and m_child.group(1).lower() == pl
    ):
        # 按 compact 偏移映回原文（去空白对齐）
        ci = 0
        split_at = 0
        for ch in raw:
            if not ch.isspace():
                if ci == m_child.start():
                    split_at = raw.index(ch, split_at)
                    break
                ci += 1
        if split_at > 0:
            return raw[:split_at].strip(), raw[split_at:].strip()
    return raw, ""


def _header_cell_origin(
    cells: Sequence[Dict[str, Any]], row: int, col: int
) -> Optional[Dict[str, Any]]:
    """表头格起点（允许 rowspan>1，不要求 row_start==row_end）。"""
    for c in cells:
        if int(c["row_start"]) == row and int(c["col_start"]) == col:
            return c
    return None


def _atomic_header_at(
    cells: Sequence[Dict[str, Any]], row: int, col: int
) -> Optional[Dict[str, Any]]:
    for c in cells:
        if (
            int(c["row_start"]) == int(c["row_end"]) == row
            and int(c["col_start"]) == int(c["col_end"]) == col
        ):
            return c
    return None


def _row_col_header_label(cells: Sequence[Dict[str, Any]], row: int, col: int) -> str:
    cell = _atomic_header_at(cells, row, col)
    if cell is None:
        return ""
    return str(cell.get("text") or "").strip()


def _column_gap_between_subheaders(
    cells: Sequence[Dict[str, Any]],
    row: int,
    col: int,
    *,
    header_end: int,
) -> bool:
    """本列下方无格，但左右邻列同层已有表头 → 父级 colspan 占位，勿纵向合并。"""
    if row > header_end or col < 0:
        return False
    left = _row_col_header_label(cells, row, col - 1)
    right = _row_col_header_label(cells, row, col + 1)
    if not left or not right:
        return False
    return not (
        _is_cell_frag(left)
        or _is_cell_frag(right)
        or _is_header_unit_line(left)
        or _is_header_unit_line(right)
    )


def _child_cols_for_lettered_parent(
    cells: Sequence[Dict[str, Any]],
    *,
    parent_letter: str,
    bottom_row: int,
    max_col: int,
) -> List[int]:
    """下层子列：同字母 (a1…) 及紧邻的「其他」等通用子头，止于下一字母父头列。"""
    child_cols: List[int] = []
    pl = parent_letter.lower()
    for col in range(max_col + 1):
        text = _row_col_header_label(cells, bottom_row, col)
        if not text:
            continue
        compact = "".join(text.split())
        child_letter = _lettered_child_letter(text)
        if child_letter == pl:
            child_cols.append(col)
            continue
        if child_cols and _GENERIC_SUBHDR_RE.fullmatch(compact):
            child_cols.append(col)
            continue
        if child_cols:
            other_parent = _lettered_parent_letter(text)
            if other_parent and other_parent != pl:
                break
            if other_parent == pl:
                child_cols.append(col)
                continue
            break
    return child_cols


def _compact_logic_columns(
    cells: List[Dict[str, Any]], drop_cols: Set[int]
) -> List[Dict[str, Any]]:
    """删除表头子列之间的幽灵逻辑列并重映射 col_*。"""
    if not drop_cols:
        return cells
    max_col = max(int(c["col_end"]) for c in cells)
    remap = {}
    new_c = 0
    for old_c in range(max_col + 1):
        if old_c in drop_cols:
            continue
        remap[old_c] = new_c
        new_c += 1
    out: List[Dict[str, Any]] = []
    for c in cells:
        nc = dict(c)
        cs, ce = int(c["col_start"]), int(c["col_end"])
        new_idxs = [remap[i] for i in range(cs, ce + 1) if i in remap]
        if not new_idxs:
            continue
        nc["col_start"] = min(new_idxs)
        nc["col_end"] = max(new_idxs)
        _refresh_cell_spans(nc)
        out.append(nc)
    return out


def _normalize_nested_column_group_headers(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    多级列组表头归一化（通用，非单表补丁）：

    1. 父表头行短 CJK 碎片 → 合并为 colspan 父头（如 酸催+化剂 → 酸催化剂）
    2. 子表头行空槽 → 按同宽兄弟列组模板补全（如 溶剂下缺 量(g)）
    """
    if not cells:
        return cells

    from ..structure.hline_repair import _nested_header_parent_sub_rows
    from ..structure.header_topology import repair_parent_header_topology
    from ..structure.tsr_refine import _derive_seps as _derive_seps_hr

    work = repair_parent_header_topology(cells, [])

    header_end = _top_header_band_end(work)
    if header_end < 1:
        return work

    row_seps, _ = _derive_seps_hr(work)
    parent_row, bottom_row = _nested_header_parent_sub_rows(work, [], row_seps)
    top_row = parent_row
    max_col = max(int(c["col_end"]) for c in work)
    body_start = header_end + 1

    def _label_at(row: int, col: int) -> str:
        cell = _atomic_header_at(work, row, col)
        return str(cell.get("text") or "").strip() if cell else ""

    def _group_top_span(cs: int) -> Tuple[int, int]:
        """从 cs 起顶行连续原子格 / colspan 父格覆盖的列区间。"""
        for c in work:
            if int(c["row_start"]) > top_row or int(c["row_end"]) < top_row:
                continue
            pcs, pce = int(c["col_start"]), int(c["col_end"])
            if pcs <= cs <= pce and pce > pcs:
                return pcs, pce
        ce = cs
        while ce + 1 <= max_col:
            if _atomic_header_at(work, top_row, ce + 1) is None:
                break
            ce += 1
        return cs, ce

    templates: Dict[int, List[str]] = {}
    col = 0
    while col <= max_col:
        cs, ce = _group_top_span(col)
        width = ce - cs + 1
        if width >= 2:
            labels = [_label_at(bottom_row, c) for c in range(cs, ce + 1)]
            filled = sum(1 for t in labels if t and not _is_header_unit_line(t))
            if filled >= width:
                templates[width] = labels
        col = ce + 1

    if not templates:
        return work

    filled_n = 0
    col = 0
    while col <= max_col:
        cs, ce = _group_top_span(col)
        width = ce - cs + 1
        if width >= 2 and width in templates:
            tmpl = templates[width]
            for off in range(width):
                ccol = cs + off
                cur = _label_at(bottom_row, ccol)
                if cur and not _is_cell_frag(cur):
                    continue
                if not tmpl[off] or _is_header_unit_line(tmpl[off]):
                    continue
                if not (
                    _column_has_data_values(
                        work, ccol, skip_rows=set(range(body_start))
                    )
                    or _column_has_body_content(work, ccol, header_end)
                ):
                    continue
                slot = _atomic_header_at(work, bottom_row, ccol)
                if slot is None:
                    ref = _atomic_header_at(work, bottom_row, cs) or _atomic_header_at(
                        work, top_row, cs
                    )
                    if ref is None:
                        continue
                    bbox = _cell_physical_bbox_x_y(ref)
                    slot = _empty_atomic_from_bbox(ref, ccol, cs, ce, bbox)
                    slot["row_start"] = bottom_row
                    slot["row_end"] = bottom_row
                    slot["row_span"] = 1
                    work.append(slot)
                slot["text"] = tmpl[off]
                filled_n += 1
        col = ce + 1

    if filled_n:
        logger.info("子表头模板补全: %d 处", filled_n)
    return work


def _normalize_tiered_column_headers(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    字母分级表头：(a) 父头跨 (a1…)/其他 子列；无子列的 (b)/清漆/溶剂 等纵向跨表头带。

    修复 TSR 把父头放在子列间隙、或 `_merge_header_empty_below` 误扩 rowspan 的情况。
    """
    if not cells:
        return cells

    header_end = _top_header_band_end(cells)
    if header_end < 1:
        return cells

    top_row = 0
    bottom_row = header_end
    max_col = max(int(c["col_end"]) for c in cells)

    parent_cells: List[Dict[str, Any]] = []
    child_cols_union: Set[int] = set()
    for c in cells:
        if c.get("_drop_render"):
            continue
        letter = _lettered_parent_letter(str(c.get("text") or ""))
        if not letter:
            continue
        child_cols = _child_cols_for_lettered_parent(
            cells,
            parent_letter=letter,
            bottom_row=bottom_row,
            max_col=max_col,
        )
        if not child_cols:
            continue
        parent_cells.append(c)
        child_cols_union.update(child_cols)
        parent_text, child_text = _split_lettered_parent_child_text(
            str(c.get("text") or "")
        )
        if child_text:
            c["text"] = parent_text
            left_col = min(child_cols)
            target = _atomic_header_at(cells, bottom_row, left_col)
            if target is c:
                target = None
            if target is None:
                bbox = _cell_physical_bbox_x_y(c)
                new_c = _empty_atomic_from_bbox(
                    c,
                    left_col,
                    int(c["col_start"]),
                    int(c["col_end"]),
                    bbox,
                )
                new_c["row_start"] = bottom_row
                new_c["row_end"] = bottom_row
                new_c["row_span"] = 1
                new_c["text"] = child_text
                cells.append(new_c)
                logger.info(
                    "分级列头: 子头文本归位 col %d ← %s",
                    left_col,
                    child_text[:24],
                )
            elif not str(target.get("text") or "").strip():
                target["text"] = child_text
            elif _lettered_child_letter(str(target.get("text") or "")) == letter:
                target["text"] = child_text
        c["row_start"] = top_row
        c["row_end"] = top_row
        c["col_start"] = min(child_cols)
        c["col_end"] = max(child_cols)
        _refresh_cell_spans(c)

    if not parent_cells:
        return cells

    gap_cols: Set[int] = set()
    if len(child_cols_union) >= 2:
        lo, hi = min(child_cols_union), max(child_cols_union)
        gap_cols = {col for col in range(lo + 1, hi) if col not in child_cols_union}
    if gap_cols:
        cells = _compact_logic_columns(cells, gap_cols)
        child_cols_union = set()
        for c in cells:
            letter = _lettered_parent_letter(str(c.get("text") or ""))
            if not letter:
                continue
            child_cols_union.update(
                _child_cols_for_lettered_parent(
                    cells,
                    parent_letter=letter,
                    bottom_row=bottom_row,
                    max_col=max(int(x["col_end"]) for x in cells),
                )
            )

    for c in cells:
        if c.get("_drop_render"):
            continue
        rs, re = int(c["row_start"]), int(c["row_end"])
        cs = int(c["col_start"])
        if rs != bottom_row or re != bottom_row:
            continue
        if cs in child_cols_union:
            continue
        text = str(c.get("text") or "").strip()
        if not text or _is_cell_frag(text) or _is_header_unit_line(text):
            continue
        from ..structure.header_topology import (
            _looks_like_kind_or_amount_leaf,
            _looks_like_numeric_metric_leaf,
        )

        if (
            _is_known_leaf_subheader(text)
            or _looks_like_kind_or_amount_leaf(text)
            or _looks_like_numeric_metric_leaf(text)
        ):
            continue
        if _EXAMPLE_COL_HEADER_RE.search(text):
            continue
        if int(c.get("col_span") or 1) > 1:
            continue
        # 顶行已有 colspan/rowspan 父头覆盖本列 → 底行叶子保持单行
        top_cover = _row_col_header_label(cells, top_row, cs)
        if top_cover and (
            _lettered_parent_letter(top_cover)
            or _looks_like_independent_header_label(top_cover)
        ):
            continue
        top_label = _row_col_header_label(cells, top_row, cs)
        if (
            top_label
            and _is_stacked_header_label(top_label)
            and _is_stacked_header_label(text)
        ):
            continue
        c["row_start"] = top_row
        c["row_end"] = bottom_row
        _refresh_cell_spans(c)

    logger.info(
        "分级列头归位: %d 个字母父头, 子列 %s",
        len(parent_cells),
        sorted(child_cols_union),
    )
    return cells


def _is_stacked_header_label(text: str) -> bool:
    """表头同列纵向叠放的中文标签（非子列代号/单位/数据）。"""
    t = (text or "").strip()
    if not t:
        return False
    compact = "".join(t.split())
    if len(compact) <= 2 or compact in {"-", "—", "_"}:
        return False
    if compact.startswith("(") or compact.startswith("（"):
        return False
    if _is_header_unit_line(t):
        return False
    if _EXAMPLE_COL_HEADER_RE.search(t):
        return False
    if _DATA_VALUE_RE.fullmatch(compact) or _LETTER_DATA_RE.fullmatch(compact):
        return False
    if _HEADER_SUBCOL_CODE_RE.fullmatch(compact):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", t))


_WRAP_HEADER_SUFFIX_RE = re.compile(
    r"(比率|含量|比例|浓度|份数|成分|单位|量\s*[\[%％]\]|质量\s*[\[%％]\])"
)
_INDEPENDENT_HDR_TAIL_RE = re.compile(
    r"(剂|物|素|基|团|料|液|体|率|值|数|型|类|项|例|酸|胺|醇|醚|酮|醛|烃)$"
)
# 同列平行系列标签：聚合物A/聚合物B、化合物1/2（非折行）
_PARALLEL_SERIES_LABEL_RE = re.compile(
    r"^(?P<stem>[\u4e00-\u9fff]{2,10})"
    r"(?P<idx>[A-Za-zＡ-Ｚａ-ｚ0-9０-９]+)$"
)


def _compact_header_text(text: str) -> str:
    return "".join((text or "").split())


def _looks_like_parallel_series_labels(upper: str, lower: str) -> bool:
    """同词干 + 不同尾缀序号/字母 → 平行行头，禁止当折行合并。"""
    u = _compact_header_text(upper)
    l = _compact_header_text(lower)
    mu = _PARALLEL_SERIES_LABEL_RE.fullmatch(u)
    ml = _PARALLEL_SERIES_LABEL_RE.fullmatch(l)
    if not mu or not ml:
        return False
    return mu.group("stem") == ml.group("stem") and mu.group("idx") != ml.group("idx")


def _looks_like_independent_header_label(text: str) -> bool:
    """像完整独立列头（如添加剂、热产碱剂），而非折行片段。"""
    compact = _compact_header_text(text)
    if len(compact) < 2:
        return False
    # 配方代号/字母子列行是长表头折行的一部分，不是独立 stacked 标签
    if re.search(r"[\(（][A-Za-zα-ωΑ-Ω]", compact):
        return False
    if re.search(r"[A-Za-z]", compact) and len(compact) > 4:
        return False
    return bool(_INDEPENDENT_HDR_TAIL_RE.search(compact))


def _looks_like_wrapped_header_lines(
    parts: Sequence[str],
    row_span: int,
) -> bool:
    """\\n 多行是单格内折行（如竖向长表头），不是每行一个独立表头。"""
    if len(parts) <= 1:
        return False
    # 字母组分父头 + 种类/量 叶子：独立上下格，非折行
    if _lettered_parent_letter(parts[0]) and _is_leaf_type_amount_subheader(
        "\n".join(parts[1:])
    ):
        return False
    # 室温放置 + 显影/感度叶子：独立上下格
    if _is_room_storage_over_eval_leaf(parts[0], "\n".join(parts[1:])):
        return False
    # 组成[质量份] + （D）着色剂：独立上下格，非折行
    if _is_composition_span_parent(parts[0]) and _lettered_parent_letter(
        "\n".join(parts[1:])
    ):
        return False
    # 折行数多于 rowspan：不可能是「每行一标签」
    if len(parts) > row_span:
        return True
    joined = _compact_header_text("".join(parts))
    if _WRAP_HEADER_SUFFIX_RE.search(joined):
        if not any(
            _WRAP_HEADER_SUFFIX_RE.search(_compact_header_text(p))
            for p in parts[:-1]
        ):
            return True
    for i in range(len(parts) - 1):
        a = _compact_header_text(parts[i])
        b = _compact_header_text(parts[i + 1])
        if len(a) < 2 or len(b) < 2:
            continue
        if a[-1] in ")）]】%％" or b[0] in "(（[【":
            continue
        if not re.search(r"[\u4e00-\u9fff]", a) or not re.search(r"[\u4e00-\u9fff]", b):
            continue
        if _looks_like_independent_header_label(a) and _looks_like_independent_header_label(b):
            continue
        # 字母父头即使 ≤8 字也不是折行前缀（如 （B）光酸产生剂）
        if _lettered_parent_letter(a):
            continue
        combined = a + b
        if len(a) <= 8 and _WRAP_HEADER_SUFFIX_RE.search(combined):
            return True
        if len(a) <= 8 and not _looks_like_independent_header_label(a):
            return True
    return False


def _looks_like_independent_stacked_header_lines(
    parts: Sequence[str],
    row_span: int,
) -> bool:
    """True = 每行一个独立表头，可按 rowspan 拆成上下格。"""
    if len(parts) < 2 or len(parts) > row_span:
        return False
    if _looks_like_wrapped_header_lines(parts, row_span):
        return False
    return all(_looks_like_independent_header_label(p) for p in parts)


def _split_stacked_header_rowspans(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """表头带内 rowspan>1 且叠放独立中文标签 → 按行拆成上下两格（如添加剂/热产碱剂）。

    亦拆「(B)光酸产生剂\\n种类\\n(量)」这类字母父头 + 叶子子表头。
    """
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if header_end < 1:
        return cells
    work = list(cells)
    added: List[Dict[str, Any]] = []
    for c in work:
        rs, re = int(c["row_start"]), int(c["row_end"])
        if re <= rs or re > header_end:
            continue
        if int(c["col_start"]) != int(c["col_end"]):
            continue
        raw = str(c.get("text") or "")
        parts = [p.strip() for p in raw.split("\n") if p.strip()]
        row_span = re - rs + 1
        if len(parts) < 2:
            continue
        lettered_leaf_split = bool(
            _lettered_parent_letter(parts[0])
            and _is_leaf_type_amount_subheader("\n".join(parts[1:]))
        )
        room_leaf_split = bool(
            _is_room_storage_over_eval_leaf(parts[0], "\n".join(parts[1:]))
        )
        if _lettered_parent_letter(raw) and not lettered_leaf_split:
            continue
        if not lettered_leaf_split and not room_leaf_split:
            if not all(_is_stacked_header_label(p) for p in parts):
                continue
            if not _looks_like_independent_stacked_header_lines(parts, row_span):
                continue
        n_parts = min(len(parts), re - rs + 1)
        if n_parts < 2:
            continue
        # 字母父头 / 室温放置中层：上行保留父名，其余叶子合并进下一格
        if (lettered_leaf_split or room_leaf_split) and n_parts >= 2:
            if room_leaf_split:
                parent_txt, _ = _peel_room_storage_and_leaf(parts[0])
                if not parent_txt.endswith("日后") and not parent_txt.endswith("日"):
                    parent_txt = parts[0].split("\n")[0].strip()
                leaf_joined = "显影时的膜减少量"
                for p in parts[1:]:
                    if _EVAL_FILM_LEAF_RE.search(_compact_hdr(p)) or "显影" in p or "膜" in p:
                        leaf_joined = "显影时的膜减少量"
                        break
                    if _EVAL_SENS_LEAF_RE.match(_compact_hdr(p)):
                        leaf_joined = "感度变化" if "化" in p or "变" in p else "感度"
                        break
                split_parts = [parent_txt or parts[0], leaf_joined]
            else:
                leaf_joined = "\n".join(parts[1:])
                split_parts = [parts[0], leaf_joined]
            n_parts = 2
            # 父头只占首行，叶子占剩余行
            c["text"] = split_parts[0]
            c["row_end"] = rs
            _refresh_cell_spans(c)
            bbox = _cell_physical_bbox_x_y(c)
            cs, ce = int(c["col_start"]), int(c["col_end"])
            pool = work + added
            leaf_row = rs + 1
            # 叶子可跨到原 row_end
            slot = _atomic_header_at(pool, leaf_row, cs)
            if slot is not None and slot is not c:
                if not str(slot.get("text") or "").strip():
                    slot["text"] = split_parts[1]
                    slot["row_end"] = re
                    _refresh_cell_spans(slot)
                continue
            new_c = _empty_atomic_from_bbox(c, cs, cs, ce, bbox)
            new_c["row_start"] = leaf_row
            new_c["row_end"] = re
            new_c["row_span"] = re - leaf_row + 1
            new_c["text"] = split_parts[1]
            added.append(new_c)
            continue
        cs, ce = int(c["col_start"]), int(c["col_end"])
        c["text"] = parts[0]
        c["row_end"] = rs
        _refresh_cell_spans(c)
        bbox = _cell_physical_bbox_x_y(c)
        pool = work + added
        for i in range(1, n_parts):
            row = rs + i
            slot = _atomic_header_at(pool, row, cs)
            if slot is not None and slot is not c:
                if not str(slot.get("text") or "").strip():
                    slot["text"] = parts[i]
                continue
            new_c = _empty_atomic_from_bbox(c, cs, cs, ce, bbox)
            new_c["row_start"] = row
            new_c["row_end"] = row
            new_c["row_span"] = 1
            new_c["text"] = parts[i]
            added.append(new_c)
    if added:
        work.extend(added)
        logger.info("split_stacked_header_rowspans: 拆 %d 格", len(added))
    return work



_TRUNCATED_HEADER_TAIL_RE = re.compile(r"(的|中|与|及|、)$")
_HEADER_CONTINUATION_HEAD_RE = re.compile(
    r"^(置\d*日后|NMP|浓度|含量|质量|[×xX]\s*10)",
    re.IGNORECASE,
)


def _compact_hdr(text: str) -> str:
    return "".join((text or "").split())


def _is_truncated_header_prefix(text: str) -> bool:
    """上格像未写完的表头前缀（以的/中等收尾）。"""
    t = _compact_hdr(text)
    if not t or len(t) > 16:
        return False
    return bool(_TRUNCATED_HEADER_TAIL_RE.search(t))


def _is_composition_span_parent(text: str) -> bool:
    """「组成[质量份]」类顶栏父头（含误粘字母叶子后的多行文本）。"""
    return bool(_COMPOSITION_SPAN_HEADER_RE.search(text or ""))


def _is_header_phrase_continuation(upper: str, lower: str) -> bool:
    """下格是上格折行续文（单位、浓度、半截词等）。"""
    # (B)组分名 / 种类(量) 是父子表头，不是折行
    if _is_lettered_parent_over_leaf_subheader(upper, lower):
        return False
    # 室温放置 / 显影·感度 是中层与叶子，不是折行
    if _is_room_storage_over_eval_leaf(upper, lower):
        return False
    u, l = _compact_hdr(upper), _compact_hdr(lower)
    if not l:
        return True
    # （D）着色剂 等字母叶子：相对组成/非字母父头绝非折行续文
    if _lettered_parent_letter(l) and not _lettered_parent_letter(u):
        return False
    if _is_composition_span_parent(u) and _lettered_parent_letter(l):
        return False
    if _is_header_unit_line(lower):
        return True
    if _HEADER_CONTINUATION_HEAD_RE.match(l):
        # 裸括号开头：仅单位行（℃）等可续；字母组分叶子已在上方排除
        if l.startswith("(") or l.startswith("（"):
            if _is_header_unit_line(lower):
                return True
            if re.match(r"^[\(（][^）\)]*[\)）]$", l):
                return True
            return False
        return True
    if _is_truncated_header_prefix(u):
        cjk = len(re.findall(r"[\u4e00-\u9fff]", l))
        if cjk <= 6 and not _looks_like_independent_header_label(l):
            return True
    if _looks_like_wrapped_header_lines([upper, lower], 2):
        return True
    return False


def _header_band_atomic_nonempty(
    cells: Sequence[Dict[str, Any]], row: int
) -> List[Dict[str, Any]]:
    """表头带内某逻辑行的原子非空格。"""
    out: List[Dict[str, Any]] = []
    for c in cells:
        if c.get("_drop_render"):
            continue
        if int(c["row_start"]) != row or int(c["row_end"]) != row:
            continue
        if str(c.get("text") or "").strip():
            out.append(c)
    return out


def _is_wrap_artifact_header_band(
    cells: Sequence[Dict[str, Any]], header_end: int
) -> bool:
    """顶表头带是否像「折行被切成假多行」而非真实嵌套二级表头。

    典型：上行几乎无独立原子标签，下行有多列叶头，部分列 rowspan 跨整段
    （介电/常数、来自R1…的比例）；上行无 colspan 父组头。
    """
    if header_end < 1 or header_end > 2:
        return False
    top = min(
        (int(c["row_start"]) for c in cells if not c.get("_drop_render")),
        default=0,
    )
    if header_end <= top:
        return False

    atomic_top = _header_band_atomic_nonempty(cells, top)
    # 上行已有真实标签/父头 → 保留多级表头
    if atomic_top:
        if any(int(c["col_end"]) > int(c["col_start"]) for c in atomic_top):
            return False
        # 上行原子非空达到 2 列以上，更像真实二级
        if len(atomic_top) >= 2:
            return False

    atomic_bot = _header_band_atomic_nonempty(cells, header_end)
    if len(atomic_bot) < 2:
        return False

    tall_wraps = 0
    for c in cells:
        if c.get("_drop_render"):
            continue
        rs, re = int(c["row_start"]), int(c["row_end"])
        if rs != top or re < header_end or re <= rs:
            continue
        if str(c.get("text") or "").strip():
            tall_wraps += 1

    # 楼梯形：上行空、下行有叶头；或夹杂折行 rowspan
    return (not atomic_top) or tall_wraps >= 1


def _collapse_wrap_artifact_header_band(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把折行造成的假多行表头压成单行，保留格内换行文本。

    几何规则：上行无独立叶头、无 colspan 父组；下行有多列叶头。将表头带内
    各列文本收拢到 top 行，身行行号上移。
    """
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    if not _is_wrap_artifact_header_band(cells, header_end):
        return cells

    work = [dict(c) for c in cells]
    top = min(int(c["row_start"]) for c in work if not c.get("_drop_render"))
    shift = header_end - top
    if shift < 1:
        return cells

    # 按列范围归并表头带文本：同 (cs,ce) 取最长文本；跨行 rowspan 保留其文案
    by_cols: Dict[Tuple[int, int], Dict[str, Any]] = {}
    drop_ids: Set[int] = set()
    for c in work:
        if c.get("_drop_render"):
            continue
        rs, re = int(c["row_start"]), int(c["row_end"])
        if rs > header_end:
            continue
        if re < top:
            continue
        cs, ce = int(c["col_start"]), int(c["col_end"])
        key = (cs, ce)
        t = str(c.get("text") or "").strip()
        prev = by_cols.get(key)
        if prev is None:
            by_cols[key] = c
            continue
        pt = str(prev.get("text") or "").strip()
        # 有字优先于空格；更长文案优先；同等时保留跨行折行格
        prev_span = int(prev["row_end"]) - int(prev["row_start"])
        cur_span = re - rs
        take_cur = False
        if t and not pt:
            take_cur = True
        elif t and pt and len(t) > len(pt):
            take_cur = True
        elif t and pt and len(t) == len(pt) and cur_span > prev_span:
            take_cur = True
        elif not t and not pt and cur_span > prev_span:
            take_cur = True
        if take_cur:
            drop_ids.add(id(prev))
            by_cols[key] = c
        else:
            drop_ids.add(id(c))

    for c in by_cols.values():
        if id(c) in drop_ids:
            continue
        c["row_start"] = top
        c["row_end"] = top
        _refresh_cell_spans(c)

    # 表头带内未被选中的格丢掉；身行上移
    out: List[Dict[str, Any]] = []
    for c in work:
        if c.get("_drop_render") or id(c) in drop_ids:
            continue
        rs, re = int(c["row_start"]), int(c["row_end"])
        if re <= header_end and rs <= header_end:
            # 仅保留已压到 top 的表头格
            if rs == top and re == top:
                out.append(c)
            continue
        if rs > header_end:
            c["row_start"] = rs - shift
            c["row_end"] = re - shift
            _refresh_cell_spans(c)
            out.append(c)
            continue
        # 跨界：缩到身段
        c["row_start"] = max(top + 1, rs) - shift
        c["row_end"] = re - shift
        if int(c["row_end"]) < int(c["row_start"]):
            continue
        _refresh_cell_spans(c)
        out.append(c)

    logger.info(
        "collapse_wrap_artifact_header_band: header_end %d→%d, cells %d→%d",
        header_end,
        top,
        len(cells),
        len(out),
    )
    return out


_LEAKED_UNIT_BODY_RE = re.compile(
    r"^(?P<unit>[\(（][^\)\n|]+[\)）])\s*[\n|]?\s*(?P<rest>\S.*)$",
    re.DOTALL,
)


def _peel_leaked_header_units_from_body(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """首个表体行若以括号单位开头（如 (%)\\n73），把单位贴回正上方表头。

    折行表头被切矮后，(%) 常落到第一身格；几何上单位应属表头叶。
    """
    if not cells:
        return cells
    header_end = _top_header_band_end(cells)
    body_row = header_end + 1
    work = [dict(c) for c in cells]
    changed = 0

    for c in work:
        if c.get("_drop_render"):
            continue
        if int(c["row_start"]) != body_row or int(c["row_end"]) != body_row:
            continue
        raw = str(c.get("text") or "").strip()
        if not raw:
            continue
        m = _LEAKED_UNIT_BODY_RE.match(raw)
        if not m:
            continue
        unit = m.group("unit").strip()
        rest = m.group("rest").strip()
        if not _is_header_unit_line(unit):
            continue
        if not rest or not (
            _looks_like_data_value(rest)
            or re.match(r"^[\d\.]+", rest)
            or re.match(r"^[○●◎△×✕✖]$", rest)
        ):
            continue
        cs, ce = int(c["col_start"]), int(c["col_end"])
        # 正上方表头：同行范围覆盖本列
        header = None
        for h in work:
            if h.get("_drop_render") or h is c:
                continue
            hrs, hre = int(h["row_start"]), int(h["row_end"])
            if hre < hrs or hre > header_end or hrs > header_end:
                continue
            if int(h["col_start"]) > cs or int(h["col_end"]) < ce:
                continue
            if not str(h.get("text") or "").strip():
                continue
            header = h
            break
        if header is None:
            continue
        ht = str(header.get("text") or "").strip()
        unit_compact = "".join(unit.split())
        if unit_compact and unit_compact in "".join(ht.split()):
            c["text"] = rest
            changed += 1
            continue
        # 贴到表头末尾：的比例 + (%) → 的比例(%)
        if ht.endswith("\n") or not ht:
            header["text"] = (ht + unit).strip()
        else:
            header["text"] = ht + unit
        c["text"] = rest
        changed += 1
        logger.info(
            "peel_leaked_header_unit: %r ← body %r → %r",
            unit,
            raw[:20],
            rest[:20],
        )

    return work if changed else cells


def _looks_like_wrapped_header_pair(upper: str, lower: str) -> bool:
    """上下非空标签像同一短语折行（非两个独立列头）。"""
    if _is_lettered_parent_over_leaf_subheader(upper, lower):
        return False
    if _is_room_storage_over_eval_leaf(upper, lower):
        return False
    if _is_composition_span_parent(upper) and _lettered_parent_letter(lower):
        return False
    if _lettered_parent_letter(lower) and not _lettered_parent_letter(upper):
        return False
    # 聚合物A/B、化合物1/2：平行系列行头，不是折行
    if _looks_like_parallel_series_labels(upper, lower):
        return False
    if _is_header_phrase_continuation(upper, lower):
        return True
    u, l = _compact_hdr(upper), _compact_hdr(lower)
    if _looks_like_independent_header_label(u) and _looks_like_independent_header_label(l):
        return False
    if len(u) <= 8 and len(l) <= 8 and not _looks_like_independent_header_label(u):
        if _lettered_parent_letter(u):
            return False
        return True
    return False


def _merge_header_empty_below(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并表头中：上方有“实文本”的单行格，且正下方同列范围是空/碎片文本的单行格；
    或上下两层均为应纵向叠在同一格的中文标签（如 P40「二苯基醚/二甲酰氯」）。

    目标修复：诸如 P100X888 的“分散液”表头被切成两层的情况。
    仅在真正的顶表头带内合并，不用表体左侧大行头去扩大截止行。
    """
    if not cells:
        return cells

    header_end = _top_header_band_end(cells)
    if header_end < 1:
        return cells

    def _is_frag_or_empty(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        compact = "".join(t.split())
        # 括号包裹的单位/注释（如 (℃)、(min)、(%)）不是碎片
        if compact.startswith("(") or compact.startswith("（"):
            return False
        # 单字母等级代号（A/B/C…）是表体数据，不可当下层空碎片吞掉
        if _LETTER_DATA_RE.fullmatch(compact):
            return False
        return len(compact) <= 2 or compact in {"-", "—", "_"}

    # 用 (row_start, col_start, col_end) 精确匹配“正下方同列范围”的格
    # （注意：某些表头“空格”在 TSR/cells 中可能根本不存在，此时 renderer 会输出空 <td>。）
    by_key: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for c in cells:
        rs, re = int(c["row_start"]), int(c["row_end"])
        cs, ce = int(c["col_start"]), int(c["col_end"])
        if rs == re:  # 只在 atomic row 内尝试“向下合并”
            by_key[(rs, cs, ce)] = c

    # 判断某个“(next_row, column_range)”区域是否被任何非空 cell 占用
    def _occupied_by_nonempty(next_row: int, cs: int, ce: int) -> bool:
        for c in cells:
            rs, re = int(c["row_start"]), int(c["row_end"])
            cols_s, cols_e = int(c["col_start"]), int(c["col_end"])
            if not (rs <= next_row <= re):
                continue
            # c 的列范围至少覆盖目标列范围
            if not (cols_s <= cs and cols_e >= ce):
                continue
            if str(c.get("text") or "").strip():
                return True
        return False

    dropped_ids: Set[int] = set()
    out: List[Dict[str, Any]] = []
    for upper in cells:
        if id(upper) in dropped_ids:
            continue

        ur = int(upper["row_start"])
        ue = int(upper["row_end"])
        uc_s = int(upper["col_start"])
        uc_e = int(upper["col_end"])
        upper_text = str(upper.get("text") or "").strip()

        # 只合并顶表头带内。下一行若已出表头带（进入表体），不得把列头/数据格向下吞掉。
        if ue + 1 > header_end:
            out.append(upper)
            continue
        is_colspan_parent = uc_s != uc_e
        if not upper_text or _is_frag_or_empty(upper_text):
            # 空角/碎片不在这里纵向合并：上下皆空就消横线会误伤「应保留的空角格线」。
            # 与右侧 rowspan 对齐的空角占位改由渲染阶段 _empty_td_rowspan_for_gap 处理。
            out.append(upper)
            continue
        # 上层 colspan 组头覆盖多列 → 子列头不得纵向叠字合并（P32 溶解性/粘度）
        under_colspan_parent = False
        if ur > 0:
            for pc in cells:
                prs, pre = int(pc["row_start"]), int(pc["row_end"])
                if not (prs <= ur - 1 <= pre):
                    continue
                pcs, pce = int(pc["col_start"]), int(pc["col_end"])
                if pce > pcs and pcs <= uc_s <= pce:
                    under_colspan_parent = True
                    break

        # 同级 colspan 不再解除 under_colspan_parent：
        # 否则会把「室温放置」中层碎片误 rowspan 进叶子行；折行续接在下方单独处理。

        # 实施例/比较例列头不得向下吞并（下层常为 A/B/C 等数据字母）
        if _EXAMPLE_COL_HEADER_RE.search(upper_text):
            out.append(upper)
            continue

        # 跨空行找续文：组合物中的 →(空)→ NMP；酸的含量 →(空)→ 单位行。
        # 即使 upper 已有 rowspan（TSR 先扩到空中层），仍可继续向下吞续文。
        cont_cell = None
        for look_row in range(ue + 1, header_end + 1):
            cand = by_key.get((look_row, uc_s, uc_e))
            if cand is None:
                if _occupied_by_nonempty(look_row, uc_s, uc_e):
                    break
                continue
            if id(cand) in dropped_ids:
                break
            ct = str(cand.get("text") or "").strip()
            if not ct or _is_frag_or_empty(ct):
                continue
            if _is_header_phrase_continuation(upper_text, ct):
                cont_cell = cand
            break
        if cont_cell is not None:
            ct = str(cont_cell.get("text") or "").strip()
            upper["text"] = upper_text + "\n" + ct if upper_text else ct
            upper["row_end"] = int(cont_cell["row_end"])
            upper["row_span"] = int(upper["row_end"]) - int(upper["row_start"]) + 1
            dropped_ids.add(id(cont_cell))
            cont_cell["_drop_render"] = True
            for mid_r in range(ue + 1, int(cont_cell["row_start"])):
                mid = by_key.get((mid_r, uc_s, uc_e))
                if mid is not None and id(mid) not in dropped_ids:
                    dropped_ids.add(id(mid))
                    mid["_drop_render"] = True
            out.append(upper)
            continue

        # colspan 父头：仅当中间行在本列范围内完全无非空格时，才向下扩 rowspan
        # （通式…化合物 下空洞）；若范围内已有 (A)/(B) 等子头则不得吞掉。
        if is_colspan_parent:
            from ..structure.header_topology import _looks_like_lettered_category_parent

            if _lettered_parent_letter(upper_text) or _looks_like_lettered_category_parent(
                upper_text
            ):
                out.append(upper)
                continue
            def _any_nonempty_overlap(next_row: int, cs: int, ce: int) -> bool:
                for c in cells:
                    if id(c) in dropped_ids or c is upper:
                        continue
                    rs, re = int(c["row_start"]), int(c["row_end"])
                    if not (rs <= next_row <= re):
                        continue
                    ocs, oce = int(c["col_start"]), int(c["col_end"])
                    if oce < cs or ocs > ce:
                        continue
                    if str(c.get("text") or "").strip():
                        return True
                return False

            expand_to = ue
            for look_row in range(ue + 1, header_end):
                if _any_nonempty_overlap(look_row, uc_s, uc_e):
                    break
                expand_to = look_row
            if expand_to > ue:
                upper["row_end"] = expand_to
                upper["row_span"] = int(upper["row_end"]) - int(upper["row_start"]) + 1
            out.append(upper)
            continue

        # 已有 rowspan 的格：除续文合并不再做空行扩展/叠字合并
        if ue != ur:
            out.append(upper)
            continue

        lower = by_key.get((ue + 1, uc_s, uc_e))
        if lower is None:
            # 下方没有 cell 对象时，renderer 会输出空 td；若没有任何非空 cell 占用该区域，则直接扩展 upper rowspan
            next_row = ue + 1
            if not _occupied_by_nonempty(next_row, uc_s, uc_e):
                if _column_gap_between_subheaders(
                    cells, next_row, uc_s, header_end=header_end
                ):
                    out.append(upper)
                    continue
                if _lettered_parent_letter(upper_text):
                    out.append(upper)
                    continue
                # colspan 父组下的中层格不要 rowspan 进叶子空位
                if under_colspan_parent:
                    out.append(upper)
                    continue
                # 残尾前缀先等续文，勿单独吞空行（否则挡掉更下方的 NMP/单位）
                if _is_truncated_header_prefix(upper_text):
                    out.append(upper)
                    continue
                upper["row_end"] = next_row
                upper["row_span"] = int(upper["row_end"]) - int(upper["row_start"]) + 1
            out.append(upper)
            continue

        lower_text = str(lower.get("text") or "").strip()

        from ..utils.label_patterns import is_named_row_label

        if is_named_row_label(lower_text):
            out.append(upper)
            continue

        if not _is_frag_or_empty(lower_text):
            from ..structure.header_topology import (
                _looks_like_kind_or_amount_leaf,
                _looks_like_numeric_metric_leaf,
            )
            from ..utils.label_patterns import looks_like_body_data_token

            if _looks_like_kind_or_amount_leaf(upper_text) and looks_like_body_data_token(
                lower_text
            ):
                out.append(upper)
                continue

            if _is_known_leaf_subheader(lower_text) or _looks_like_kind_or_amount_leaf(
                lower_text
            ):
                if not _lettered_parent_letter(upper_text):
                    out.append(upper)
                    continue
            if _looks_like_numeric_metric_leaf(lower_text):
                out.append(upper)
                continue
            if (
                _is_stacked_header_label(upper_text)
                and _is_stacked_header_label(lower_text)
            ):
                # 平行系列行头（聚合物A/B）禁止合并；其余沿用折行/边界启发式
                if _looks_like_parallel_series_labels(upper_text, lower_text):
                    out.append(upper)
                    continue
                # 独立双标签保持分列；折行短语对才合并
                if (
                    int(lower["row_start"]) == header_end
                    and ur < header_end
                    and int(lower["col_start"]) == uc_s
                    and not under_colspan_parent
                    and not _looks_like_wrapped_header_pair(upper_text, lower_text)
                ):
                    out.append(upper)
                    continue
                if under_colspan_parent and not _looks_like_wrapped_header_pair(
                    upper_text, lower_text
                ):
                    out.append(upper)
                    continue
                upper["text"] = (
                    upper_text + "\n" + lower_text
                    if upper_text
                    else lower_text
                )
                upper["row_end"] = int(lower["row_end"])
                upper["row_span"] = int(upper["row_end"]) - int(upper["row_start"]) + 1
                dropped_ids.add(id(lower))
                out.append(upper)
                continue
            out.append(upper)
            continue

        # 空/碎片下层：colspan 父头下的中层格不得 rowspan 进叶子行
        if under_colspan_parent:
            out.append(upper)
            continue
        # 残尾前缀不单独吞空行
        if _is_truncated_header_prefix(upper_text):
            out.append(upper)
            continue

        # 合并：把 lower 的行范围并入 upper，丢弃 lower
        upper["row_end"] = int(lower["row_end"])
        upper["row_span"] = int(upper["row_end"]) - int(upper["row_start"]) + 1
        dropped_ids.add(id(lower))
        out.append(upper)

    return out

def _escape_cell_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # 保留格内换行
    parts = t.split("\n")
    return "<br>".join(html.escape(p) for p in parts)


def _empty_td_rowspan_for_gap(
    r: int,
    c: int,
    *,
    n_rows: int,
    origin: Dict[Tuple[int, int], Dict[str, Any]],
    covered: Set[Tuple[int, int]],
) -> int:
    """空位占位 rowspan：仅当右侧邻格从本行起 rowspan>1，且本列对应行均为空位。

    用于消除「聚合物」左侧被拆成两个空 td 而凭空产生的横线；
    不根据「上下皆空」消线——右侧若是两行独立格，仍输出两个空 td。
    """
    right = origin.get((r, c + 1))
    if right is None:
        return 1
    if not str(right.get("text") or "").strip():
        return 1
    rs = int(right["row_start"])
    re = int(right["row_end"])
    if rs != r:
        return 1
    span = re - rs + 1
    if span <= 1:
        return 1
    for rr in range(r, min(re, n_rows - 1) + 1):
        if (rr, c) in covered or (rr, c) in origin:
            return 1
    return min(span, n_rows - r)


def cells_to_html_table(
    cells: List[Dict[str, Any]],
    *,
    compress_empty: bool = True,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """
    将带逻辑拓扑的单元格列表渲染为单个 <table>。

    占位矩阵：起始格写 td（带 rowspan/colspan），被 span 覆盖的位置跳过。
    """
    if not cells:
        return ""

    work = [dict(c) for c in cells]
    work = _split_example_header_rowspans(work)
    from ..structure.header_topology import repair_parent_header_topology as _topo_pre
    from ..structure.tsr_refine import resolve_logic_grid_conflicts

    # 先按子表头列组恢复父头 colspan，再消解重叠，避免 (A)(B)(C) 等被拼成宽格
    work = _topo_pre(work, list(text_boxes or []))
    work = resolve_logic_grid_conflicts(work, text_boxes)
    work = _resolve_logic_overlaps(work, text_boxes)
    # 先丢掉全空幽灵行（P24 顶上空行），再算表头带 / 拆身列 colspan
    work = _drop_fully_empty_rows(work)
    work = _drop_leading_header_only_columns(work)
    work = _split_label_over_data_columns(work)
    work = _extend_sparse_section_trailing_colspans(work)
    work = _split_header_over_index_column(work)
    work = _unglue_composition_side_and_span(work)
    work = _merge_split_composition_span_fragments(work)
    work = _collapse_header_empty_corners(work)
    work = drop_evidenceless_columns(work)
    work = _align_orphan_left_anchor_headers(work)
    from ..structure.header_topology import repair_parent_header_topology

    work = repair_parent_header_topology(work, list(text_boxes or []))
    work = _drop_body_empty_columns(work)
    work = _merge_leading_empty_into_label(work)
    work = _merge_split_body_side_labels(work)
    work = _merge_leading_label_gaps(work)
    work = drop_evidenceless_rows(work)
    work = _drop_fully_empty_rows(work)
    work = drop_noise_rows(work)
    work = _split_merged_example_body_rows(work)
    work = _peel_glued_example_index_columns(work)
    work = _restore_composition_parent_header(work)
    work = _extend_composition_parent_colspan(work)
    if compress_empty:
        work = compress_empty_logic_columns(work)
        work = compress_empty_logic_rows(work)
    work = _split_mismerged_unit_rowspans(work)
    work = _merge_header_unit_rows(work)
    work = _normalize_nested_column_group_headers(work)
    work = _normalize_tiered_column_headers(work)
    from ..structure.header_topology import repair_parent_header_topology as _topo_late

    work = _topo_late(work, list(text_boxes or []))
    work = _merge_formula_parent_colspans(work)
    work = _ensure_composition_parent_header(work)
    work = _extend_composition_parent_colspan(work)
    work = _promote_tall_header_rowspans(work)
    work = _collapse_wrap_artifact_header_band(work)
    work = _peel_leaked_header_units_from_body(work)
    work = _split_stacked_header_rowspans(work)
    work = _repair_room_storage_eval_headers(work)
    work = _fold_acid_times_shard_column(work)
    work = _merge_header_empty_below(work)
    work = _extend_composition_parent_colspan(work)
    work = _extend_sparse_section_trailing_colspans(work)
    work = _repair_room_storage_eval_headers(work)
    work = _retract_value_rowspan_across_side_indexes(work)
    work = [c for c in work if not c.get("_drop_render")]
    if not work:
        return ""

    max_row = max(int(c["row_end"]) for c in work)
    max_col = max(int(c["col_end"]) for c in work)
    n_rows = max_row + 1
    n_cols = max_col + 1

    # (r,c) -> cell dict at origin; covered positions marked
    origin: Dict[Tuple[int, int], Dict[str, Any]] = {}
    covered: Set[Tuple[int, int]] = set()

    # 去重：同一逻辑矩形只保留一份（文本更长者优先）
    by_logic: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    for cell in work:
        key = _cell_key(cell)
        prev = by_logic.get(key)
        if prev is None:
            by_logic[key] = cell
            continue
        t_new = str(cell.get("text") or "")
        t_old = str(prev.get("text") or "")
        if len(t_new) > len(t_old):
            by_logic[key] = cell

    for cell in by_logic.values():
        rs, re = int(cell["row_start"]), int(cell["row_end"])
        cs, ce = int(cell["col_start"]), int(cell["col_end"])
        if rs < 0 or cs < 0 or rs >= n_rows or cs >= n_cols:
            continue
        # 若起点已被覆盖：把文本并入覆盖者
        if (rs, cs) in covered or (rs, cs) in origin:
            text = str(cell.get("text") or "").strip()
            if text:
                owner = origin.get((rs, cs))
                if owner is None:
                    # 找任意覆盖该点的已登记格
                    for (or_, oc_), oc in origin.items():
                        ore, oce = int(oc["row_end"]), int(oc["col_end"])
                        if or_ <= rs <= ore and oc_ <= cs <= oce:
                            owner = oc
                            break
                if owner is not None:
                    prev = str(owner.get("text") or "").strip()
                    if text not in prev:
                        owner["text"] = (prev + " " + text).strip() if prev else text
                        logger.warning(
                            "渲染冲突，文本并入 (%s,%s): %r",
                            owner.get("row_start"),
                            owner.get("col_start"),
                            text[:40],
                        )
            continue
        origin[(rs, cs)] = cell
        for r in range(rs, min(re, n_rows - 1) + 1):
            for c in range(cs, min(ce, n_cols - 1) + 1):
                if (r, c) != (rs, cs):
                    covered.add((r, c))

    lines: List[str] = ['<table border="1" cellspacing="0" cellpadding="4">']
    for r in range(n_rows):
        lines.append("<tr>")
        for c in range(n_cols):
            if (r, c) in covered:
                continue
            cell = origin.get((r, c))
            if cell is None:
                # 右侧邻格已 rowspan 覆盖多行、且本列对应行均为空位时，空角与之对齐，
                # 避免在「聚合物」左侧画出原稿没有的横线。依据邻格结构，非「上下空→消线」。
                stub_span = _empty_td_rowspan_for_gap(
                    r, c, n_rows=n_rows, origin=origin, covered=covered
                )
                if stub_span > 1:
                    lines.append(f'<td rowspan="{stub_span}"></td>')
                    for rr in range(r + 1, r + stub_span):
                        covered.add((rr, c))
                else:
                    lines.append("<td></td>")
                continue
            rs, re = int(cell["row_start"]), int(cell["row_end"])
            cs, ce = int(cell["col_start"]), int(cell["col_end"])
            row_span = max(re - rs + 1, 1)
            col_span = max(ce - cs + 1, 1)
            attrs = []
            if row_span > 1:
                attrs.append(f'rowspan="{row_span}"')
            if col_span > 1:
                attrs.append(f'colspan="{col_span}"')
            attr_s = (" " + " ".join(attrs)) if attrs else ""
            text = _escape_cell_text(str(cell.get("text") or ""))
            lines.append(f"<td{attr_s}>{text}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def cells_to_html(
    cells: List[Dict[str, Any]],
    *,
    split_subtables: bool = True,
    compress_empty: bool = True,
    lift_captions: bool = True,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """多子表切分后各自生成 HTML，表题提升为段落。"""
    if split_subtables:
        subtables = split_cells_into_subtables(cells)
    else:
        subtables = [cells] if cells else []

    parts: List[str] = []
    for sub in subtables:
        caption = ""
        work = sub
        if lift_captions:
            caption, work = extract_caption_row(work)
        if not any(str(c.get("text") or "").strip() for c in work):
            if caption:
                parts.append(f"<p>{html.escape(caption)}</p>")
            continue
        table_html = cells_to_html_table(
            work, compress_empty=compress_empty, text_boxes=text_boxes
        )
        if not table_html:
            if caption:
                parts.append(f"<p>{html.escape(caption)}</p>")
            continue
        if caption:
            parts.append(f"<p>{html.escape(caption)}</p>\n{table_html}")
        else:
            parts.append(table_html)
    return "\n\n".join(parts)


def _drop_duplicate_caption_paragraphs(table_html: str, prefix_keys: Set[str]) -> str:
    """去掉与前缀 free_text 重复的表内题注段。"""
    if not prefix_keys or not table_html:
        return table_html
    kept: List[str] = []
    for line in table_html.split("\n"):
        if line.startswith("<p>") and line.endswith("</p>"):
            inner = html.unescape(line[3:-4])
            if _caption_norm_key(inner) in prefix_keys:
                continue
        kept.append(line)
    return "\n".join(kept)


def build_html_output(
    cells: List[Dict[str, Any]],
    free_texts: List[Dict[str, Any]],
    *,
    split_subtables: bool = True,
    compress_empty: bool = True,
    text_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """游离文本前缀 + HTML 表格。结构失败时只输出游离文本块。"""
    prefix = format_free_texts(free_texts)
    prefix_keys = {
        _caption_norm_key(line)
        for line in prefix.splitlines()
        if _caption_norm_key(line)
    }
    paras = [html.escape(line) for line in prefix.splitlines() if line.strip()] if prefix else []
    prefix_html = "\n".join(f"<p>{p}</p>" for p in paras)

    has_cell_text = any(str(c.get("text") or "").strip() for c in cells)
    if not has_cell_text:
        return prefix_html

    table = cells_to_html(
        cells,
        split_subtables=split_subtables,
        compress_empty=compress_empty,
        text_boxes=text_boxes,
    )
    if prefix_keys:
        table = _drop_duplicate_caption_paragraphs(table, prefix_keys)
    if prefix_html and table:
        return prefix_html + "\n\n" + table
    return prefix_html or table or ""


# 专利实验表常见列头（含感光组合物表：碱溶/敏感度/密合…）
_NORMAL_TABLE_HEADER_RE = re.compile(
    r"(组合物|组成\s*[\[［]|灵敏?度|敏感度|聚合物|单体|"
    r"碱溶|交联|溶剂|密合|醌二叠氮|耐化学|剥离|通式|树脂|"
    r"显影|感光|清漆|判定)"
)
_EXAMPLE_LABEL_TOKEN_RE = re.compile(
    r"(实[施試]例|比[较較]例|合成例|对照例|参考例)"
)
# 误转 90° 后首行常变成等级字母堆：A / B / A<br>A / C B<br>A A
_GRADE_STACK_CELL_RE = re.compile(
    r"^[ABC](?:\s*[ABC]){0,3}$|^[ABC](?:/[ABC])+$",
    re.IGNORECASE,
)
_RECIPE_MARKER_RE = re.compile(
    r"\b(?:GBL|HMOM|PGMEA|NMP)\b|(?:^|[^A-Za-z])(?:A|B|C)-\d",
    re.IGNORECASE,
)
# 配方矩阵侧栏：成分代号 / 溶剂行标 / 分组父头（几何上应在表体左侧纵排）
_COMPONENT_ROW_LABEL_RE = re.compile(
    r"(?:"
    r"\(?[A-Ea-e]\)?\s*[-－–]?\s*\d{1,2}"
    r"|聚合物|引发剂|单体|溶剂|敏化剂|抑制剂|成分"
    r"|NMP|GBL|DMSO"
    r"|焦点|截面|电特性|合格|不合格"
    r")"
)
# 横置 OCR 常见乱码碎片（非正常专利表词）
_SIDEWAYS_OCR_GARBAGE_RE = re.compile(
    r"(成\d|后房|品房|全房|房合|电话|产分|产\))"
)


def _html_td_plain(cell_html: str) -> str:
    s = re.sub(r"<br\s*/?>", "", cell_html or "")
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", "", s)


def _html_table_row_plains(table_html: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for tr in re.findall(r"<tr>(.*?)</tr>", table_html or "", re.DOTALL):
        tds = re.findall(r"<td[^>]*>([\s\S]*?)</td>", tr)
        rows.append([_html_td_plain(t) for t in tds])
    return rows


def _count_body_example_row_labels(rows: Sequence[Sequence[str]]) -> int:
    """表体首列是实施例/比较例行标签的行数（直立实验表的强信号）。"""
    if len(rows) < 2:
        return 0
    n = 0
    for plains in rows[1:]:
        if not plains:
            continue
        head = plains[0]
        if _EXAMPLE_LABEL_TOKEN_RE.search(head) and len(head) <= 16:
            n += 1
    return n


def _count_row0_example_labels(rows: Sequence[Sequence[str]]) -> int:
    """首行实施例/比较例单元格数（配方矩阵列头的强信号）。"""
    if not rows:
        return 0
    return sum(
        1 for p in rows[0] if _EXAMPLE_LABEL_TOKEN_RE.search(p or "")
    )


def _count_body_component_row_labels(rows: Sequence[Sequence[str]]) -> int:
    """
    表体左侧（前两列）成分/溶剂/指标行标签数。

    与「实施例作列头」正交：配方矩阵直立时侧栏纵排 (A)-1 / 聚合物 / NMP，
    真侧躺碎表则侧栏多为数字或 OCR 乱码。
    """
    if len(rows) < 3:
        return 0
    n = 0
    for plains in rows[1:]:
        if not plains:
            continue
        left = " ".join(plains[:2])
        if _EXAMPLE_LABEL_TOKEN_RE.search(left):
            continue
        if _COMPONENT_ROW_LABEL_RE.search(left):
            n += 1
    return n


def _looks_like_composition_matrix(rows: Sequence[Sequence[str]]) -> bool:
    """
    几何拓扑：实施例横作列头 + 侧栏成分纵排 + 足够行数 → 正常配方矩阵，非侧躺。
    """
    row0_ex = _count_row0_example_labels(rows)
    if row0_ex < 6:
        return False
    n_rows = len(rows)
    if n_rows < 8:
        return False
    return _count_body_component_row_labels(rows) >= 4


def html_structure_health(html_text: str) -> float:
    """
    表格结构健康分（越高越好）。定向重试时用于比较，避免「误报 broken 的好表」
    被「漏报 broken 的侧躺碎表」替换。
    """
    if not html_text or "<table" not in html_text:
        return -100.0
    first = re.search(r"<table[^>]*>(.*?)</table>", html_text, re.DOTALL)
    if not first:
        return -50.0
    table = first.group(1)
    rows = _html_table_row_plains(table)
    n_rows = len(rows)
    n_cols = max((len(r) for r in rows), default=0)
    score = 0.0
    if not html_output_looks_broken(html_text):
        score += 20.0
    else:
        score -= 15.0
    body_labels = _count_body_example_row_labels(rows)
    score += min(body_labels, 25) * 1.2
    # 配方矩阵：实施例作列头 + 侧栏成分纵排（P186 类）
    if _looks_like_composition_matrix(rows):
        score += 18.0
    else:
        comp_side = _count_body_component_row_labels(rows)
        row0_ex = _count_row0_example_labels(rows)
        score += min(comp_side, 12) * 0.8
        if row0_ex >= 6:
            score += min(row0_ex, 20) * 0.35
    # 直立实验表：行数明显多于列数
    if n_rows >= 10 and n_rows >= n_cols:
        score += 6.0
    # 侧躺配方倾倒：极少行、极多列
    if n_rows <= 6 and n_cols >= 12:
        score -= 12.0
    if _NORMAL_TABLE_HEADER_RE.search(table[:2000] or ""):
        score += 4.0
    # 表内几乎看不到行标签，却到处是配方溶剂/等级字母
    ex_all = len(_EXAMPLE_LABEL_TOKEN_RE.findall(table))
    recipe = len(_RECIPE_MARKER_RE.findall(table))
    if ex_all <= 1 and recipe >= 8 and n_cols >= 10:
        score -= 10.0
    # 横置 OCR 乱码：无实施例/成分词却有大量「房/成N」碎片
    garbage = len(_SIDEWAYS_OCR_GARBAGE_RE.findall(table))
    if ex_all == 0 and garbage >= 3:
        score -= 20.0
    elif ex_all == 0 and n_rows >= 8 and not _NORMAL_TABLE_HEADER_RE.search(table):
        if n_cols > n_rows * 2.0:
            score -= 12.0
    if n_rows >= 12 and n_cols >= 5 and n_rows > n_cols * 1.15:
        score += 10.0
    return score


def html_output_looks_broken(html_text: str) -> bool:
    """
    检测侧躺/行列颠倒等灾难性 HTML（大量空行、异常 rowspan、实施例横排成表）。

    用于在误转 90° 等场景下触发定向重试，避免污染正常表格输出。

    关键（P26X199 类）：
    - 直立感光组合物表：表体大量「实施例N」行标签 +「碱溶性树脂/敏感度」列头
      → 旧逻辑把「实施例≥8 且无组合物/聚合物」误判为侧躺，触发 90° 重试后更糟。
    - 误转 90° 后：极少行、极多列，首行等级字母堆、单元格内 GBL/HMOM 竖叠，
      且几乎无实施例行标签 → 必须判坏。
    """
    if not html_text or "<table" not in html_text:
        return False
    empty_tr = html_text.count("<tr>\n</tr>") + html_text.count("<tr></tr>")
    # 大量空 <tr>：rowspan 把整行撑空，是侧躺碎网格的强信号
    if empty_tr >= 5:
        return True
    # 大量游离短段落（侧躺后 OCR 掉出表外）
    free_paras = re.findall(r"<p>([^<]{1,12})</p>", html_text)
    if len(free_paras) >= 8 and sum(1 for p in free_paras if re.fullmatch(r"[A-Za-z0-9+＋.]+|100|A\+?", p)) >= 6:
        return True
    first_table = re.search(r"<table[^>]*>(.*?)</table>", html_text, re.DOTALL)
    if not first_table:
        return False
    table = first_table.group(1)
    rows = _html_table_row_plains(table)
    n_rows = len(rows)
    n_cols = max((len(r) for r in rows), default=0)
    body_labels = _count_body_example_row_labels(rows)

    # 游离短段：仅当行列比也异常时才判坏（避免英文表 caption 误杀）
    if len(free_paras) >= 5:
        blob = re.sub(r"\s+", "", html_text)
        if not re.search(r"(组合物|组成\[|聚合物|单体\s*[\[［]|实施例|合成例)", blob):
            if n_rows < 8 or n_cols > n_rows * 1.8:
                return True

    # 直立实验表：表体首列一串实施例/比较例 → 健康，勿因「表内实施例多」误杀
    if body_labels >= 5 and empty_tr < 5:
        return False

    # 直立配方矩阵：实施例作列头 + 侧栏成分纵排 → 健康（勿当侧躺）
    if _looks_like_composition_matrix(rows) and empty_tr < 5:
        return False

    # 直立宽表（语言无关）：行数≥列数且足够大 → 健康
    if n_rows >= 10 and n_cols >= 4 and n_rows >= n_cols and empty_tr < 5:
        return False

    chunk = table[:2500]
    ex_hits = len(re.findall(r"实[施試]例", chunk))
    has_normal_header = bool(_NORMAL_TABLE_HEADER_RE.search(chunk))
    comp_side = _count_body_component_row_labels(rows)

    # 横置 OCR 乱码大表：无实施例/正常表头，却有大量「房/成N」碎片
    if (
        ex_hits == 0
        and not has_normal_header
        and len(_SIDEWAYS_OCR_GARBAGE_RE.findall(table)) >= 3
        and n_rows >= 8
    ):
        return True

    # 误转 90° 结构侧躺（语言无关）：列数远大于行数
    if (
        n_rows >= 3
        and n_cols >= 10
        and n_cols > n_rows * 2.3
        and body_labels <= 1
        and empty_tr >= 1
    ):
        return True

    # 误转 90° 配方倾倒：极少行 + 极多列 + 等级字母/溶剂堆，几乎无行标签
    if n_rows <= 6 and n_cols >= 12 and body_labels <= 1:
        grade_cells = 0
        for plains in rows[:2]:
            for p in plains:
                if _GRADE_STACK_CELL_RE.fullmatch(p or ""):
                    grade_cells += 1
        recipe_hits = len(_RECIPE_MARKER_RE.findall(table))
        if grade_cells >= 6 or recipe_hits >= 8:
            return True
        # 首行几乎全是单字母/空/短碎片
        if rows:
            shortish = sum(
                1
                for p in rows[0]
                if (not p)
                or _GRADE_STACK_CELL_RE.fullmatch(p)
                or (len(p) <= 2 and re.fullmatch(r"[A-Za-z0-9品]+", p or ""))
            )
            if shortish >= max(8, int(len(rows[0]) * 0.6)):
                return True

    # 仅当「大量实施例却完全不见列头词、也无侧栏成分」才判坏
    # （表体行标签 / 配方矩阵已在上方 early-return）
    if ex_hits >= 8 and not has_normal_header and body_labels < 3 and comp_side < 3:
        return True

    if rows:
        row0 = rows[0]
        row0_ex = _count_row0_example_labels(rows)
        # 首行多个「实施例」但侧栏无成分拓扑、且行很少 → 真侧躺列头
        if row0_ex >= 6 and comp_side < 3 and n_rows <= 8:
            return True
        # 首行多数为纯数字/空，且整表缺少表头词 → 横置碎表
        num_or_empty = sum(
            1
            for p in row0
            if not p or re.fullmatch(r"\d+(?:\.\d+)?", p)
        )
        if (
            len(row0) >= 8
            and num_or_empty >= max(6, int(len(row0) * 0.7))
            and not _NORMAL_TABLE_HEADER_RE.search(chunk)
            and not _EXAMPLE_LABEL_TOKEN_RE.search(chunk)
        ):
            return True
    large_rowspan = len(re.findall(r'rowspan="(?:[6-9]|1[0-9]+)"', chunk))
    if large_rowspan >= 8 and empty_tr >= 2:
        return True
    return False