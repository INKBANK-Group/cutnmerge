"""表格图像方向归正：投影剖面定轴向 + OCR 置信度消歧 180°。

必须在 OCR 之前完成轴向旋转；180° 消歧可在首次 OCR 之后按需再跑一次。

独立 CLI 用法（纠正结果原地覆盖原图，原图不可恢复）:
    python src/preprocess/orient.py           # 批量纠正 data/input/ 下全部图片
    python src/preprocess/orient.py 图片路径   # 仅纠正该图片
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ROTATE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """按顺时针角度旋转（0/90/180/270）。"""
    angle = int(angle) % 360
    if angle == 0:
        return image
    code = _ROTATE.get(angle)
    if code is None:
        raise ValueError(f"不支持的旋转角: {angle}")
    return cv2.rotate(image, code)


def _projection_periodicity(profile: np.ndarray) -> float:
    """投影剖面周期性：相邻差分能量 + 零值间隙数。"""
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 8:
        return 0.0
    k = max(3, min(9, p.size // 40 * 2 + 1))
    kernel = np.ones(k, dtype=np.float64) / k
    sm = np.convolve(p, kernel, mode="same")
    mx = float(sm.max())
    if mx <= 1e-6:
        return 0.0
    sm = sm / mx
    diff = np.diff(sm)
    energy = float(np.sum(diff * diff))
    thresh = 0.08
    low = sm < thresh
    gaps = 0
    in_gap = False
    min_gap = max(2, p.size // 80)
    run = 0
    for v in low:
        if v:
            run += 1
            if not in_gap and run >= min_gap:
                in_gap = True
                gaps += 1
        else:
            run = 0
            in_gap = False
    return energy * (1.0 + 0.15 * gaps)


def estimate_axis_rotation(image: np.ndarray) -> int:
    """
    弱几何先验：仅在「去线后字块明显偏高」时建议 90，否则 0。

    真正的轴向以 OCR 框宽高比 / 置信度为准（见 ensure_upright_axis）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 15), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 15)))
    text_mask = cv2.subtract(
        binary,
        cv2.bitwise_or(
            cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk),
            cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk),
        ),
    )
    row_score = _projection_periodicity(text_mask.sum(axis=1).astype(np.float64))
    col_score = _projection_periodicity(text_mask.sum(axis=0).astype(np.float64))
    h_line = float(cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk).sum()) / 255.0 / max(w, 1)
    v_line = float(cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk).sum()) / 255.0 / max(h, 1)
    logger.info(
        "方向轴向先验: textR=%.3f textC=%.3f hLine=%.1f vLine=%.1f",
        row_score,
        col_score,
        h_line,
        v_line,
    )
    # 侧躺：文字呈竖向条带（row 周期强）且竖线墨迹更重
    if row_score > col_score * 1.35 and v_line > h_line * 1.2:
        return 90
    return 0


def _mean_ocr_score(text_boxes: list) -> float:
    if not text_boxes:
        return 0.0
    scores = [float(tb.get("score") or 0.0) for tb in text_boxes]
    return float(np.mean(scores)) if scores else 0.0


def _text_baseline_angle_deg(poly: np.ndarray) -> float:
    """OCR 四边形最长边方向角，归一化到 [0, 90]（0=横排，90=竖排）。"""
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 4:
        return 0.0
    best_len = 0.0
    best_ang = 0.0
    for i in range(4):
        p0, p1 = pts[i], pts[(i + 1) % 4]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        leng = math.hypot(dx, dy)
        if leng <= best_len:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        while ang > 90.0:
            ang -= 180.0
        while ang < -90.0:
            ang += 180.0
        best_len = leng
        best_ang = abs(ang)
    return best_ang


def _weighted_median_baseline_angle(
    text_boxes: list,
    *,
    image: Optional[np.ndarray] = None,
) -> float:
    """按文本长度加权的中位基线角；排除左侧窄列竖排框。"""
    if not text_boxes:
        return 0.0
    img_w = float(image.shape[1]) if image is not None and image.size else 0.0
    samples: List[Tuple[float, float]] = []
    for tb in text_boxes:
        poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
        if poly.size < 4:
            continue
        bw = float(poly[:, 0].max() - poly[:, 0].min())
        bh = float(poly[:, 1].max() - poly[:, 1].min())
        cx = float(0.5 * (poly[:, 0].min() + poly[:, 0].max()))
        if img_w > 0 and cx < img_w * 0.12 and bh > bw * 2.0 and bw < img_w * 0.10:
            continue
        text_len = len(str(tb.get("text") or "").strip())
        weight = max(1.0, min(float(text_len), 10.0))
        samples.append((weight, _text_baseline_angle_deg(poly)))
    if not samples:
        return 0.0
    samples.sort(key=lambda t: t[1])
    total_w = sum(w for w, _ in samples)
    half = total_w * 0.5
    acc = 0.0
    for w, ang in samples:
        acc += w
        if acc >= half:
            return ang
    return samples[-1][1]


def _box_aspect_upright_ratio(
    text_boxes: list,
    *,
    image: Optional[np.ndarray] = None,
) -> float:
    """按文本长度加权的横长框占比（长文本框权重更高，天然压制单字符干扰）。"""
    if not text_boxes:
        return 0.0

    img_w = float(image.shape[1]) if image is not None and image.size else 0.0
    total_weight = 0.0
    upright_weight = 0.0

    for tb in text_boxes:
        poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
        if poly.size < 4:
            continue
        w = float(poly[:, 0].max() - poly[:, 0].min())
        h = float(poly[:, 1].max() - poly[:, 1].min())
        cx = float(0.5 * (poly[:, 0].min() + poly[:, 0].max()))
        if img_w > 0 and cx < img_w * 0.12 and h > w * 2.0 and w < img_w * 0.10:
            continue

        text_len = len(str(tb.get("text") or "").strip())
        weight = max(1.0, min(float(text_len), 10.0))

        total_weight += weight
        if w >= h * 0.85:
            upright_weight += weight

    return (upright_weight / total_weight) if total_weight > 0 else 0.0


def _orientation_quality(
    text_boxes: list,
    *,
    image: Optional[np.ndarray] = None,
) -> float:
    """综合分：置信度 + 横长框占比。"""
    if not text_boxes:
        return 0.0
    return _mean_ocr_score(text_boxes) + 0.25 * _box_aspect_upright_ratio(
        text_boxes, image=image
    )


def _grid_line_strengths(image: np.ndarray) -> Tuple[float, float]:
    """返回 (hline_score, vline_score)，大致为线墨迹占比。"""
    if image is None or getattr(image, "size", 0) == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return 0.0, 0.0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 25, 15), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 25, 15)))
    hmask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vmask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    area = float(h * w)
    return float(np.count_nonzero(hmask)) / area, float(np.count_nonzero(vmask)) / area


def has_strong_upright_wired_grid(image: np.ndarray) -> bool:
    """
    有线网格整体直立：横竖线都明显，且横线不弱于竖线。

    用于否决「仅因竖排表头字」导致的整图 90° 误转（含横版专利表 P26X100）。
    """
    if image is None or getattr(image, "size", 0) == 0:
        return False
    hs, vs = _grid_line_strengths(image)
    if hs < 0.004 or vs < 0.003:
        return False
    return hs >= vs * 0.55


def _caption_band_is_horizontal(
    text_boxes: list,
    image: np.ndarray,
    *,
    angle_thresh: float = 20.0,
) -> bool:
    """图像顶部题注区（如 [0122] [Table 1]）基线是否近水平。"""
    if image is None or not text_boxes:
        return False
    h = float(image.shape[0])
    top_cut = h * 0.15
    angles: List[float] = []
    for tb in text_boxes:
        poly = np.asarray(tb.get("polygon"), dtype=np.float64).reshape(-1, 2)
        if poly.size < 4:
            continue
        cy = float(poly[:, 1].mean())
        if cy > top_cut:
            continue
        if not str(tb.get("text") or "").strip():
            continue
        angles.append(_text_baseline_angle_deg(poly))
    if not angles:
        return False
    return float(np.median(angles)) <= angle_thresh


def _structure_grid_score(
    image: np.ndarray,
    *,
    orig_landscape: bool = False,
) -> float:
    """轻量线网格结构分（越高越好），用于 0°/90° 投票。"""
    ih, iw = image.shape[:2]
    hs, vs = _grid_line_strengths(image)
    score = (hs + vs) * 100.0
    if orig_landscape and iw > ih * 1.1:
        if hs >= vs * 0.55 and hs > 0.004:
            score += 12.0
        elif vs > hs * 1.25:
            score -= 10.0
    elif hs >= vs * 0.55 and hs > 0.004:
        score += 6.0
    return score


def looks_sideways_wired_grid(image: np.ndarray) -> bool:
    """
    有线表被整页侧躺成竖图：高明显大于宽，且竖线墨迹显著强于横线。

    原表横线旋转后变成竖线 → vs≫hs。与「直立表 + 竖排表头字」相反：
    后者 hs 仍不弱于 vs。用于 OCR 单字框 aspect 看不出侧躺时仍触发 try90。
    """
    if image is None or getattr(image, "size", 0) == 0:
        return False
    h, w = image.shape[:2]
    if h < w * 1.15:
        return False
    hs, vs = _grid_line_strengths(image)
    if vs < 0.008:
        return False
    return vs > hs * 1.35


def looks_sideways(
    text_boxes: list,
    *,
    score_thresh: float = 0.55,
    image: Optional[np.ndarray] = None,
) -> bool:
    """OCR 结果是否像侧躺（基线角 + 横长框占比 + 置信度）。"""
    if len(text_boxes) < 5:
        return True

    baseline = _weighted_median_baseline_angle(text_boxes, image=image)
    if baseline >= 55.0:
        return True
    if baseline <= 18.0:
        return False

    aspect = _box_aspect_upright_ratio(text_boxes, image=image)
    score = _mean_ocr_score(text_boxes)

    if aspect < 0.50:
        return True
    if aspect < 0.65 and score < score_thresh:
        return True
    return False


def ensure_upright_axis(
    image: np.ndarray,
    text_boxes: list,
    *,
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    score_thresh: float = 0.55,
    cache_extra_base: str = "orient=0",
) -> Tuple[np.ndarray, int, list]:
    """
    若当前 OCR 结果像侧躺，则试转 90° 并取质更好者。

    有线竖表（网格直立、仅表头竖排字）跳过 try90，避免整图旋转打乱网格。
    有线表整页侧躺（竖线主导的窄高图）即使 OCR aspect 正常也试转 90°。
    """
    from ..core.models import predict_texts

    ocr_sideways = looks_sideways(text_boxes, score_thresh=score_thresh, image=image)
    grid_sideways = looks_sideways_wired_grid(image)
    if not ocr_sideways and not grid_sideways:
        return image, 0, text_boxes

    ih, iw = image.shape[:2]
    orig_landscape = iw > ih * 1.1

    if has_strong_upright_wired_grid(image) and not grid_sideways:
        logger.info(
            "方向轴向: 检测到直立有线网格，跳过 try90（避免竖排表头误转整图）"
        )
        return image, 0, text_boxes

    if _caption_band_is_horizontal(text_boxes, image) and not grid_sideways:
        logger.info("方向轴向: 顶部题注水平，跳过 try90")
        return image, 0, text_boxes

    score0 = _structure_grid_score(image, orig_landscape=orig_landscape)
    score90 = _structure_grid_score(
        rotate_image(image, 90), orig_landscape=orig_landscape
    )
    if score0 >= score90 + 4.0 and not grid_sideways:
        logger.info(
            "方向轴向: 结构投票保持 0° (grid0=%.1f grid90=%.1f)",
            score0,
            score90,
        )
        return image, 0, text_boxes

    rotated = rotate_image(image, 90)
    boxes90 = predict_texts(
        rotated,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=f"{cache_extra_base}|try90",
    )
    q0 = _orientation_quality(text_boxes, image=image)
    q90 = _orientation_quality(boxes90, image=rotated)

    # 结构投票：明显偏好某一角度时优先采纳
    if abs(score0 - score90) >= 6.0:
        if score90 > score0 and (grid_sideways or q90 >= q0 - 0.08):
            logger.info(
                "方向轴向: 结构投票采纳 90° (grid0=%.1f grid90=%.1f q0=%.3f q90=%.3f)",
                score0,
                score90,
                q0,
                q90,
            )
            return rotated, 90, boxes90
        if score0 > score90:
            logger.info(
                "方向轴向: 结构投票保持 0° (grid0=%.1f grid90=%.1f q0=%.3f q90=%.3f)",
                score0,
                score90,
                q0,
                q90,
            )
            return image, 0, text_boxes

    # 网格已判定侧躺：旋转后若横线恢复主导，优先采纳 90°
    if grid_sideways:
        hs90, vs90 = _grid_line_strengths(rotated)
        if hs90 >= vs90 * 0.55 and (q90 >= q0 - 0.05 or q90 > 0.35):
            logger.info(
                "方向轴向: 侧躺有线网格，采纳 90° (q0=%.3f q90=%.3f hs90=%.4f vs90=%.4f)",
                q0,
                q90,
                hs90,
                vs90,
            )
            return rotated, 90, boxes90

    hs0, vs0 = _grid_line_strengths(image)
    hs90, vs90 = _grid_line_strengths(rotated)
    if hs0 > 0.004 and vs0 > 0.003 and hs0 >= vs0 * 0.5:
        if vs90 >= hs90 * 0.9 and q90 < q0 + 0.08:
            logger.info(
                "方向轴向 OCR 比对: q0=%.3f q90=%.3f（网格惩罚，保持 0°）",
                q0,
                q90,
            )
            return image, 0, text_boxes

    # 横版原图：提高采纳 90° 门槛
    if orig_landscape and q90 <= q0 + 0.10 and not grid_sideways:
        logger.info(
            "方向轴向: 横版表 q90 优势不足，保持 0° (q0=%.3f q90=%.3f)",
            q0,
            q90,
        )
        return image, 0, text_boxes

    logger.info("方向轴向 OCR 比对: q0=%.3f q90=%.3f", q0, q90)

    if q90 > q0:
        return rotated, 90, boxes90
    return image, 0, text_boxes


def parse_orientation_mode(mode: Union[str, int]) -> Union[str, int]:
    """归一化 CLI/API 的 orientation 参数。"""
    mode_s = str(mode).strip().lower()
    if mode_s in {"auto", "none", "off", "disable"}:
        return "none" if mode_s in {"none", "off", "disable"} else "auto"
    forced = int(mode_s) % 360
    if forced not in {0, 90, 180, 270}:
        raise ValueError(f"orientation 仅支持 auto/none/0/90/180/270，收到 {mode!r}")
    return forced


def apply_orientation_axis(
    image: np.ndarray,
    *,
    mode: Union[str, int] = "auto",
) -> Tuple[np.ndarray, int, str]:
    """
    强制角度立即旋转；auto 不做几何预转（交给 OCR 轴向校验）。

    Returns:
        (image, axis_angle, mode_kind)  mode_kind in {auto, none, forced}
    """
    if image is None or getattr(image, "size", 0) == 0:
        return image, 0, "none"

    parsed = parse_orientation_mode(mode)
    if parsed == "none":
        return image, 0, "none"
    if parsed != "auto":
        out = rotate_image(image, int(parsed))
        logger.info("方向强制旋转 %d°", int(parsed))
        return out, int(parsed), "forced"

    # auto：保留原图，由 ensure_upright_axis 在 OCR 后决定是否转 90
    return image, 0, "auto"


def maybe_flip_180_by_ocr(
    image: np.ndarray,
    text_boxes: list,
    *,
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    score_thresh: float = 0.55,
    cache_extra_base: str = "orient",
) -> Tuple[np.ndarray, int, list]:
    """
    根据已有 OCR 结果决定是否再转 180°；需要时重跑 OCR。

    Returns:
        (image, added_180, text_boxes)
    """
    from ..core.models import predict_texts

    s0 = _mean_ocr_score(text_boxes)
    a0 = _box_aspect_upright_ratio(text_boxes, image=image)
    if s0 >= score_thresh and a0 >= 0.55 and len(text_boxes) >= 8:
        logger.info("方向 180 消歧: keep score=%.3f aspect=%.3f", s0, a0)
        return image, 0, text_boxes

    ih, iw = image.shape[:2]
    if iw > ih * 1.15 and _weighted_median_baseline_angle(text_boxes, image=image) <= 20.0:
        logger.info("方向 180 消歧: 横版表基线水平，跳过 flip")
        return image, 0, text_boxes

    flipped = rotate_image(image, 180)
    boxes1 = predict_texts(
        flipped,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=f"{cache_extra_base}_flip180",
    )
    s1 = _mean_ocr_score(boxes1)
    a1 = _box_aspect_upright_ratio(boxes1, image=flipped)
    score0 = s0 + 0.15 * a0
    score1 = s1 + 0.15 * a1
    logger.info(
        "方向 180 消歧: s0=%.3f a0=%.3f vs s1=%.3f a1=%.3f", s0, a0, s1, a1
    )
    if score1 > score0:
        return flipped, 180, boxes1
    return image, 0, text_boxes


def correct_orientation(
    image: np.ndarray,
    *,
    mode: Union[str, int] = "auto",
    ocr_engine=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    score_thresh: float = 0.55,
) -> Tuple[np.ndarray, int, Optional[list]]:
    """
    兼容入口：OCR 轴向校验 + 180 消歧。
    """
    image, angle, kind = apply_orientation_axis(image, mode=mode)
    if kind != "auto":
        return image, angle, None

    from ..core.models import predict_texts

    boxes = predict_texts(
        image,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        cache_extra=f"orient_ax{angle}=0",
    )
    image, axis_delta, boxes = ensure_upright_axis(
        image,
        boxes,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        score_thresh=score_thresh,
        cache_extra_base=f"orient_ax{angle}",
    )
    angle = (angle + axis_delta) % 360
    image, flip, boxes = maybe_flip_180_by_ocr(
        image,
        boxes,
        ocr_engine=ocr_engine,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        score_thresh=score_thresh,
        cache_extra_base=f"orient_ax{angle}",
    )
    total = (angle + flip) % 360
    return image, total, boxes


def cli_main(argv: Optional[List[str]] = None) -> int:
    """
    命令行入口：python orient.py [图片路径]

    - 无参数：批量纠正 data/input/ 下全部图片的方向（原地覆盖原图）
    - 带路径：仅纠正该图片（原地覆盖）

    依赖 .env 中的 PADDLEOCR_ACCESS_TOKEN（云端 OCR 消歧 180°）。
    """
    import argparse
    import sys
    from pathlib import Path

    from src.core.config import load_env
    from src.core.models import load_ocr
    from src.utils.image_io import imread_unicode, imwrite_unicode

    ROOT = Path(__file__).resolve().parents[2]
    INPUT_DIR = ROOT / "data" / "input"
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    parser = argparse.ArgumentParser(
        description="表格图像方向归正：纠正后原地覆盖原图（原图不可恢复，请先备份）",
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="图片路径；省略时批量纠正 data/input/ 下全部图片",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="禁用 OCR 本地缓存，每次都请求云端"
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="强制重新请求云端 OCR 并覆盖本地缓存",
    )
    args = parser.parse_args(argv)

    if args.path:
        path = Path(args.path)
        if not path.is_file():
            alt = ROOT / args.path
            if alt.is_file():
                path = alt
            else:
                print(f"错误: 找不到图片 {args.path}", file=sys.stderr)
                return 1
        images = [path]
    else:
        if not INPUT_DIR.is_dir():
            print(f"错误: 默认输入目录不存在: {INPUT_DIR}", file=sys.stderr)
            return 1
        images = sorted(
            p
            for p in INPUT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            print(f"错误: {INPUT_DIR} 下没有图片", file=sys.stderr)
            return 1

    # 仅展示本模块的方向判定日志（如「方向轴向 OCR 比对」），
    # 其余模块保持默认 WARNING，避免云端 OCR 轮询日志刷屏
    orient_logger = logging.getLogger(__name__)
    if not orient_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        orient_logger.addHandler(handler)
        orient_logger.setLevel(logging.INFO)
        orient_logger.propagate = False

    print(f"[info] 将纠正 {len(images)} 张图片并原地覆盖原图（原图不可恢复，请先备份）")
    try:
        load_env()
        ocr = load_ocr()
    except Exception as exc:  # noqa: BLE001
        print(
            f"错误: OCR 初始化失败（请检查 .env 的 PADDLEOCR_ACCESS_TOKEN）: {exc}",
            file=sys.stderr,
        )
        return 1

    ok, rotated, failed = 0, 0, 0
    for i, img_path in enumerate(images, start=1):
        print(f"[info] ({i}/{len(images)}) {img_path.name}")
        try:
            image = imread_unicode(str(img_path))
            if image is None:
                raise RuntimeError("图片读取失败")
            corrected, total_angle, _ = correct_orientation(
                image,
                ocr_engine=ocr,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
            )
            if total_angle == 0:
                print("[info]   方向已正确，无需旋转")
            else:
                if not imwrite_unicode(str(img_path), corrected):
                    raise RuntimeError("图片写入失败")
                print(f"[info]   已旋转 {total_angle}° 并覆盖: {img_path.name}")
                rotated += 1
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[error] 处理失败 {img_path.name}: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    print(f"[info] 完成: 成功 {ok}（旋转 {rotated} 张），失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 以脚本方式直接运行（python orient.py）时：
    # 1) 把项目根目录加入 sys.path，使包内绝对导入可用；
    # 2) 以包内模块身份重新导入本模块——否则本模块内部的
    #    "from ..core.models import ..." 相对导入会因无父包而失败。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.preprocess.orient import cli_main as _run_cli

    raise SystemExit(_run_cli())
