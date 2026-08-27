"""云端 PaddleOCR 加载与推断。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .config import get_settings

_ocr_client = None


class CloudOcrEngine:
    """HTTP 云端 OCR 占位对象（兼容 pipeline 的 ocr_engine 传参）。"""

    def __repr__(self) -> str:  # noqa: D401
        return "CloudOcrEngine(http_v2)"


def load_ocr():
    """
    初始化云端 OCR（HTTP 提交/轮询/下载）。

    Token / 模型名等均从项目根目录 .env 读取；启动时校验 Token。
    """
    global _ocr_client
    if _ocr_client is not None:
        return _ocr_client

    get_settings().require_token()
    _ocr_client = CloudOcrEngine()
    return _ocr_client


def predict_texts(
    image: Union[str, np.ndarray],
    ocr_engine=None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_extra: Optional[str] = None,
    save_artifacts: Optional[bool] = None,
    artifact_stem: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    使用云端 PaddleOCR（HTTP）提取整图文本块（含检测坐标）。

    - 本地预处理后上传；坐标映回原图
    - use_cache：命中 data/cache/ 则跳过云端；refresh_cache 强制重拉并覆盖

    Returns:
        文本块列表，每项含 polygon / text / score / top_left(x,y)（原图坐标）
    """
    from pathlib import Path

    from ..ocr.cloud_ocr import run_cloud_ocr
    from ..ocr.ocr_cache import load_ocr_cache, save_ocr_cache

    if ocr_engine is None:
        load_ocr()

    if use_cache and not refresh_cache:
        cached = load_ocr_cache(image, extra=cache_extra)
        if cached is not None:
            return cached

    settings = get_settings()
    do_artifacts = (
        settings.save_ocr_artifacts if save_artifacts is None else bool(save_artifacts)
    )
    stem = artifact_stem
    filename = "image.jpg"
    if stem is None and isinstance(image, str):
        stem = Path(image).stem
        filename = Path(image).name
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            filename = f"{stem}.jpg"

    text_boxes = run_cloud_ocr(
        image,
        filename=filename,
        save_artifacts=do_artifacts and bool(stem),
        artifact_stem=stem,
        artifact_dir=settings.ocr_artifact_dir,
    )

    if use_cache or refresh_cache:
        try:
            save_ocr_cache(image, text_boxes, extra=cache_extra)
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning("写入 OCR 缓存失败: %s", exc)
    return text_boxes
