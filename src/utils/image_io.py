"""图像读写工具（支持 Unicode 路径）。"""

from __future__ import annotations

import cv2
import numpy as np


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """cv2.imwrite 在 Windows 中文路径下会失败，改用 imencode + tofile。"""
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(path)
    return True


def imread_unicode(path: str) -> np.ndarray | None:
    """Unicode 安全读图。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None
