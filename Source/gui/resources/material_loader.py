"""material_loader.py - Đọc thư viện vật liệu Wavefront MTL."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from .model_loader import AssetError

log = logging.getLogger("bloxorz.assets")

Color = Tuple[int, int, int]


@dataclass
class Material:
    name: str
    diffuse: Color = (200, 200, 200)    # Kd
    opacity: float = 1.0                # d (Tr = 1 - d)
    illum: int = 2                      # 0 = màu cố định, không chịu ánh sáng
    texture_ref: Optional[str] = None   # map_Kd nguyên văn trong file

    @property
    def lit(self) -> bool:
        return self.illum != 0


DEFAULT_MATERIAL = Material("default")


def _color(args) -> Color:
    return tuple(max(0, min(255, round(float(a) * 255))) for a in args[:3])  # type: ignore[return-value]


def parse_mtl(path: Path) -> Dict[str, Material]:
    """Đọc file MTL. Dòng sai được bỏ qua (có log) để một lỗi nhỏ không làm mất cả thư viện vật liệu.
    Raise AssetError nếu file thiếu hoặc không có vật liệu nào."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AssetError(f"{path}: không đọc được ({exc})") from exc
    out: Dict[str, Material] = {}
    cur: Optional[Material] = None
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tag, _, rest = line.partition(" ")
        args = rest.split()
        try:
            if tag == "newmtl":
                cur = out[rest.strip()] = Material(rest.strip())
            elif cur is None:
                continue
            elif tag == "Kd":
                cur.diffuse = _color(args)
            elif tag == "d":
                cur.opacity = float(args[0])
            elif tag == "Tr":
                cur.opacity = 1.0 - float(args[0])
            elif tag == "illum":
                cur.illum = int(args[0])
            elif tag == "map_Kd":
                # map_Kd có thể kèm tùy chọn (-s 1 1 1 ...); tên file là phần cuối
                cur.texture_ref = rest.strip().split(" -")[-1].split()[-1] if " -" in f" {rest}" else rest.strip()
        except (ValueError, IndexError):
            log.warning("%s: dòng %d sai định dạng, bỏ qua: %s", path, line_no, line)
    if not out:
        raise AssetError(f"{path}: không có vật liệu nào")
    return out
