"""asset_manager.py - Điểm truy cập duy nhất tới tài nguyên đồ họa."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

from .asset_cache import AssetCache
from .material_loader import DEFAULT_MATERIAL, Material, parse_mtl
from .model_loader import AssetError, Model, parse_obj
from .paths import ASSET_ROOT, resolve_reference
from .texture_loader import Texture, load_texture

log = logging.getLogger("bloxorz.assets")


@dataclass
class LoadedModel:
    model: Model
    materials: Dict[str, Material]
    textures: Dict[str, Texture] = field(default_factory=dict)
    source: str = "file"

    def material(self, name: str) -> Material:
        return self.materials.get(name, DEFAULT_MATERIAL)


class AssetManager:
    def __init__(self, root: Path = ASSET_ROOT) -> None:
        self.root = Path(root).resolve()
        self.models: AssetCache[LoadedModel] = AssetCache()
        self.material_libs: AssetCache[Dict[str, Material]] = AssetCache()
        self.textures: AssetCache[Texture] = AssetCache()

    def load_texture(self, ref: str, base_dir: Optional[Path] = None) -> Optional[Texture]:
        path = resolve_reference(ref, base_dir or self.root / "textures", self.root, ("textures",))
        if path is None:
            return self.textures.get_or_load(f"missing:{ref}", lambda: self._missing("texture", ref))
        return self.textures.get_or_load(str(path), lambda: self._try(load_texture, path))

    def load_materials(self, ref: str, base_dir: Optional[Path] = None) -> Optional[Dict[str, Material]]:
        path = resolve_reference(ref, base_dir or self.root / "models", self.root, ("models", "materials"))
        if path is None:
            return self.material_libs.get_or_load(f"missing:{ref}", lambda: self._missing("MTL", ref))
        return self.material_libs.get_or_load(str(path), lambda: self._try(parse_mtl, path))

    def load_model(self, name: str, fallback: Optional[Callable[[], LoadedModel]] = None,
                   required: bool = False) -> Optional[LoadedModel]:
        def load() -> Optional[LoadedModel]:
            path = self.root / "models" / f"{name}.obj"
            try:
                model = parse_obj(path) if path.is_file() else None
            except AssetError as exc:
                log.warning("Model hỏng, dùng phương án dự phòng: %s", exc)
                model = None
            if model is None:
                return fallback() if fallback is not None else None
            return self._link(model, path.parent, fallback)

        loaded = self.models.get_or_load(name, load)
        if loaded is None and required:
            raise AssetError(f"Thiếu model bắt buộc: {name}")
        if loaded is None:
            log.warning("Thiếu model tùy chọn: %s", name)
        return loaded

    def _link(self, model: Model, base_dir: Path, fallback: Optional[Callable[[], LoadedModel]]) -> LoadedModel:
        libs = self.load_materials(model.material_lib, base_dir) if model.material_lib else None
        if libs is None:
            libs = fallback().materials if fallback is not None else {}
        textures: Dict[str, Texture] = {}
        for material in libs.values():
            if material.texture_ref:
                tex = self.load_texture(material.texture_ref, base_dir)
                if tex is not None:
                    textures[material.name] = tex
        return LoadedModel(model, libs, textures, "file")

    @staticmethod
    def _try(loader, path):
        try:
            return loader(path)
        except AssetError as exc:
            log.warning("Bỏ qua tài nguyên lỗi: %s", exc)
            return None

    @staticmethod
    def _missing(kind: str, ref: str):
        log.warning("Không tìm thấy %s: %s", kind, ref)
        return None


_DEFAULT: Optional[AssetManager] = None


def get_assets() -> AssetManager:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = AssetManager()
    return _DEFAULT

