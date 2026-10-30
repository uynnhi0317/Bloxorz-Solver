"""Quản lý tài nguyên đồ họa (OBJ, MTL, texture) có cache, độc lập với logic game và giao diện."""
from .asset_cache import AssetCache
from .asset_manager import AssetManager, LoadedModel, get_assets
from .material_loader import Material, parse_mtl
from .model_loader import AssetError, Face, Model, ModelBuilder, parse_obj, save_obj
from .paths import ASSET_ROOT, resolve_reference
from .texture_loader import Texture, load_texture

__all__ = ["AssetCache", "AssetError", "AssetManager", "ASSET_ROOT", "Face", "LoadedModel", "Material", "Model",
           "ModelBuilder", "Texture", "get_assets", "load_texture", "parse_mtl", "parse_obj", "resolve_reference",
           "save_obj"]
