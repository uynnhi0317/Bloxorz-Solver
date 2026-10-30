"""Kiểm thử Resource Manager: OBJ, MTL, texture, cache, đường dẫn, phương án dự phòng."""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from gui import noface, prop_models
from gui.mesh_render import compile_model
from gui.resources import (ASSET_ROOT, AssetCache, AssetError, AssetManager, get_assets, load_texture, parse_mtl,
                           parse_obj, resolve_reference)
from gui.resources import asset_manager as am
from gui.resources.material_loader import DEFAULT_MATERIAL

SOURCE_DIR = Path(__file__).resolve().parent.parent
QUAD_OBJ = """# một hình vuông
mtllib set.mtl
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
o plate
usemtl red
f 1/1 2/2 3/3 4/4
"""
RED_MTL = "newmtl red\nKd 1 0 0\nd 0.5\nillum 0\nmap_Kd ../textures/pattern.png\n"


@pytest.fixture
def assets(tmp_path):
    """Thư mục tài nguyên giả: models/quad.obj + set.mtl + textures/pattern.png (xanh lá)."""
    (tmp_path / "models").mkdir()
    (tmp_path / "textures").mkdir()
    (tmp_path / "models" / "quad.obj").write_text(QUAD_OBJ, encoding="utf-8")
    (tmp_path / "models" / "set.mtl").write_text(RED_MTL, encoding="utf-8")
    img = pygame.Surface((4, 4))
    img.fill((0, 200, 0))
    pygame.image.save(img, str(tmp_path / "textures" / "pattern.png"))
    return tmp_path


# ------------------------------------------------------------------ OBJ
def test_parse_obj_reads_geometry_uv_material_and_part(assets):
    model = parse_obj(assets / "models" / "quad.obj")
    assert model.vertices.shape == (4, 3) and model.uvs.shape == (4, 2)
    assert model.material_lib == "set.mtl"
    face = model.faces[0]
    assert face.vertices == (0, 1, 2, 3) and face.uvs == (0, 1, 2, 3)
    assert face.material == "red" and face.part == "plate"


def test_parse_obj_accepts_all_face_formats_and_negative_indices(tmp_path):
    path = tmp_path / "a.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf 1 2 3\nf 1//1 2//1 3//1\nf -3 -2 -1\nf 1/1/1 2/1/1 3/1/1\n")
    model = parse_obj(path)
    assert [f.vertices for f in model.faces] == [(0, 1, 2)] * 4
    assert model.faces[0].uvs == () and model.faces[3].uvs == (0, 0, 0)


@pytest.mark.parametrize("content", ["", "# chỉ có chú thích\n", "v 0 0 0\n", "v a b c\nf 1 1 1\n", "v 0 0 0\nf 1 2 3\n",
                                     "v 0 0 0\nv 1 1 1\nf 1 2\n", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n"])
def test_parse_obj_rejects_invalid_files(tmp_path, content):
    path = tmp_path / "bad.obj"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(AssetError):
        parse_obj(path)


def test_parse_obj_rejects_missing_file_and_wrong_extension(tmp_path):
    with pytest.raises(AssetError):
        parse_obj(tmp_path / "nothing.obj")
    other = tmp_path / "model.txt"
    other.write_text("v 0 0 0\n")
    with pytest.raises(AssetError):
        parse_obj(other)


# ------------------------------------------------------------------ MTL
def test_parse_mtl_reads_material_properties(assets):
    mat = parse_mtl(assets / "models" / "set.mtl")["red"]
    assert mat.diffuse == (255, 0, 0)
    assert mat.opacity == 0.5
    assert not mat.lit
    assert mat.texture_ref == "../textures/pattern.png"


def test_parse_mtl_skips_bad_lines_and_understands_options(tmp_path):
    path = tmp_path / "x.mtl"
    path.write_text("newmtl a\nKd oops\nillum 2\nmap_Kd -s 1 1 1 wood.png\nnewmtl b\nTr 0.25\n")
    mats = parse_mtl(path)
    assert mats["a"].diffuse == DEFAULT_MATERIAL.diffuse   # dòng sai: giữ giá trị mặc định
    assert mats["a"].texture_ref == "wood.png"
    assert mats["b"].opacity == 0.75


def test_parse_mtl_without_materials_is_an_error(tmp_path):
    path = tmp_path / "empty.mtl"
    path.write_text("# rỗng\n", encoding="utf-8")
    with pytest.raises(AssetError):
        parse_mtl(path)


# ------------------------------------------------------------------ đường dẫn
def test_resolve_reference_handles_relative_backslash_and_foreign_absolute_paths(assets):
    base = assets / "models"
    target = (assets / "textures" / "pattern.png").resolve()
    assert resolve_reference("../textures/pattern.png", base, assets) == target
    assert resolve_reference("..\\textures\\pattern.png", base, assets) == target
    # đường dẫn tuyệt đối của máy khác: chỉ dùng tên file, tìm trong thư mục tài nguyên
    assert resolve_reference("C:\\Users\\someone\\Desktop\\pattern.png", base, assets, ("textures",)) == target
    assert resolve_reference("/home/someone/pattern.png", base, assets, ("textures",)) == target


def test_resolve_reference_never_leaves_asset_root(assets, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.png"
    outside.write_bytes(b"x")
    assert resolve_reference(str(outside), assets / "models", assets) is None
    assert resolve_reference("../../../../../../secret.png", assets / "models", assets) is None
    assert resolve_reference("", assets / "models", assets) is None


def test_assets_load_when_run_from_another_working_directory(tmp_path):
    code = ("import os; os.environ['SDL_VIDEODRIVER'] = 'dummy'\n"
            "from gui.resources import get_assets\n"
            "m = get_assets().load_model('torii')\n"
            "print(m.source, len(m.textures) > 0, get_assets().load_texture('water.png') is not None)\n")
    env = dict(os.environ, PYTHONPATH=str(SOURCE_DIR))
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "file True True"


# ------------------------------------------------------------------ nối OBJ -> MTL -> texture
def test_load_model_links_material_and_texture(assets):
    loaded = AssetManager(assets).load_model("quad")
    assert loaded.source == "file"
    assert loaded.material("red").diffuse == (255, 0, 0)
    assert loaded.textures["red"].pixels[0, 0].tolist() == [0, 200, 0]


def test_compiled_color_is_diffuse_times_texture(assets):
    mesh = compile_model(AssetManager(assets).load_model("quad"))
    assert mesh.colors[0].tolist() == [0.0, 0.0, 0.0]     # đỏ x xanh lá
    (assets / "models" / "set.mtl").write_text("newmtl red\nKd 1 1 1\nmap_Kd ../textures/pattern.png\n")
    mesh = compile_model(AssetManager(assets).load_model("quad"))
    assert mesh.colors[0].tolist() == [0.0, 200.0, 0.0]


def test_texture_formats_and_broken_images(assets):
    assert load_texture(assets / "textures" / "pattern.png").size == (4, 4)
    (assets / "textures" / "broken.png").write_bytes(b"not an image")
    (assets / "textures" / "note.txt").write_text("x")
    manager = AssetManager(assets)
    assert manager.load_texture("broken.png") is None
    with pytest.raises(AssetError):
        load_texture(assets / "textures" / "note.txt")


# ------------------------------------------------------------------ cache
def test_each_resource_is_loaded_only_once(assets, monkeypatch):
    calls = {"obj": 0, "mtl": 0, "tex": 0}
    real_obj, real_mtl, real_tex = am.parse_obj, am.parse_mtl, am.load_texture

    def count(name, fn):
        def wrapper(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return wrapper
    monkeypatch.setattr(am, "parse_obj", count("obj", real_obj))
    monkeypatch.setattr(am, "parse_mtl", count("mtl", real_mtl))
    monkeypatch.setattr(am, "load_texture", count("tex", real_tex))
    manager = AssetManager(assets)
    first = manager.load_model("quad")
    assert manager.load_model("quad") is first
    assert manager.load_texture("pattern.png") is first.textures["red"]   # cùng một đối tượng, không có bản sao
    assert calls == {"obj": 1, "mtl": 1, "tex": 1}
    assert manager.stats()["models"] == {"items": 1, "loads": 1, "hits": 1}


def test_failed_resources_are_not_retried(assets, monkeypatch):
    calls = []
    monkeypatch.setattr(am, "load_texture", lambda p: calls.append(p) or (_ for _ in ()).throw(AssetError("hỏng")))
    manager = AssetManager(assets)
    for _ in range(5):
        assert manager.load_texture("pattern.png") is None
    assert len(calls) == 1


def test_cache_does_not_release_resources_in_use():
    cache = AssetCache()
    cache.acquire("a", lambda: 1)
    cache.acquire("a", lambda: 1)
    cache.get_or_load("b", lambda: 2)
    assert cache.evict("a") is False           # còn nơi đang dùng
    assert cache.purge() == ["b"]              # chỉ giải phóng cái không ai giữ
    assert "a" in cache
    cache.release("a")
    assert cache.ref_count("a") == 1 and cache.evict("a") is False
    cache.release("a")
    assert cache.evict("a") is True and "a" not in cache


# ------------------------------------------------------------------ phương án dự phòng
def test_missing_or_broken_obj_uses_fallback(assets):
    manager = AssetManager(assets)
    fallback = prop_models.build_torii
    assert manager.load_model("nothing", fallback=fallback).source == "fallback"
    (assets / "models" / "bad.obj").write_text("v x y z\n")
    assert manager.load_model("bad", fallback=fallback).source == "fallback"


def test_required_and_optional_models_are_distinguished(assets):
    manager = AssetManager(assets)
    assert manager.load_model("nothing") is None
    with pytest.raises(AssetError):
        AssetManager(assets).load_model("nothing", required=True)


def test_missing_mtl_uses_default_material(assets):
    (assets / "models" / "set.mtl").unlink()
    loaded = AssetManager(assets).load_model("quad")
    assert loaded.source == "file"
    assert loaded.material("red") is DEFAULT_MATERIAL
    compile_model(loaded)   # vẫn dùng được


def test_missing_mtl_prefers_materials_of_fallback(assets):
    (assets / "models" / "set.mtl").unlink()
    loaded = AssetManager(assets).load_model("quad", fallback=prop_models.build_torii)
    assert "torii_red" in loaded.materials


def test_missing_texture_keeps_diffuse_color(assets):
    (assets / "textures" / "pattern.png").unlink()
    loaded = AssetManager(assets).load_model("quad")
    assert loaded.textures == {}
    assert compile_model(loaded).colors[0].tolist() == [255.0, 0.0, 0.0]


# ------------------------------------------------------------------ tài nguyên thật của game
@pytest.mark.parametrize("name,build", [("noface_whole", lambda: noface.build_model("whole")),
                                        ("noface_head", lambda: noface.build_model("head")),
                                        ("noface_body", lambda: noface.build_model("body")),
                                        ("torii", prop_models.build_torii)])
def test_shipped_models_load_from_file_and_match_code(name, build):
    manager = AssetManager()
    loaded = manager.load_model(name)
    assert loaded.source == "file"
    assert loaded.textures, "phải có ít nhất một texture được gắn"
    source = build().model
    # file trên đĩa phải khớp hình học trong code: quên chạy make_assets.py sẽ bị bắt ở đây
    assert len(loaded.model.faces) == len(source.faces)
    assert np.allclose(loaded.model.vertices, source.vertices, atol=1e-4)
    assert {f.material for f in loaded.model.faces} <= set(loaded.materials)


def test_game_textures_exist_and_asset_root_is_inside_project():
    assert ASSET_ROOT.is_dir() and SOURCE_DIR in ASSET_ROOT.parents
    for name in ("stone.png", "torii_wood.png", "cloak.png", "water.png"):
        assert AssetManager().load_texture(name) is not None


def test_game_runs_with_empty_asset_folder(tmp_path, monkeypatch):
    """Không có file tài nguyên nào: mọi thứ phải chạy bằng phương án dự phòng."""
    from gui import mesh_render, renderer as renderer_module, theme
    from game import Board
    manager = AssetManager(tmp_path)
    monkeypatch.setattr(am, "_DEFAULT", manager)
    monkeypatch.setattr(mesh_render, "_MESHES", {})
    pygame.init()
    pygame.display.set_mode((800, 600))
    board = Board.from_dict({"start": [0, 0], "tiles": [[{"type": "floor"}] * 3 + [{"type": "goal"}]]})
    surf = pygame.Surface((800, 600))
    bg = theme.Background((800, 600))
    bg.draw_under(surf)
    r = renderer_module.Renderer(board, pygame.Rect(0, 0, 800, 600))
    r.water_pass = bg.draw_water
    r.draw(surf, board.initial_state)
    assert manager.stats()["models"]["items"] >= 1   # Vô Diện; bàn chơi không còn model vật dựng nào
    assert all(m.source == "fallback" for m in manager.models._items.values() if m is not None)


def test_switching_stages_does_not_reload_resources():
    from gui.app import App
    app = App()
    app.start_stage(0)
    app.update(16)
    app.draw()
    before = {k: v["loads"] for k, v in get_assets().stats().items()}
    for i in (1, 2, 1, 0):
        app.start_stage(i)
        app.update(16)
        app.draw()
    assert {k: v["loads"] for k, v in get_assets().stats().items()} == before
