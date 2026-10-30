"""theme.py - Theme "Đền thần bên biển" (tranh vẽ tay kiểu Ghibli: pastel, xanh ngọc, nâu gỗ): bảng màu + nền động.

Mọi hình ảnh đều vẽ bằng code (không cần file ảnh ngoài), nên không có rủi ro thiếu asset.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pygame

from .resources import get_assets

Color = Tuple[int, int, int]

# ---- bảng màu gốc của concept: pastel mộc mạc, ít bão hòa
SAKURA_LIGHT: Color = (244, 214, 208)  # #F4D6D0
SAKURA_DEEP: Color = (214, 150, 146)   # #D69692
WOOD: Color = (160, 116, 80)           # #A07450
TORII_RED: Color = (184, 88, 68)       # #B85844 son đỏ đã phai
VOID_DARK: Color = (52, 70, 72)        # #344648
WATER_DEEP: Color = (62, 128, 132)     # #3E8084
LEAF: Color = (140, 170, 112)          # #8CAA70
CREAM: Color = (253, 246, 230)         # #FDF6E6

# ---- màu thế giới (renderer dùng)
PALETTE = {
    "floor": (208, 196, 168),          # đá lát sân, ngả màu giấy cũ
    "moss": (150, 174, 122),           # rêu (lá cây dịu lại để ô Floor vẫn trung tính)
    "seam": (166, 152, 128),           # đường ron giữa các phiến đá
    "fragile": (168, 116, 70),         # nền đống lá
    "leaves": ((222, 176, 104), (200, 142, 84), (232, 200, 134), (170, 108, 70), (170, 160, 100)),
    "bridge": (164, 112, 74),          # ván gỗ nâu ấm
    "bridge_dark": (108, 74, 50),      # thanh đỡ dưới ván cầu
    "bridge_ghost": (112, 96, 78),     # ván cầu đóng chìm dưới nước
    "goal": (186, 170, 140),           # đá sân quanh hố dịch chuyển
    "goal_rim": (112, 84, 62),         # vành gỗ quanh hố
    "goal_hole": (30, 66, 72),
    "goal_inner": (56, 118, 118),
    "goal_swirl": (206, 238, 222),     # xoáy sáng trong hố
    "torii": TORII_RED,
    "torii_cap": (78, 62, 54),
    "switch_base": (146, 132, 110),
    "soft": (244, 200, 196),
    "soft_off": (176, 140, 134),
    "heavy": (176, 208, 136),          # tre non
    "heavy_off": (126, 140, 100),
    "split": CREAM,
    "split_off": (188, 178, 158),
    "block": (62, 54, 58),             # thân Vô Diện: than ấm, không đen tuyền
    "cube_active": (76, 66, 68),
    "cube_idle": (50, 44, 48),
    "mask": (250, 244, 230),
    "mask_mark": (150, 104, 136),      # vệt tím trên mặt nạ
    "mask_eye": (54, 46, 46),
}

# ---- màu giao diện (controls/app dùng)
UI = {
    "bg": VOID_DARK, "bar": (74, 58, 46), "panel": (96, 76, 58), "border": (168, 128, 90),
    "text": CREAM, "muted": (232, 220, 198), "accent": (246, 214, 176),
    "accent_dark": (126, 92, 66), "win": (250, 226, 176), "lose": (226, 140, 104),
    "wood": (132, 94, 62), "wood_hover": (156, 114, 78), "wood_press": (110, 76, 50),
    "wood_edge": (84, 58, 40), "wood_off": (128, 118, 104), "text_off": (206, 198, 184),
    "cord": TORII_RED, "ok": (176, 210, 140), "fail": (232, 130, 110), "dim": (52, 44, 38),
    "ink": (84, 56, 38), "ink_soft": (104, 72, 50),   # chữ vẽ trực tiếp lên nền trời/biển sáng
}
SKY_TOP: Color = (124, 186, 212)
WATER_ALPHA_NEAR, WATER_ALPHA_FAR = 92, 215   # độ đục lớp nước: gần trong, xa đục (0..255)
RIPPLE_GAIN = 14                              # độ sáng gợn sóng (0..255)
SUN_X = 0.76                                  # vị trí mặt trời theo bề ngang: vệt lấp lánh trên biển nằm ngay dưới
SERIF = "palatinolinotype,palatino,garamond,georgia,timesnewroman,dejavuserif,liberationserif,freeserif"

_GLOWS: Dict[tuple, pygame.Surface] = {}


def glow_sprite(radius: int, color: Color, gain: float = 1.0) -> pygame.Surface:
    """Chấm sáng mềm (nền đen) để vẽ cộng (BLEND_RGB_ADD). Có cache."""
    radius = max(1, int(radius))
    key = (radius, tuple(color), round(gain, 2))
    if key not in _GLOWS:
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        a = np.clip(1 - np.sqrt(x * x + y * y) / radius, 0, 1) ** 2 * gain
        arr = (a[..., None] * np.array(color)).clip(0, 255).astype(np.uint8)
        _GLOWS[key] = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
    return _GLOWS[key]


# ------------------------------------------------------------------ cánh hoa
def sakura_points(cx: float, cy: float, radius: float, petal: int, phase: float = -math.pi / 2):
    """Đa giác của cánh thứ `petal` trong bông hoa 5 cánh, có khía nhỏ ở đầu cánh."""
    base = phase + petal * 2 * math.pi / 5
    prof = ((0.12, 0.0), (0.46, 0.5), (0.82, 0.42), (1.0, 0.16), (0.88, 0.0), (1.0, -0.16),
            (0.82, -0.42), (0.46, -0.5))
    return [(cx + radius * d * math.cos(base + a), cy + radius * d * math.sin(base + a)) for d, a in prof]


_PETAL_FRAMES = 24
_PETAL_SQUASH = (1.0, 0.68, 0.36)      # cánh hoa lật trong gió = co một chiều
_PETALS: Dict[tuple, pygame.Surface] = {}


def petal_sprite(size: int, angle_idx: int, squash_idx: int, color: Color, alpha: int = 255) -> pygame.Surface:
    """Một cánh hoa rơi đã xoay sẵn (bảng tra theo góc) để không phải rotate mỗi frame."""
    key = (size, angle_idx % _PETAL_FRAMES, squash_idx, color, alpha)
    if key not in _PETALS:
        ss, d = 4, size * 2 + 2
        big = pygame.Surface((d * ss, d * ss), pygame.SRCALPHA)
        ang = 2 * math.pi * key[1] / _PETAL_FRAMES
        ca, sa, sq = math.cos(ang), math.sin(ang), _PETAL_SQUASH[squash_idx]
        pts = []
        for t in range(16):  # dáng giọt nước, đầu có khía
            u = t / 16 * 2 * math.pi
            x = math.cos(u) * size
            y = math.sin(u) * size * 0.58 * (1 - 0.22 * math.cos(u))
            if abs(u - math.pi) < 0.25:
                x *= 0.72
            y *= sq
            pts.append(((d / 2 + x * ca - y * sa) * ss, (d / 2 + x * sa + y * ca) * ss))
        pygame.draw.polygon(big, (*color, alpha), pts)
        _PETALS[key] = pygame.transform.smoothscale(big, (d, d))
    return _PETALS[key]


def flower_sprite(radius: int, color: Color = SAKURA_LIGHT, core: Color = SAKURA_DEEP) -> pygame.Surface:
    """Bông hoa anh đào 5 cánh nhỏ (dấu trang trí trong UI)."""
    key = ("flower", radius, color, core)
    if key not in _PETALS:
        ss, d = 4, radius * 2 + 2
        big = pygame.Surface((d * ss, d * ss), pygame.SRCALPHA)
        c = d * ss / 2
        for i in range(5):
            pygame.draw.polygon(big, color, sakura_points(c, c, radius * ss, i))
        pygame.draw.circle(big, core, (c, c), radius * ss * 0.24)
        _PETALS[key] = pygame.transform.smoothscale(big, (d, d))
    return _PETALS[key]


# ------------------------------------------------------------------ nền tĩnh (dựng 1 lần theo kích thước)
def _paper_grain(w: int, h: int, seed: int) -> np.ndarray:
    """Hệ số nhân độ sáng (w, h) giả nét cọ ngang và vân giấy, để các mảng màu không phẳng như đồ họa máy."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(0, 255, (max(2, w // 48), max(2, h // 5), 1)).repeat(3, axis=2).astype(np.uint8)
    brush = pygame.surfarray.array3d(pygame.transform.smoothscale(pygame.surfarray.make_surface(small), (w, h)))[..., 0]
    return 1 + 0.07 * (brush / 255 - 0.5) + rng.normal(0, 0.012, (w, h))


def _ridge(rng: random.Random, w: int, base: float, amp: float, n: int = 5) -> List[Tuple[float, float]]:
    """Đường đồi núi xa: tổng vài sóng sin tần số nguyên, nên hai đầu ảnh nối liền nhau."""
    waves = [(rng.randint(1, 4), rng.uniform(0, 6.28), rng.uniform(0.3, 1.0)) for _ in range(n)]
    norm = sum(a for _, _, a in waves)
    return [(x, base - amp * sum(a * math.sin(f * x / w * 6.28 + p) for f, p, a in waves) / norm)
            for x in range(0, w + 16, 16)]


def _torii_silhouette(s: pygame.Surface, cx: float, foot: float, h: float, color) -> None:
    """Bóng cổng torii lớn đứng dưới hồ."""
    w = h * 0.82
    post = max(2, int(h * 0.075))
    for dx in (-w * 0.36, w * 0.36):
        pygame.draw.polygon(s, color, [(cx + dx - post, foot), (cx + dx + post, foot),
                                       (cx + dx + post * 0.7, foot - h * 0.88), (cx + dx - post * 0.7, foot - h * 0.88)])
    pygame.draw.rect(s, color, (cx - w * 0.44, foot - h * 0.72, w * 0.88, max(2, h * 0.05)))
    pygame.draw.polygon(s, color, [(cx - w * 0.6, foot - h * 1.0), (cx - w * 0.5, foot - h * 0.9),
                                   (cx + w * 0.5, foot - h * 0.9), (cx + w * 0.6, foot - h * 1.0),
                                   (cx + w * 0.3, foot - h * 0.96), (cx - w * 0.3, foot - h * 0.96)])


def _pagoda_silhouette(s: pygame.Surface, cx: float, foot: float, h: float, color) -> None:
    """Bóng chùa ba tầng trên đồi."""
    levels, w = 3, h * 0.34
    step = h / (levels + 0.4)
    for i in range(levels):
        y = foot - i * step
        ww = w * (1 - 0.2 * i)
        pygame.draw.rect(s, color, (cx - ww * 0.55, y - step * 0.75, ww * 1.1, step * 0.75))
        pygame.draw.polygon(s, color, [(cx - ww * 1.2, y - step * 0.62), (cx - ww * 0.55, y - step),
                                       (cx + ww * 0.55, y - step), (cx + ww * 1.2, y - step * 0.62)])
    pygame.draw.line(s, color, (cx, foot - levels * step), (cx, foot - levels * step - step * 0.5), max(1, int(h * 0.02)))


def _canopy(s: pygame.Surface, rng: random.Random, cx: float, cy: float, spread: float, k: float) -> None:
    """Tán anh đào ở góc màn hình: cành tối + các chùm hoa tròn mờ, không rực để khỏi át bản đồ."""
    branch = (104, 78, 60, 255)
    tips = []
    grow = 1 if cx < s.get_width() / 2 else -1   # cành mọc từ góc vào giữa màn hình
    for _ in range(5):
        x, y = cx, cy
        ang = rng.uniform(0.25, 1.1)
        for _ in range(4):
            nx = x + grow * math.cos(ang) * spread * 0.2
            ny = y + abs(math.sin(ang)) * spread * 0.16
            pygame.draw.line(s, branch, (x, y), (nx, ny), max(1, int(6 * k)))
            x, y = nx, ny
            ang += rng.uniform(-0.45, 0.45)
            tips.append((x, y))
    shades = ((236, 186, 190, 170), (244, 206, 206, 170), (250, 226, 220, 150))
    for x, y in tips:
        for _ in range(9):
            r = rng.uniform(8, 26) * k
            px, py = x + rng.gauss(0, 30 * k), y + rng.gauss(0, 20 * k)
            blob = pygame.Surface((int(2 * r) + 2, int(2 * r) + 2), pygame.SRCALPHA)
            pygame.draw.circle(blob, rng.choice(shades), (r, r), r)
            s.blit(blob, (px - r, py - r))


def _cloud(s: pygame.Surface, rng: random.Random, cx: float, base: float, width: float) -> None:
    """Mây tích vẽ tay: đáy phẳng, các cụm bông chồng ba lớp màu (bóng tím xám, thân kem, đỉnh trắng)."""
    ss, h = 2, width * 0.62
    cw, ch = int(width * ss), int(h * ss)
    layer = pygame.Surface((cw, ch), pygame.SRCALPHA)
    n = rng.randint(7, 10)
    puffs = []
    for i in range(n):
        t = (i + 0.5) / n
        bell = math.sin(t * math.pi) ** 0.8
        rad = width * rng.uniform(0.07, 0.13) * (0.45 + bell)
        puffs.append((t * width, h - rad * rng.uniform(0.3, 0.9) - bell * h * 0.22, rad))
    passes = (((186, 196, 218, 255), 0.0, 1.0), ((252, 242, 224, 255), -0.16, 0.92), ((255, 252, 244, 255), -0.3, 0.7))
    for col, lift, grow in passes:
        for x, y, rad in puffs:
            pygame.draw.circle(layer, col, (x * ss, (y + lift * rad) * ss), rad * grow * ss)
    img = pygame.transform.smoothscale(layer, (int(width), int(h)))
    s.blit(img, (cx - width / 2, base - h))


def _sky_panorama(w: int, sky_h: int, k: float, rng: random.Random) -> Tuple[pygame.Surface, pygame.Surface]:
    """Bầu trời + đồi + chùa + torii, cạnh dưới là đường chân trời.
    Trả về (ảnh trời, bóng soi trên nước)."""
    t = np.linspace(0, 1, sky_h)[None, :, None]
    top, low = np.array(SKY_TOP), np.array((250, 234, 208))
    sky = pygame.surfarray.make_surface(np.broadcast_to(top + (low - top) * t ** 2.2, (w, sky_h, 3)).astype(np.uint8))
    mx, my, mr = int(w * SUN_X), sky_h - int(150 * k), max(10, int(40 * k))
    sky.blit(glow_sprite(mr * 5, (70, 58, 34), 0.9), (mx - mr * 5, my - mr * 5), special_flags=pygame.BLEND_RGB_ADD)
    pygame.draw.circle(sky, (255, 250, 232), (mx, my), mr)
    for fx, base, width in ((0.10, 62, 330), (0.40, 34, 270), (0.60, 104, 210), (0.93, 48, 300)):
        _cloud(sky, rng, w * fx, sky_h - base * k, width * k)
    land = pygame.Surface((w, sky_h), pygame.SRCALPHA)
    for base, amp, col, line in ((26, 34, (160, 194, 186, 255), (128, 164, 158, 255)),
                                 (6, 20, (122, 166, 146, 255), (92, 134, 116, 255))):
        ridge = _ridge(rng, w, sky_h - base * k, amp * k)
        pygame.draw.polygon(land, col, [(0, sky_h)] + ridge + [(w, sky_h)])
        pygame.draw.aalines(land, line, False, ridge)   # nét bút chì chạy theo sống đồi
    _pagoda_silhouette(land, w * 0.17, sky_h - 8 * k, 112 * k, (96, 104, 98, 255))
    _torii_silhouette(land, w * 0.86, sky_h, 96 * k, (190, 98, 78, 255))
    sky.blit(land, (0, 0))
    painted = pygame.surfarray.array3d(sky) * _paper_grain(w, sky_h, 5)[..., None]
    sky = pygame.surfarray.make_surface(painted.clip(0, 255).astype(np.uint8))

    # bóng nước: lật ảnh xuống, gợn ngang, nhạt dần theo độ xa đường chân trời
    arr = pygame.surfarray.array3d(sky)
    n = min(sky_h, int(240 * k) + 8)
    ys = np.arange(n)
    shift = (np.sin(ys * 0.34) * (1 + ys * 0.05)).astype(int)
    flipped = arr[:, sky_h - 1 - ys].copy()
    for i in range(n):
        flipped[:, i] = np.roll(flipped[:, i], shift[i], axis=0)
    mirror = pygame.Surface((w, n), pygame.SRCALPHA)
    rgb = pygame.surfarray.pixels3d(mirror)
    rgb[...] = flipped
    del rgb
    a = pygame.surfarray.pixels_alpha(mirror)
    a[...] = (130 * (1 - ys / n) ** 1.6).astype(np.uint8)[None, :]
    del a
    return sky, mirror


def _lake_bed(w: int, h: int) -> pygame.Surface:
    """Đáy hồ sáng màu lục lam: chính độ tương phản với lớp nước phía trên tạo cảm giác nước trong."""
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h), indexing="ij")
    shallow, deep = np.array((150, 206, 190)), np.array((84, 150, 148))
    blotch = 0.5 + 0.25 * np.sin(x * 19 + np.sin(y * 7) * 2) + 0.25 * np.sin(y * 23 + np.sin(x * 11) * 2)
    mix = (y ** 0.8 * 0.8 + blotch * 0.2)[..., None]
    img = (deep + (shallow - deep) * mix) * _paper_grain(w, h, 6)[..., None]
    vignette = np.clip(1.08 - 0.36 * np.abs(x * 2 - 1) ** 2.4 - 0.2 * np.abs(y * 2 - 1) ** 3, 0, 1)[..., None]
    return pygame.surfarray.make_surface((img * vignette).clip(0, 255).astype(np.uint8))


def _water_sheet(w: int, h: int) -> pygame.Surface:
    """Lớp nước trong suốt, dòng đầu là đường chân trời. Nhìn gần (mép dưới màn hình) trong, nhìn xa đục dần
    và pha màu trời, đúng hiệu ứng Fresnel của mặt nước thật."""
    t = np.linspace(0, 1, h)[None, :, None]
    near, far = np.array((98, 184, 176)), np.array((176, 216, 208))
    fresnel = (1 - t) ** 2.4
    sheet = pygame.Surface((w, h), pygame.SRCALPHA)
    rgb = pygame.surfarray.pixels3d(sheet)
    rgb[...] = (near + (far - near) * fresnel).astype(np.uint8)
    del rgb
    a = pygame.surfarray.pixels_alpha(sheet)
    a[...] = (WATER_ALPHA_NEAR + (WATER_ALPHA_FAR - WATER_ALPHA_NEAR) * fresnel[..., 0]).astype(np.uint8)
    del a
    return sheet


def _ripple_layer(w: int, h: int) -> pygame.Surface:
    """Gợn sóng để vẽ cộng lên nước: lấy từ textures/water.png, thiếu thì tự sinh."""
    tex = get_assets().load_texture("water.png")
    if tex is not None:
        tile = pygame.surfarray.make_surface(tex.pixels)
    else:
        x, y = np.meshgrid(np.linspace(0, 6.28, 256), np.linspace(0, 6.28, 256), indexing="ij")
        crest = np.clip(np.sin(x * 5 + np.sin(y * 3) * 2) * np.sin(y * 9 + np.sin(x * 2)), 0, 1) ** 3
        tile = pygame.surfarray.make_surface((crest[..., None] * 230).astype(np.uint8).repeat(3, axis=2))
    tile = pygame.transform.smoothscale(tile, (tile.get_width() * 2, tile.get_height()))   # dẹt theo chiều dọc như mặt phẳng nhìn nghiêng
    out = pygame.Surface((w, h))
    for x0 in range(0, w, tile.get_width()):
        for y0 in range(0, h, tile.get_height()):
            out.blit(tile, (x0, y0))
    out.fill((RIPPLE_GAIN, RIPPLE_GAIN, RIPPLE_GAIN), special_flags=pygame.BLEND_RGB_MULT)
    return out


def _wave_layer(w: int, h: int, k: float, seed: int, gain: float) -> pygame.Surface:
    """Các hàng sóng kiểu tranh vẽ tay để vẽ cộng lên nước: nét cong thon hai đầu, gần to thưa, xa mảnh dày.
    Lặp được theo x. Hai lớp trôi ngược chiều nhau tạo mặt nước lăn tăn."""
    ss = 2
    big = pygame.Surface((w * ss, h * ss))
    rng = random.Random(seed)
    tint = np.array((172, 214, 216)) * gain

    def stroke(x: float, y: float, ln: float, arch: float, thick: float, level: float, swell) -> None:
        for dx in (-w, 0, w):
            pts = []
            us = [-1 + 2 * i / 10 for i in range(11)]
            for u in us:
                pts.append(((x + dx + u * ln) * ss, (y + swell(x + u * ln) - arch * (1 - u * u)) * ss))
            for u in reversed(us):
                pts.append(((x + dx + u * ln) * ss, (y + swell(x + u * ln) - arch * (1 - u * u) + thick * (1 - u * u)) * ss))
            pygame.draw.polygon(big, tuple(int(v * level) for v in tint), pts)

    y = 2.0
    while y < h:
        t = y / h
        persp = 0.14 + t ** 1.25
        freq, phase, amp = rng.randint(2, 6), rng.uniform(0, 6.28), rng.uniform(2, 6) * persp * k

        def swell(x: float, freq=freq, phase=phase, amp=amp) -> float:
            return amp * math.sin(2 * math.pi * freq * x / w + phase)

        x = rng.uniform(0, w)
        end = x + w
        while x < end:
            ln = max(4.0, rng.uniform(26, 96) * persp * k)
            level = rng.uniform(0.5, 1.0) * (0.55 + 0.45 * t)
            arch, thick = ln * rng.uniform(0.03, 0.09), rng.uniform(1.0, 2.6) * persp * k + 0.5
            stroke((x + ln) % w, y, ln, arch, thick * 2.4, level * 0.26, swell)   # quầng mờ quanh nét sóng
            stroke((x + ln) % w, y, ln, arch, thick, level, swell)
            x += 2 * ln + rng.uniform(0.5, 2.6) * ln
        y += (4 + 30 * t) * k * rng.uniform(0.8, 1.25)
    return pygame.transform.smoothscale(big, (w, h))


def _sparkle(radius: int, level: int) -> pygame.Surface:
    """Ngôi sao lấp lánh bốn cánh trên nền đen để vẽ cộng. level từ 1 đến 4. Có cache."""
    key = ("sparkle", radius, level)
    if key not in _GLOWS:
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(float)
        ax, ay = np.abs(x) / radius, np.abs(y) / radius
        ray = np.exp(-3.5 * ax) * np.exp(-(y / 0.9) ** 2) + np.exp(-3.5 * ay) * np.exp(-(x / 0.9) ** 2)
        core = np.exp(-(x * x + y * y) / max(1.2, radius * 0.26) ** 2)
        a = np.clip(ray + core, 0, 1) * np.clip(1.15 - np.sqrt(ax * ax + ay * ay), 0, 1) * level / 4
        arr = (a[..., None] * np.array((255, 248, 226))).clip(0, 255).astype(np.uint8)
        _GLOWS[key] = pygame.surfarray.make_surface(arr)
    return _GLOWS[key]


def _mist(w: int, h: int, rng: random.Random, peak: int) -> pygame.Surface:
    """Dải sương mù: mờ theo chiều dọc và có độ dày thay đổi theo chiều ngang; lặp được theo x."""
    x = np.arange(w)[:, None]
    y = np.linspace(-1, 1, h)[None, :]
    wave = sum(rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * (x / w) * rng.randint(1, 4) + rng.uniform(0, 6.28))
               for _ in range(4))
    thick = np.clip(0.55 + 0.35 * wave / 2, 0.0, 1.0)
    alpha = (peak * thick * np.exp(-(y / 0.55) ** 2)).clip(0, 255).astype(np.uint8)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    rgb = pygame.surfarray.pixels3d(s)
    rgb[...] = (250, 244, 232)
    del rgb
    a = pygame.surfarray.pixels_alpha(s)
    a[...] = alpha
    del a
    return s


class Background:
    """Nền động, vẽ làm hai lượt để có chỗ cho những gì nằm dưới mặt nước:
        draw_under: đáy hồ.
        (renderer vẽ ván cầu chìm, khối đang rơi)
        draw_water: lớp nước trong suốt + sóng + bóng soi + ánh lấp lánh, trời, sương, cánh hoa, tán hoa.
    Luôn vẽ TRƯỚC bàn chơi nên không bao giờ che bản đồ. draw() gộp cả hai cho các màn không có bàn chơi."""
    N_FALLING = 34
    N_FLOATING = 10                       # tổng 44 hạt
    N_GLINTS = 56
    N_SPARKLES = 36
    TINTS: Sequence[Color] = (SAKURA_LIGHT, SAKURA_DEEP, (250, 232, 226))

    def __init__(self, size: Tuple[int, int], seed: int = 7):
        self.rng = random.Random(seed)
        self.t = 0.0
        self.resize(size)

    def resize(self, size: Tuple[int, int]) -> None:
        w, h = size
        self.size = size
        self.k = min(w / 1536, h / 864)
        self.sky_h = max(2, int(h * 0.62))
        self.sky, self.mirror = _sky_panorama(w, self.sky_h, self.k, random.Random(11))
        self.bed = _lake_bed(w, h)
        self.sheet = _water_sheet(w, h)
        self.ripples = _ripple_layer(w, h)
        tree = pygame.Surface((w, h), pygame.SRCALPHA)
        rng = random.Random(11)
        _canopy(tree, rng, -20, -10, 380 * self.k, self.k)
        _canopy(tree, rng, w + 20, -30, 300 * self.k, self.k * 0.85)
        box = tree.get_bounding_rect()
        self.canopy = (tree.subsurface(box).copy(), box.topleft)
        mist_h = max(8, int(h * 0.16))
        self.mists = [(_mist(w, mist_h, self.rng, 46), 0.0, 9.0), (_mist(w, mist_h, self.rng, 22), 0.34, -5.0)]
        water_h = max(1, h - self.horizon)
        self.waves = (_wave_layer(w, water_h, self.k, 21, 0.42), _wave_layer(w, water_h, self.k, 34, 0.26))
        self.glints = [self._new_glint(i % 2 == 0) for i in range(self.N_GLINTS)]
        self.sparkles = [self._new_sparkle(i % 2 == 0) for i in range(self.N_SPARKLES)]
        scale = max(0.45, self.k)
        self.sizes = tuple(max(3, int(s * scale)) for s in (5, 7, 10))
        # falling: x, y (tỉ lệ), vy, vx, biên độ lắc, tần số lắc, pha, góc, tốc độ xoay, cỡ, màu, alpha
        self.falling = [self._new_petal(self.rng.random()) for _ in range(self.N_FALLING)]
        # floating: x, y, vx, pha, góc, cỡ, màu
        self.floating = [[self.rng.random(), self.rng.uniform(0.34, 0.97), self.rng.uniform(0.003, 0.012) * self.rng.choice((-1, 1)),
                          self.rng.uniform(0, 6.28), self.rng.randrange(_PETAL_FRAMES), self.rng.randint(0, 1),
                          self.rng.randrange(len(self.TINTS))] for _ in range(self.N_FLOATING)]

    @property
    def horizon(self) -> int:
        return int(self.size[1] * 0.24)

    def _water_spot(self, under_sun: bool) -> Tuple[float, float]:
        """Vị trí trên mặt hồ (x, y tỉ lệ). under_sun: nằm trong vệt sáng dưới mặt trời, càng gần càng loe rộng."""
        y = self.rng.uniform(0.01, 1.0)
        if under_sun:
            return self.rng.gauss(SUN_X, 0.02 + 0.09 * y) % 1.0, y
        return self.rng.random(), y

    def _new_glint(self, under_sun: bool) -> tuple:
        x, y = self._water_spot(under_sun)
        return x, y, self.rng.uniform(0, 6.28), self.rng.uniform(0.7, 2.0), int(self.rng.uniform(8, 18) + 34 * y * self.rng.random())

    def _new_sparkle(self, under_sun: bool) -> tuple:
        x, y = self._water_spot(under_sun)
        radius = max(4, int((self.rng.uniform(6, 11) + 16 * y * self.rng.random()) * max(0.6, self.k)))
        return x, y, self.rng.uniform(0, 6.28), self.rng.uniform(1.1, 3.2), radius

    def _new_petal(self, y: float) -> list:
        r = self.rng
        far = r.random() < 0.45  # hạt xa: nhỏ, chậm, nhạt -> tạo chiều sâu
        return [r.random(), y, r.uniform(0.028, 0.05) if far else r.uniform(0.05, 0.095), r.uniform(0.004, 0.028),
                r.uniform(0.004, 0.02), r.uniform(0.5, 1.3), r.uniform(0, 6.28), r.uniform(0, _PETAL_FRAMES),
                r.uniform(3, 11) * r.choice((-1, 1)), 0 if far else r.randint(1, 2), r.randrange(len(self.TINTS)),
                130 if far else 225]

    def _glint(self, half_len: int, level: int) -> pygame.Surface:
        """Vệt sáng ngang 1 px, mờ dần hai đầu, vẽ cộng lên mặt hồ."""
        key = ("glint", half_len, level)
        if key not in _GLOWS:
            x = np.linspace(-1, 1, max(2, 2 * half_len))
            a = (1 - x * x)[:, None, None] * level
            arr = (a * np.array((0.8, 0.85, 1.1))).clip(0, 255).astype(np.uint8)
            _GLOWS[key] = pygame.surfarray.make_surface(arr)
        return _GLOWS[key]

    def update(self, dt_ms: float) -> None:
        dt = dt_ms / 1000
        self.t += dt
        for p in self.falling:
            p[1] += p[2] * dt
            p[0] = (p[0] + p[3] * dt) % 1.05
            p[7] += p[8] * dt
            if p[1] > 1.04:
                p[0], p[1] = self.rng.random(), -0.04
        for p in self.floating:
            p[0] = (p[0] + p[2] * dt) % 1.0

    def _blit_wrapped(self, surf: pygame.Surface, img: pygame.Surface, x: float, y: int, flags: int = 0) -> None:
        w = self.size[0]
        x = int(x) % w
        surf.blit(img, (x, y), special_flags=flags)
        surf.blit(img, (x - w, y), special_flags=flags)

    def draw_under(self, surf: pygame.Surface) -> None:
        surf.blit(self.bed, (0, 0))

    def draw_water(self, surf: pygame.Surface) -> None:
        w, h = self.size
        hy = self.horizon
        # mặt nước
        surf.blit(self.sheet, (0, hy))
        clip = surf.get_clip()
        surf.set_clip(pygame.Rect(0, hy, w, h - hy))
        self._blit_wrapped(surf, self.ripples, self.t * 7, 0, pygame.BLEND_RGB_ADD)
        self._blit_wrapped(surf, self.mirror, 0, hy)
        self._blit_wrapped(surf, self.waves[0], self.t * 11, hy, pygame.BLEND_RGB_ADD)
        self._blit_wrapped(surf, self.waves[1], -self.t * 7, hy + int(1.5 * math.sin(self.t * 0.8)), pygame.BLEND_RGB_ADD)
        surf.set_clip(clip)
        # trời
        if hy > 0:
            top = hy - self.sky_h
            if top > 0:
                surf.fill(SKY_TOP, (0, 0, w, top))
            self._blit_wrapped(surf, self.sky, 0, top)
        for img, offset, speed in self.mists:
            self._blit_wrapped(surf, img, self.t * speed, hy + int(offset * (h - hy)) - img.get_height() // 2)
        for gx, gy, phase, speed, length in self.glints:  # gợn sáng lấp lánh trên hồ
            lvl = max(0.0, math.sin(self.t * speed + phase))
            if lvl > 0.05:
                ln = int(length * max(0.5, self.k))
                c = int(5 * lvl) * 9                    # lượng tử hóa để cache sprite nhỏ
                surf.blit(self._glint(ln, c), (gx * w - ln, hy + (h - hy) * gy), special_flags=pygame.BLEND_RGB_ADD)
        for sx, sy, phase, speed, radius in self.sparkles:  # chỉ lóe lên chốc lát rồi tắt
            lvl = int(max(0.0, math.sin(self.t * speed + phase)) ** 4 * 4.99)
            if lvl:
                surf.blit(_sparkle(radius, lvl), (sx * w - radius, hy + (h - hy) * sy - radius), special_flags=pygame.BLEND_RGB_ADD)
        for x, y, _, phase, ang, sz, tint in self.floating:  # cánh hoa trôi trên mặt hồ (dẹt)
            py = hy + (h - hy) * y
            bob = math.sin(self.t * 0.9 + phase) * 2
            surf.blit(petal_sprite(self.sizes[sz], int(ang), 2, self.TINTS[tint], 190), (x * w, py + bob))
        for p in self.falling:
            x = (p[0] + p[4] * math.sin(self.t * p[5] + p[6])) * w
            squash = int(abs(math.sin(self.t * p[5] * 1.3 + p[6])) * 2.99)
            surf.blit(petal_sprite(self.sizes[p[9]], int(p[7]), squash, self.TINTS[p[10]], p[11]), (x, p[1] * h))
        surf.blit(*self.canopy)

    def draw(self, surf: pygame.Surface) -> None:
        self.draw_under(surf)
        self.draw_water(surf)
