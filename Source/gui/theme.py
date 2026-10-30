"""theme.py - Theme "Đền thần bên hồ" (Nhật Bản cổ, hoa anh đào): bảng màu + nền động.

Mọi hình ảnh đều vẽ bằng code (không cần file ảnh ngoài), nên không có rủi ro thiếu asset.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pygame

Color = Tuple[int, int, int]

# ---- bảng màu gốc của concept
SAKURA_LIGHT: Color = (247, 198, 208)  # #F7C6D0
SAKURA_DEEP: Color = (229, 143, 166)   # #E58FA6
WOOD: Color = (164, 117, 81)           # #A47551
TORII_RED: Color = (200, 50, 60)       # #C8323C
VOID_DARK: Color = (30, 42, 68)        # #1E2A44
WATER_DEEP: Color = (47, 75, 110)      # #2F4B6E
LEAF: Color = (127, 176, 105)          # #7FB069
CREAM: Color = (255, 246, 233)         # #FFF6E9

# ---- màu thế giới (renderer dùng)
PALETTE = {
    "floor": (150, 146, 130),          # đá lát sân
    "moss": (112, 146, 90),            # rêu (lá cây dịu lại để ô Floor vẫn trung tính)
    "seam": (112, 108, 96),            # đường ron giữa các phiến đá
    "fragile": (186, 110, 44),         # nền đống lá
    "leaves": ((236, 176, 72), (224, 136, 52), (244, 202, 96), (206, 104, 44)),
    "bridge": TORII_RED,
    "bridge_dark": (140, 34, 44),      # thanh đỡ, lan can
    "bridge_ghost": (150, 58, 74),     # ván cầu đóng chìm dưới nước
    "goal": (88, 92, 108),             # đá sân đền quanh cổng
    "goal_hole": (12, 16, 34),
    "goal_inner": (40, 54, 92),
    "torii": TORII_RED,
    "torii_cap": (42, 34, 40),
    "lantern": (168, 164, 150),
    "lantern_light": (255, 214, 140),
    "switch_base": (92, 92, 92),
    "soft": SAKURA_LIGHT,
    "soft_off": (168, 112, 126),
    "heavy": (168, 214, 128),          # tre non
    "heavy_off": (92, 120, 76),
    "split": CREAM,
    "split_off": (196, 186, 170),
    "block": (32, 28, 36),             # thân Vô Diện
    "cube_active": (42, 36, 48),
    "cube_idle": (24, 22, 30),
    "mask": (250, 244, 232),
    "mask_mark": (116, 72, 136),       # vệt tím trên mặt nạ
    "mask_eye": (24, 20, 28),
    "gold": (232, 190, 92),            # đồng vàng của phần thân
}

# ---- màu giao diện (controls/app dùng)
UI = {
    "bg": VOID_DARK, "bar": (28, 32, 52), "panel": (46, 44, 62), "border": (150, 108, 76),
    "text": CREAM, "muted": (216, 204, 194), "accent": SAKURA_LIGHT,
    "accent_dark": (120, 70, 86), "win": SAKURA_LIGHT, "lose": (242, 134, 96),
    "wood": (120, 82, 54), "wood_hover": (148, 104, 70), "wood_press": (98, 66, 44),
    "wood_edge": (70, 44, 30), "wood_off": (72, 68, 70), "text_off": (150, 146, 146),
    "cord": TORII_RED, "ok": (156, 214, 124), "fail": (244, 112, 104), "dim": (14, 18, 34),
}
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
def _ridge(rng: random.Random, w: int, base: float, amp: float, n: int = 5) -> List[Tuple[float, float]]:
    """Đường đồi núi xa: tổng vài sóng sin."""
    waves = [(rng.uniform(0.6, 3.2), rng.uniform(0, 6.28), rng.uniform(0.3, 1.0)) for _ in range(n)]
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
    branch = (34, 28, 40, 255)
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
    shades = ((178, 112, 138, 150), (206, 134, 156, 150), (229, 158, 178, 130))
    for x, y in tips:
        for _ in range(9):
            r = rng.uniform(8, 26) * k
            px, py = x + rng.gauss(0, 30 * k), y + rng.gauss(0, 20 * k)
            blob = pygame.Surface((int(2 * r) + 2, int(2 * r) + 2), pygame.SRCALPHA)
            pygame.draw.circle(blob, rng.choice(shades), (r, r), r)
            s.blit(blob, (px - r, py - r))


def _backdrop(w: int, h: int, horizon: int, rng: random.Random) -> pygame.Surface:
    """Bầu trời hoàng hôn tím, đồi + chùa + torii, mặt hồ soi bóng (dựng bằng numpy)."""
    k = h / 864
    # bầu trời
    t = np.linspace(0, 1, max(1, horizon))[None, :, None]
    top, low = np.array((22, 26, 52)), np.array((104, 84, 118))
    sky = top + (low - top) * t ** 1.5
    img = np.empty((w, h, 3), dtype=float)
    img[:, :horizon] = sky
    # mặt hồ
    t = np.linspace(0, 1, max(1, h - horizon))[None, :, None]
    lake_top, lake_bot = np.array(WATER_DEEP) * 1.1, np.array(VOID_DARK) * 0.82
    img[:, horizon:] = lake_top + (lake_bot - lake_top) * t ** 0.7
    base = pygame.surfarray.make_surface(img.clip(0, 255).astype(np.uint8))

    # vật thể trên đường chân trời vẽ riêng để còn lật làm bóng nước
    land = pygame.Surface((w, h), pygame.SRCALPHA)
    mx, my, mr = int(w * 0.76), int(horizon * 0.42), max(10, int(40 * k))
    base.blit(glow_sprite(mr * 5, (150, 120, 130), 0.9), (mx - mr * 5, my - mr * 5), special_flags=pygame.BLEND_RGB_ADD)
    pygame.draw.circle(base, (250, 232, 214), (mx, my), mr)
    far = [(0, h)] + _ridge(rng, w, horizon - 26 * k, 34 * k) + [(w, h)]
    pygame.draw.polygon(land, (66, 62, 96, 255), far)
    near = [(0, h)] + _ridge(rng, w, horizon - 6 * k, 20 * k) + [(w, h)]
    pygame.draw.polygon(land, (42, 40, 70, 255), near)
    _pagoda_silhouette(land, w * 0.17, horizon - 8 * k, 112 * k, (36, 32, 60, 255))
    _torii_silhouette(land, w * 0.86, horizon + 18 * k, 96 * k, (112, 36, 52, 255))
    base.blit(land, (0, 0), (0, 0, w, horizon + int(18 * k)))

    # bóng nước: lật phần trên đường chân trời xuống, gợn ngang, nhạt dần
    arr = pygame.surfarray.array3d(base).astype(float)
    n = min(horizon, h - horizon)
    ys = np.arange(n)
    shift = (np.sin(ys * 0.34) * (1 + ys * 0.05)).astype(int)
    fade = (0.42 * (1 - ys / max(1, n)) ** 1.4)[None, :, None]
    src = arr[:, horizon - 1 - ys]
    for i in range(n):
        src[:, i] = np.roll(src[:, i], shift[i], axis=0)
    arr[:, horizon:horizon + n] = arr[:, horizon:horizon + n] * (1 - fade) + src * fade
    # vignette tối hai mép để mắt tập trung vào bàn chơi
    x = np.linspace(-1, 1, w)[:, None, None]
    yy = np.linspace(-1, 1, h)[None, :, None]
    arr *= np.clip(1.08 - 0.42 * np.abs(x) ** 2.4 - 0.26 * np.abs(yy) ** 3, 0.0, 1.0)
    out = pygame.surfarray.make_surface(arr.clip(0, 255).astype(np.uint8))

    # tán hoa hai góc trên
    tree = pygame.Surface((w, h), pygame.SRCALPHA)
    _canopy(tree, rng, -20, -10, 380 * k, k)
    _canopy(tree, rng, w + 20, -30, 300 * k, k * 0.85)
    out.blit(tree, (0, 0))
    return out


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
    rgb[...] = (214, 208, 226)
    del rgb
    a = pygame.surfarray.pixels_alpha(s)
    a[...] = alpha
    del a
    return s


class Background:
    """Nền động: hồ + bóng đền (tĩnh), sương trôi, gợn sáng, cánh hoa rơi và cánh hoa trôi trên nước.
    Luôn vẽ TRƯỚC bàn chơi nên không bao giờ che bản đồ."""
    N_FALLING = 34
    N_FLOATING = 10                       # tổng 44 hạt
    TINTS: Sequence[Color] = (SAKURA_LIGHT, SAKURA_DEEP, (252, 222, 228))

    def __init__(self, size: Tuple[int, int], seed: int = 7):
        self.rng = random.Random(seed)
        self.t = 0.0
        self.resize(size)

    def resize(self, size: Tuple[int, int]) -> None:
        w, h = size
        self.size = size
        self.k = min(w / 1536, h / 864)
        self.horizon = max(1, int(h * 0.24))
        self.backdrop = _backdrop(w, h, self.horizon, random.Random(11))
        mist_h = max(8, int(h * 0.16))
        self.mists = [(_mist(w, mist_h, self.rng, 46), self.horizon - mist_h // 2, 9.0),
                      (_mist(w, mist_h, self.rng, 26), int(h * 0.56), -5.0)]
        self.glints = [(self.rng.uniform(0.04, 0.96), self.rng.uniform(0.3, 1.0), self.rng.uniform(0, 6.28),
                        self.rng.uniform(0.5, 1.4), self.rng.randint(14, 46)) for _ in range(20)]
        scale = max(0.45, self.k)
        self.sizes = tuple(max(3, int(s * scale)) for s in (5, 7, 10))
        # falling: x, y (tỉ lệ), vy, vx, biên độ lắc, tần số lắc, pha, góc, tốc độ xoay, cỡ, màu, alpha
        self.falling = [self._new_petal(self.rng.random()) for _ in range(self.N_FALLING)]
        # floating: x, y, vx, pha, góc, cỡ, màu
        self.floating = [[self.rng.random(), self.rng.uniform(0.34, 0.97), self.rng.uniform(0.003, 0.012) * self.rng.choice((-1, 1)),
                          self.rng.uniform(0, 6.28), self.rng.randrange(_PETAL_FRAMES), self.rng.randint(0, 1),
                          self.rng.randrange(len(self.TINTS))] for _ in range(self.N_FLOATING)]

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

    def draw(self, surf: pygame.Surface) -> None:
        w, h = self.size
        surf.blit(self.backdrop, (0, 0))
        for img, y, speed in self.mists:
            x = int(self.t * speed) % w
            surf.blit(img, (x, y))
            surf.blit(img, (x - w, y))
        for gx, gy, phase, speed, length in self.glints:  # gợn sáng lấp lánh trên hồ
            lvl = max(0.0, math.sin(self.t * speed + phase))
            if lvl > 0.05:
                y = self.horizon + (h - self.horizon) * gy
                ln = int(length * max(0.5, self.k))
                c = int(5 * lvl) * 9                    # lượng tử hóa để cache sprite nhỏ
                surf.blit(self._glint(ln, c), (gx * w - ln, y), special_flags=pygame.BLEND_RGB_ADD)
        s_small = self.sizes
        for x, y, _, phase, ang, sz, tint in self.floating:  # cánh hoa trôi trên mặt hồ (dẹt)
            bob = math.sin(self.t * 0.9 + phase) * 2
            spr = petal_sprite(s_small[sz], int(ang), 2, self.TINTS[tint], 190)
            surf.blit(spr, (x * w, y * h + bob))
        for p in self.falling:
            x = (p[0] + p[4] * math.sin(self.t * p[5] + p[6])) * w
            squash = int(abs(math.sin(self.t * p[5] * 1.3 + p[6])) * 2.99)
            spr = petal_sprite(s_small[p[9]], int(p[7]), squash, self.TINTS[p[10]], p[11])
            surf.blit(spr, (x, p[1] * h))
