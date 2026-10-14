import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union
from .state import Orientation, State

"""
    Quy ước ký hiệu trong map của board:
    + `.`: sàn thường
    + `#`: ô trống
    + `G`: đích đến
    + `F`: gạch dễ vỡ
    + `S`: soft switch
    + `H`: heavy switch
    + `X`: split switch
    + `B`: cầu
"""

# Quy định loại tile
class TileType(str, Enum):
    FLOOR = "floor"
    VOID = "void"
    GOAL = "goal"
    FRAGILE = "fragile"
    SWITCH = "switch"
    BRIDGE = "bridge"

# Quy định loại switch
class SwitchKind(str, Enum):
    SOFT = "soft"
    HEAVY = "heavy"
    SPLIT = "split"

# Quy định cách ảnh hướng của switch lên cầu
class SwitchEffect(str, Enum):
    """Ảnh hưởng của switch lên cầu"""
    TOGGLE = "toggle"
    PERMANENT = "permanent"

@dataclass(frozen = True, slots = True)
class Tile:
    """
        Lớp chứa thông tin của 1 tile:
        + kind: loại tile
        + tile_id: mã định danh cho tile (trong trường hợp là cầu cần có id để switch kết nối)
        + switch_kind: loại switch
        + effect: ảnh hưởng của switch 
        + targets: chứa các nhóm cầu mà switch điều khiển
        + allows_standing: ô đó có cho phép block đứng không
        + metadata: thông tin bổ sung dạng key, value (trong trường hợp cần dùng)
    """
    kind: TileType
    tile_id: Optional[str] = None
    switch_kind: Optional[SwitchKind] = None
    effect: Optional[SwitchEffect] = None
    targets: Tuple[str, ...] = ()
    allows_standing: bool = True
    metadata: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Khởi tạo board và check dữ liệu đầu vào"""
        if self.kind == TileType.SWITCH:
            if self.switch_kind is None:
                raise ValueError("Initial board: Switch phải có thông tin về loại")
            
            if self.switch_kind != SwitchKind.SPLIT and self.effect is None:
                raise ValueError("Initial board: Switch phải có effect toggle/permanent")
            
            if not self.targets and self.switch_kind != SwitchKind.SPLIT:
                raise ValueError("Initial board: Switch không tách block phải có cầu kết nối")
            
        elif self.switch_kind is not None or self.effect is not None or self.targets:
            raise ValueError("Initial board: Thông tin liên quan đến switch không hợp lệ")
        
        if self.kind == TileType.BRIDGE and not self.tile_id:
            raise ValueError("Initial board: Cầu phải có mã định danh")

    @classmethod
    def from_spec(cls, spec: Union[str, Mapping[str, Any]]) -> "Tile":
        """Dựng tile dựa trên thông số của mapping hoặc ký tự"""
        if isinstance(spec, str):
            # Trường hợp là các tile không kích hoạt thì không cần setup các thông số liên quan
            # Xử lý riêng đối với switch và cầu
            aliases = {".": TileType.FLOOR, "#": TileType.VOID, "G": TileType.GOAL, "F": TileType.FRAGILE, 
                       **{kind.value: kind for kind in TileType}}

            if spec in aliases:
                kind = aliases[spec]
                return cls(kind, allows_standing=(kind != TileType.FRAGILE))
            
            if spec == "S":
                return cls(TileType.SWITCH, tile_id="switch", switch_kind=SwitchKind.SOFT, effect=SwitchEffect.TOGGLE, targets=("bridge",))
            
            if spec == "H":
                return cls(TileType.SWITCH, tile_id="switch", switch_kind=SwitchKind.HEAVY, effect=SwitchEffect.TOGGLE, targets=("bridge",))

            if spec == "X":
                return cls(TileType.SWITCH, tile_id="switch", switch_kind=SwitchKind.SPLIT)
            
            if spec == "B":
                return cls(TileType.BRIDGE, tile_id="bridge")
            
            raise ValueError(f"Tile spec: Không xác định được thông số tile")

        if not isinstance(spec, Mapping):
            raise TypeError("Tile spec: Loại tile dữ liệu đầu vào không hợp lệ")

        # Đọc thông số tile từ mapping
        kind = TileType(spec["type"])

        common = {"tile_id": spec.get("id"),
                  "allows_standing": bool(spec.get("allows_standing", kind != TileType.FRAGILE)),
                  "metadata": tuple(sorted((str(k), str(v)) for k, v in spec.get("metadata", {}).items()))}

        # Trường hợp tile là switch thì phải xử lý các thuộc tính liên quan đến kích hoạt
        if kind == TileType.SWITCH:
            switch_kind = SwitchKind(spec.get("switch", "soft"))
            effect_value = spec.get("effect")

            if effect_value is None and switch_kind != SwitchKind.SPLIT:
                effect_value = "toggle"

            return cls(kind, switch_kind = switch_kind, 
                       effect = SwitchEffect(effect_value) if effect_value is not None else None,
                       targets = tuple(spec.get("targets", ())), **common)

        # Trường hợp tile không kích hoạt
        return cls(kind, **common)


@dataclass(frozen = True, slots = True)
class Board:
    """
        Lớp board - Sàn chơi
        + tiles: các ô trên sàn chơi
        + start: ô bắt đầu
        + name: tên màn chơi
        + bridge_ids: danh sách id của tất cả cầu trên sàn chơi
        + switch_ids: danh sách id của tất cả switch trên sàn chơi
        + initial_closed_bridges: các cầu đóng ở trạng thái ban đầu
        + initial_bridge_mask: dãy bit thể hiện cầu đóng/mở ở trạng thái ban đầu
    """
    tiles: Tuple[Tuple[Tile, ...], ...]
    start: Tuple[int, int]
    name: str = "Untitled"

    # Mặc định các cầu được mở
    initial_closed_bridges: Tuple[str, ...] = ()
    bridge_ids: Tuple[str, ...] = field(init = False)
    switch_ids: Tuple[str, ...] = field(init = False)
    initial_bridge_mask: int = field(init = False)

    def __post_init__(self) -> None:
        """Khởi tạo sàn chơi và chuẩn hóa/kiểm tra dữ liệu"""
        # Thiết lập thông số cho tiles và ô bắt đầu
        object.__setattr__(self, "tiles", tuple(tuple(row) for row in self.tiles))
        object.__setattr__(self, "start", tuple(self.start))
        object.__setattr__(self, "initial_closed_bridges", tuple(self.initial_closed_bridges))

        # Kiểm tra thông số cho tiles và ô bắt đầu
        if not self.tiles or not self.tiles[0]:
            raise ValueError("Initial board: Sàn chơi phải chứa ít nhất 1 ô")
        
        width = len(self.tiles[0])

        if any(len(row) != width for row in self.tiles):
            raise ValueError("Initial board: Các hàng phải có cùng số ô")
        
        if not self.in_bounds(self.start):
            raise ValueError(f"Initial board: Vị trí bắt đầu nằm ngoài sàn chơi {self.start}")
        
        if self.tile_at(self.start).kind == TileType.VOID:
            raise ValueError("Initial board: Vị trí bắt đầu không thể nằm ở ô trống")

        # Thiết lập id cho cầu và switch
        bridge_ids = tuple(sorted({t.tile_id for row in self.tiles for t in row if t.kind == TileType.BRIDGE and t.tile_id}))
        switch_ids = tuple(sorted({t.tile_id for row in self.tiles for t in row if t.kind == TileType.SWITCH and t.tile_id}))
        object.__setattr__(self, "bridge_ids", bridge_ids)
        object.__setattr__(self, "switch_ids", switch_ids)

        # Raise error trong trường hợp targets của switch là cầu mà id không nằm trong bridge_ids
        known = set(bridge_ids)
        for row in self.tiles:
            for tile in row:
                if tile.kind == TileType.SWITCH and tile.switch_kind != SwitchKind.SPLIT:
                    missing = set(tile.targets) - known
                    if missing:
                        raise ValueError(f"Initial board: Switch điều khiển cầu không có trong sàn chơi: {sorted(missing)}")

        # Tạo trạng thái các cây cầu đóng/mở bằng chuỗi bit nhị phân
        # - Bit i được bật: Bridge thứ i tương ứng được đóng
        # - Bit i tắt: Bridge thứ i tương ứng mở
        unknown_initial_bridges = set(self.initial_closed_bridges) - set(bridge_ids)
        if unknown_initial_bridges:
            raise ValueError(f"Initial board: Trạng thái cầu ban đầu tham chiếu ID không tồn tại: {sorted(unknown_initial_bridges)}")
        
        if len(set(self.initial_closed_bridges)) != len(self.initial_closed_bridges):
            raise ValueError("Initial board: initial_closed_bridges không được chứa ID trùng")
        
        initial_mask = 0
        for bridge_id in self.initial_closed_bridges:
            initial_mask |= 1 << bridge_ids.index(bridge_id)
        object.__setattr__(self, "initial_bridge_mask", initial_mask)

    @property
    def height(self) -> int:
        """Lấy chiều cao của sàn chơi"""
        return len(self.tiles)

    @property
    def width(self) -> int:
        """Lấy chiều rộng của sàn chơi"""
        return len(self.tiles[0])

    def in_bounds(self, coord: Tuple[int, int]) -> bool:
        """Kiểm tra 1 tọa độ thuộc trong sàn chơi"""
        row, col = coord
        return 0 <= row < len(self.tiles) and 0 <= col < len(self.tiles[0])

    def tile_at(self, coord: Tuple[int, int]) -> Tile:
        """Lấy thông tin tile khi có tọa độ"""
        row, col = coord
        if not self.in_bounds(coord):
            return Tile(TileType.VOID)
        return self.tiles[row][col]

    def bridge_index(self, bridge_id: str) -> int:
        """Tra cứu thứ tự cầu theo bridge_id"""
        try:
            return self.bridge_ids.index(bridge_id)
        except ValueError as exc:
            raise KeyError(f"Bridge: Cầu không xác định: {bridge_id}") from exc

    @property
    def initial_state(self) -> State:
        """Tạo cấu hình ban đầu của sàn chơi với trạng thái khởi tạo của các cây cầu"""
        return State(self.start[0], self.start[1], Orientation.STANDING, bridge_mask=self.initial_bridge_mask)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Board":
        """
            Gắn thông tin dữ liệu sàn chơi từ file Mapping.
            
            Cấu trúc file Mapping (từ JSON):
            {
                "name": ...
                "start": ...,
                "tiles": [
                    [
                        {"type": ...},
                        {"type": ..., ...},
                        {"type": ..., "supports_standing": ..., ...}
                    ],
                    [
                        ...
                    ],
                    [
                        ...
                    ]
                ]
            }

            Các trường trong Mapping (JSON):
            - name: tên màn chơi
            - start: vị trí bắt đầu
            - initial_closed_bridges: danh sách ID cầu bắt đầu ở trạng thái đóng,
              các cầu còn lại bắt đầu mở
            - tiles: mảng 2 chiều gồm các ô, mỗi tile có các trường sau:
                + type: loại ô
                + id: mã định danh ô
                + allows_standing: cho phép đứng trên ô không
                + metadata: thông tin bổ sung
                + switch: loại công tắc
                + effect: ảnh hưởng của công tắc lên cầu
                + targets: danh sách ID nhóm cầu bị tác động (nếu tile là switch)
        """
        # Lấy thông tin từng ô
        rows = data.get("tiles")
        if not isinstance(rows, list) or not rows:
            raise ValueError("JSON Content: JSON sai - tiles[] phải là list khác rỗng")
        tiles = tuple(tuple(Tile.from_spec(cell) for cell in row) for row in rows)

        # Tạo điểm khởi đầu
        start = tuple(data["start"])
        if len(start) != 2:
            raise ValueError("JSON Content: Điểm khởi đầu phải là [row, column]")

        # Tạo danh sách cầu đóng khởi tạo
        initial_closed_bridges = data.get("initial_closed_bridges", ())
        if not isinstance(initial_closed_bridges, (list, tuple)):
            raise ValueError("JSON Content: initial_closed_bridges phải là danh sách bridge ID")
        
        return cls(tiles = tiles, start = (int(start[0]), int(start[1])), name = str(data.get("name", "Untitled")),
                   initial_closed_bridges = tuple(str(bridge_id) for bridge_id in initial_closed_bridges))

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> "Board":
        """Đọc file JSON text hoặc có đường dẫn dẫn tới"""
        # Đọc file JSON nằm theo đường dẫn
        if isinstance(source, Path):
            data = json.loads(source.read_text(encoding="utf-8"))
        else:
            candidate = Path(source)
            if "\n" not in source and candidate.exists():
                data = json.loads(candidate.read_text(encoding="utf-8"))
            else:
                # Load text data (source là data thô)
                data = json.loads(source)
        return cls.from_dict(data)

    @classmethod
    def from_text(cls, text: str, start: Optional[Tuple[int, int]] = None, name: str = "Untitled") -> "Board":
        """
            Gán thông tin board từ dữ liệu text
            Xem 'P' như điểm bắt đầu của block
        """
        # Đọc dữ liệu từng line trong file
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Text read: Text rỗng")

        # Lấy điểm bắt đầu (`P`)
        if start is None:
            found = [(r, c) for r, line in enumerate(lines) for c, ch in enumerate(line) if ch == "P"]
            if len(found) != 1:
                raise ValueError("Text read: Chỉ có 1 điểm bắt đầu (`P`) duy nhất")
            start = found[0]

        # Lấy thông tin từng ô trong sàn chơi
        rows = []
        for line in lines:
            row = []
            for ch in line:
                row.append(Tile(TileType.FLOOR) if ch == "P" else Tile.from_spec(ch))
            rows.append(tuple(row))

        return cls(tuple(rows), start, name)

    @classmethod
    def from_text_file(cls, source: Union[str, Path], start: Optional[Tuple[int, int]] = None, name: str = "Untitled") -> "Board":
        """Load file .txt (text) lưu sàn chơi"""
        # Đọc file text trên đường dẫn
        if isinstance(source, Path):
            text = source.read_text(encoding = "utf-8")
        elif "\n" not in source and "\r" not in source and Path(source).is_file():
            text = Path(source).read_text(encoding = "utf-8")
        else:
            # Đọc file text thô
            text = source

        return cls.from_text(text, start = start, name = name)
