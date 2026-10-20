import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple, Union
from .state import Orientation, State, Coord
from .context import GameContext
from .tile import Tile, TileType, SwitchKind, SwitchEffect, PermanentState

VOID_TILE = Tile(TileType.VOID)

@dataclass(frozen = True, slots = True)
class Board:
    """
        Lớp board - Sàn chơi
        + tiles: các ô trên sàn chơi
        + start: ô bắt đầu
        + name: tên màn chơi
        + bridge_ids: danh sách id của tất cả cầu trên sàn chơi
        + initial_closed_bridges: các cầu đóng ở trạng thái ban đầu
        + initial_bridge_mask: dãy bit thể hiện cầu đóng/mở ở trạng thái ban đầu
    """
    tiles: Tuple[Tuple[Tile, ...], ...]
    start: Coord
    goal: Coord = field(init = False)
    name: str = "Untitled"

    # Mặc định các cầu được mở
    initial_closed_bridges: Tuple[str, ...] = ()
    bridge_ids: Tuple[str, ...] = field(init = False)
    initial_bridge_mask: int = field(init = False)

    bridge_bit: Mapping[str, int] = field(init = False) # ID cầu -> chuỗi bit tương ứng

    # Tọa độ switch -> (m, n, p)
    # switch_masks lưu theo thứ tự (toggle_mask, close_mask, open_mask).
    # Bit bật trong mọi mask tương ứng với các cầu được tác động.
    switch_masks: Mapping[Coord, Tuple[int, int, int]] = field(init = False)

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

        # Thiết lập thông tin về đích 
        goals = [(r, c) for r, row in enumerate(self.tiles) for c, tile in enumerate(row) if tile.kind == TileType.GOAL]
        if len(goals) != 1:
            raise ValueError(f"Initial board: Sàn chơi phải có đúng 1 ô goal, hiện có {len(goals)}")
        object.__setattr__(self, "goal", goals[0])


        # Thiết lập id cho cầu và switch
        bridge_ids = tuple(sorted({t.tile_id for row in self.tiles for t in row if t.kind == TileType.BRIDGE and t.tile_id}))
        object.__setattr__(self, "bridge_ids", bridge_ids)

        bridge_bit = {bid: 1 << i for i, bid in enumerate(bridge_ids)}
        object.__setattr__(self, "bridge_bit", bridge_bit)

        # Raise error trong trường hợp targets của switch là cầu mà id không nằm trong bridge_ids
        known = set(bridge_ids)
        for row in self.tiles:
            for tile in row:
                if tile.kind == TileType.SWITCH and tile.switch_kind != SwitchKind.SPLIT:
                    missing = set(tile.targets) - known
                    if missing:
                        raise ValueError(f"Initial board: Switch điều khiển cầu không có trong sàn chơi: {sorted(missing)}")

        # Thiết lập switch_masks
        switch_masks = {}
        for r, row in enumerate(self.tiles):
            for c, tile in enumerate(row):
                if tile.kind != TileType.SWITCH or tile.switch_kind == SwitchKind.SPLIT:
                    continue

                m = 0
                for bid in tile.targets:
                    m |= bridge_bit[bid]

                if tile.effect == SwitchEffect.TOGGLE:
                    switch_masks[(r, c)] = (m, 0, 0)
                elif tile.permanent_state == PermanentState.CLOSE:
                    switch_masks[(r, c)] = (0, m, 0)
                else:
                    switch_masks[(r, c)] = (0, 0, m)
        object.__setattr__(self, "switch_masks", switch_masks)

        # Kiểm tra đích dịch chuyển của split switch
        for r, row in enumerate(self.tiles):
            for c, tile in enumerate(row):
                if tile.kind != TileType.SWITCH or tile.switch_kind != SwitchKind.SPLIT:
                    continue
                for dest in tile.destinations:
                    if not self.in_bounds(dest):
                        raise ValueError(f"Initial board: Split switch tại {(r, c)} có đích {dest} nằm ngoài sàn chơi")
                    dest_tile = self.tile_at(dest)
                    if dest_tile.kind == TileType.VOID:
                        raise ValueError(f"Initial board: Split switch tại {(r, c)} có đích {dest} nằm trên ô trống")
                    if dest_tile.kind == TileType.FRAGILE:
                        raise ValueError(f"Initial board: Split switch tại {(r, c)} có đích {dest} là gạch dễ vỡ (cube đứng sẽ làm vỡ)")

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
            initial_mask |= bridge_bit[bridge_id]
        object.__setattr__(self, "initial_bridge_mask", initial_mask)

    @property
    def height(self) -> int:
        """Lấy chiều cao của sàn chơi"""
        return len(self.tiles)

    @property
    def width(self) -> int:
        """Lấy chiều rộng của sàn chơi"""
        return len(self.tiles[0])

    def in_bounds(self, coord: Coord) -> bool:
        """Kiểm tra 1 tọa độ thuộc trong sàn chơi"""
        row, col = coord
        return 0 <= row < len(self.tiles) and 0 <= col < len(self.tiles[0])

    def tile_at(self, coord: Coord) -> Tile:
        """Lấy thông tin tile khi có tọa độ"""
        row, col = coord
        if not self.in_bounds(coord):
            return VOID_TILE
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

    @property
    def initial_context(self) -> GameContext:
        """Tạo context ban đầu để quản lý quyền điều khiển ngoài State"""
        return GameContext(self.initial_state)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Board":
        """
            Gắn thông tin dữ liệu sàn chơi từ file Mapping.

            Các trường trong Mapping (JSON):
            - name: tên màn chơi
            - start: vị trí bắt đầu
            - goal: đích đến
            - initial_closed_bridges: danh sách ID cầu bắt đầu ở trạng thái đóng, các cầu còn lại bắt đầu mở
            - tiles: mảng 2 chiều gồm các ô, mỗi tile có các trường sau:
                + type: loại ô
                + id: mã định danh ô
                + metadata: thông tin bổ sung
                + switch: loại công tắc
                + effect: ảnh hưởng của công tắc lên cầu
                + set_to: trạng thái cầu switch PERMANENT đặt (open/closed, mặc định open)
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
        # source là chuỗi JSON text
        if isinstance(source, str) and source.lstrip().startswith("{"):
            text = source
            origin = "Chuỗi JSON"
        else: # source là path str hoặc là Path
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"JSON Content: Không tìm thấy file stage: {path}")
            text = path.read_text(encoding="utf-8")
            origin = str(path)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON Content: {origin} không phải JSON hợp lệ: {exc}") from exc

        if not isinstance(data, Mapping):
            raise ValueError(f"JSON Content: {origin} phải là một object JSON ở cấp ngoài cùng")

        return cls.from_dict(data)
