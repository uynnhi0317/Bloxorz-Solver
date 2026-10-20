from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple, Literal
from .state import Coord

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

# Trạng thái đóng/mở cố định của SwitchPermanent
class PermanentState(str, Enum):
    OPEN = "open"
    CLOSE = "close"

@dataclass(frozen = True, slots = True)
class Tile:
    """
        Lớp chứa thông tin của 1 tile:
        + kind: loại tile
        + tile_id: mã định danh cho tile (trong trường hợp là cầu cần có id để switch kết nối)
        + switch_kind: loại switch
        + effect: ảnh hưởng của switch 
        + targets: chứa các nhóm cầu mà switch điều khiển
        + permanent_state: trạng thái cầu mà switch PERMANENT đặt
        + destinations: 2 ô spawn block con khi block đi vào split switch
        + metadata: thông tin bổ sung dạng key, value (trong trường hợp cần dùng)
    """
    kind: TileType
    tile_id: Optional[str] = None
    switch_kind: Optional[SwitchKind] = None
    effect: Optional[SwitchEffect] = None
    permanent_state: Optional[PermanentState] = None
    targets: Tuple[str, ...] = ()
    destinations: Tuple[Coord, ...] = ()
    metadata: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Khởi tạo board và check dữ liệu đầu vào"""
        if self.kind == TileType.SWITCH:
            if self.switch_kind is None:
                raise ValueError("Initial board: Switch phải có thông tin về loại")
            
            if self.switch_kind != SwitchKind.SPLIT and self.effect is None:
                raise ValueError("Initial board: Switch phải có effect toggle/permanent")
            
            if self.switch_kind != SwitchKind.SPLIT and not self.targets:
                raise ValueError("Initial board: Switch không tách block phải có cầu kết nối")

            if self.switch_kind == SwitchKind.SPLIT and (len(self.destinations) != 2 or self.destinations[0] == self.destinations[1]):
                raise ValueError("Initial board: Split switch phải có đúng 2 tọa độ đích khác nhau")

            if self.switch_kind == SwitchKind.SPLIT and self.targets:
                raise ValueError("Initial board: Split switch không điều khiển cầu")
            
            if self.switch_kind == SwitchKind.SPLIT and self.effect is not None:
                raise ValueError("Initial board: Split switch không có effect")

            if self.effect == SwitchEffect.PERMANENT and self.permanent_state is None:
                raise ValueError("Initial board: Switch PERMANENT phải có permanent_state")

            if self.effect != SwitchEffect.PERMANENT and self.permanent_state is not None:
                raise ValueError("Initial board: permanent_state chỉ hợp lệ với switch PERMANENT")
            
        elif self.switch_kind is not None or self.effect is not None or self.permanent_state is not None or self.targets:
            raise ValueError("Initial board: Thông tin liên quan đến switch không hợp lệ")
        
        if self.kind == TileType.BRIDGE and not self.tile_id:
            raise ValueError("Initial board: Cầu phải có mã định danh")

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "Tile":
        """Dựng tile dựa trên thông số của mapping"""
        if not isinstance(spec, Mapping):
            raise TypeError("Tile spec: Loại dữ liệu đầu vào của tile không hợp lệ")

        # Đọc thông số tile từ mapping
        kind = TileType(spec["type"])

        # Lấy các thuộc tính chung của tile
        common = {"tile_id": spec.get("id"), "metadata": tuple(sorted((str(k), str(v)) for k, v in spec.get("metadata", {}).items()))}

        # Trường hợp tile là switch thì phải xử lý các thuộc tính liên quan đến kích hoạt
        if kind == TileType.SWITCH:
            switch_kind = SwitchKind(spec.get("switch", "soft"))
            effect_value = spec.get("effect")
            destinations = tuple((int(r), int(c)) for r, c in spec.get("destinations", ()))

            if effect_value is None and switch_kind != SwitchKind.SPLIT:
                effect_value = "toggle"

            effect = SwitchEffect(effect_value) if effect_value is not None else None
            permanent_state = None
            if effect == SwitchEffect.PERMANENT:
                set_to = spec.get("set_to", "open")
                try:
                    permanent_state = PermanentState(set_to)
                except ValueError as exc:
                    raise ValueError("Tile spec: set_to của switch PERMANENT phải là 'open' hoặc 'closed'") from exc

            return cls(kind, switch_kind = switch_kind, 
                       effect = effect,
                       permanent_state = permanent_state,
                       targets = tuple(spec.get("targets", ())), 
                       destinations = destinations, **common)

        # Trường hợp tile không kích hoạt
        return cls(kind, **common)