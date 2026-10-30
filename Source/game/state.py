from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

# Khai báo tọa độ điểm neo của block - Quy ước:
# + Nếu block đứng thẳng: Tọa độ block lấy theo ô tại nó đang đứng
# + Nếu block nằm ngang: Tọa độ lấy theo ô bên trên trái cùng trong 2 ô bị chiếm
Coord = Tuple[int, int]

# Hướng đặt block
class Orientation(str, Enum):
    STANDING = "standing"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

@dataclass(frozen = True, slots = True)
class State:
    """
        Lớp State: Lưu trạng thái của game trong mỗi bước di chuyển
        + row: chỉ số hàng của điểm neo
        + column: chỉ số cột của điểm neo
        + orientation: hướng đặt của block
        + bridge_mask: số nguyên đặc trưng cho dãy bit
            - Bit i bật: Bridge thứ i tương ứng đóng, không cho block di chuyển
            - Bit i tắt: Bridge thứ i tương ứng mở, cho phép block di chuyển
        + split_positions: Lưu tọa độ chuẩn hóa của 2 khối block trong trường hợp bị chia thành 2 khối nhỏ
    """
    row: int
    column: int
    orientation: Orientation = Orientation.STANDING
    bridge_mask: int = 0
    split_positions: Optional[Tuple[Coord, Coord]] = None

    def __post_init__(self) -> None:
        """Khởi tạo và kiểm tra trạng thái board và check đầu vào dữ liệu"""
        object.__setattr__(self, "orientation", Orientation(self.orientation))

        if self.split_positions is not None:
            a, b = (tuple(p) for p in self.split_positions)
            if a == b:
                raise ValueError("Initial state: 2 block con không được cùng 1 ô")
            if b < a:
                a, b = b, a
            pos = (a, b)
            object.__setattr__(self, "split_positions", pos)
            object.__setattr__(self, "row", a[0])
            object.__setattr__(self, "column", a[1])
            object.__setattr__(self, "orientation", Orientation.STANDING)
        if self.bridge_mask < 0:
            raise ValueError("Initial state: bridge_mask không âm")

    @property
    def position(self) -> Coord:
        """Lấy tọa độ điểm neo của block"""
        return self.row, self.column

    @property
    def occupied_cells(self) -> Tuple[Coord, ...]:
        """Lấy các ô bị block chiếm"""
        if self.split_positions is not None:
            return self.split_positions
        
        if self.orientation == Orientation.STANDING:
            return ((self.row, self.column),)
        
        if self.orientation == Orientation.HORIZONTAL:
            return ((self.row, self.column), (self.row, self.column + 1))
        
        return ((self.row, self.column), (self.row + 1, self.column))

    @classmethod
    def split(cls, first: Coord, second: Coord, bridge_mask: int = 0) -> "State":
        """Chuyển trạng thái mới cho block được phân chia"""
        return cls(first[0], first[1], Orientation.STANDING, bridge_mask, (first, second))

    def toggle_bridge(self, bridge_index: int) -> "State":
        """Đảo trạng thái cầu (đóng sang mở và ngược lại)."""
        if bridge_index < 0:
            raise ValueError("Bridge toggle: Chỉ số của bridge không âm")
        
        mask = self.bridge_mask ^ (1 << bridge_index)
        return State(self.row, self.column, self.orientation, mask, self.split_positions)