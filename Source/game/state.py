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
            - Bit i được bật: Bridge thứ i tương ứng được đóng
            - Bit i tắt: Bridge thứ i tương ứng mở
        + split_positions: Lưu tọa độ 2 khối block trong trường hợp bị chia thành 2 khối nhỏ
        + controlled_block: block nào đang được điều khiển
            - Bằng 0 nếu điều khiển block không bị phân tách/bị phân tách nhưng không phải block con bị dịch chuyển
            - Bằng 1 nếu điều khiển block con bị phân tách và dịch chuyển đến ô chỉ định
    """
    row: int
    column: int
    orientation: Orientation = Orientation.STANDING
    bridge_mask: int = 0
    split_positions: Optional[Tuple[Coord, Coord]] = None
    controlled_block: int = 0

    def __post_init__(self) -> None:
        """Khởi tạo và kiểm tra trạng thái board và check đầu vào dữ liệu"""
        object.__setattr__(self, "orientation", Orientation(self.orientation))

        if self.split_positions is not None:
            a, b = (tuple(p) for p in self.split_positions)
            if a == b:
                raise ValueError("Initial state: 2 block con không được cùng 1 ô")
            if self.controlled_block not in (0, 1):
                raise ValueError("Initial state: controlled_block chỉ được phép 0/1")

            # Đồng nhất state - Nếu có 2 block con a và b thì việc đổi control giữa 2 block không làm thay đổi state
            ctrl = self.controlled_block
            if b < a:
                a, b, ctrl = b, a, 1 - ctrl
            pos = (a, b)
            object.__setattr__(self, "split_positions", pos)
            object.__setattr__(self, "controlled_block", ctrl)
            object.__setattr__(self, "row", pos[ctrl][0])
            object.__setattr__(self, "column", pos[ctrl][1])
            object.__setattr__(self, "orientation", Orientation.STANDING)
        elif self.controlled_block != 0:
            raise ValueError("Initial state: controlled_block phải bằng 0 khi không split block")
        if self.bridge_mask < 0:
            raise ValueError("Initial state: bridge_mask không âm")

    @property
    def position(self) -> Coord:
        """Lấy tọa độ điểm neo của block"""
        return self.row, self.column

    @property
    def is_split(self) -> bool:
        """Kiểm tra block có được phân tách không"""
        return self.split_positions is not None

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
    def split(cls, first: Coord, second: Coord, controlled_block: int = 0, bridge_mask: int = 0) -> "State":
        """Chuyển trạng thái mới cho block được phân chia"""
        return cls(first[0], first[1], Orientation.STANDING, bridge_mask, (first, second), controlled_block)

    def with_bridge(self, bridge_index: int, enabled: bool) -> "State":
        """
            Chuyển trạng thái khi cầu được mở/đóng
            enable=True/False -> cầu đóng/mở
        """
        if bridge_index < 0:
            raise ValueError("State: Chỉ số của bridge không âm")

        # Nếu cần mở cầu i -> AND với ~ (bit 1 thứ i)
        # Nếu cần đóng cầu i -> OR với bit 1 thứ i
        bit = 1 << bridge_index
        mask = self.bridge_mask | bit if enabled else self.bridge_mask & ~bit

        return State(self.row, self.column, self.orientation, mask, self.split_positions, self.controlled_block)