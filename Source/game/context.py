from dataclasses import dataclass
from typing import Optional
from .state import Coord, State

@dataclass(frozen=True, slots=True)
class GameContext:
    """
        Thông tin điều khiển tách biệt khỏi cấu hình vật lý của State
        Chứa trạng thái của game và cube đang được điều khiển
    """
    state: State
    selected_cube: Optional[Coord] = None

    def __post_init__(self) -> None:
        if self.state.is_split:
            selected = tuple(self.selected_cube) if self.selected_cube is not None else self.state.split_positions[0]
            if selected not in self.state.split_positions:
                raise ValueError("Game context: selected_cube phải là một trong hai block con")
            object.__setattr__(self, "selected_cube", selected)
        elif self.selected_cube is not None:
            raise ValueError("Game context: selected_cube chỉ hợp lệ khi State đang split")

    def switch_control(self) -> "GameContext":
        """Chọn cube con còn lại mà không làm thay đổi State"""
        if not self.state.is_split:
            return self
        other = next(cell for cell in self.state.split_positions if cell != self.selected_cube)
        return GameContext(self.state, other)