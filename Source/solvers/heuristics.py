import math
from collections import deque
from typing import Callable, Dict, List

from game.board import Board
from game.rules import DIRECTION_DELTAS
from game.state import Coord, Orientation, State
from game.tile import SwitchKind, TileType
from solvers.ucs import MIN_MOVE_COST

Heuristic = Callable[[State], float]
HeuristicFactory = Callable[[Board], Heuristic]
Passable = Callable[[Coord], bool]


def _distances_to_goal(board: Board, passable: Passable) -> Dict[Coord, int]:
    """Số ô ít nhất từ mỗi ô tới goal trên lưới 4 hướng
    """
    # teleport_sources[D]: danh sách các switch dịch chuyển tới D
    teleport_sources: Dict[Coord, List[Coord]] = {}
    for r, row in enumerate(board.tiles):
        for c, tile in enumerate(row):
            if tile.switch_kind == SwitchKind.SPLIT:
                for dest in tile.destinations:
                    teleport_sources.setdefault(dest, []).append((r, c))

    # BFS ngược từ goal
    dist = {board.goal: 0}
    queue = deque([board.goal])
    while queue:
        cell = queue.popleft()
        d = dist[cell]
        for source in teleport_sources.get(cell, ()):
            if d < dist.get(source, math.inf):
                dist[source] = d
                queue.appendleft(source)  # Cạnh giá 0 vào đầu hàng đợi
        for dr, dc in DIRECTION_DELTAS.values():
            neighbor = (cell[0] + dr, cell[1] + dc)
            if passable(neighbor) and d + 1 < dist.get(neighbor, math.inf):
                dist[neighbor] = d + 1
                queue.append(neighbor)  # Cạnh giá 1 vào cuối hàng đợi
    return dist


def _make_heuristic(
    board: Board, passable: Passable, orientation_penalty: bool
) -> Heuristic:
    """h = MIN_MOVE_COST * ceil((khoảng cách ô gần goal nhất + phạt) / 2)."""
    dist = _distances_to_goal(board, passable)

    def heuristic(state: State) -> float:
        reachable = [dist[c] for c in state.occupied_cells if c in dist]
        if not reachable:
            return math.inf  # Không ô nào nối được tới goal
        not_standing = (state.split_positions is not None
                        or state.orientation != Orientation.STANDING)
        penalty = int(orientation_penalty and not_standing)
        # Mỗi action kéo ô gần goal nhất lại gần thêm tối đa 2 ô
        return MIN_MOVE_COST * math.ceil((min(reachable) + penalty) / 2)

    return heuristic


def manhattan_heuristic(board: Board) -> Heuristic:
    """Manhattan/2: lưới không vật cản, không xét orientation."""
    return _make_heuristic(board, board.in_bounds, orientation_penalty=False)


def orientation_heuristic(board: Board) -> Heuristic:
    """Manhattan/2 cộng phạt orientation."""
    return _make_heuristic(board, board.in_bounds, orientation_penalty=True)


def point_object_heuristic(board: Board) -> Heuristic:
    """Point-object (chỉ đi trên ô khác void)/2 cộng phạt orientation."""
    def not_void(cell: Coord) -> bool:
        return board.tile_at(cell).kind != TileType.VOID

    return _make_heuristic(board, not_void, orientation_penalty=True)