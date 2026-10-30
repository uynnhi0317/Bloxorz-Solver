"""Test cho 4 solver (BFS, DFS, UCS, A*) và các heuristic"""

import pytest

from game.board import Board
from game.rules import DIRECTION_DELTAS, as_move, get_next_state, is_goal
from game.state import Orientation
from game.tile import TileType
from solvers.astar import AStarSolver
from solvers.base import BaseSolver
from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.heuristics import (
    manhattan_heuristic,
    orientation_heuristic,
    point_object_heuristic,
)
from solvers.ucs import (
    FRAGILE_COST,
    NORMAL_COST,
    SWITCH_COST,
    UCSSolver,
    move_cost,
    solution_cost,
)
from tests.conftest import (
    make_row_board,
    make_split_required_board,
    make_switch_inline_board,
    make_unsolvable_board,
)

SOLVERS = {
    "BFS": BFSSolver(),
    "DFS": DFSSolver(),
    "UCS": UCSSolver(),
    "A*": AStarSolver(),
}
HEURISTICS = {
    "manhattan": manhattan_heuristic,
    "orientation": orientation_heuristic,
    "point_object": point_object_heuristic,
}
STAGES = [f"stage_{i:02d}" for i in range(1, 16)]


#  Helpers
def _move(dr: int, dc: int):
    """Lấy Move theo độ lệch (hàng, cột) từ DIRECTION_DELTAS, khỏi phụ thuộc tên member."""
    return next(
        as_move(m) for m, d in DIRECTION_DELTAS.items() if tuple(d) == (dr, dc)
    )


UP = _move(-1, 0)
RIGHT = _move(0, 1)


def _path_states(board, moves):
    """Danh sách state từ start tới cuối chuỗi moves (kèm start)."""
    states = [board.initial_state]
    for m in moves:
        nxt = get_next_state(board, states[-1], m)
        assert nxt is not None, "Chuỗi nước đi không hợp lệ"
        states.append(nxt)
    return states


@pytest.fixture(scope="module", params=STAGES)
def stage_run(request):
    """Chạy 4 solver một lần cho mỗi stage rồi dùng lại ở mọi test bên dưới."""
    board = request.getfixturevalue(request.param)
    return board, {name: s.solve(board) for name, s in SOLVERS.items()}



# Chạy trên cả 15 stage thật
class TestAllStages:
    def test_every_solver_finds_valid_solution(self, stage_run):
        board, results = stage_run
        for name, res in results.items():
            assert res.is_solved, f"{name} không tìm ra lời giải"
            assert BaseSolver.verify_solution(board, res.moves), (
                f"{name} trả về chuỗi nước đi không hợp lệ"
            )
            assert res.time >= 0
            assert res.peak_memory >= 0
            assert res.expanded_nodes >= 0

    def test_bfs_has_fewest_steps(self, stage_run):
        _, results = stage_run
        best = results["BFS"].solution_length
        for name, res in results.items():
            assert best <= res.solution_length, f"BFS dài hơn {name}"

    def test_ucs_has_lowest_cost(self, stage_run):
        board, results = stage_run
        ucs_cost = solution_cost(board, results["UCS"].moves)
        for name, res in results.items():
            assert ucs_cost <= solution_cost(board, res.moves), f"UCS đắt hơn {name}"

    def test_astar_cost_equals_ucs(self, stage_run):
        board, results = stage_run
        assert solution_cost(board, results["A*"].moves) == solution_cost(
            board, results["UCS"].moves
        )

    @pytest.mark.parametrize("make_h", HEURISTICS.values(), ids=list(HEURISTICS))
    def test_heuristic_admissible_along_optimal_path(self, stage_run, make_h):
        """h(s) <= chi phí tối ưu thật từ s tới goal.

        Phần đuôi của đường đi tối ưu (UCS) cũng tối ưu từ state tương ứng nên
        chi phí còn lại tại mỗi bước chính là chi phí thật. Đồng thời h(goal) = 0.
        """
        board, results = stage_run
        h = make_h(board)
        states = _path_states(board, results["UCS"].moves)
        costs = [
            move_cost(board, states[i], states[i + 1]) for i in range(len(states) - 1)
        ]

        remaining = sum(costs)
        for i, state in enumerate(states):
            assert h(state) <= remaining, (
                f"Không admissible tại bước {i}: h={h(state)} > còn lại={remaining}"
            )
            if i < len(costs):
                remaining -= costs[i]

        assert is_goal(board, states[-1])
        assert h(states[-1]) == 0

    def test_astar_expands_no_more_than_ucs(self, stage_run):
        _, results = stage_run
        assert results["A*"].expanded_nodes <= results["UCS"].expanded_nodes


@pytest.mark.parametrize("stage_name", ["stage_01", "stage_04", "stage_07", "stage_11"])
@pytest.mark.parametrize("make_h", HEURISTICS.values(), ids=list(HEURISTICS))
def test_astar_with_each_heuristic_matches_ucs(stage_name, make_h, request):
    """Cả ba heuristic đều phải cho A* ra đúng chi phí tối ưu của UCS."""
    board = request.getfixturevalue(stage_name)
    ucs = UCSSolver().solve(board)
    astar = AStarSolver(heuristic=make_h).solve(board)
    assert astar.is_solved
    assert solution_cost(board, astar.moves) == solution_cost(board, ucs.moves)



# Cơ chế đặc biệt
class TestMechanics:
    @pytest.mark.parametrize("name", list(SOLVERS))
    def test_never_stands_upright_on_fragile(self, name, stage_04):
        res = SOLVERS[name].solve(stage_04)
        assert res.is_solved
        for state in _path_states(stage_04, res.moves):
            if state.split_positions is None and state.orientation == Orientation.STANDING:
                for cell in state.occupied_cells:
                    assert stage_04.tile_at(cell).kind != TileType.FRAGILE, (
                        f"{name} đứng thẳng trên ô fragile tại {cell}"
                    )

    @pytest.mark.parametrize("name", list(SOLVERS))
    def test_bridge_must_be_opened_by_switch(self, name):
        board = make_switch_inline_board()
        assert board.initial_state.bridge_mask != 0  # cầu đóng sẵn
        res = SOLVERS[name].solve(board)
        assert res.is_solved
        assert BaseSolver.verify_solution(board, res.moves)

    @pytest.mark.parametrize("name", list(SOLVERS))
    def test_split_board_requires_splitting(self, name):
        """Board này chỉ giải được nếu block tách đôi rồi điều khiển từng cube."""
        board = make_split_required_board()
        res = SOLVERS[name].solve(board)
        assert res.is_solved
        assert BaseSolver.verify_solution(board, res.moves)
        states = _path_states(board, res.moves)
        assert any(s.split_positions is not None for s in states), (
            f"Lời giải của {name} không hề tách block"
        )


# Trường hợp biên
class TestEdgeCases:
    @pytest.mark.parametrize("name", list(SOLVERS))
    def test_unsolvable_board_returns_none(self, name):
        res = SOLVERS[name].solve(make_unsolvable_board())
        assert res.is_solved is False
        assert res.moves is None
        assert res.solution_length == 0

    @pytest.mark.parametrize("name", list(SOLVERS))
    def test_start_already_at_goal(self, name):
        board = Board.from_dict({"start": [0, 0], "tiles": [[{"type": "goal"}]]})
        res = SOLVERS[name].solve(board)
        assert res.is_solved is True
        assert res.moves == ()

    def test_unreachable_cell_heuristic_is_inf(self):
        board = Board.from_dict({
            "start": [0, 0],
            "tiles": [[{"type": "floor"}, {"type": "void"}, {"type": "goal"}]],
        })
        assert point_object_heuristic(board)(board.initial_state) == float("inf")


# Chi phí nước đi

class TestCosts:
    def test_normal_move(self):
        board = make_row_board(num_floors=2)
        start = board.initial_state
        child = get_next_state(board, start, RIGHT)
        assert move_cost(board, start, child) == NORMAL_COST

    def test_move_that_opens_bridge(self):
        board = make_switch_inline_board()
        start = board.initial_state
        child = get_next_state(board, start, RIGHT)  # nằm đè switch ở (0,2)
        assert child.bridge_mask != start.bridge_mask
        assert move_cost(board, start, child) == SWITCH_COST

    def test_block_lying_on_fragile(self):
        floor = {"type": "floor"}
        board = Board.from_dict({
            "start": [0, 0],
            "tiles": [[floor, {"type": "fragile"}, floor, {"type": "goal"}]],
        })
        start = board.initial_state
        child = get_next_state(board, start, RIGHT)  # chiếm (0,1) fragile và (0,2)
        assert child is not None
        assert move_cost(board, start, child) == FRAGILE_COST

    def test_solution_cost_sums_move_costs(self):
        board = make_row_board(num_floors=2)
        assert solution_cost(board, (RIGHT, RIGHT)) == 2 * NORMAL_COST

    def test_solution_cost_rejects_invalid_moves(self):
        board = make_row_board(num_floors=2)
        with pytest.raises(ValueError, match="không hợp lệ"):
            solution_cost(board, (UP,))  # UP rơi ra ngoài map