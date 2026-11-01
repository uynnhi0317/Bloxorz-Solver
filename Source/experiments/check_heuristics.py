"""Per stage: C*, h(s0), h(s0)/C*, and consistency violations over the whole reachable graph."""
import heapq
import math
from itertools import count
from pathlib import Path

from game import get_successors, is_goal, load_stage
from solvers.heuristics import (manhattan_heuristic, orientation_heuristic,
                                point_object_heuristic)
from solvers.ucs import move_cost

SOURCE = Path(__file__).resolve().parent.parent
GROUPS = ("simple", "fragile", "switch", "combined")
HEURISTICS = {"manhattan": manhattan_heuristic,
              "orient": orientation_heuristic,
              "point": point_object_heuristic}
EPS = 1e-9


def explore(board):
    """Dijkstra over the FULL reachable graph. Returns (C*, states, edges)."""
    start = board.initial_state
    best = {start: 0}
    tie = count()
    pq = [(0, next(tie), start)]
    edges, c_star = [], math.inf
    while pq:
        g, _, s = heapq.heappop(pq)
        if g > best[s]:
            continue
        if is_goal(board, s):          # goal is terminal, do not expand
            c_star = min(c_star, g)
            continue
        for _, child in get_successors(board, s):
            c = move_cost(board, s, child)
            edges.append((s, c, child))
            if child not in best or g + c < best[child]:
                best[child] = g + c
                heapq.heappush(pq, (g + c, next(tie), child))
    return c_star, list(best), edges


def main():
    stages = []
    for grp in GROUPS:
        stages += sorted((SOURCE / "stages" / grp).glob("*.json"))

    print(f"{'stage':<26}{'C*':>5} | " +
          " | ".join(f"{n:>9} h0  h0/C*  viol" for n in HEURISTICS))
    for path in stages:
        board = load_stage(path)
        c_star, states, edges = explore(board)
        row = f"{path.parent.name + '/' + path.stem:<26}{c_star:>5} | "
        cells = []
        for make in HEURISTICS.values():
            h = make(board)
            viol = sum(1 for s, c, t in edges if h(s) > c + h(t) + EPS)
            viol += sum(1 for s in states if is_goal(board, s) and h(s) != 0)
            h0 = h(board.initial_state)
            cells.append(f"{h0:>12.0f}  {h0 / c_star:>5.2f}  {viol:>4}")
        print(row + " | ".join(cells))


if __name__ == "__main__":
    main()