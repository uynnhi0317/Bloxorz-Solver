"""Script chạy benchmark cho các Bloxorz Solver.

Có hai chế độ chạy:
- Có timeout (mặc định 60 giây mỗi cặp solver/stage): mỗi lần chạy nằm trong một tiến trình con,
  quá giờ thì bị dừng và ghi nhận status = "timeout".
- Không timeout: đặt timeout = 0 hoặc None, khi đó solver chạy thẳng trong tiến trình chính, 
  không giới hạn thời gian.

Cách chạy (từ thư mục Source/):
    python -m experiments.run_benchmark
    python -m experiments.run_benchmark --timeout 20

Kết quả được ghi vào experiments/results/benchmark_results.csv
"""

import argparse
import csv
import importlib
import multiprocessing as mp
import queue
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from game.board import Board
from game.stage_loader import StageSource, load_stage
from solvers.base import BaseSolver, SolveResult

SOURCE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_STAGES_DIR = SOURCE_DIR / "stages"
DEFAULT_RESULTS_CSV = SOURCE_DIR / "experiments" / "results" / "benchmark_results.csv"

# Thời gian tối đa (giây) cho mỗi cặp (solver, stage); 0 hoặc None --> không giới hạn
DEFAULT_TIMEOUT = 0

# Thứ tự các nhóm màn chơi theo độ khó tăng dần
STAGE_CATEGORIES = ("simple", "fragile", "switch", "combined")


@dataclass(frozen = True, slots = True)
class BenchmarkRecord:
    """
        Một dòng bản ghi kết quả khi chạy 1 solver trên 1 stage cụ thể.
    """
    stage_category: str
    stage_name: str
    solver_name: str
    is_solved: bool
    solution_length: int
    expanded_nodes: int
    time_sec: float
    peak_memory_bytes: int
    is_valid: Optional[bool] = None
    status: str = "ok"  # ok | timeout | error

    @property
    def time_ms(self) -> float:
        """Thời gian chạy quy đổi ra ms"""
        return self.time_sec * 1000.0

    @property
    def peak_memory_kb(self) -> float:
        """Đỉnh bộ nhớ quy đổi ra KB"""
        return self.peak_memory_bytes / 1024.0


def measure_single(
    solver: BaseSolver,
    stage: StageSource,
    category: str = "custom",
    verify: bool = True,
) -> BenchmarkRecord:
    """
        Hàm đo hiệu năng chung của 1 thuật toán (solver) trên 1 màn chơi.
        + solver: thuật toán cần đo (kế thừa từ BaseSolver)
        + stage: đối tượng Board, đường dẫn file JSON, chuỗi JSON hoặc dict
        + category: nhóm độ khó của màn chơi (simple, fragile, switch, combined)
        + verify: tự động kiểm tra tính hợp lệ của lời giải nếu tìm thấy
    """
    board = stage if isinstance(stage, Board) else load_stage(stage)
    result: SolveResult = solver.solve(board)

    is_valid: Optional[bool] = None
    if verify and result.is_solved and result.moves is not None:
        is_valid = BaseSolver.verify_solution(board, result.moves)
        if not is_valid:
            print(f"Warning: Lời giải của {solver.name} trên màn '{board.name}' không hợp lệ!")

    return BenchmarkRecord(
        stage_category = category,
        stage_name = board.name,
        solver_name = solver.name,
        is_solved = result.is_solved,
        solution_length = result.solution_length,
        expanded_nodes = result.expanded_nodes,
        time_sec = result.time,
        peak_memory_bytes = result.peak_memory,
        is_valid = is_valid,
    )

def _child(solver:BaseSolver, stage: StageSource, category: str, verify: bool, out) -> None:
    """Chạy trong tiến trình con: đo xong thì bỏ kết quả (dạng dict) vào queue."""
    try:
        out.put(("ok", asdict(measure_single(solver, stage, category, verify))))
    except BaseException as exc:
        out.put(("error", f"{type(exc).__name__}: {exc}"))

def measure_with_timeout(
        solver:BaseSolver,
        stage: StageSource,
        category: str,
        verify: bool,
        timeout: float,
        stage_name: str
) -> BenchmarkRecord:
    """ Tương tự như measure_single() nhưng chạy trong tiến trình con và bị giết nếu quá timeout
    Quá giờ hoặc lỗi --> trả về record có isSolved=False và status="timeout"/"error"
    NOTE: solver và stage phải pickle được (instance của class ở cấp module, stage là Path/str/dict)
            || pickle: biến object thành chuỗi byte để gửi đi nơi khác, rồi dựng lại object từ đầu bên kia. 
            ||         Ở đây là gửi từ tiến trình mẹ đến tiến trình con
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target = _child, args = (solver, stage, category, verify, q), daemon = True)
    proc.start()

    try:
        kind, payload = q.get(timeout=timeout)
    except queue.Empty:
        kind, payload = ("timeout", None) if proc.is_alive() else ("error", "tiến trình con thoát bất thường")
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=5)

    if kind == "ok":
        return BenchmarkRecord(**payload)
    if kind == "error":
        print(f"Warning: {solver.name} trên '{stage.name} bị lỗi: {payload}")
    return BenchmarkRecord(
        stage_category = category,
        stage_name = stage_name,
        solver_name = solver.name,
        is_solved = False,
        solution_length = 0,
        expanded_nodes = 0,
        time_sec = timeout if kind == "timeout" else 0.0,
        peak_memory_bytes = 0,
        is_valid = None,
        status = kind
    )

def discover_stages(stages_dir: Path = DEFAULT_STAGES_DIR) -> List[tuple[str, Path]]:
    """
        Quét toàn bộ các file màn chơi (.json) trong thư mục stages/
        theo thứ tự các nhóm: simple -> fragile -> switch -> combined.
    """
    found_stages: List[tuple[str, Path]] = []
    if not stages_dir.is_dir():
        return found_stages

    # Quét các thư mục theo đúng thứ tự độ khó đã định nghĩa
    for category in STAGE_CATEGORIES:
        cat_dir = stages_dir / category
        if cat_dir.is_dir():
            for json_path in sorted(cat_dir.glob("*.json")):
                found_stages.append((category, json_path))

    # Quét thêm các thư mục con khác (nếu có ngoài 4 nhóm chuẩn)
    for sub_dir in sorted(stages_dir.iterdir()):
        if sub_dir.is_dir() and sub_dir.name not in STAGE_CATEGORIES:
            for json_path in sorted(sub_dir.glob("*.json")):
                found_stages.append((sub_dir.name, json_path))

    return found_stages


def run_benchmarks(
    solvers: Sequence[BaseSolver],
    stages: Optional[Iterable[tuple[str, StageSource]]] = None,
    verify: bool = True,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
) -> List[BenchmarkRecord]:
    """
        Chạy toàn bộ danh sách solvers qua toàn bộ danh sách stages.
        + solvers: danh sách các thuật toán [BFSSolver(), DFSSolver(), UCSSolver(), AStarSolver()]
        + stages: danh sách các cặp (category, stage_source). Nếu để None sẽ tự quét thư mục stages/
    """
    if stages is None:
        stages = discover_stages()

    records: List[BenchmarkRecord] = []
    for category, stage_source in stages:
        board = stage_source if isinstance(stage_source, Board) else load_stage(stage_source)
        for solver in solvers:
            if timeout:
                record = measure_with_timeout(solver, stage_source, category, verify, timeout, board.name)
            else:
                record = measure_single(solver, board, category = category, verify = verify)

            records.append(record)
            print(
                f"[{category}/{board.name}] {solver.name:<14} | "
                f"{record.status:<7} | "
                f"Solved: {str(record.is_solved):<5} | "
                f"Valid: {str(record.is_valid):<5} | "
                f"Len: {record.solution_length:<4} | "
                f"Nodes: {record.expanded_nodes:<7} | "
                f"Time: {record.time_ms:>8.2f} ms | "
                f"Mem: {record.peak_memory_kb:>8.2f} KB"
            )

    return records


def save_results_to_csv(
    records: Iterable[BenchmarkRecord],
    output_path: Path = DEFAULT_RESULTS_CSV,
) -> None:
    """
        Xuất danh sách kết quả đo hiệu năng ra file CSV.
    """
    output_path.parent.mkdir(parents = True, exist_ok = True)

    fieldnames = [
        "stage_category",
        "stage_name",
        "solver",
        "is_solved",
        "is_valid",
        "status",
        "solution_length",
        "expanded_nodes",
        "time_sec",
        "time_ms",
        "peak_memory_bytes",
        "peak_memory_kb",
    ]

    with output_path.open("w", newline = "", encoding = "utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "stage_category": rec.stage_category,
                "stage_name": rec.stage_name,
                "solver": rec.solver_name,
                "is_solved": rec.is_solved,
                "is_valid": rec.is_valid,
                "status": rec.status,
                "solution_length": rec.solution_length,
                "expanded_nodes": rec.expanded_nodes,
                "time_sec": f"{rec.time_sec:.6f}",
                "time_ms": f"{rec.time_ms:.3f}",
                "peak_memory_bytes": rec.peak_memory_bytes,
                "peak_memory_kb": f"{rec.peak_memory_kb:.2f}",
            })


def build_solvers() -> List[BaseSolver]:
    """
        Đăng ký các solver sẽ chạy benchmark. Solver nào chưa cài đặt thì bỏ qua và báo, không làm sập cả benchmark.
        Thêm solver mới: thêm một dòng _add(...) bên dưới.
    """
    solvers: List[BaseSolver] = []

    def _add(module: str, cls: str) -> None:
        try:
            solvers.append(getattr(importlib.import_module(module), cls)())
        except (ImportError, AttributeError) as exc:
            print(f"[bỏ qua] {cls}: chưa dùng được ({type(exc).__name__}: {exc})")

    _add("solvers.bfs", "BFSSolver")
    _add("solvers.dfs", "DFSSolver")
    _add("solvers.ucs", "UCSSolver")

    # add A*: chạy 3 heuristic để so sánh; đặt name riêng để CSV phân biệt được các dòng
    try:
        from solvers.astar import AStarSolver
        from solvers import heuristics as hs
        for label, factory in (("A* point", hs.point_object_heuristic),
                               ("A* manhattan", hs.manhattan_heuristic),
                               ("A* orient", hs.orientation_heuristic)):
            solver = AStarSolver(factory)
            solver.name = label
            solvers.append(solver)
    except ImportError as exc:
        print(f"[bỏ qua] AStarSolver: chưa dùng được ({exc})")

    return solvers


def main() -> None:
    """Điểm khởi chạy khi gọi: python -m experiments.run_benchmark (chạy từ thư mục Source/)"""
    parser = argparse.ArgumentParser(description = "Benchmark các solver Bloxorz")
    parser.add_argument("--timeout", type = float, default = DEFAULT_TIMEOUT,
                        help = "giây tối đa cho mỗi cặp (solver, stage); 0 = không giới hạn")
    args = parser.parse_args()

    solvers = build_solvers()

    if not solvers:
        print("Chưa có solver nào dùng được để chạy benchmark.")
        return

    records = run_benchmarks(solvers, timeout = args.timeout)
    if records:
        save_results_to_csv(records)
        print(f"\nĐã lưu kết quả benchmark tại: {DEFAULT_RESULTS_CSV}")


if __name__ == "__main__":
    main()