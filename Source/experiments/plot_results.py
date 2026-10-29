"""Vẽ biểu đồ và tạo bảng tổng hợp từ kết quả benchmark.

Đầu vào: experiments/results/benchmark/benchmark_results.csv
Đầu ra: experiments/results/ chia làm 2 thư mục con:
    tables/: các bảng (.md, .csv)
    charts/: các biểu đồ
    - summary_overall.png       : các chỉ số trong METRICS, mỗi thuật toán một cột
                                  (tổng hợp trên các stage mà mọi solver đều giải được)
    - detail_<category>.png     : các chỉ số theo từng stage trong nhóm (simple/fragile/switch/combined)
    - cactus_overall.png        : cactus plot các chỉ số trên toàn bộ stage
    - strip_plot.png            : strip plot, x = solver, y = giá trị đo trên từng stage (màu = nhóm stage)
    - summary_table.csv/.md     : bảng tổng hợp, mỗi dòng là một thuật toán (trên tập stage chung)
    - per_stage_table.csv/.md   : bảng chi tiết theo từng (stage, solver)
"""

import argparse
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd

plt.rcParams["axes.titlesize"] = 13  # tiêu đề các biểu đồ con

SOURCE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = SOURCE_DIR / "experiments" / "results"
DEFAULT_CSV = RESULTS_DIR / "benchmark" / "benchmark_results.csv"
TABLES_SUBDIR = "tables"
CHARTS_SUBDIR = "charts"

# Thứ tự các nhóm màn chơi theo độ khó tăng dần
STAGE_CATEGORIES = ("simple", "fragile", "switch", "combined")

# cột CSV -> (nhãn trục, luôn dùng trục log?, định dạng số trong bảng)
# Độ dài, chi phí tự chuyển sang log khi lớn nhất / nhỏ nhất > AUTO_LOG_RATIO
METRICS: Dict[str, Tuple[str, bool, str]] = {
    "time_ms": ("Thời gian tìm kiếm (ms)", True, "{:,.2f}"),
    "peak_memory_kb": ("Bộ nhớ đỉnh (KB)", True, "{:,.1f}"),
    "expanded_nodes": ("Số node mở rộng", True, "{:,.0f}"),
    "solution_length": ("Độ dài lời giải (số bước)", False, "{:,.1f}"),
    "solution_cost": ("Chi phí lời giải", False, "{:,.1f}"),
}

REQUIRED_COLUMNS = {
    "stage_category", "stage_name", "solver", "is_solved",
    "solution_length", "expanded_nodes", "time_sec", "peak_memory_bytes",
}

# Bố cục phần đầu/cuối hình, tính theo inch
TITLE_SIZE = 18
_TITLE_TOP = 0.10      # mép trên -> tiêu đề chung
_LEGEND_TOP = 0.50     # mép trên -> legend
_BODY_TOP = 0.95       # mép trên -> vùng biểu đồ (gồm tiêu đề các biểu đồ con)
_BODY_TOP_NO_LEGEND = 0.65  # như _BODY_TOP nhưng cho hình không có legend
_NOTE_BOTTOM = 0.40    # chừa ở dưới cho chú thích

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
AUTO_LOG_RATIO = 50


# Đọc và chuẩn hoá dữ liệu
def load_results(csv_path: Path) -> pd.DataFrame:
    """Đọc CSV, ép kiểu và thêm cột solved_ok (dòng đủ điều kiện đưa vào số liệu)."""
    if not csv_path.is_file():
        raise SystemExit(f"Không tìm thấy file kết quả: {csv_path}\nHãy chạy run_benchmark trước.")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"CSV thiếu các cột: {sorted(missing)}")
    if df.empty:
        raise SystemExit("CSV không có dòng dữ liệu nào.")

    # Các cột có thể không có ở bản CSV cũ
    if "status" not in df.columns:
        df["status"] = "ok"
    if "is_valid" not in df.columns:
        df["is_valid"] = ""
    if "time_ms" not in df.columns:
        df["time_ms"] = df["time_sec"] * 1000.0
    if "peak_memory_kb" not in df.columns:
        df["peak_memory_kb"] = df["peak_memory_bytes"] / 1024.0
    if "solution_cost" not in df.columns:
        df["solution_cost"] = np.nan

    df["status"] = df["status"].fillna("ok").astype(str).str.strip().str.lower()
    df["is_solved"] = df["is_solved"].astype(str).str.strip().str.lower() == "true"
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Lời giải không hợp lệ thì không được tính vào số liệu
    invalid = df["is_valid"].astype(str).str.strip().str.lower() == "false"
    if invalid.any():
        bad = df.loc[invalid, ["stage_name", "solver"]].drop_duplicates()
        print(f"Warning: {int(invalid.sum())} dòng có lời giải không hợp lệ, đã loại khỏi số liệu:")
        for _, r in bad.iterrows():
            print(f"    - {r['solver']} trên {r['stage_name']}")

    df["is_invalid"] = invalid
    df["solved_ok"] = df["is_solved"] & (df["status"] == "ok") & ~invalid
    return df


def mark_common_stages(df: pd.DataFrame, solvers: Sequence[str]) -> int:
    """Thêm cột in_common: stage mà mọi solver đều giải được. Trả về số stage chung"""
    ok = df[df["solved_ok"]]
    counts = ok.groupby(["stage_category", "stage_name"])["solver"].nunique()
    common = set(counts[counts == len(solvers)].index)
    df["in_common"] = [(c, s) in common for c, s in zip(df["stage_category"], df["stage_name"])]
    if not common:
        print("Warning: không có stage nào mà mọi solver đều giải được, bảng tổng hợp sẽ trống")
    return len(common)


def ordered_categories(df: pd.DataFrame) -> List[str]:
    present = list(df["stage_category"].drop_duplicates())
    known = [c for c in STAGE_CATEGORIES if c in present]
    return known + sorted(c for c in present if c not in STAGE_CATEGORIES)


def ordered_solvers(df: pd.DataFrame) -> List[str]:
    """Giữ thứ tự solver theo lần xuất hiện đầu tiên trong CSV"""
    return list(df["solver"].drop_duplicates())


def solver_colors(solvers: Sequence[str]) -> Dict[str, str]:
    """Màu cố định cho mỗi solver"""
    base = ["#4C78A8", "#54A24B", "#B279A2", "#72B7B2", "#9D755D", "#BAB0AC"]
    astar_shades = ["#F58518", "#C2570A", "#FFB566", "#8C3B00", "#FFD2A1"]
    colors: Dict[str, str] = {}
    i_base = i_astar = 0
    for s in solvers:
        if s.startswith("A*"):
            colors[s] = astar_shades[i_astar % len(astar_shades)]
            i_astar += 1
        else:
            colors[s] = base[i_base % len(base)]
            i_base += 1
    return colors


def _geomean(values: pd.Series) -> float:
    v = values.dropna().clip(lower=1e-9)
    return float(np.exp(np.log(v).mean())) if len(v) else float("nan")


# Bảng tổng hợp
def build_overall_summary(df: pd.DataFrame, solvers: Sequence[str]) -> pd.DataFrame:
    """
        Mỗi dòng là 1 solver.
        + solved/total: độ phủ trên toàn bộ stage
        + các chỉ số: chỉ tính trên các stage mà mọi solver đều giải được
    """
    rows = []
    for solver in solvers:
        sub = df[df["solver"] == solver]
        solved = sub[sub["solved_ok"]]
        ok = sub[sub["solved_ok"] & sub["in_common"]]
        rows.append({
            "solver": solver,
            "total": len(sub),
            "solved": len(solved),
            "invalid": int(sub["is_invalid"].sum()),
            "timeout": int((sub["status"] == "timeout").sum()),
            "error": int((sub["status"] == "error").sum()),
            "time_ms": _geomean(ok["time_ms"]),
            "peak_memory_kb": _geomean(ok["peak_memory_kb"]),
            "expanded_nodes": _geomean(ok["expanded_nodes"]),
            "solution_length": float(ok["solution_length"].mean()) if len(ok) else float("nan"),
            "solution_cost": float(ok["solution_cost"].mean()) if len(ok) else float("nan"),
        })
    return pd.DataFrame(rows)


def _fmt(value, pattern: str) -> str:
    return "–" if value is None or (isinstance(value, float) and np.isnan(value)) else pattern.format(value)


def _md_table(rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _status_label(row: pd.Series) -> str:
    if row["solved_ok"]:
        return "ok"
    if row["is_invalid"]:
        return "Invalid"
    if row["status"] == "timeout":
        return "Timeout"
    if row["status"] == "error":
        return "Error"
    return "N/A"


def write_tables(df: pd.DataFrame, overall: pd.DataFrame, categories: Sequence[str],
                 n_common: int, out_dir: Path) -> None:
    # Bảng tổng hợp: mỗi dòng là một thuật toán, tính trên toàn bộ stage
    overall.round(4).to_csv(out_dir / "summary_table.csv", index = False, encoding = "utf-8")
    print(f"  -> {out_dir / 'summary_table.csv'}")

    md = ["# Bảng tổng hợp theo thuật toán",
          "",
          "Mỗi dòng là một thuật toán. ",
          "",
          f"Các chỉ số được tính trên {n_common} stage mà mọi solver đều giải được. ",
          "",
          "Thời gian, bộ nhớ, số node: trung bình nhân. Độ dài và chi phí lời giải: trung bình cộng. ",
          "",
          "`Solved` = số stage giải được / tổng số stage (độ phủ trên toàn bộ stage).",
          ""]
    header = ["Solver", "Solved", "Timeout", "Error", "Invalid", "Time (ms)", "Memory (KB)",
              "Expanded nodes", "Length", "Cost"]
    rows = []
    for _, r in overall.iterrows():
        rows.append([
            r["solver"], f"{r['solved']}/{r['total']}", str(r["timeout"]), str(r["error"]), str(r["invalid"]),
            _fmt(r["time_ms"], METRICS["time_ms"][2]),
            _fmt(r["peak_memory_kb"], METRICS["peak_memory_kb"][2]),
            _fmt(r["expanded_nodes"], METRICS["expanded_nodes"][2]),
            _fmt(r["solution_length"], METRICS["solution_length"][2]),
            _fmt(r["solution_cost"], METRICS["solution_cost"][2]),
        ])
    md += [_md_table(rows, header), ""]
    (out_dir / "summary_table.md").write_text("\n".join(md), encoding = "utf-8")
    print(f"  -> {out_dir / 'summary_table.md'}")

    # Bảng chi tiết theo (stage, solver)
    detail = df.copy()
    detail["result"] = detail.apply(_status_label, axis = 1)
    cols = ["stage_category", "stage_name", "solver", "result",
            "solution_length", "solution_cost", "expanded_nodes", "time_ms", "peak_memory_kb"]
    detail[cols].to_csv(out_dir / "per_stage_table.csv", index = False, encoding = "utf-8")
    print(f"  -> {out_dir / 'per_stage_table.csv'}")

    md = ["# Bảng chi tiết theo từng stage", ""]
    header = ["Stage", "Solver", "Kết quả", "Length", "Cost", "Expanded nodes", "Time (ms)", "Memory (KB)"]
    for cat in categories:
        md += [f"## Nhóm `{cat}`", ""]
        rows = []
        for _, r in detail[detail["stage_category"] == cat].iterrows():
            ok = r["solved_ok"]
            rows.append([
                r["stage_name"], r["solver"], r["result"],
                _fmt(r["solution_length"], "{:,.0f}") if ok else "–",
                _fmt(r["solution_cost"], "{:,.0f}") if ok else "–",
                _fmt(r["expanded_nodes"], "{:,.0f}") if ok else "–",
                _fmt(r["time_ms"], "{:,.2f}") if ok else "–",
                _fmt(r["peak_memory_kb"], "{:,.1f}") if ok else "–",
            ])
        md += [_md_table(rows, header), ""]
    (out_dir / "per_stage_table.md").write_text("\n".join(md), encoding = "utf-8")
    print(f"  -> {out_dir / 'per_stage_table.md'}")

# Vẽ biểu đồ
def _plain_number(value: float, _pos = None) -> str:
    """Số thường"""
    if value >= 1:
        return f"{value:,.0f}"
    return f"{value:g}"


def _plain_log_axis(ax) -> None:
    """Trục y log, nhãn là số cụ thể (1, 2, 5, 10, 20, 50, ...)"""
    lo, hi = ax.get_ylim()
    decades = np.log10(hi / lo) if lo > 0 else 6
    if decades < 1.2:
        subs = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    elif decades <= 3:
        subs = (1, 2, 5)
    elif decades <= 5:
        subs = (1, 3)
    else:
        subs = (1,)
    ax.yaxis.set_major_locator(ticker.LogLocator(base = 10, subs = subs, numticks = 30))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_plain_number))


def _is_log(col: str, values) -> bool:
    """Dùng trục log nếu METRICS yêu cầu, hoặc khi các giá trị dương chênh nhau quá AUTO_LOG_RATIO lần"""
    if METRICS[col][1]:
        return True
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0], dtype = float)
    return len(v) > 1 and v.max() / v.min() > AUTO_LOG_RATIO


def _axis_title(label: str, log: bool) -> str:
    return label + (" - trục log" if log else "")


def _grid_shape() -> Tuple[int, int]:
    """(số hàng, số cột) đủ chỗ cho mọi chỉ số trong METRICS"""
    n = len(METRICS)
    ncols = 2 if n <= 4 else 3
    return math.ceil(n / ncols), ncols


def _metric_grid(fig_width: float, row_height: float = 4.5):
    """Tạo lưới biểu đồ con theo số chỉ số trong METRICS, ẩn ô thừa"""
    n = len(METRICS)
    nrows, ncols = _grid_shape()
    fig, axes = plt.subplots(nrows, ncols, figsize = (fig_width, row_height * nrows + 1))
    axes = list(np.atleast_1d(axes).ravel())
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def _grouped_bars(
    ax,
    labels: Sequence[str],
    series: Dict[str, List[float]],
    colors: Dict[str, str],
    log: bool,
    marks: Optional[Dict[Tuple[str, int], str]] = None,
    tick_rotation: Optional[int] = None,
) -> None:
    """
        Vẽ cột nhóm: mỗi nhãn trên trục x có một cột cho mỗi solver.
        marks[(solver, idx)] = "Timeout" | "Error" | "N/A" | "–": chữ ghi ở chỗ không có cột.
    """
    n = max(len(series), 1)
    width = 0.8 / n
    x = np.arange(len(labels))
    marks = marks or {}

    finite = [v for vals in series.values() for v in vals if np.isfinite(v) and (v > 0 or not log)]
    for i, (name, vals) in enumerate(series.items()):
        pos = x - 0.4 + width * (i + 0.5)
        for j, v in enumerate(vals):
            if np.isfinite(v) and (v > 0 or not log):
                ax.bar(pos[j], v, width, color = colors[name], edgecolor = "black", linewidth = 0.4,
                       label = name if j == 0 else None)
            else:
                text = marks.get((name, j), "–")
                ax.text(pos[j], 0.01, text, transform = ax.get_xaxis_transform(),
                        ha = "center", va = "bottom", rotation = 90, fontsize = 7,
                        color = "#C00000" if text in ("Timeout", "Error", "Invalid") else "#666666")

    ax.set_xticks(x)
    if tick_rotation is None:
        tick_rotation = 25 if len(labels) > 4 else 0
    ax.set_xticklabels(labels, rotation = tick_rotation, ha = "right" if tick_rotation else "center")
    ax.grid(axis = "y", linestyle = ":", alpha = 0.6)
    ax.set_axisbelow(True)
    if log:
        ax.set_yscale("log")
        if finite:
            ax.set_ylim(min(finite) / 4, max(finite) * 3)
        _plain_log_axis(ax)
    elif finite:
        ax.set_ylim(0, max(finite) * 1.12)


def _legend_anchor(fig) -> Tuple[float, float]:
    return 0.5, 1 - _LEGEND_TOP / fig.get_figheight()


def _finish_layout(fig, legend: bool = True) -> None:
    """Căn các biểu đồ con vào vùng giữa tiêu đề/legend ở trên và chú thích ở dưới"""
    h = fig.get_figheight()
    body_top = _BODY_TOP if legend else _BODY_TOP_NO_LEGEND
    fig.tight_layout(pad = 0.5, h_pad = 1.2, w_pad = 1.5, rect = (0, _NOTE_BOTTOM / h, 1, 1 - body_top / h))


def _title(fig, title: str, note: Optional[str] = None) -> None:
    """Tiêu đề chung (in đậm, trên cùng) và chú thích nhỏ (dưới cùng) cho cả hình"""
    h = fig.get_figheight()
    # fig.text để tight_layout không tự chừa thêm khoảng trống cho tiêu đề
    fig.text(0.5, 1 - _TITLE_TOP / h, title, ha = "center", va = "top", fontsize = TITLE_SIZE, fontweight = "bold")
    if note:
        fig.text(0.5, 0.05 / h, note, ha = "center", va = "bottom", fontsize = 9, color = "#444444")


def _legend(fig, solvers: Sequence[str], colors: Dict[str, str]) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor = colors[s], edgecolor = "black", linewidth = 0.4)
               for s in solvers]
    fig.legend(handles, solvers, loc = "upper center", bbox_to_anchor = _legend_anchor(fig),
               ncol = min(len(solvers), 6), frameon = False)


def _save(fig, out_dir: Path, name: str, dpi: int, show: bool) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi = dpi, bbox_inches = "tight")
    print(f"  -> {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_summary(overall, solvers, colors, n_common, out_dir, dpi, show) -> None:
    """Biểu đồ tổng hợp: mỗi chỉ số một biểu đồ con, trục x = thuật toán, mỗi thuật toán một cột"""
    fig, axes = _metric_grid(20)
    x = np.arange(len(solvers))
    tick_labels = []
    for solver in solvers:
        row = overall[overall["solver"] == solver].iloc[0]
        tick_labels.append(f"{solver}\n({row['solved']}/{row['total']})")

    for ax, (col, (label, _, pattern)) in zip(axes, METRICS.items()):
        vals = []
        for solver in solvers:
            row = overall[overall["solver"] == solver].iloc[0]
            vals.append(float(row[col]))
        log = _is_log(col, vals)
        finite = [v for v in vals if np.isfinite(v) and (v > 0 or not log)]

        for i, (solver, v) in enumerate(zip(solvers, vals)):
            if np.isfinite(v) and (v > 0 or not log):
                ax.bar(i, v, 0.65, color = colors[solver], edgecolor = "black", linewidth = 0.5, zorder = 3)
                ax.annotate(pattern.format(v), (i, v), xytext = (0, 3), textcoords = "offset points",
                            ha = "center", va = "bottom", fontsize = 9)
            else:
                row = overall[overall["solver"] == solver].iloc[0]
                text = ("Timeout" if row["timeout"] > 0 else "Error" if row["error"] > 0
                        else "Invalid" if row["invalid"] > 0 else "N/A")
                ax.text(i, 0.01, text, transform = ax.get_xaxis_transform(), ha = "center", va = "bottom",
                        rotation = 90, fontsize = 8, color = "#C00000")

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation = 0, fontsize = 9)
        ax.set_title(_axis_title(label, log))
        ax.set_ylabel(label)
        ax.grid(axis = "y", linestyle = ":", alpha = 0.6, zorder = 0)
        ax.set_axisbelow(True)
        if finite:
            if log:
                ax.set_yscale("log")
                ax.set_ylim(min(finite) / 4, max(finite) * 5)
                _plain_log_axis(ax)
            else:
                ax.set_ylim(0, max(finite) * 1.2)
    _title(fig, "Tổng hợp hiệu năng của các thuật toán",
           note = f"Tính trên {n_common} stage mà mọi solver đều giải được. Thời gian, bộ nhớ, số node: "
                  "trung bình nhân; độ dài và chi phí lời giải: trung bình cộng. "
                  "(n/N) = số stage giải được / tổng số stage")
    _finish_layout(fig, legend = False)
    _save(fig, out_dir, "summary_overall", dpi, show)


def _short_stage_label(name: str) -> str:
    """Chỉ giữ 'Stage <số>' làm nhãn, ví dụ 'Stage 04 [Fragile] - Dual Path Shortcut' -> 'Stage 04'."""
    m = re.match(r"\s*stage\s*(\d+)", name, flags = re.IGNORECASE)
    if m:
        return f"Stage {m.group(1)}"
    return re.sub(r"\s*\[[^\]]*\]", "", name).strip()


def _stage_tick_labels(stages: Sequence[str], fig_width: float) -> Tuple[List[str], int]:
    """Nhãn trục x cho các stage; trả về (nhãn, góc xoay). Nhãn ngắn thì để nằm ngang."""
    short = [_short_stage_label(s) for s in stages]
    slot_inches = fig_width * 0.8 / _grid_shape()[1] / max(len(stages), 1)  # bề ngang mỗi stage trên một biểu đồ con
    chars_per_line = int(slot_inches * 9)
    if max(len(t) for t in short) <= chars_per_line:
        return short, 0
    if chars_per_line >= 14:  # tên dài: ngắt dòng
        return [textwrap.fill(t, width = chars_per_line) for t in short], 0
    return short, 35


def plot_category_detail(df, category, solvers, colors, out_dir, dpi, show) -> None:
    """Biểu đồ chi tiết một nhóm: mỗi stage một cụm cột, đủ 4 chỉ số"""
    sub = df[df["stage_category"] == category]
    stages = list(sub["stage_name"].drop_duplicates())
    ncols = _grid_shape()[1]
    width = max(20.0, 0.55 * len(stages) * len(solvers) * ncols / 2)
    fig_width = min(width, 36.0)
    fig, axes = _metric_grid(fig_width)
    tick_labels, tick_rotation = _stage_tick_labels(stages, fig_width)

    for ax, (col, (label, _, _)) in zip(axes, METRICS.items()):
        series, marks = {}, {}
        for solver in solvers:
            vals = []
            for j, stage in enumerate(stages):
                rec = sub[(sub["solver"] == solver) & (sub["stage_name"] == stage)]
                if rec.empty:
                    vals.append(float("nan"))
                    marks[(solver, j)] = "–"
                    continue
                r = rec.iloc[0]
                if r["solved_ok"]:
                    vals.append(float(r[col]))
                else:
                    vals.append(float("nan"))
                    marks[(solver, j)] = _status_label(r)
            series[solver] = vals
        log = _is_log(col, [v for vals in series.values() for v in vals])
        _grouped_bars(ax, tick_labels, series, colors, log, marks, tick_rotation)
        ax.set_title(_axis_title(label, log))
        ax.set_ylabel(label)
    _legend(fig, solvers, colors)
    _title(fig, f"Hiệu năng chi tiết của các thuật toán theo từng stage – nhóm '{category}'")
    _finish_layout(fig)
    _save(fig, out_dir, f"detail_{category}", dpi, show)


def _draw_cactus(ax, sub: pd.DataFrame, solvers, colors, col: str, label: str) -> None:
    """Vẽ một cactus plot cho chỉ số `col` trên tập dòng `sub`"""
    total = sub[["stage_category", "stage_name"]].drop_duplicates().shape[0]
    drawn = False
    for i, solver in enumerate(solvers):
        vals = (sub[(sub["solver"] == solver) & sub["solved_ok"]][col]
                .dropna().clip(lower = 1e-9).sort_values().to_numpy())
        if len(vals) == 0:
            continue
        drawn = True
        ax.plot(np.arange(1, len(vals) + 1), vals, color = colors[solver],
                marker = MARKERS[i % len(MARKERS)], markersize = 5, linewidth = 1.6, alpha = 0.9)

    if not drawn:
        ax.text(0.5, 0.5, "Không có stage nào được giải", transform = ax.transAxes,
                ha = "center", va = "center", color = "#666666")
    log = _is_log(col, sub.loc[sub["solved_ok"], col])
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom = 0)
    ax.set_xlim(0.5, max(total, 1) + 0.5)
    if total <= 25:
        ax.set_xticks(range(1, total + 1))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer = True))
    ax.set_xlabel(f"Số stage đã giải được (tổng {total})")
    ax.set_ylabel(label)
    ax.grid(True, which = "major", linestyle = ":", alpha = 0.6)
    ax.set_axisbelow(True)

    # Đường timeout
    if col == "time_ms":
        t_out = sub.loc[sub["status"] == "timeout", "time_ms"].dropna()
        if len(t_out):
            ax.axhline(t_out.max(), color = "#C00000", linestyle = "--", linewidth = 1)
            ax.text(0.99, t_out.max(), "timeout", transform = ax.get_yaxis_transform(),
                    ha = "right", va = "bottom", fontsize = 8, color = "#C00000")
    if log:
        _plain_log_axis(ax)


def _cactus_legend(fig, solvers, colors) -> None:
    handles = [plt.Line2D([0], [0], color = colors[s], marker = MARKERS[i % len(MARKERS)],
                          markersize = 5, linewidth = 1.6) for i, s in enumerate(solvers)]
    fig.legend(handles, solvers, loc = "upper center", bbox_to_anchor = _legend_anchor(fig),
               ncol = min(len(solvers), 6), frameon = False)


def plot_cactus_overall(df, solvers, colors, out_dir, dpi, show) -> None:
    """Cactus plot trên toàn bộ stage: mỗi chỉ số trong METRICS một biểu đồ con"""
    fig, axes = _metric_grid(20, row_height = 5)
    for ax, (col, (label, _, _)) in zip(axes, METRICS.items()):
        _draw_cactus(ax, df, solvers, colors, col, label)
        ax.set_title(f"Cactus plot - {_axis_title(label, _is_log(col, df.loc[df['solved_ok'], col]))}")
    _cactus_legend(fig, solvers, colors)
    _title(fig, "Cactus plot - số stage giải được theo từng chỉ số",
           note = "Đường thấp hơn và dài hơn thì tốt hơn; stage được sắp tăng dần theo chỉ số của từng solver")
    _finish_layout(fig)
    _save(fig, out_dir, "cactus_overall", dpi, show)


CATEGORY_COLORS = ("#59A14F", "#EDC948", "#F28E2B", "#E15759", "#B07AA1", "#76B7B2")  # xanh -> đỏ ~ dễ -> khó


def plot_strip(df, categories, solvers, out_dir, dpi, show) -> None:
    """
    Strip plot: trục x = solver, trục y = giá trị đo; mỗi chấm là kết quả trên một stage.
    Màu chấm = nhóm stage (dễ -> khó), vạch đen = trung vị của solver, x đỏ = stage không giải được.
    Mỗi chỉ số trong METRICS một biểu đồ con.
    """
    rng = np.random.default_rng(0)  # jitter cố định để lần chạy nào cũng ra cùng một hình
    cat_color = {c: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, c in enumerate(categories)}
    x = np.arange(len(solvers))

    fig, axes = _metric_grid(20, row_height = 5)

    for ax, (col, (label, _, _)) in zip(axes, METRICS.items()):
        log = _is_log(col, df.loc[df["solved_ok"], col])
        all_vals: List[float] = []
        for i, solver in enumerate(solvers):
            sub = df[df["solver"] == solver]
            ok = sub[sub["solved_ok"]].dropna(subset = [col])
            if log:
                ok = ok[ok[col] > 0]
            if len(ok):
                jitter = rng.uniform(-0.18, 0.18, len(ok))
                ax.scatter(i + jitter, ok[col], s = 38, c = [cat_color[c] for c in ok["stage_category"]],
                           edgecolor = "black", linewidth = 0.5, alpha = 0.9, zorder = 3)
                ax.hlines(ok[col].median(), i - 0.3, i + 0.3, color = "black", linewidth = 2, zorder = 4)
                all_vals.extend(ok[col].tolist())

            failed = int((~sub["solved_ok"]).sum())
            if failed:
                ax.plot(i + rng.uniform(-0.18, 0.18, failed), np.full(failed, 0.97), "x",
                        transform = ax.get_xaxis_transform(), color = "#C00000",
                        markersize = 7, markeredgewidth = 1.6, zorder = 5, clip_on = False)
                # Dấu x nằm theo tỉ lệ khung, không phải giá trị đo -> ghi chú để tránh đọc nhầm
                ax.annotate(f"{failed} không giải", (i, 0.97), xycoords = ax.get_xaxis_transform(),
                            xytext = (10, 0), textcoords = "offset points", va = "center",
                            fontsize = 7, color = "#C00000", annotation_clip = False)

        ax.set_xlim(-0.6, len(solvers) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(solvers, rotation = 20, ha = "right")
        ax.set_ylabel(label)
        ax.set_title(_axis_title(label, log))
        ax.grid(axis = "y", linestyle = ":", alpha = 0.6, zorder = 0)
        ax.set_axisbelow(True)
        if log:
            ax.set_yscale("log")
        if all_vals:  # chừa khoảng trống phía trên cho dấu x đỏ
            lo, hi = min(all_vals), max(all_vals)
            ax.set_ylim(lo / 3, hi * 12) if log else ax.set_ylim(0, hi * 1.2)
        if log:
            _plain_log_axis(ax)

    handles = [plt.Line2D([0], [0], marker = "o", linestyle = "", markerfacecolor = cat_color[c],
                          markeredgecolor = "black", markersize = 8) for c in categories]
    names = [f"nhóm {c}" for c in categories]
    handles += [plt.Line2D([0], [0], color = "black", linewidth = 2),
                plt.Line2D([0], [0], marker = "x", linestyle = "", color = "#C00000", markersize = 8,
                           markeredgewidth = 1.6)]
    names += ["trung vị", "không giải được"]
    fig.legend(handles, names, loc = "upper center", bbox_to_anchor = _legend_anchor(fig),
               ncol = min(len(names), 6), frameon = False)
    _title(fig, "Phân bố kết quả đo của từng thuật toán trên các stage",
           note = "Mỗi chấm = 1 stage; màu = nhóm stage; vạch đen = trung vị; "
                  "dấu x đỏ đặt ở mép trên, không phải giá trị đo")
    _finish_layout(fig)
    _save(fig, out_dir, "strip_plot", dpi, show)

def main() -> None:
    parser = argparse.ArgumentParser(description = "Vẽ biểu đồ & bảng từ kết quả benchmark Bloxorz")
    parser.add_argument("--csv", type = Path, default = DEFAULT_CSV, help = "file CSV kết quả benchmark")
    parser.add_argument("--out", type = Path, default = RESULTS_DIR,
                        help = "thư mục gốc kết quả; bảng lưu ở <out>/tables, biểu đồ ở <out>/charts")
    parser.add_argument("--dpi", type = int, default = 150)
    parser.add_argument("--show", action = "store_true", help = "hiện cửa sổ biểu đồ sau khi lưu")
    args = parser.parse_args()

    df = load_results(args.csv)
    categories = ordered_categories(df)
    solvers = ordered_solvers(df)
    n_common = mark_common_stages(df, solvers)
    colors = solver_colors(solvers)
    overall = build_overall_summary(df, solvers)

    tables_dir = args.out / TABLES_SUBDIR
    charts_dir = args.out / CHARTS_SUBDIR
    tables_dir.mkdir(parents = True, exist_ok = True)
    charts_dir.mkdir(parents = True, exist_ok = True)
    print(f"Đọc {len(df)} dòng | {len(solvers)} solver | nhóm: {', '.join(categories)} | "
          f"stage chung: {n_common}")
    print("Đang ghi bảng và biểu đồ:")

    write_tables(df, overall, categories, n_common, tables_dir)
    plot_summary(overall, solvers, colors, n_common, charts_dir, args.dpi, args.show)
    for cat in categories:
        plot_category_detail(df, cat, solvers, colors, charts_dir, args.dpi, args.show)
    plot_cactus_overall(df, solvers, colors, charts_dir, args.dpi, args.show)
    plot_strip(df, categories, solvers, charts_dir, args.dpi, args.show)

    print(f"\nHoàn tất trích xuất:\n  - Bảng: {tables_dir}\n  - Biểu đồ: {charts_dir}")


if __name__ == "__main__":
    main()