# Bloxorz Solver

## Tổng quan

Dự án giải bài toán Bloxorz sử dụng các thuật toán tìm kiếm: BFS, DFS, UCS, A*.
Giao diện đồ hoạ được xây dựng bằng Pygame, hỗ trợ visualize quá trình giải và benchmark so sánh hiệu năng các thuật toán.

---

## Cấu trúc dự án

```
Bloxorz-Solver/
├── PLAN.md                            # Tài liệu kế hoạch & quy tắc đồ án (file này)
├── .gitignore                         # Danh sách file/thư mục Git bỏ qua
│
└── Source/                            # Toàn bộ mã nguồn
    ├── main.py                        # Khởi chạy chương trình
    ├── README.md                      # Hướng dẫn cài đặt và chạy chương trình
    ├── requirements.txt               # Danh sách thư viện phụ thuộc (pygame, psutil, numpy)
    │
    ├── game/                          # Package xử lý logic trò chơi
    │   ├── __init__.py                # Đánh dấu thư mục là Python package
    │   ├── state.py                   # Định nghĩa State: vị trí block, orientation,
    │   │                              #   trạng thái bridge, trạng thái split
    │   ├── board.py                   # Định nghĩa các loại Tile (normal, fragile,
    │   │                              #   switch, goal, void) và load bản đồ stage
    │   ├── rules.py                   # Bảng chuyển đổi orientation khi di chuyển,
    │   │                              #   kiểm tra move hợp lệ, xử lý fragile tile,
    │   │                              #   switch, split block, và goal test
    │   └── stage_loader.py            # Đọc dữ liệu stage từ file JSON/text,
    │                                  #   parse thành đối tượng Board
    │
    ├── solvers/                       # Package chứa các thuật toán tìm kiếm
    │   ├── __init__.py                # Đánh dấu thư mục là Python package
    │   ├── base.py                    # Lớp trừu tượng (interface) cho mọi solver,
    │   │                              #   định nghĩa SolveResult (time, memory,
    │   │                              #   expanded nodes, solution length)
    │   ├── bfs.py                     # Thuật toán Breadth-First Search (BFS)
    │   ├── dfs.py                     # Thuật toán Depth-First Search (DFS)
    │   ├── ucs.py                     # Thuật toán Uniform Cost Search (UCS)
    │   ├── astar.py                   # Thuật toán A* Search
    │   └── heuristics.py              # Các hàm heuristic dùng cho A*
    │                                  #   (Manhattan distance, v.v.)
    │
    ├── gui/                           # Package giao diện đồ hoạ (Pygame)
    │   ├── __init__.py                # Đánh dấu thư mục là Python package
    │   ├── app.py                     # Vòng lặp chính của GUI, quản lý các nút
    │   │                              #   New Game / Restart / Solve (BFS/DFS/UCS/A*)
    │   ├── renderer.py                # Vẽ các thành phần: tile, block, switch, bridge
    │   ├── animations.py              # Hiệu ứng animation cho block di chuyển
    │   ├── controls.py                # Xử lý input từ bàn phím / chuột
    │   └── assets/                    # Thư mục chứa hình ảnh, sprite, font
    │       └── .gitkeep               # Giữ thư mục rỗng trong Git
    │
    ├── stages/                        # Dữ liệu các level (bản đồ)
    │   ├── simple/                    # Level cơ bản — chỉ có tile thường
    │   ├── fragile/                   # Level có fragile tile (ô dễ vỡ)
    │   ├── switch/                    # Level có switch & bridge
    │   └── combined/                  # Level kết hợp nhiều loại tile
    │
    ├── experiments/                   # Benchmark & phân tích hiệu năng
    │   ├── run_benchmark.py           # Chạy mọi solver trên mọi stage,
    │   │                              #   xuất kết quả ra file CSV
    │   ├── plot_results.py            # Đọc CSV và vẽ biểu đồ so sánh
    │   │                              #   (thời gian, bộ nhớ, số node mở rộng)
    │   └── results/                   # Thư mục chứa output (CSV, PNG)
    │       └── (*.csv, *.png)         # Bị .gitignore, không được track
    │
    └── tests/                         # Unit tests (pytest)
        ├── test_tiles.py              # Test các loại tile: tạo, thuộc tính,
        │                              #   hành vi khi block đứng/nằm trên tile
        ├── test_transitions.py        # Test bảng chuyển đổi orientation
        │                              #   khi di chuyển (UP/DOWN/LEFT/RIGHT)
        └── test_solvers.py            # Test các solver: tìm được lời giải đúng,
                                       #   trả về None khi không có lời giải
```

---

## Quy tắc làm việc với Git

### 1. Quy tắc đặt tên nhánh

Sử dụng format: `<loại>/<mô-tả-ngắn>`

| Tiền tố (prefix) | Mục đích | Ví dụ |
|---|---|---|
| `feat/` | Thêm chức năng mới | `feat/bfs-solver` |
| `fix/` | Sửa lỗi | `fix/fragile-tile-crash` |
| `ref/` | Tái cấu trúc code (không đổi hành vi) | `ref/state-hashable` |
| `test/` | Thêm / sửa test | `test/solver-edge-cases` |
| `docs/` | Cập nhật tài liệu | `docs/update-readme` |
| `exp/` | Thử nghiệm benchmark / phân tích | `exp/heuristic-compare` |

**Quy tắc chung:**
- Viết **lowercase**, dùng dấu gạch ngang `-` để phân tách từ
- Mô tả **ngắn gọn, rõ ràng**
- **Không** dùng dấu cách, ký tự đặc biệt, tiếng Việt có dấu
- Ví dụ hợp lệ: `feat/astar-solver`, `fix/goal-test-bug`
- Ví dụ **không** hợp lệ: `Feat/Add A* Solver`, `fix_bug`, `nhanh-moi`

### 2. Luồng làm việc với nhánh

```
main
 ├── feature/bfs-solver
 ├── feature/dfs-solver
 ├── feature/gui-renderer
 └── fix/fragile-tile-crash
```

- **`main`**: Nhánh chính duy nhất, chứa code mới nhất của dự án
- **Feature/fix branches**: Tạo từ `main`, merge lại `main` khi hoàn thành

**Quy trình:**

```bash
# 1. Tạo nhánh mới từ main
git checkout main
git pull origin main
git checkout -b feature/ten-tinh-nang

# 2. Code & commit thường xuyên
git add .
git commit -m "feat: mo ta ngan"

# 3. Push nhánh lên remote
git push origin feat/ten-tinh-nang

# 4. Tạo Pull Request vào main, chờ review

# 5. Sau khi merge, xoá nhánh
git branch -d feat/ten-tinh-nang
```

### 3. Quy tắc viết commit message

Sử dụng format: `<type>: <mô tả ngắn>`

Có thể thêm scope để rõ ràng hơn: <type>(<scope>): <mô tả ngắn>

#### Các `type` được sử dụng

| Type | Ý nghĩa | Ví dụ |
|---|---|---|
| `feat` | Thêm chức năng mới | `feat(solver): implement BFS solver` |
| `fix` | Sửa lỗi | `fix(rules): handle block fall off edge` |
| `refactor` | Tái cấu trúc (không đổi hành vi) | `refactor(state): make State hashable` |
| `test` | Thêm / sửa test | `test(solver): add BFS edge case tests` |
| `docs` | Cập nhật tài liệu | `docs: update PLAN.md with git rules` |
| `style` | Format code (không đổi logic) | `style: fix indentation in renderer.py` |
| `chore` | Cập nhật config, dependencies | `chore: add psutil to requirements.txt` |
| `perf` | Cải thiện hiệu năng | `perf(astar): optimize open set with heap` |

#### `scope` (phạm vi) — tuỳ chọn

| Scope | Áp dụng cho |
|---|---|
| `solver` | Module `solvers/` |
| `game` | Module `game/` |
| `gui` | Module `gui/` |
| `stage` | Dữ liệu stage trong `stages/` |
| `experiment` | Module `experiments/` |
| `test` | Module `tests/` |
| *(bỏ trống)* | Thay đổi chung, không thuộc module cụ thể |

#### Quy tắc chung cho commit message

- **Không** kết thúc bằng dấu chấm `.`
- Mỗi commit nên là **một thay đổi logic duy nhất** — không gộp nhiều việc không liên quan

#### Ví dụ commit messages tốt

```bash
# Feature đơn giản
git commit -m "feat(solver): implement DFS with iterative stack"

# Chore
git commit -m "chore: update .gitignore for experiment results"
```

#### Ví dụ commit messages **không tốt**

```bash
# ❌ Quá mơ hồ
git commit -m "fix bug"
git commit -m "update code"

# ❌ Không theo format
git commit -m "Fixed the BFS solver bug."
git commit -m "added new feature"

# ❌ Gộp nhiều việc
git commit -m "add BFS, fix renderer, update README"
```

---

## Phân công

Trong file sheet sẽ được phân công cụ thể

## Cách chạy

```bash
cd Source
pip install -r requirements.txt
python main.py
```

### Chạy tests

```bash
cd Source
python -m pytest tests/ -v
```

### Chạy benchmark

```bash
cd Source
python -m experiments.run_benchmark
python -m experiments.plot_results
```