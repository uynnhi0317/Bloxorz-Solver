# Bloxorz Solver

> Đồ án 1 môn **CSC14003 – Nhập môn Trí tuệ Nhân tạo** · Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM

Bản cài đặt lại trò chơi **Bloxorz** bằng Python/Pygame, kèm bộ giải tự động
dùng **BFS, DFS, UCS và A\*** với giao diện replay lời giải và so sánh hiệu năng.

<p align="center">
  <img src="docs/demo.gif" alt="Gif Demo" width="700">
</p>

**Video demo:** <link YouTube>

## Mục lục
- [1. Giới thiệu](#1-giới-thiệu) 
- [2. Tính năng](#2-tính-năng)
- [3. Cài đặt & chạy](#3-cài-đặt--chạy)
- [4. Cấu trúc thư mục](#4-cấu-trúc-thư-mục) 
- [5. Thành viên](#5-Thành-viên) 
- [6. Tham khảo](#6-tham-khảo)

## 1. Giới thiệu
**Bloxorz** là trò chơi lăn khối hộp 1×1×2 trên sàn gạch; mục tiêu là đưa khối
**đứng thẳng** vào ô đích. Vì orientation của khối là một phần của state, bài toán rất
phù hợp để thực hành các thuật toán tìm kiếm.

Dự án gồm: game có GUI, bộ solver (BFS, DFS, UCS, A*), và bộ thực nghiệm so sánh hiệu năng.

## 2. Tính năng
- **Gameplay:** lăn khối theo 4 hướng, hỗ trợ gạch dễ vỡ, cầu, công tắc
  (soft / heavy / split) và 4 nhóm màn chơi tăng dần độ khó. Luật chi tiết có trong
  mục *Tutorial* của game.
- **Solver:** chọn thuật toán qua nút **Solver** trên HUD. Hiển thị thời gian tìm kiếm,
  bộ nhớ đỉnh, số node mở rộng, độ dài lời giải; replay lời giải có Pause / Prev / Next.

**Điều khiển**
<div align="center">

| Phím / thao tác | Chức năng |
|---|---|
| Mũi tên hoặc W A S D | Lăn khối |
| Space | Đổi cube đang điều khiển (khi khối đã tách) |
| R | Chơi lại màn |
| Esc | Tạm dừng |
| Solver (HUD) | Mở panel chọn thuật giải |

</div>

## 3. Cài đặt & chạy
### Yêu cầu
- Python 3.10 – 3.12
- Các thư viện trong `requirements.txt` (pygame, numpy, psutil, pytest, matplotlib, pandas)

### Cài đặt
**Clone repo**
```bash
git clone https://github.com/uynnhi0317/Bloxorz-Solver.git
cd Bloxorz-Solver/Source
```
**Tạo và kích hoạt môi trường ảo**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```
**Cài đặt thư viện**
```bash
pip install -r requirements.txt
```
### Chạy chương trình
> Chạy các lệnh sau khi đứng từ thư mục Source/ của dự án

**Trò chơi**
```bash
python main.py 
```
**Test - Kiểm tra logic game và thuật giải**
```bash
python -m pytest tests/ -v
```
**Benchmark - Đo số liệu các thuật giải và vẽ biểu đồ**
```bash 
python -m experiments.run_benchmark             # Đo số liệu
python -m experiments.plot_results              # Vẽ biểu đồ
```

## 4. Cấu trúc thư mục
```
Bloxorz-Solver
│   .gitignore
│   PLAN.md
│
└───Source                                      # Mã nguồn dự án
    │   main.py
    │   README.md
    │   requirements.txt
    │
    ├───experiments                             # Scripts đo đạc các thuật giải,
    │   │                                         trực quan hóa dữ liệu
    │   │
    │   └───results                             # Lưu trữ số liệu và biểu đồ
    │
    ├───game                                    # Logic trò chơi
    │
    ├───gui                                     # Đồ họa, hoạt ảnh
    │   │
    │   └───assets                              # Tài nguyên trò chơi
    │       │
    │       ├───scripts                         # Script sinh ra bộ thumbnail của các màn chơi
    │       │
    │       ├───thumbs                          # Ảnh thumbnail của các màn chơi
    │       │
    │       └───tutorial                        # Hướng dẫn chơi
    │
    ├───solvers                                 # Thuật toán giải màn chơi
    │
    ├───stages                                  # Bản đồ của các màn chơi
    │
    └───tests                                   # Scripts kiểm tra các tính năng của dự án,
                                                  mẫu chung cho bản đồ màn chơi
```

## 5. Thành viên
<div align="center">

| Họ tên | MSSV | GitHub |
|---|---|---|
| Trần Tuấn Kiệt | 24120359 | [tkiet06-us](https://github.com/tkiet06-us)|
| Trần Ngô Uyên Nhi | 24120210 | [uynnhi0317](https://github.com/uynnhi0317)|
| Dương Ngọc Minh Thư | 24120144 | [dnmthuw](https://github.com/dnmthuw)|
| Hoàng Diễm Phương | 24120125 | [dp-24ctt3](https://github.com/dp-24ctt3)|

</div>

## 6. Tham khảo
- Bloxorz – Coolmath Games: https://www.coolmathgames.com/0-bloxorz
