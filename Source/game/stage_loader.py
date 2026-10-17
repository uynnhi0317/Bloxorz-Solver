from pathlib import Path
from typing import Any, Mapping, Union
from .board import Board

"""
    Dữ liệu màn chơi có thể lấy từ 3 nguồn:
    + JSON text
    + Đường dẫn đến file
    + Cấu trúc ánh xạ Mapping
"""
StageSource = Union[str, Path, Mapping[str, Any]]

def load_stage(source: StageSource) -> Board:
    """
        Nạp stage từ đường dẫn, JSON text hoặc mapping đã parse.
    """
    if isinstance(source, Mapping):
        return Board.from_dict(source)
    if isinstance(source, (str, Path)):
        return Board.from_json(source)
    raise TypeError("Stage loader: source phải là đường dẫn, JSON text hoặc mapping")
