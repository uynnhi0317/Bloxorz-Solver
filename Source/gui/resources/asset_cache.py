"""asset_cache.py - Cache cho một loại tài nguyên: mỗi khóa chỉ tải một lần, có đếm tham chiếu.

Lỗi tải cũng được nhớ (negative cache): tài nguyên hỏng không bị đọc lại từ ổ đĩa và không ghi log lặp mỗi frame.
"""
from __future__ import annotations

from typing import Callable, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class AssetCache(Generic[T]):
    def __init__(self) -> None:
        self._items: Dict[str, Optional[T]] = {}
        self._refs: Dict[str, int] = {}
        self.hits = 0
        self.loads = 0

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def get_or_load(self, key: str, loader: Callable[[], Optional[T]]) -> Optional[T]:
        """Trả về tài nguyên đã cache; chưa có thì gọi loader đúng một lần (loader trả None = tải lỗi)."""
        if key in self._items:
            self.hits += 1
            return self._items[key]
        self.loads += 1
        value = loader()
        self._items[key] = value
        return value

    def acquire(self, key: str, loader: Callable[[], Optional[T]]) -> Optional[T]:
        """Như get_or_load, đồng thời đánh dấu đang được dùng để purge() không giải phóng."""
        value = self.get_or_load(key, loader)
        self._refs[key] = self._refs.get(key, 0) + 1
        return value

    def release(self, key: str) -> None:
        count = self._refs.get(key, 0)
        if count <= 1:
            self._refs.pop(key, None)
        else:
            self._refs[key] = count - 1

    def ref_count(self, key: str) -> int:
        return self._refs.get(key, 0)

    def evict(self, key: str) -> bool:
        """Bỏ một tài nguyên khỏi cache. Từ chối (False) nếu còn nơi đang dùng."""
        if self._refs.get(key, 0) > 0 or key not in self._items:
            return False
        del self._items[key]
        return True

    def purge(self) -> List[str]:
        """Giải phóng mọi tài nguyên không còn được dùng; trả về các khóa đã bỏ."""
        removed = [k for k in self._items if self._refs.get(k, 0) == 0]
        for k in removed:
            del self._items[k]
        return removed
