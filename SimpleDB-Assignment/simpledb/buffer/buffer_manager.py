"""
Buffer Manager for the database.
"""

from typing import List
from simpledb.disk.disk_manager import DiskManager
from simpledb.disk.page import Page
from simpledb.heap.page_id import PageId
from simpledb.buffer.buffer_frame import BufferFrame
from simpledb.buffer.replacement.replacer import Replacer


class BufferAccessException(Exception):
    """Raised when buffer access fails."""
    pass


class BufferManager:
    """Buffer Manager for the database."""

    def __init__(self, max_frames: int, replacer: Replacer, disk_manager: DiskManager):
        """Initialize the BufferManager."""
        self.replacer = replacer
        self.max_frames = max_frames
        self.frames: List[BufferFrame] = []
        self.cache_hits = 0
        self.disk_manager = disk_manager
        self.page_accesses = 0

    def get_page(self, page_id: PageId) -> Page:
        """Get a page from the buffer."""
        self.page_accesses += 1
        found = self._find_frame_by_page_id(page_id)
        
        # Cache hit
        if found is not None:
            self.replacer.notify(self.frames, found)
            self.cache_hits += 1
            return found.get_page()
        
        # Cache miss - get replacement frame
        frame = self._get_buffer_frame()
        self._replace_frame_in_buffer(page_id, frame)
        return frame.get_page()

    def _replace_frame_in_buffer(self, page_id: PageId, current: BufferFrame) -> None:
        """Replace a given BufferFrame with a new page."""
        if current.is_dirty():
            self.flush_page(current)
        
        replacement = Page()
        try:
            self.disk_manager.read_page(page_id, replacement)
            current.set_page(page_id, replacement)
        except Exception as e:
            raise BufferAccessException(str(e))
        
        self.replacer.notify(self.frames, current)

    def flush_dirty(self) -> int:
        """Flush all dirty frames in the buffer manager."""
        flush_count = 0
        for frame in self.frames:
            if frame.is_dirty():
                self.flush_page(frame)
                flush_count += 1
        return flush_count

    def flush_page(self, frame: BufferFrame) -> None:
        """Write a page to disk."""
        try:
            self.disk_manager.write_page(frame.get_page_id(), frame.content)
        except Exception as e:
            raise BufferAccessException(str(e))
        frame.unpin()
        frame.set_dirty(False)

    def get_new_page(self) -> PageId:
        """Create a new page on disk."""
        return self.disk_manager.allocate_page()

    def get_total_disk_pages(self) -> int:
        """Get the total number of pages allocated on disk."""
        return self.disk_manager.get_num_pages()

    def pin(self, page_id: PageId) -> None:
        """Pin a page to the buffer."""
        found = self._find_frame_by_page_id(page_id)
        if found is not None:
            found.pin()

    def mark_dirty(self, page_id: PageId) -> None:
        """Mark a page as dirty."""
        found = self._find_frame_by_page_id(page_id)
        if found is not None:
            found.set_dirty(True)

    def unpin(self, page_id: PageId, is_dirty: bool) -> None:
        """Unpin a page from the buffer."""
        found = self._find_frame_by_page_id(page_id)
        if found is not None:
            if is_dirty:
                found.set_dirty(True)
            found.unpin()

    def _find_frame_by_page_id(self, page_id: PageId) -> BufferFrame:
        """Find the frame containing the given page ID."""
        for frame in self.frames:
            if frame.contains(page_id):
                return frame
        return None

    def _get_buffer_frame(self) -> BufferFrame:
        """Get the next buffer frame to be replaced."""
        if len(self.frames) < self.max_frames:
            frame = BufferFrame()
            self.frames.append(frame)
            return frame
        return self.replacer.choose(self.frames)

    def get_cache_hits(self) -> int:
        """Get the number of cache hits."""
        return self.cache_hits

    def get_num_pinned(self) -> int:
        """Get the number of pinned pages."""
        return sum(1 for frame in self.frames if frame.is_pinned())

    def get_page_accesses(self) -> int:
        """Get the number of page accesses."""
        return self.page_accesses

    def print_stats(self):
        bstatus = f"** Buffer Manager ({self.get_cache_hits()} cache hits, {self.get_page_accesses()} page accesses, {self.get_num_pinned()} pinned pages):\n" 
        for frame in self.frames:
            bstatus += f" {frame.get_page_id()}"
            if frame.is_dirty():
                bstatus += "*"
            if frame.is_pinned():
                bstatus += f"^{frame.pinned}"
        print(bstatus)
