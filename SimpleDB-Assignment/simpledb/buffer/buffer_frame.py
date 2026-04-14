"""
Buffer Frame representing a slot in the buffer pool.
"""

from simpledb.disk.page import Page
from simpledb.heap.page_id import PageId


class BufferFrameDirtyException(Exception):
    """Raised when trying to replace a dirty frame."""
    pass


class AllBufferFramesPinnedException(Exception):
    """Raised when all buffer frames are pinned."""
    pass


class BufferFrame:
    """Describes the contents of a buffer frame."""

    def __init__(self):
        """Initialize a BufferFrame."""
        self.content = Page()
        self.page_id = PageId()
        self.is_dirty_flag = False
        self.pinned = 0

    def is_empty(self) -> bool:
        """Check whether current frame is valid or empty."""
        return not self.page_id.is_valid()

    def is_dirty(self) -> bool:
        """Check whether the current frame is dirty."""
        return self.is_dirty_flag

    def set_dirty(self, is_dirty: bool) -> None:
        """Mark the current frame as dirty."""
        self.is_dirty_flag = is_dirty

    def is_pinned(self) -> bool:
        """Check whether the current frame is pinned."""
        return self.pinned > 0

    def pin(self) -> None:
        """Mark the current frame as pinned."""
        self.pinned += 1

    def unpin(self) -> None:
        """Reduce the current frame's pin count."""
        if self.is_pinned():
            self.pinned -= 1

    def contains(self, pid: PageId) -> bool:
        """Check whether this frame contains a certain page."""
        return self.page_id == pid

    def get_page_id(self) -> PageId:
        """Get the PageId of this frame."""
        return PageId(self.page_id.get())

    def get_page(self) -> Page:
        """Get the Page content of current frame."""
        self.pin()
        return self.content

    def set_page(self, pid: PageId, page: Page) -> None:
        """Store a given Page in this frame."""
        if self.is_dirty():
            raise BufferFrameDirtyException()
        self.content.copy(page)
        self.page_id.set(pid.get())
        self.pinned = 0
        self.is_dirty_flag = False
