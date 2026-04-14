"""
Replacer interface for buffer replacement strategies.
"""

from abc import ABC, abstractmethod
from typing import List
from simpledb.buffer.buffer_frame import BufferFrame, AllBufferFramesPinnedException


class Replacer(ABC):
    """Interface for buffer replacement strategy."""

    @abstractmethod
    def get_name(self) -> str:
        """Get a unique string for the replacement policy."""
        pass

    @abstractmethod
    def choose(self, pool: List[BufferFrame]) -> BufferFrame:
        """Choose a BufferFrame to replace."""
        pass

    @abstractmethod
    def notify(self, pool: List[BufferFrame], frame: BufferFrame) -> None:
        """Called whenever a frame is accessed."""
        pass
