"""
Random Replacer implementation.
"""

import random
from typing import List
from simpledb.buffer.buffer_frame import BufferFrame, AllBufferFramesPinnedException
from simpledb.buffer.replacement.replacer import Replacer


class RandomReplacer(Replacer):
    """Demo Random replacement policy."""

    def get_name(self) -> str:
        """Get the name of the replacement policy."""
        return "Random"

    def choose(self, pool: List[BufferFrame]) -> BufferFrame:
        """Choose a random frame to replace."""
        assert len(pool) > 0 # "Expects a pool of at least size 1";

        # Pick a random, unpinned index - but avoid infinity loop...
        count = 0
        random_index = random.randint(0, len(pool) - 1)
        while pool[random_index].is_pinned():
            random_index = random.randint(0, len(pool) - 1)
            count += 1
            if count > 5 * len(pool):
                raise AllBufferFramesPinnedException()  

        return pool[random_index]

    def notify(self, pool: List[BufferFrame], frame: BufferFrame) -> None:
        """Nothing to be done here for Random."""
        pass