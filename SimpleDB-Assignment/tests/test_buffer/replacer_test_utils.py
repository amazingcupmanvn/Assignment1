"""
Test Utilities for Unit Tests of Buffer Replacement Policies.
"""

from simpledb.buffer.buffer_frame import BufferFrame

"""
  Created by Scott Sidwell on 20/07/15.
  Updated by Chris Natoli and Bryn Jeffries
  Converted to Python by Uwe Roehm, 25/03/2026
"""
class TestUtils():

    # Create a buffer pool with a specified number of frames and
    # populate each with a page.
    @staticmethod
    def make_pool(num):
        pool = [BufferFrame() for _ in range(num)]
        # Set page IDs to make them non-empty
        for i, frame in enumerate(pool):
            frame.page_id.set(i + 1)
        return pool

    # Call the replacer's notify method multiple times on frames of a pool
    #     @param replacer      Replacer to call
    #     @param pool          Buffer pool
    #     @param timesToNotify Number of times to call notify() on each frame
    @staticmethod
    def notify_many(replacer, pool, *counts):
        for frame, c in zip(pool, counts):
            for _ in range(c):
                replacer.notify(pool, frame)

    # Simulate loading a page into a frame, which should reset the clock count and notify the replacer
    @staticmethod
    def simulate_page_load(replacer, pool, frame, new_page_id):
        frame.page_id.set(new_page_id)
        frame.set_clock_count(0)
        replacer.notify(pool, frame)
