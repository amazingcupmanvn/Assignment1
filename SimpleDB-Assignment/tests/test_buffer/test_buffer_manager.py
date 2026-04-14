"""
Tests for BufferManager.
"""

import unittest
import tempfile
import os
from simpledb.buffer.buffer_manager import BufferManager
from simpledb.buffer.replacement.random_replacer import RandomReplacer
from simpledb.disk.disk_manager import DiskManager
from simpledb.heap.page_id import PageId


class TestBufferManager(unittest.TestCase):
    """Test BufferManager functionality."""

    def setUp(self):
        """Set up test buffer manager with temp database."""
        self.fd, self.db_name = tempfile.mkstemp(suffix='.db')
        self.disk_manager = DiskManager(self.db_name, num_pages=10, tmp_file=os.fdopen(self.fd, 'r+b'))
        self.replacer = RandomReplacer()
        self.buffer_manager = BufferManager(max_frames=3, replacer=self.replacer, disk_manager=self.disk_manager)

    def tearDown(self):
        """Clean up test database."""
        os.close(self.fd)
        if os.path.exists(self.db_name):
            os.remove(self.db_name)

    def test_initialization(self):
        """Test buffer manager initialization."""
        self.assertEqual(self.buffer_manager.max_frames, 3)
        self.assertEqual(len(self.buffer_manager.frames), 0)
        self.assertEqual(self.buffer_manager.get_cache_hits(), 0)
        self.assertEqual(self.buffer_manager.get_page_accesses(), 0)

    def test_get_page_cache_miss(self):
        """Test getting a page that isn't in buffer (cache miss)."""
        page_id = PageId(1)
        page = self.buffer_manager.get_page(page_id)
        self.assertIsNotNone(page)
        self.assertEqual(len(self.buffer_manager.frames), 1)
        self.assertEqual(self.buffer_manager.get_page_accesses(), 1)
        self.assertEqual(self.buffer_manager.get_cache_hits(), 0)

    def test_get_page_cache_hit(self):
        """Test getting a page that's already in buffer (cache hit)."""
        page_id = PageId(1)
        page1 = self.buffer_manager.get_page(page_id)
        page2 = self.buffer_manager.get_page(page_id)
        self.assertEqual(page1, page2)
        self.assertEqual(self.buffer_manager.get_page_accesses(), 2)
        self.assertEqual(self.buffer_manager.get_cache_hits(), 1)

    def test_pin_and_unpin(self):
        """Test pinning and unpinning pages."""
        page_id = PageId(1)
        self.buffer_manager.get_page(page_id)
        self.assertEqual(self.buffer_manager.get_num_pinned(), 1)  # get_page pins it
        
        self.buffer_manager.pin(page_id)
        self.assertEqual(self.buffer_manager.get_num_pinned(), 1)  # we pinned the same page... should still be 1
        self.assertEqual(self.buffer_manager.frames[0].pinned, 2)  # pinned count should be 2

        self.buffer_manager.unpin(page_id, False)
        self.assertEqual(self.buffer_manager.get_num_pinned(), 1)
        self.assertEqual(self.buffer_manager.frames[0].pinned, 1) 

    def test_mark_dirty(self):
        """Test marking a page as dirty."""
        page_id = PageId(1)
        self.buffer_manager.get_page(page_id)
        frame = self.buffer_manager._find_frame_by_page_id(page_id)
        self.assertFalse(frame.is_dirty())
        
        self.buffer_manager.mark_dirty(page_id)
        self.assertTrue(frame.is_dirty())

    def test_flush_dirty(self):
        """Test flushing dirty pages."""
        page_id = PageId(1)
        self.buffer_manager.get_page(page_id)
        self.buffer_manager.mark_dirty(page_id)
        
        flushed = self.buffer_manager.flush_dirty()
        self.assertEqual(flushed, 1)
        frame = self.buffer_manager._find_frame_by_page_id(page_id)
        self.assertFalse(frame.is_dirty())

    def test_get_new_page(self):
        """Test allocating a new page."""
        new_page_id = self.buffer_manager.get_new_page()
        self.assertTrue(new_page_id.is_valid())
        self.assertEqual(self.buffer_manager.get_total_disk_pages(), 11)  # Started with 10, added 1

    def test_replacement_when_full(self):
        """Test frame replacement when buffer is full; assuming Random replacer."""
        # Fill buffer
        page_ids = [PageId(i) for i in range(1, 4)]
        for pid in page_ids:
            self.buffer_manager.get_page(pid)      # frame also gets automatically pinned in get_page()
            self.buffer_manager.unpin(pid, False)  # Unpin to allow replacement

        self.assertEqual(len(self.buffer_manager.frames), 3)

        if self.replacer.get_name() == "Random":
            seen = set()
            for _ in range(50):
                ch = self.buffer_manager.replacer.choose(self.buffer_manager.frames)
                seen.add(ch.page_id.get())
            self.assertGreater(len(seen), 1)

    def test_stats(self):
        """Test statistics methods."""
        page_id = PageId(1)
        self.buffer_manager.get_page(page_id)  # Miss
        self.buffer_manager.get_page(page_id)  # Hit
        
        self.assertEqual(self.buffer_manager.get_page_accesses(), 2)
        self.assertEqual(self.buffer_manager.get_cache_hits(), 1)


if __name__ == '__main__':
    unittest.main()
