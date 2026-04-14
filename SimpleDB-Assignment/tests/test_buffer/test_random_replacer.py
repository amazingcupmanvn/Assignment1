"""
Tests for RandomReplacer.
"""

import unittest
from simpledb.buffer.replacement.random_replacer import RandomReplacer
from simpledb.buffer.buffer_frame import BufferFrame, AllBufferFramesPinnedException
from tests.test_buffer.replacer_test_utils import TestUtils


class TestRandomReplacer(unittest.TestCase):

    def setUp(self):
        self.replacer = RandomReplacer()
        self.pool = TestUtils.make_pool(10)

    def test_get_name(self):
        """Test replacer name."""
        self.assertEqual(self.replacer.get_name(), "Random")

    def test_choose_is_random(self):
        """Test random replacer chooser different frames even if nothing changed."""
        seen = set()
        for _ in range(50):
            ch = self.replacer.choose(self.pool)
            self.assertIsInstance(ch, BufferFrame)
            seen.add(ch.page_id.get())
        self.assertGreater(len(seen), 1)

    def test_choose_all_pinned_raises(self):
        """Test that choosing when all frames are pinned raises exception."""
        for f in self.pool:
            f.pin()
        with self.assertRaises(AllBufferFramesPinnedException):
            self.replacer.choose(self.pool)

if __name__ == "__main__":
    unittest.main()
