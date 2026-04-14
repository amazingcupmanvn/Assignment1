"""
Unit tests for HeaderPage class.
"""

import unittest
from simpledb.disk.header_page import HeaderPage
from simpledb.main.database_constants import DatabaseConstants
from simpledb.heap.page_id import PageId


class TestHeaderPage(unittest.TestCase):
    """Test cases for HeaderPage."""

    def test_initialise(self):
        """Test initialisation of HeaderPage."""
        page = HeaderPage()
        page.initialise()
        self.assertEqual(page.get_magic(), DatabaseConstants.PAGE_MAGIC)
        self.assertEqual(page.get_version_type(), DatabaseConstants.HEADER_PAGE_TYPE)
        self.assertEqual(page.get_next_page().get(), DatabaseConstants.INVALID_PAGE_ID)
        self.assertEqual(page.get_num_slots(), 0)
        self.assertEqual(page.get_free_start(), HeaderPage.SLOT_DIR_START)
        self.assertEqual(page.get_free_end(), DatabaseConstants.PAGE_SIZE)
        self.assertEqual(page.get_schema_name(), "SimpleDB Catalog")

    def test_set_get_next_page(self):
        """Test set and get next page."""
        page = HeaderPage()
        page.set_next_page(PageId(42))
        self.assertEqual(page.get_next_page().get(), 42)

    def test_add_get_entry(self):
        """Test set and get catalog entry."""
        page = HeaderPage()
        page.initialise()
        page_id = PageId(100)
        entry = "test_table"
        slot_id = page.add_entry(entry, page_id)
        self.assertNotEqual(slot_id, -1)
        retrieved_page_id = PageId()
        retrieved_entry = page.get_entry(slot_id, retrieved_page_id)
        self.assertEqual(retrieved_page_id.get(), 100)
        self.assertEqual(retrieved_entry, "test_table")

    def test_get_entry_invalid_slot(self):
        """Test get catalog entry for invalid slot."""
        page = HeaderPage()
        page.initialise()
        retrieved_page_id = PageId()
        with self.assertRaises(AssertionError):
            page.get_entry(0, retrieved_page_id)


if __name__ == '__main__':
    unittest.main()