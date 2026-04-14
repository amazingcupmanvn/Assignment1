"""
Unit tests for SlottedPage class.
"""

import unittest
from simpledb.disk.slotted_page import SlottedPage
from simpledb.main.database_constants import DatabaseConstants
from simpledb.heap.page_id import PageId


class TestSlottedPage(unittest.TestCase):
    """Test cases for SlottedPage."""

    def test_initialise(self):
        """Test initialisation of SlottedPage."""
        page = SlottedPage()
        page.initialise(DatabaseConstants.HEADER_PAGE_TYPE)
        self.assertEqual(page.get_magic(), DatabaseConstants.PAGE_MAGIC)
        self.assertEqual(page.get_version_type(), DatabaseConstants.HEADER_PAGE_TYPE)
        self.assertEqual(page.get_next_page().get(), DatabaseConstants.INVALID_PAGE_ID)
        self.assertEqual(page.get_num_slots(), 0)
        self.assertEqual(page.get_free_start(), SlottedPage.SLOT_DIR_START)
        self.assertEqual(page.get_free_end(), DatabaseConstants.PAGE_SIZE)
        self.assertEqual(page.get_schema_name(), "")

    def test_set_get_magic(self):
        """Test set and get magic."""
        page = SlottedPage()
        page.set_magic(12345)
        self.assertEqual(page.get_magic(), 12345)

    def test_set_get_version_type(self):
        """Test set and get version/type."""
        page = SlottedPage()
        page.set_version_type(DatabaseConstants.DATA_PAGE_TYPE)
        self.assertEqual(page.get_version_type(), DatabaseConstants.DATA_PAGE_TYPE)

    def test_set_get_next_page(self):
        """Test set and get next page."""
        page = SlottedPage()
        page.set_next_page(PageId(42))
        self.assertEqual(page.get_next_page().get(), 42)

    def test_set_get_num_slots(self):
        """Test set and get num slots."""
        page = SlottedPage()
        page.set_num_slots(10)
        self.assertEqual(page.get_num_slots(), 10)

    def test_set_get_free_start_end(self):
        """Test set and get free start/end."""
        page = SlottedPage()
        page.set_free_start(100)
        page.set_free_end(200)
        self.assertEqual(page.get_free_start(), 100)
        self.assertEqual(page.get_free_end(), 200)

    def test_set_get_name(self):
        """Test set and get name."""
        page = SlottedPage()
        page.set_schema_name("test_table")
        self.assertEqual(page.get_schema_name(), "test_table")

    def test_set_name_too_long(self):
        """Test setting name that is too long."""
        page = SlottedPage()
        with self.assertRaises(ValueError):
            page.set_schema_name("a" * (DatabaseConstants.MAX_TABLE_NAME_LENGTH + 1))

    def test_slot_management(self):
        """Test slot offset set/get."""
        page = SlottedPage()
        page.set_num_slots(5)
        page.set_slot_offset(2, 100)
        self.assertEqual(page.get_slot_offset(2), 100)
        with self.assertRaises(IndexError):
            page.get_slot_offset(5)

    def test_find_free_slot(self):
        """Test finding free slot."""
        page = SlottedPage()
        page.set_num_slots(3)
        page.set_slot_offset(0, 10)
        page.set_slot_offset(1, 0)  # free
        page.set_slot_offset(2, 20)
        self.assertEqual(page.find_free_slot(), 1)

    def test_allocate_slot(self):
        """Test allocating a slot."""
        page = SlottedPage()
        page.initialise(DatabaseConstants.DATA_PAGE_TYPE)
        slot_id = page.allocate_slot(10)
        self.assertEqual(slot_id, 0)
        self.assertEqual(page.get_num_slots(), 1)
        self.assertEqual(page.get_slot_offset(0), DatabaseConstants.PAGE_SIZE - 10)

    def test_allocate_too_long(self):
        """Test allocating a slot that is too long."""
        page = SlottedPage()
        page.initialise(DatabaseConstants.DATA_PAGE_TYPE)
        with self.assertRaises(OverflowError):
            slot_id = page.allocate_slot(DatabaseConstants.PAGE_SIZE)  # too long
            self.assertEqual(slot_id, -1)
        

if __name__ == '__main__':
    unittest.main()