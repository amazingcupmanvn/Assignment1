"""
Tests for database manager and catalog functionality.
"""

import unittest
import os
import tempfile
from simpledb.main.database_manager import DatabaseManager
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.heap.tuple import Tuple


class TestDatabaseManager(unittest.TestCase):
    """Test database manager functionality."""

    def setUp(self):
        """Set up test database."""
        self.db_name = tempfile.NamedTemporaryFile(suffix='.db').name
        self.dbms = DatabaseManager(self.db_name)

    def tearDown(self):
        """Clean up test database."""
        if os.path.exists(self.db_name):
            os.remove(self.db_name)

    def test_create_schema(self):
        """Test schema creation."""
        schema = TupleDesc()
        schema.add_string("name").add_integer("age")
        self.dbms.get_catalog().add_schema(schema, "test_table")
        
        retrieved = self.dbms.get_catalog().read_schema("test_table")
        self.assertEqual(schema, retrieved)

    def test_insert_and_read(self):
        """Test inserting and reading tuples."""
        schema = TupleDesc()
        schema.add_string("name").add_integer("age")
        self.dbms.get_catalog().add_schema(schema, "people")
        
        table = self.dbms.get_heap_file("people")
        
        # Insert rows
        with table.inserter() as inserter:
            inserter.insert(["Alice", 30])
            inserter.insert(["Bob", 25])
        
        # Read back
        rows = []
        for tuple_obj in table.iterator():
            rows.append(tuple_obj)
        
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].get_column("name"), "Alice")
        self.assertEqual(rows[0].get_column("age"), 30)


if __name__ == '__main__':
    unittest.main()
