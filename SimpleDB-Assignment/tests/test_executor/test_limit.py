"""
Tests for the Limit executor operator.
"""

import unittest
from simpledb.executor.limit.limit import Limit
from simpledb.access.read.access_iterator import AccessIterator
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.heap.tuple import Tuple


class SimpleAccessIterator(AccessIterator):
    """Simple mock iterator for testing."""

    def __init__(self, tuples, schema):
        """Initialize with tuples and schema."""
        self.tuples = tuples
        self.schema = schema
        self.idx = 0
        self.marked_idx = 0

    def close(self):
        """Close the iterator."""
        pass

    def get_schema(self):
        """Get the schema."""
        return self.schema

    def mark(self):
        """Mark current position."""
        self.marked_idx = self.idx

    def reset(self):
        """Reset to marked position."""
        self.idx = self.marked_idx

    def has_next(self):
        """Check if there are more tuples."""
        return self.idx < len(self.tuples)

    def __next__(self):
        """Get the next tuple."""
        if not self.has_next():
            raise StopIteration()
        item = self.tuples[self.idx]
        self.idx += 1
        return item


class TestLimit(unittest.TestCase):
    """Test the Limit iterator operator."""

    def setUp(self):
        """Set up test data."""
        self.schema = TupleDesc().add_string("name").add_integer("age")
        self.rows = [
            Tuple(self.schema, ["Alice", 30]),
            Tuple(self.schema, ["Bob", 25]),
            Tuple(self.schema, ["Carol", 40]),
            Tuple(self.schema, ["Dan", 22]),
        ]

    def test_limit_returns_correct_count(self):
        """Test that Limit returns only up to the defined count."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        limit_iter = Limit(src_iter, 3)

        collected = []
        while limit_iter.has_next():
            collected.append(next(limit_iter))

        self.assertEqual(len(collected), 3)
        self.assertEqual(collected[0].get_column("name"), "Alice")
        self.assertEqual(collected[1].get_column("name"), "Bob")
        self.assertEqual(collected[2].get_column("name"), "Carol")

        # Should not yield more after limit
        self.assertFalse(limit_iter.has_next())
        limit_iter.close()

    def test_limit_zero(self):
        """Test Limit with zero limit returns no tuples."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        limit_iter = Limit(src_iter, 0)

        self.assertFalse(limit_iter.has_next())
        limit_iter.close()

    def test_limit_greater_than_size(self):
        """Test Limit greater than available rows returns all rows."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        limit_iter = Limit(src_iter, 10)

        collected = list(limit_iter)
        self.assertEqual(len(collected), 4)
        limit_iter.close()

    def test_limit_schema_passthrough(self):
        """Test that Limit correctly returns the source schema."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        limit_iter = Limit(src_iter, 2)

        schema = limit_iter.get_schema()
        self.assertEqual(schema, self.schema)
        limit_iter.close()


if __name__ == '__main__':
    unittest.main()
