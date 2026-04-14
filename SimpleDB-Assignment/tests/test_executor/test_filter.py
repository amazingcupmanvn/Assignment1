"""
Tests for the filter operator predicates.
"""

import unittest
from simpledb.executor.filter.filter import Filter, Predicate
from simpledb.executor.filter.equals import Equals
from simpledb.executor.filter.range import GreaterThan, GreaterThanEquals, LessThan, LessThanEquals
from simpledb.executor.filter.not_modifier import NotModifier
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


class TestEqualsFilter(unittest.TestCase):
    """Test the Equals predicate."""

    def setUp(self):
        """Set up test data."""
        self.schema = TupleDesc().add_string("name").add_integer("age").add_boolean("is_student").add_double("gpa")
        self.rows = [
            Tuple(self.schema, ["Alice", 30, True, 84.6]),
            Tuple(self.schema, ["Bob",   25, True, 81.2]),
            Tuple(self.schema, ["Carol", 30, False, 0.0]),
            Tuple(self.schema, ["Dan",   22, True, 67.9]),
        ]

    def test_equals_integer(self):
        """Test Equals filter on integer column."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", Equals(30))
        collected = list(filter_iter)
        self.assertEqual(len(collected), 2)
        self.assertEqual(collected[0].get_column("name"), "Alice")
        self.assertEqual(collected[1].get_column("name"), "Carol")

    def test_equals_double(self):
        """Test Equals filter on a double column."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "gpa", Equals(81.2))
        collected = list(filter_iter)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].get_column("name"), "Bob")

    def test_equals_string(self):
        """Test Equals filter on string column."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "name", Equals("Bob"))
        collected = list(filter_iter)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].get_column("age"), 25)

    def test_equals_boolean(self):
        """Test Equals filter on boolean column."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "is_student", Equals(True))
        collected = list(filter_iter)
        self.assertEqual(len(collected), 3)
        filter_iter.reset()
        filter_iter = Filter(src_iter, "is_student", Equals(False))
        collected = list(filter_iter)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].get_column("name"), "Carol")

    def test_equals_no_match(self):
        """Test Equals filter with no matching rows."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", Equals(99))
        collected = list(filter_iter)
        self.assertEqual(len(collected), 0)
        src_iter.reset()
        filter_iter = Filter(src_iter, "gpa", Equals(81))
        collected = list(filter_iter)
        self.assertEqual(len(collected), 0)


class TestRangeFilters(unittest.TestCase):
    """Test range-based predicates."""

    def setUp(self):
        """Set up test data."""
        self.schema = TupleDesc().add_string("name").add_integer("age").add_boolean("is_student").add_double("gpa")
        self.rows = [
            Tuple(self.schema, ["Alice", 30, True, 84.6]),
            Tuple(self.schema, ["Bob",   25, True, 81.2]),
            Tuple(self.schema, ["Carol", 35, False, 0.0]),
            Tuple(self.schema, ["Dan",   22, True, 67.9]),
        ]

    def test_greater_than(self):
        """Test GreaterThan filter."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", GreaterThan(28))

        collected = list(filter_iter)
        self.assertEqual(len(collected), 2)
        ages = [t.get_column("age") for t in collected]
        self.assertIn(30, ages)
        self.assertIn(35, ages)

    def test_greater_than_zero(self):
        """Test GreaterThan filter with limit 0."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", GreaterThan(0))
        collected = list(filter_iter)
        self.assertEqual(len(collected), 4)

    def test_greater_than_negative(self):
        """Test GreaterThan filter with negativ lower limit."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", GreaterThan(-5))
        collected = list(filter_iter)
        self.assertEqual(len(collected), 4)

    def test_greater_than_equals(self):
        """Test GreaterThanEquals filter."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", GreaterThanEquals(30))

        collected = list(filter_iter)
        self.assertEqual(len(collected), 2)
        ages = [t.get_column("age") for t in collected]
        self.assertIn(30, ages)
        self.assertIn(35, ages)

    def test_less_than(self):
        """Test LessThan filter."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", LessThan(28))

        collected = list(filter_iter)
        self.assertEqual(len(collected), 2)
        ages = [t.get_column("age") for t in collected]
        self.assertIn(25, ages)
        self.assertIn(22, ages)

    def test_less_than_equals(self):
        """Test LessThanEquals filter."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "gpa", LessThanEquals(82))

        collected = list(filter_iter)
        self.assertEqual(len(collected), 3)
        gpas = [t.get_column("gpa") for t in collected]
        self.assertIn(81.2, gpas)
        self.assertIn(67.9, gpas)
        self.assertIn(0, gpas)

    def test_range_with_strings(self):
        """Test range filter on string column."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "name", GreaterThan("Bob"))

        collected = list(filter_iter)
        self.assertEqual(len(collected), 2)
        names = [t.get_column("name") for t in collected]
        self.assertIn("Carol", names)
        self.assertIn("Dan", names)

    def test_unsupported_range_filter(self):
        """Test unsupported range filter."""
        from simpledb.executor.filter.range import UnsupportedOperationError
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "is_student", GreaterThan(False))
        with self.assertRaises(UnsupportedOperationError):
            result = list(filter_iter)


class TestNotModifier(unittest.TestCase):
    """Test the NotModifier predicate."""

    def setUp(self):
        """Set up test data."""
        self.schema = TupleDesc().add_string("name").add_integer("age").add_boolean("is_student").add_double("gpa")
        self.rows = [
            Tuple(self.schema, ["Alice", 30, True, 84.6]),
            Tuple(self.schema, ["Bob",   25, True, 81.2]),
            Tuple(self.schema, ["Carol", 30, False, 0.0]),
            Tuple(self.schema, ["Dan",   22, True, 67.9]),
        ]

    def test_not_equals(self):
        """Test NOT Equals filter."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", NotModifier(Equals(30)))

        collected = list(filter_iter)
        self.assertEqual(len(collected), 2)
        ages = [t.get_column("age") for t in collected]
        self.assertNotIn(30, ages)

    def test_not_greater_than(self):
        """Test NOT GreaterThan filter (essentially LessThanEquals)."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", NotModifier(GreaterThan(28)))

        collected = list(filter_iter)
        self.assertEqual(len(collected), 2)
        ages = [t.get_column("age") for t in collected]
        self.assertIn(25, ages)
        self.assertIn(22, ages)


class TestFilterOperator(unittest.TestCase):
    """General tests for the Filter operator."""

    def setUp(self):
        """Set up test data."""
        self.schema = TupleDesc().add_string("name").add_integer("age").add_boolean("is_student").add_double("gpa")
        self.rows = [
            Tuple(self.schema, ["Alice", 30, True, 84.6]),
            Tuple(self.schema, ["Bob",   25, True, 81.2]),
            Tuple(self.schema, ["Carol", 35, False, 0.0]),
        ]

    def test_schema_passthrough(self):
        """Test that Filter correctly passes through the schema."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", Equals(30))

        schema = filter_iter.get_schema()
        self.assertEqual(schema, self.schema)

    def test_iterator_protocol(self):
        """Test that Filter correctly implements iterator protocol."""
        src_iter = SimpleAccessIterator(self.rows, self.schema)
        filter_iter = Filter(src_iter, "age", GreaterThan(25))

        # Should be iterable
        result = []
        for tuple_obj in filter_iter:
            result.append(tuple_obj)

        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
