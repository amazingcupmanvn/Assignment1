"""
Tests for ordering operators.
"""

import unittest
from simpledb.executor.ordering.in_memory_order_by import InMemoryOrderBy
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.heap.tuple import Tuple
from simpledb.access.read.access_iterator import AccessIterator


class SimpleAccessIterator(AccessIterator):
    def __init__(self, tuples, schema):
        self.tuples = tuples
        self.schema = schema
        self.idx = 0

    def close(self):
        pass

    def get_schema(self):
        return self.schema

    def mark(self):
        pass

    def reset(self):
        pass

    def has_next(self):
        return self.idx < len(self.tuples)

    def __next__(self):
        if not self.has_next():
            raise StopIteration()
        result = self.tuples[self.idx]
        self.idx += 1
        return result

    def __iter__(self):
        return self


class TestInMemoryOrderBy(unittest.TestCase):
    def test_sort_by_single_string_column(self):
        schema = TupleDesc().add_string("name").add_integer("age")
        rows = [
            Tuple(schema, ["Bob", 20]),
            Tuple(schema, ["Alice", 30]),
            Tuple(schema, ["Carol", 25])
        ]

        iterator = SimpleAccessIterator(rows, schema)
        ordered = InMemoryOrderBy(iterator, ["name"])

        output = [t.get_column("name") for t in ordered]
        self.assertEqual(output, ["Alice", "Bob", "Carol"])

    def test_sort_by_single_int_column(self):
        schema = TupleDesc().add_string("name").add_integer("age")
        rows = [
            Tuple(schema, ["Bob", 20]),
            Tuple(schema, ["Alice", 30]),
            Tuple(schema, ["Carol", 25])
        ]
        iterator = SimpleAccessIterator(rows, schema)
        ordered = InMemoryOrderBy(iterator, ["age"])

        output = [t.get_column("name") for t in ordered]
        self.assertEqual(output, ["Bob", "Carol", "Alice"])

    def test_sort_by_single_double_column(self):
        schema = TupleDesc().add_string("name").add_double("gpa")
        rows = [
            Tuple(schema, ["Bob", 50.1]),
            Tuple(schema, ["Alice", 80.5]),
            Tuple(schema, ["Carol", 80.4])
        ]
        iterator = SimpleAccessIterator(rows, schema)
        ordered = InMemoryOrderBy(iterator, ["gpa"])

        output = [t.get_column("name") for t in ordered]
        self.assertEqual(output, ["Bob", "Carol", "Alice"])

    def test_sort_by_single_boolean_column(self):
        schema = TupleDesc().add_string("name").add_boolean("is_student")
        rows = [
            Tuple(schema, ["Bob", False]),
            Tuple(schema, ["Alice", True]),
            Tuple(schema, ["Carol", True])
        ]
        iterator = SimpleAccessIterator(rows, schema)
        ordered = InMemoryOrderBy(iterator, ["is_student"])

        output = [t.get_column("name") for t in ordered]
        self.assertEqual(output, ["Bob", "Alice", "Carol"])

    def test_sort_by_two_columns(self):
        schema = TupleDesc().add_string("name").add_integer("age")
        rows = [
            Tuple(schema, ["Alice", 30]),
            Tuple(schema, ["Bob", 25]),
            Tuple(schema, ["Alice", 20])
        ]

        iterator = SimpleAccessIterator(rows, schema)
        ordered = InMemoryOrderBy(iterator, ["name", "age"])

        output = [(t.get_column("name"), t.get_column("age")) for t in ordered]
        self.assertEqual(output, [("Alice", 20), ("Alice", 30), ("Bob", 25)])


if __name__ == '__main__':
    unittest.main()
