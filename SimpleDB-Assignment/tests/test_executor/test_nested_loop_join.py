import unittest
from simpledb.executor.join.nested_loop_join import NestedLoopJoin
from simpledb.parser.join_args import JoinArgs
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.access.read.access_iterator import UnsupportedOperationError


class TestNestedLoopJoin(unittest.TestCase):
    def setUp(self):
        # left schema (id, name)
        self.left_schema = TupleDesc().add_integer("id").add_string("name")
        # right schema (fk, value)
        self.right_schema = TupleDesc().add_integer("fk").add_integer("value")

        self.join_condition = JoinArgs("dummy", "id", "fk")

    def make_iterator(self, schema, rows):
        from simpledb.heap.tuple import Tuple
        from simpledb.access.read.access_iterator import AccessIterator
        class ListAccessIterator(AccessIterator):
            def __init__(self, schema, rows):
                self.schema = schema
                self.tuples = [Tuple(schema, values=row) for row in rows]
                self.idx = 0
                self.bookmark = 0

            def get_schema(self):
                return self.schema

            def close(self):
                pass

            def mark(self):
                self.bookmark = self.idx

            def reset(self):
                self.idx = self.bookmark

            def has_next(self):
                return self.idx < len(self.tuples)

            def __iter__(self):
                return self

            def __next__(self):
                if self.idx >= len(self.tuples):
                    raise StopIteration()
                t = self.tuples[self.idx]
                self.idx += 1
                return t

        return ListAccessIterator(schema, rows)

    def test_join_matching_rows(self):
        left_rows = [
            [1, "Alice"],
            [2, "Bob"],
            [3, "Carol"],
        ]
        right_rows = [
            [1, 100],
            [2, 200],
            [4, 400],
        ]
        left_it = self.make_iterator(self.left_schema, left_rows)
        right_it = self.make_iterator(self.right_schema, right_rows)

        join = NestedLoopJoin(left_it, right_it, self.join_condition)

        out = [tuple_.row for tuple_ in join]  # joined tuples
        self.assertEqual(out, [
            [1, "Alice", 1, 100],
            [2, "Bob", 2, 200],
        ])
        self.assertFalse(join.has_next())

    def test_join_no_matches(self):
        left_rows = [[10, "X"], [20, "Y"]]
        right_rows = [[1, 100], [2, 200]]
        left_it = self.make_iterator(self.left_schema, left_rows)
        right_it = self.make_iterator(self.right_schema, right_rows)

        join = NestedLoopJoin(left_it, right_it, self.join_condition)
        self.assertEqual(list(join), [])

    def test_has_next_and_next_work(self):
        left_rows = [[1, "A"], [2, "B"]]
        right_rows = [[2, 20], [1, 10]]
        join = NestedLoopJoin(
            self.make_iterator(self.left_schema, left_rows),
            self.make_iterator(self.right_schema, right_rows),
            self.join_condition,
        )

        self.assertTrue(join.has_next())
        first = next(join)
        self.assertEqual(first.row, [1, "A", 1, 10])
        self.assertTrue(join.has_next())
        second = next(join)
        self.assertEqual(second.row, [2, "B", 2, 20])
        self.assertFalse(join.has_next())
        with self.assertRaises(StopIteration):
            next(join)

    def test_mark_reset_not_supported(self):
        left_it = self.make_iterator(self.left_schema, [[1, "A"]])
        right_it = self.make_iterator(self.right_schema, [[1, 1]])
        join = NestedLoopJoin(left_it, right_it, self.join_condition)
        with self.assertRaises(UnsupportedOperationError):
            join.reset()
        with self.assertRaises(UnsupportedOperationError):
            join.mark()


if __name__ == "__main__":
    unittest.main()