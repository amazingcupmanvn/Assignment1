import unittest
import tempfile
import os
from simpledb.executor.join.hash_join import HashJoin
from simpledb.executor.join.nested_loop_join import NestedLoopJoin
from simpledb.parser.join_args import JoinArgs
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.access.read.access_iterator import UnsupportedOperationError


class TestHashJoin(unittest.TestCase):
    def setUp(self):
        # left schema (id, name)
        self.left_schema = TupleDesc().add_integer("id").add_string("name")
        # right schema (fk, value)
        self.right_schema = TupleDesc().add_integer("fk").add_integer("value")

        self.join_condition = JoinArgs("dummy", "id", "fk")

    def make_iterator(self, schema, rows):
        """Helper: wraps a list of rows in an AccessIterator."""
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

    # --- correctness tests ---

    def test_join_matching_rows(self):
        """Basic join: two matching rows, one non-matching."""
        left_rows  = [[1, "Alice"], [2, "Bob"], [3, "Carol"]]
        right_rows = [[1, 100], [2, 200], [4, 400]]

        join = HashJoin(
            self.make_iterator(self.left_schema, left_rows),
            self.make_iterator(self.right_schema, right_rows),
            self.join_condition,
        )
        out = [t.row for t in join]
        self.assertEqual(out, [[1, "Alice", 1, 100], [2, "Bob", 2, 200]])

    def test_join_no_matches(self):
        """No rows should be returned when no keys match."""
        left_rows  = [[10, "X"], [20, "Y"]]
        right_rows = [[1, 100], [2, 200]]

        join = HashJoin(
            self.make_iterator(self.left_schema, left_rows),
            self.make_iterator(self.right_schema, right_rows),
            self.join_condition,
        )
        self.assertEqual(list(join), [])

    def test_empty_left_table(self):
        """Empty left table should produce no results."""
        join = HashJoin(
            self.make_iterator(self.left_schema, []),
            self.make_iterator(self.right_schema, [[1, 100]]),
            self.join_condition,
        )
        self.assertEqual(list(join), [])

    def test_empty_right_table(self):
        """Empty right table should produce no results."""
        join = HashJoin(
            self.make_iterator(self.left_schema, [[1, "Alice"]]),
            self.make_iterator(self.right_schema, []),
            self.join_condition,
        )
        self.assertEqual(list(join), [])

    def test_multiple_left_rows_same_key(self):
        """Multiple left rows with the same key should all match."""
        left_rows  = [[1, "Alice"], [1, "Adam"], [2, "Bob"]]
        right_rows = [[1, 100]]

        join = HashJoin(
            self.make_iterator(self.left_schema, left_rows),
            self.make_iterator(self.right_schema, right_rows),
            self.join_condition,
        )
        out = [t.row for t in join]
        self.assertEqual(len(out), 2)
        self.assertIn([1, "Alice", 1, 100], out)
        self.assertIn([1, "Adam",  1, 100], out)

    def test_multiple_right_rows_same_key(self):
        """Multiple right rows with the same key should all match."""
        left_rows  = [[1, "Alice"]]
        right_rows = [[1, 100], [1, 200]]

        join = HashJoin(
            self.make_iterator(self.left_schema, left_rows),
            self.make_iterator(self.right_schema, right_rows),
            self.join_condition,
        )
        out = [t.row for t in join]
        self.assertEqual(len(out), 2)
        self.assertIn([1, "Alice", 1, 100], out)
        self.assertIn([1, "Alice", 1, 200], out)

    def test_has_next_and_next_work(self):
        """has_next / next sequence should behave correctly."""
        left_rows  = [[1, "A"], [2, "B"]]
        right_rows = [[2, 20], [1, 10]]

        join = HashJoin(
            self.make_iterator(self.left_schema, left_rows),
            self.make_iterator(self.right_schema, right_rows),
            self.join_condition,
        )
        self.assertTrue(join.has_next())
        first = next(join)
        self.assertIn(first.row, [[1, "A", 1, 10], [2, "B", 2, 20]])
        self.assertTrue(join.has_next())
        second = next(join)
        self.assertIn(second.row, [[1, "A", 1, 10], [2, "B", 2, 20]])
        self.assertFalse(join.has_next())
        with self.assertRaises(StopIteration):
            next(join)

    def test_mark_reset_not_supported(self):
        """mark() and reset() should raise UnsupportedOperationError."""
        join = HashJoin(
            self.make_iterator(self.left_schema, [[1, "A"]]),
            self.make_iterator(self.right_schema, [[1, 1]]),
            self.join_condition,
        )
        with self.assertRaises(UnsupportedOperationError):
            join.mark()
        with self.assertRaises(UnsupportedOperationError):
            join.reset()

    # --- performance comparison test ---

    def test_hash_join_fewer_right_scans_than_nested_loop(self):
        """
        HashJoin should scan the right table exactly once regardless of left size.
        NestedLoopJoin scans the right table once per left row.
        We verify this by counting how many times right rows are accessed.
        """
        from simpledb.heap.tuple import Tuple
        from simpledb.access.read.access_iterator import AccessIterator

        class CountingIterator(AccessIterator):
            """Iterator that counts how many times next() is called."""
            def __init__(self, schema, rows):
                self.schema = schema
                self.tuples = [Tuple(schema, values=r) for r in rows]
                self.idx = 0
                self.bookmark = 0
                self.access_count = 0

            def get_schema(self): return self.schema
            def close(self): pass
            def mark(self): self.bookmark = self.idx
            def reset(self): self.idx = self.bookmark
            def has_next(self): return self.idx < len(self.tuples)
            def __iter__(self): return self

            def __next__(self):
                if self.idx >= len(self.tuples):
                    raise StopIteration()
                self.access_count += 1
                t = self.tuples[self.idx]
                self.idx += 1
                return t

        left_rows  = [[i, f"name{i}"] for i in range(5)]
        right_rows = [[i, i * 10]     for i in range(5)]

        # HashJoin
        right_hash = CountingIterator(self.right_schema, right_rows)
        hash_join  = HashJoin(
            self.make_iterator(self.left_schema, left_rows),
            right_hash,
            self.join_condition,
        )
        list(hash_join)
        hash_right_accesses = right_hash.access_count

        # NestedLoopJoin
        right_nlj = CountingIterator(self.right_schema, right_rows)
        nlj = NestedLoopJoin(
            self.make_iterator(self.left_schema, left_rows),
            right_nlj,
            self.join_condition,
        )
        list(nlj)
        nlj_right_accesses = right_nlj.access_count

        # HashJoin scans right once (5 accesses); NLJ scans it once per left row (up to 25)
        self.assertLess(hash_right_accesses, nlj_right_accesses)


    # --- string key tests (matches real demo: class = id) ---

    def test_string_key_join(self):
        """Join on string keys — matches the real Students JOIN Tutors scenario."""
        left_schema  = TupleDesc().add_string("name").add_string("class")
        right_schema = TupleDesc().add_string("id").add_string("tutor")
        condition    = JoinArgs("dummy", "class", "id")

        left_rows  = [["Alice", "INFO1103"], ["Bob", "INFO1903"], ["Carol", "ELEC1601"]]
        right_rows = [["INFO1103", "Joshua"], ["INFO1903", "Steven"], ["COMP2129", "Maxwell"]]

        join = HashJoin(
            self.make_iterator(left_schema, left_rows),
            self.make_iterator(right_schema, right_rows),
            condition,
        )
        out = [t.row for t in join]
        self.assertEqual(len(out), 2)
        self.assertIn(["Alice", "INFO1103", "INFO1103", "Joshua"], out)
        self.assertIn(["Bob",   "INFO1903", "INFO1903", "Steven"], out)

    def test_string_key_no_match(self):
        """String key join where no classes match."""
        left_schema  = TupleDesc().add_string("name").add_string("class")
        right_schema = TupleDesc().add_string("id").add_string("tutor")
        condition    = JoinArgs("dummy", "class", "id")

        join = HashJoin(
            self.make_iterator(left_schema, [["Alice", "COMP9999"]]),
            self.make_iterator(right_schema, [["INFO1103", "Joshua"]]),
            condition,
        )
        self.assertEqual(list(join), [])

    def test_string_key_multiple_tutors_per_class(self):
        """Multiple tutors for one class — both should appear in output."""
        left_schema  = TupleDesc().add_string("name").add_string("class")
        right_schema = TupleDesc().add_string("id").add_string("tutor")
        condition    = JoinArgs("dummy", "class", "id")

        left_rows  = [["Alice", "INFO1103"], ["Bob", "INFO1103"]]
        right_rows = [["INFO1103", "Joshua"], ["INFO1103", "Scott"]]

        join = HashJoin(
            self.make_iterator(left_schema, left_rows),
            self.make_iterator(right_schema, right_rows),
            condition,
        )
        out = [t.row for t in join]
        # 2 students × 2 tutors = 4 rows
        self.assertEqual(len(out), 4)
        self.assertIn(["Alice", "INFO1103", "INFO1103", "Joshua"], out)
        self.assertIn(["Alice", "INFO1103", "INFO1103", "Scott"],  out)
        self.assertIn(["Bob",   "INFO1103", "INFO1103", "Joshua"], out)
        self.assertIn(["Bob",   "INFO1103", "INFO1103", "Scott"],  out)

    # --- correctness: HashJoin must produce same rows as NestedLoopJoin ---

    def test_matches_nested_loop_join_output(self):
        """HashJoin and NestedLoopJoin must produce identical rows (order may differ)."""
        left_rows  = [[i, f"n{i}"] for i in range(8)]
        right_rows = [[i % 4, i * 10] for i in range(6)]

        hj_out = sorted(
            [t.row for t in HashJoin(
                self.make_iterator(self.left_schema, left_rows),
                self.make_iterator(self.right_schema, right_rows),
                self.join_condition,
            )]
        )
        nlj_out = sorted(
            [t.row for t in NestedLoopJoin(
                self.make_iterator(self.left_schema, left_rows),
                self.make_iterator(self.right_schema, right_rows),
                self.join_condition,
            )]
        )
        self.assertEqual(hj_out, nlj_out)

    def test_matches_nlj_string_keys(self):
        """HashJoin and NestedLoopJoin agree on string-key joins."""
        left_schema  = TupleDesc().add_string("name").add_string("class")
        right_schema = TupleDesc().add_string("id").add_string("tutor")
        condition    = JoinArgs("dummy", "class", "id")

        left_rows  = [
            ["Michael", "INFO1103"], ["Jan",     "INFO1903"],
            ["Roger",   "INFO1103"], ["Rachael", "ELEC1601"],
        ]
        right_rows = [
            ["INFO1103", "Joshua"], ["INFO1103", "Scott"],
            ["COMP2129", "Maxwell"], ["INFO1903", "Steven"],
        ]

        hj_out  = sorted([t.row for t in HashJoin(
            self.make_iterator(left_schema, left_rows),
            self.make_iterator(right_schema, right_rows),
            condition,
        )])
        nlj_out = sorted([t.row for t in NestedLoopJoin(
            self.make_iterator(left_schema, left_rows),
            self.make_iterator(right_schema, right_rows),
            condition,
        )])
        self.assertEqual(hj_out, nlj_out)

    # --- edge cases ---

    def test_has_next_idempotent(self):
        """Calling has_next() multiple times should not advance the iterator."""
        left_rows  = [[1, "A"]]
        right_rows = [[1, 99]]
        join = HashJoin(
            self.make_iterator(self.left_schema, left_rows),
            self.make_iterator(self.right_schema, right_rows),
            self.join_condition,
        )
        self.assertTrue(join.has_next())
        self.assertTrue(join.has_next())
        self.assertTrue(join.has_next())
        result = next(join)
        self.assertEqual(result.row, [1, "A", 1, 99])
        self.assertFalse(join.has_next())

    def test_single_row_match(self):
        """One row on each side that matches."""
        join = HashJoin(
            self.make_iterator(self.left_schema,  [[42, "Solo"]]),
            self.make_iterator(self.right_schema, [[42, 999]]),
            self.join_condition,
        )
        out = [t.row for t in join]
        self.assertEqual(out, [[42, "Solo", 42, 999]])

    def test_many_to_many_join(self):
        """Many-to-many: 3 left × 3 right with same key = 9 output rows."""
        left_rows  = [[1, "A"], [1, "B"], [1, "C"]]
        right_rows = [[1, 10], [1, 20], [1, 30]]
        join = HashJoin(
            self.make_iterator(self.left_schema, left_rows),
            self.make_iterator(self.right_schema, right_rows),
            self.join_condition,
        )
        out = [t.row for t in join]
        self.assertEqual(len(out), 9)

    def test_both_tables_empty(self):
        """Both tables empty — no results."""
        join = HashJoin(
            self.make_iterator(self.left_schema,  []),
            self.make_iterator(self.right_schema, []),
            self.join_condition,
        )
        self.assertEqual(list(join), [])

    # --- end-to-end test with real DatabaseManager ---

    def test_end_to_end_with_real_database(self):
        """Run HashJoin through the full stack with a real DatabaseManager."""
        from simpledb.main.database_manager import DatabaseManager
        from simpledb.executor.query_planner import QueryPlanner
        from simpledb.parser.query import Query

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        os.unlink(tmp.name)

        try:
            dbms = DatabaseManager(tmp.name)

            student_schema = TupleDesc()
            student_schema.add_string("name").add_string("class")
            dbms.get_catalog().add_schema(student_schema, "Students")
            with dbms.get_heap_file("Students").inserter() as ins:
                ins.insert(["Alice", "INFO1103"])
                ins.insert(["Bob",   "INFO1903"])
                ins.insert(["Carol", "ELEC9999"])  # no matching tutor

            tutor_schema = TupleDesc()
            tutor_schema.add_string("id").add_string("tutor")
            dbms.get_catalog().add_schema(tutor_schema, "Tutors")
            with dbms.get_heap_file("Tutors").inserter() as ins:
                ins.insert(["INFO1103", "Joshua"])
                ins.insert(["INFO1903", "Steven"])

            planner  = QueryPlanner(dbms, use_hash_join=True)
            sql      = "SELECT name, tutor FROM Students JOIN Tutors ON class = id"
            query    = Query.generate_query(sql)
            query.validate(dbms.get_catalog())
            iterator = planner.create_execution_plan(planner.create_logical_plan(query))

            out = sorted([t.row for t in iterator])
            iterator.close()
            dbms.close()

            # SELECT name, tutor projects down to just these 2 columns
            self.assertEqual(out, [
                ["Alice", "Joshua"],
                ["Bob",   "Steven"],
            ])
        finally:
            if os.path.exists(tmp.name):
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
