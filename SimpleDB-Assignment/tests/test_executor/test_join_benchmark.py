"""
Performance benchmark and evaluation: HashJoin vs NestedLoopJoin.

Tests five dataset sizes spanning different page-count regimes using a real
on-disk SimpleDB database.  For each tier the test:

  1. Inserts deterministic rows into a fresh temporary database.
  2. Runs HashJoin, then NestedLoopJoin, over the same data.
  3. Asserts both produce identical (correct) results.
  4. For the buffer-exceeding tier, asserts HashJoin uses <= disk page
     accesses than NestedLoopJoin (its core I/O advantage).

Schema
------
  Left  table : (id INTEGER, name STRING)    -- 20 B/record  -> 44 records/page
  Right table : (fk INTEGER, value INTEGER)  --  8 B/record  -> 97 records/page

Join condition: Left.id = Right.fk  (equi-join, 1:1 matching rows)

Dataset tiers
-------------
  Tier 1 – tiny        : both tables sub-page (< 1 page each)
  Tier 2 – small       : both tables ~1 page each
  Tier 3 – medium      : left ~5 pages, right ~3 pages  (all fit in 32-frame buffer)
  Tier 4 – large       : left ~10 pages, right ~5 pages (all fit in 32-frame buffer)
  Tier 5 – right_heavy : left ~5 pages, right ~40 pages (RIGHT EXCEEDS 32-frame buffer)
             -> HashJoin reads right ONCE; NestedLoopJoin re-reads right
                for every left row, causing many extra disk page reads once
                right pages are evicted by the buffer replacement policy.

Page-access measurement
-----------------------
  DiskManager.page_accesses is reset to 0 immediately before each join
  execution so that insert I/O is excluded from the comparison.

Run with:
    python -m pytest tests/test_executor/test_join_benchmark.py -v -s
or:
    python -m unittest tests.test_executor.test_join_benchmark -v
"""

import os
import time
import tempfile
import unittest

from simpledb.main.database_manager import DatabaseManager
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.disk.data_page import DataPage
from simpledb.main.database_constants import DatabaseConstants
from simpledb.executor.join.hash_join import HashJoin
from simpledb.executor.join.nested_loop_join import NestedLoopJoin
from simpledb.parser.join_args import JoinArgs


# ---------------------------------------------------------------------------
# Schemas & join condition
# ---------------------------------------------------------------------------

def _left_schema() -> TupleDesc:
    """id INTEGER (4 B) + name STRING (2+14 B) = 20 B/record."""
    return TupleDesc().add_integer("id").add_string("name")


def _right_schema() -> TupleDesc:
    """fk INTEGER (4 B) + value INTEGER (4 B) = 8 B/record."""
    return TupleDesc().add_integer("fk").add_integer("value")


# Left column "id" joins to right column "fk"
_JOIN_COND = JoinArgs("Right", "id", "fk")

# Compute per-page capacities once from the actual schema sizes so that
# tier row-counts are derived from reality rather than magic numbers.
# Formula (from DataPage.get_max_records_on_page):
#   (PAGE_SIZE - PAGE_HEADER_SIZE) // (tuple_bytes + SLOT_ENTRY_SIZE)
# = (1024 - 48) // (record_size + 2)
_LEFT_RPP  = DataPage.get_max_records_on_page(_left_schema())   # 44 records/page
_RIGHT_RPP = DataPage.get_max_records_on_page(_right_schema())  # 97 records/page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db():
    """Create a fresh temporary database.  Returns (path, DatabaseManager)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)          # let DatabaseManager create it fresh
    return path, DatabaseManager(path)


def _populate(dbms: DatabaseManager, left_n: int, right_n: int) -> None:
    """
    Insert left_n rows into 'Left' and right_n rows into 'Right'.

    Left row  i : (id=i,  name="n{i:06d}")   -- 7-char name, fits MAX_STRING_LENGTH=14
    Right row i : (fk=i,  value=i*10)

    All left rows with id < right_n match exactly one right row (1:1 equi-join).
    Rows with id >= right_n have no match and are excluded from join output.
    """
    dbms.get_catalog().add_schema(_left_schema(), "Left")
    with dbms.get_heap_file("Left").inserter() as ins:
        for i in range(left_n):
            ins.insert([i, f"n{i:06d}"])

    dbms.get_catalog().add_schema(_right_schema(), "Right")
    with dbms.get_heap_file("Right").inserter() as ins:
        for i in range(right_n):
            ins.insert([i, i * 10])

    # Flush all dirty pages to disk before join execution so that
    # subsequent page_accesses measurements reflect only join I/O.
    dbms.get_buffer_manager().flush_dirty()


def _run_join(dbms: DatabaseManager, join_cls):
    """
    Execute join_cls over Left JOIN Right ON id = fk.

    Resets DiskManager.page_accesses to 0 before executing so the count
    captures only the join's own disk reads (insert I/O is excluded).

    Returns
    -------
    sorted_rows  : list of tuples — sorted for deterministic comparison
    elapsed_s    : wall-clock execution time in seconds
    page_accesses: total disk page reads during the join
    """
    dbms.get_disk_manager().page_accesses = 0

    left_iter  = dbms.get_heap_file("Left").iterator()
    right_iter = dbms.get_heap_file("Right").iterator()
    join = join_cls(left_iter, right_iter, _JOIN_COND)

    t0   = time.perf_counter()
    rows = [tuple(t.row) for t in join]
    t1   = time.perf_counter()

    accesses = dbms.get_disk_manager().get_page_accesses()
    join.close()
    return sorted(rows), t1 - t0, accesses


def _expected(left_n: int, right_n: int):
    """
    Ground-truth join output, sorted.

    Joined schema (from TupleDesc.join, no column-name overlap):
      (id, name, fk, value)

    Matching rows: those where id == fk, i.e. i in range(min(left_n, right_n)).
    """
    return sorted((i, f"n{i:06d}", i, i * 10) for i in range(min(left_n, right_n)))


def _npages(n: int, rpp: int) -> int:
    """Ceiling division: how many pages to store n records at rpp records/page."""
    return (n + rpp - 1) // rpp


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestJoinPerformanceBenchmark(unittest.TestCase):
    """
    Evaluates HashJoin vs NestedLoopJoin across five dataset sizes that span
    sub-page, single-page, multi-page, and buffer-exceeding regimes.

    Results are printed as a formatted table after all tests complete (via
    tearDownClass).  Run with '-s' / '--capture=no' to see the table in the
    terminal.
    """

    # Class-level list accumulates one dict per benchmark run.
    _results = []

    @classmethod
    def setUpClass(cls):
        cls._results = []

    @classmethod
    def tearDownClass(cls):
        _print_report(cls._results)

    # ------------------------------------------------------------------
    # Core benchmark runner
    # ------------------------------------------------------------------

    def _bench(self, label: str, left_n: int, right_n: int,
               assert_hj_wins_pages: bool = False):
        """
        Run both join algorithms over (left_n × right_n) rows and record metrics.

        Parameters
        ----------
        label               : human-readable tier name printed in the report
        left_n              : rows to insert into Left table
        right_n             : rows to insert into Right table
        assert_hj_wins_pages: when True, assert HJ page_accesses <= NLJ page_accesses.
                              Only set for tiers where the right table exceeds the
                              buffer, making the single-right-scan advantage observable.
        """
        path, dbms = _make_db()
        try:
            _populate(dbms, left_n, right_n)

            # HashJoin runs first (cold buffer after inserts).
            hj_rows,  hj_t,  hj_pg  = _run_join(dbms, HashJoin)

            # NestedLoopJoin runs second (buffer partially warmed by HashJoin).
            # page_accesses is reset to 0 inside _run_join before execution,
            # so only NLJ's own I/O is captured.
            nlj_rows, nlj_t, nlj_pg = _run_join(dbms, NestedLoopJoin)

        finally:
            # Always clean up the temporary database file.
            try:
                dbms.close()
            except Exception:
                pass
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass

        # --- correctness: both algorithms must produce identical output ---
        expected = _expected(left_n, right_n)
        self.assertEqual(
            hj_rows, expected,
            f"[{label}] HashJoin produced incorrect output"
        )
        self.assertEqual(
            nlj_rows, expected,
            f"[{label}] NestedLoopJoin produced incorrect output"
        )

        # --- I/O advantage assertion (buffer-exceeding tiers only) ---
        if assert_hj_wins_pages:
            self.assertLessEqual(
                hj_pg, nlj_pg,
                f"[{label}] Expected HashJoin ({hj_pg} page accesses) to use "
                f"<= disk pages than NestedLoopJoin ({nlj_pg} page accesses). "
                f"HashJoin reads the right table once during build; "
                f"NestedLoopJoin re-reads it for each left row, causing extra "
                f"I/O when right pages are evicted from the buffer."
            )

        # Accumulate for the printed report.
        TestJoinPerformanceBenchmark._results.append(dict(
            label    = label,
            left_n   = left_n,
            right_n  = right_n,
            left_pg  = _npages(left_n,  _LEFT_RPP),
            right_pg = _npages(right_n, _RIGHT_RPP),
            hj_pg    = hj_pg,
            nlj_pg   = nlj_pg,
            hj_t     = hj_t,
            nlj_t    = nlj_t,
        ))

    # ------------------------------------------------------------------
    # Tier tests (0N_ prefix forces alphabetical / size-ascending order)
    # ------------------------------------------------------------------

    def test_01_tiny(self):
        """
        Both tables sub-page (< 1 page each).
        ~22 left rows, ~48 right rows — trivially small.

        At this scale, HashJoin's hash-table construction overhead may
        outweigh any benefit, so no page-access assertion is made.
        Expected: similar performance; both complete in < 1 ms.
        """
        self._bench(
            "tiny  (sub-page, ~0.5 pg each)",
            left_n  = _LEFT_RPP  // 2,    # ~22 rows,  < 1 page
            right_n = _RIGHT_RPP // 2,    # ~48 rows,  < 1 page
            assert_hj_wins_pages=False,
        )

    def test_02_small(self):
        """
        Both tables fill exactly one page each.
        44 left rows, 97 right rows.

        All data fits comfortably within the 32-frame buffer.
        HashJoin and NestedLoopJoin should produce similar page access counts.
        """
        self._bench(
            "small (1 page each)",
            left_n  = _LEFT_RPP,    # 44 rows,  1 page
            right_n = _RIGHT_RPP,   # 97 rows,  1 page
            assert_hj_wins_pages=False,
        )

    def test_03_medium(self):
        """
        Left ~5 pages (220 rows), right ~3 pages (291 rows).
        Combined 8 pages — well within the 32-frame buffer.

        Once right pages load on the first NLJ scan, they stay cached.
        Both joins make ~L+R disk reads; HashJoin's CPU advantage
        (O(1) hash lookup vs O(R) linear probe) becomes visible here.
        """
        self._bench(
            "medium (5L / 3R pages)",
            left_n  = _LEFT_RPP  * 5,     # 220 rows,  5 pages
            right_n = _RIGHT_RPP * 3,     # 291 rows,  3 pages
            assert_hj_wins_pages=False,
        )

    def test_04_large(self):
        """
        Left ~10 pages (440 rows), right ~5 pages (485 rows).
        Combined 15 pages — still comfortably within the 32-frame buffer.

        NLJ right pages remain cached after the first scan, so page access
        counts are again similar to HashJoin.  Wall-clock gap widens as the
        O(1) hash lookup advantage over linear scan accumulates.
        """
        self._bench(
            "large (10L / 5R pages)",
            left_n  = _LEFT_RPP  * 10,    # 440 rows, 10 pages
            right_n = _RIGHT_RPP * 5,     # 485 rows,  5 pages
            assert_hj_wins_pages=False,
        )

    def test_05_right_heavy(self):
        """
        Left ~5 pages (220 rows), right ~40 pages (3880 rows).
        Right table EXCEEDS the 32-frame buffer by 25% (40 > 32).

        HashJoin reads right exactly once during the build phase (40 page
        reads), builds an in-memory hash table, then probes left sequentially
        (5 page reads).  Total: ~45 disk reads regardless of left size.

        NestedLoopJoin resets the right iterator for every left row (220
        resets × 40 right pages).  Because right (40 pages) does not fit in
        the 32-frame buffer, the random replacer evicts right pages during
        each scan.  Subsequent scans must re-read evicted pages from disk,
        producing dramatically more I/O than HashJoin.

        Estimated page accesses:
          HashJoin      : ~45  (40 build + 5 probe)
          NestedLoopJoin: >> 45  (evictions cause repeated right-page reads)

        Assert: HashJoin page accesses <= NestedLoopJoin page accesses.
        """
        self._bench(
            "right_heavy (5L / 40R pages, R > buffer)",
            left_n  = _LEFT_RPP  * 5,     # 220 rows,   5 pages
            right_n = _RIGHT_RPP * 40,    # 3880 rows, 40 pages  (> 32-frame buffer)
            assert_hj_wins_pages=True,
        )


# ---------------------------------------------------------------------------
# Formatted report printed after all tests
# ---------------------------------------------------------------------------

def _print_report(results: list) -> None:
    if not results:
        return

    W = 108
    buf_frames = DatabaseConstants.MAX_BUFFER_FRAMES
    buf_kb     = buf_frames * DatabaseConstants.PAGE_SIZE // 1024

    print("\n" + "=" * W)
    print("  JOIN PERFORMANCE REPORT: HashJoin vs NestedLoopJoin")
    print("=" * W)
    print(
        f"  {'Tier':<38} {'LRows':>6} {'RRows':>6} {'LPg':>4} {'RPg':>4}"
        f"  {'HJ-pg':>6} {'NLJ-pg':>7}  {'HJ-s':>8} {'NLJ-s':>8}"
        f"  {'pg NLJ/HJ':>10}  {'t NLJ/HJ':>9}"
    )
    print("-" * W)

    for r in results:
        pg_ratio = (r["nlj_pg"] / r["hj_pg"]
                    if r["hj_pg"] > 0 else float("inf"))
        t_ratio  = (r["nlj_t"]  / r["hj_t"]
                    if r["hj_t"]  > 0 else float("inf"))
        exceeds  = "  [R > buf]" if r["right_pg"] > buf_frames else ""
        print(
            f"  {r['label']:<38} {r['left_n']:>6} {r['right_n']:>6} "
            f"{r['left_pg']:>4} {r['right_pg']:>4}"
            f"  {r['hj_pg']:>6} {r['nlj_pg']:>7}  "
            f"{r['hj_t']:>8.4f} {r['nlj_t']:>8.4f}"
            f"  {pg_ratio:>9.2f}x  {t_ratio:>8.2f}x"
            f"{exceeds}"
        )

    print("=" * W)
    print(
        f"  Buffer : {buf_frames} frames x {DatabaseConstants.PAGE_SIZE} B = {buf_kb} KB  |  "
        f"Left schema : {_LEFT_RPP} rec/page ({_left_schema().get_max_tuple_length()} B/rec)  |  "
        f"Right schema: {_RIGHT_RPP} rec/page ({_right_schema().get_max_tuple_length()} B/rec)"
    )
    print(
        f"  Columns: Left(id INT, name STR) JOIN Right(fk INT, value INT) ON id=fk"
        f"  ->  joined schema (id, name, fk, value)"
    )
    print("=" * W + "\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
