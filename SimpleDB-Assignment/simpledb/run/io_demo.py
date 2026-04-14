"""
I/O Demo: Shows page reads/writes for HashJoin vs NestedLoopJoin side by side.

This file lets you see exactly how many page accesses each join algorithm
makes on the same dataset, explaining where the performance difference comes from.

Two counters are shown:
  - Logical page requests : every time the buffer manager is asked for a page
                            (includes cache hits — pages already in RAM)
  - Physical disk I/O     : only actual reads/writes to the .db file
                            (excludes cache hits)

Run from project root:
    python -B -m simpledb.run.io_demo
"""

import tempfile
import os
from simpledb.main.database_manager import DatabaseManager
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.executor.query_planner import QueryPlanner
from simpledb.parser.query import Query


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def build_dbms() -> DatabaseManager:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.unlink(tmp.name)
    dbms = DatabaseManager(tmp.name)
    dbms._tmp_path = tmp.name
    return dbms


def cleanup(dbms: DatabaseManager) -> None:
    dbms.close()
    path = getattr(dbms, "_tmp_path", None)
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except Exception:
            pass


def insert_table(dbms, name, schema, rows):
    dbms.get_catalog().add_schema(schema, name)
    with dbms.get_heap_file(name).inserter() as ins:
        for row in rows:
            ins.insert(row)
    dbms.get_buffer_manager().flush_dirty()


def snapshot(dbms):
    """Capture current logical + physical I/O counters."""
    bm = dbms.get_buffer_manager()
    dm = dbms.get_disk_manager()
    return {
        "logical":  bm.get_page_accesses(),
        "cache_hits": bm.get_cache_hits(),
        "physical": dm.get_page_accesses(),
    }


def delta(before, after):
    """Return the difference between two snapshots."""
    return {
        "logical":    after["logical"]    - before["logical"],
        "cache_hits": after["cache_hits"] - before["cache_hits"],
        "physical":   after["physical"]   - before["physical"],
    }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_join_once(dbms, use_hash, sql):
    """Execute a single join query and return the result rows."""
    query = Query.generate_query(sql)
    query.validate(dbms.get_catalog())
    planner = QueryPlanner(dbms, use_hash_join=use_hash)
    logical = planner.create_logical_plan(query)
    iterator = planner.create_execution_plan(logical)
    rows = list(iterator)
    iterator.close()
    return rows


def measure(label, use_hash, student_rows, tutor_rows, sql, repeat=1):
    """
    Build a fresh database, insert the data, run the join `repeat` times,
    and print a detailed I/O breakdown.
    """
    student_schema = TupleDesc().add_string("name").add_string("class")
    tutor_schema   = TupleDesc().add_string("id").add_string("tutor")

    dbms = build_dbms()
    insert_table(dbms, "Students", student_schema, student_rows)
    insert_table(dbms, "Tutors",   tutor_schema,   tutor_rows)

    # Warm up — first run may include setup costs, measure from clean slate
    snap_before = snapshot(dbms)

    result_rows = None
    for _ in range(repeat):
        result_rows = run_join_once(dbms, use_hash, sql)

    snap_after = snapshot(dbms)
    d = delta(snap_before, snap_after)

    cache_misses = d["logical"] - d["cache_hits"]

    print(f"    {label}")
    print(f"      Logical page requests : {d['logical']:>5}  (over {repeat} run(s) = {d['logical']/repeat:.1f} avg)")
    print(f"        of which cache hits : {d['cache_hits']:>5}  (page already in RAM — no disk read needed)")
    print(f"        of which cache miss : {cache_misses:>5}  (had to go to disk)")
    print(f"      Physical disk I/O     : {d['physical']:>5}  (actual reads+writes to .db file)")
    print(f"      Result rows returned  : {len(result_rows)}")

    cleanup(dbms)
    return d


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

def compare(title, student_rows, tutor_rows, sql, repeat=5):
    print()
    print(f"  {'='*60}")
    print(f"  Scenario: {title}")
    print(f"  Students: {len(student_rows)}  |  Tutors: {len(tutor_rows)}  |  Runs: {repeat}")
    print(f"  {'='*60}")

    nlj = measure("NestedLoopJoin", use_hash=False,
                  student_rows=student_rows, tutor_rows=tutor_rows,
                  sql=sql, repeat=repeat)

    hj  = measure("HashJoin      ", use_hash=True,
                  student_rows=student_rows, tutor_rows=tutor_rows,
                  sql=sql, repeat=repeat)

    diff = nlj["logical"] - hj["logical"]
    pct  = (diff / nlj["logical"] * 100) if nlj["logical"] > 0 else 0
    winner = "HashJoin" if hj["logical"] < nlj["logical"] else (
             "NLJ"      if nlj["logical"] < hj["logical"] else "TIE")

    print(f"  --> Winner: {winner}  |  HashJoin saved {diff} logical requests ({pct:+.1f}%)")


def main():
    sql = "SELECT name, tutor FROM Students JOIN Tutors ON class = id"

    print()
    print("=" * 62)
    print("  I/O Demo — HashJoin vs NestedLoopJoin page access breakdown")
    print("=" * 62)
    print()
    print("  What are we measuring?")
    print("  ----------------------")
    print("  Logical page requests = every time an iterator asks the")
    print("  buffer manager for a page (get_page call in buffer_manager.py).")
    print()
    print("  Physical disk I/O = only when that page was NOT already")
    print("  in the buffer's RAM frames (actual file read in disk_manager.py).")
    print()
    print("  NLJ resets the right iterator for every left row, causing")
    print("  repeated get_page calls. HashJoin reads each table exactly once.")

    # --- Scenario 1: small, easy to follow ---
    classes = ["Math", "Science", "English"]
    students = [
        ["Alice",   "Math"],
        ["Bob",     "Science"],
        ["Charlie", "Math"],
        ["Diana",   "English"],
        ["Eve",     "Science"],
    ]
    tutors = [
        ["Math",    "Dr. Smith"],
        ["Science", "Dr. Jones"],
        ["English", "Dr. Lee"],
    ]
    compare("Small dataset (5 students, 3 tutors)", students, tutors, sql, repeat=5)

    # --- Scenario 2: more students, bigger gap ---
    classes2 = [f"C{i:02d}" for i in range(5)]
    students2 = [[f"Stu{i:03d}", classes2[i % 5]] for i in range(15)]
    tutors2   = [[cls, f"Tutor{j}"] for j, cls in enumerate(classes2)]
    compare("Medium dataset (15 students, 5 tutors)", students2, tutors2, sql, repeat=10)

    # --- Scenario 3: no matches (worst case for NLJ — still scans everything) ---
    left_students  = [[f"L{i}", "LeftClass"]  for i in range(8)]
    right_tutors   = [["RightClass", f"T{j}"] for j in range(4)]
    compare("No matches — disjoint keys", left_students, right_tutors, sql, repeat=5)

    print()
    print("=" * 62)
    print("  Why does HashJoin always win here?")
    print()
    print("  NLJ cost  = pages(Students) + rows(Students) x pages(Tutors)")
    print("  Hash cost = pages(Students) + pages(Tutors)  [read each once]")
    print()
    print("  As long as Students has more than 1 row, HashJoin is cheaper.")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
