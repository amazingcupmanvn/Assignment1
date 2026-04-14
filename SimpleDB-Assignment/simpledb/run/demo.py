"""
Demo for the SimpleDB Database System.
"""

from simpledb.main.database_manager import DatabaseManager
from simpledb.main.database_constants import DatabaseConstants
from simpledb.executor.query_engine import QueryEngine
from simpledb.main.catalog.tuple_desc import TupleDesc
import argparse

STUDENT_ROWS_SMALL = [
    ["Michael", 19, "INFO1103", True],
    ["Jan", 18, "INFO1903", False],
    ["Roger", 20, "INFO1103", True],
    ["Rachael", 21, "ELEC1601", False]
]

TUTOR_ROWS_SMALL = [
    ["INFO1103", "Joshua"],
    ["INFO1103", "Scott"],
    ["COMP2129", "Maxwell"],
    ["INFO1903", "Steven"]
]


def insert_rows(table, rows):
    """Insert rows into a table."""
    with table.inserter() as inserter:
        for row in rows:
            inserter.insert(row)


def main():
    """Process command-line argument"""
    argparser = argparse.ArgumentParser(description="SimpleDB demo")
    argparser.add_argument("-d", "--dbfile", metavar="FILNAME", help="name of database file",   default=DatabaseConstants.DEFAULT_DB_NAME,   type=str)
    argparser.add_argument("-b", "--buffer", metavar="SIZE", help="number of buffer frames", default=DatabaseConstants.MAX_BUFFER_FRAMES, type=int)
    args = argparser.parse_args()

    """Run the demo."""
    try:
        dbms = DatabaseManager(args.dbfile, args.buffer)
    except Exception as e:
        print(f"Error initializing DBMS: {e}")
        return  
    
    # Create Test Schema for students
    student_schema = TupleDesc()
    student_schema.add_string("name").add_integer("age").add_string("class").add_boolean("male")
    dbms.get_catalog().add_schema(student_schema, "Students")
    students = dbms.get_heap_file("Students")
    
    # Create Test Schema for tutors
    tutor_schema = TupleDesc()
    tutor_schema.add_string("id").add_string("tutor")
    dbms.get_catalog().add_schema(tutor_schema, "Tutors")
    tutors = dbms.get_heap_file("Tutors")
    
    # Insert rows (only if db file without data yet)
    if students.is_empty():
        insert_rows(students, STUDENT_ROWS_SMALL)
    if tutors.is_empty():
        insert_rows(tutors, TUTOR_ROWS_SMALL)
    
    # Run query engine
    query_engine = QueryEngine(dbms)
    query_engine.run()
    
    # Flush dirty pages
    dbms.get_buffer_manager().flush_dirty()


if __name__ == "__main__":
    main()
