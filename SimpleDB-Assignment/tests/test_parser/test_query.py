"""
Tests for SQL query parsing functionality.
"""

import unittest
from simpledb.parser.query import Query
from simpledb.main.catalog.catalog import Catalog
from simpledb.main.catalog.tuple_desc import TupleDesc

class TestQueryParsing(unittest.TestCase):
    """Test SQL query parsing."""

    def test_simple_select(self):
        """Test parsing a simple SELECT query."""
        query = Query.generate_query("SELECT name, age FROM Students;")
        self.assertIsNotNone(query)
        self.assertEqual(query.get_table_name(), "Students")
        self.assertTrue(query.get_projected_columns() == ["name", "age"])
        self.assertFalse(query.has_join_arguments())
        self.assertFalse(query.has_filter_arguments())
        self.assertFalse(query.has_limit_clause())
        self.assertFalse(query.has_orderby_clause())

    def test_join_query_parsing(self):
        """Test parsing a JOIN query."""
        query = Query.generate_query("SELECT name, age FROM students JOIN courses ON id = class_id;")
        self.assertIsNotNone(query)
        self.assertEqual(query.get_table_name(), "students")
        self.assertTrue(query.get_projected_columns() == ["name", "age"])
        self.assertTrue(query.has_join_arguments())
        join_args = query.get_join_args()
        self.assertIsNotNone(join_args)
        self.assertEqual(join_args.get_join_table(), "courses")
        self.assertEqual(join_args.get_left_column(), "id")
        self.assertEqual(join_args.get_right_column(), "class_id")
        self.assertFalse(query.has_filter_arguments())
        self.assertFalse(query.has_limit_clause())
        self.assertFalse(query.has_orderby_clause())

    def test_sfw_query_parsing(self):
        """Test parsing a SFW query."""
        query = Query.generate_query("SELECT name, age FROM people WHERE age > 20;")
        self.assertIsNotNone(query)
        self.assertEqual(query.get_table_name(), "people")
        self.assertTrue(query.get_projected_columns() == ["name", "age"])
        self.assertTrue(query.has_filter_arguments())
        filter_args = query.get_filter_args()
        self.assertIsInstance(filter_args, list)
        self.assertTrue(len(filter_args) == 1)
        self.assertEqual(f"{filter_args[0]}", "age > 20")
        self.assertFalse(query.has_join_arguments())
        self.assertFalse(query.has_limit_clause())
        self.assertFalse(query.has_orderby_clause())

    def test_filter_clause_parsing(self):
        """Test parsing a complex WHERE clause."""
        query = Query.generate_query("SELECT name FROM Persons WHERE age >= 18 AND age > 19 AND age <> 22 AND age != 23 AND age <= 50 AND age < 51")
        self.assertIsNotNone(query)
        self.assertEqual(query.get_table_name(), "Persons")
        self.assertTrue(query.get_projected_columns() == ["name"])
        self.assertTrue(query.has_filter_arguments())
        filter_args = query.get_filter_args()
        self.assertIsInstance(filter_args, list)
        self.assertTrue(len(filter_args) == 6)
        self.assertEqual(f"{filter_args[0]}", "age >= 18")
        self.assertEqual(f"{filter_args[1]}", "age > 19")
        self.assertEqual(f"{filter_args[2]}", "age <> 22")
        self.assertEqual(f"{filter_args[3]}", "age <> 23")
        self.assertEqual(f"{filter_args[4]}", "age <= 50")
        self.assertEqual(f"{filter_args[5]}", "age < 51")
        self.assertFalse(query.has_join_arguments())
        self.assertFalse(query.has_limit_clause())
        self.assertFalse(query.has_orderby_clause())

    def test_select_limit(self):
        """Test parsing a simple SELECT-LIMIT query."""
        query = Query.generate_query("SELECT age, name FROM persons LIMIT 3;")
        self.assertIsNotNone(query)
        self.assertEqual(query.get_table_name(), "persons")
        self.assertTrue(query.get_projected_columns() == ["age", "name"])
        self.assertTrue(query.has_limit_clause())
        self.assertEqual(query.get_limit(), 3)
        self.assertFalse(query.has_join_arguments())
        self.assertFalse(query.has_filter_arguments())
        self.assertFalse(query.has_orderby_clause())

    def test_select_orderby(self):
        """Test parsing a simple SELECT-ORDER BY query."""
        query = Query.generate_query("SELECT name, age, addr FROM students ORDER BY age")
        self.assertIsNotNone(query)
        self.assertEqual(query.get_table_name(), "students")
        self.assertTrue(query.get_projected_columns() == ["name", "age", "addr"])
        self.assertTrue(query.has_orderby_clause())
        self.assertTrue(query.get_orderby_columns() == ["age"])
        self.assertFalse(query.has_join_arguments())
        self.assertFalse(query.has_filter_arguments())
        self.assertFalse(query.has_limit_clause())

    def test_case_insensitivity(self):
        """Test case- and space-insensitivity of SQL query parsing."""
        query = Query.generate_query("Select Name, Age, Id  from Students Join Courses on Class_id = Id  WhErE age >=18  and  grade <> F  order BY Age  limit 10")
        self.assertIsNotNone(query)
        self.assertEqual(query.get_table_name(), "Students")
        self.assertTrue(query.get_projected_columns() == ["name", "age", "id"])
        self.assertTrue(query.has_join_arguments())
        join_args = query.get_join_args()
        self.assertIsNotNone(join_args)
        self.assertEqual(join_args.get_join_table(), "Courses")
        self.assertEqual(join_args.get_left_column(), "class_id")
        self.assertEqual(join_args.get_right_column(), "id")
        self.assertTrue(query.has_filter_arguments())
        filter_args = query.get_filter_args()
        self.assertIsInstance(filter_args, list)
        self.assertTrue(len(filter_args) == 2)
        self.assertEqual(f"{filter_args[0]}", "age >= 18")
        self.assertEqual(f"{filter_args[1]}", "grade <> F")
        self.assertTrue(query.has_orderby_clause())
        self.assertTrue(query.get_orderby_columns() == ["age"])
        self.assertTrue(query.has_limit_clause())
        self.assertEqual(query.get_limit(), 10)

    def test_validate_query(self):
        catalog = Catalog()
        student_schema = TupleDesc()
        student_schema.add_string("name").add_integer("age").add_string("course_id")
        catalog.add_schema(student_schema, "Students")
        course_schema = TupleDesc()
        course_schema.add_string("id").add_string("title")
        catalog.add_schema(course_schema, "Courses")
        query = Query.generate_query("SELECT name, age FROM Students JOIN Courses ON course_id = id WHERE age >= 18 ORDER BY name LIMIT 5")
        self.assertIsNotNone(query)
        error = query.validate(catalog)
        self.assertIsNone(error)
    
if __name__ == '__main__':
    unittest.main()
