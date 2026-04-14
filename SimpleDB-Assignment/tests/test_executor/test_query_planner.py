"""
Tests for the QueryPlanner class.
"""

import unittest
import tempfile
import os
from simpledb.main.database_manager import DatabaseManager
from simpledb.executor.query_planner import QueryPlanner, LogicalPlanNode
from simpledb.parser.query import Query
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.heap.tuple import Tuple
from simpledb.parser.filter_args import Comparison


class TestQueryPlanner(unittest.TestCase):
    """Test the QueryPlanner functionality."""

    def setUp(self):
        """Set up test database."""
        self.db_name = tempfile.NamedTemporaryFile(suffix='.db').name
        self.dbms = DatabaseManager(self.db_name)
        self.planner = QueryPlanner(self.dbms)

        # Create a test table
        schema = TupleDesc().add_string("name").add_integer("age")
        self.dbms.get_catalog().add_schema(schema, "people")

        table = self.dbms.get_heap_file("people")
        with table.inserter() as inserter:
            inserter.insert(["Alice", 30])
            inserter.insert(["Bob", 25])
            inserter.insert(["Carol", 35])

    def tearDown(self):
        """Clean up test database."""
        if os.path.exists(self.db_name):
            os.remove(self.db_name)

    def test_create_logical_plan_simple_select(self):
        """Test creating a logical plan for a simple SELECT query."""
        query = Query.generate_query("SELECT name, age FROM people;")
        logical_plan = self.planner.create_logical_plan(query)

        # Should have access -> project structure
        self.assertEqual(logical_plan.operation, "project")
        self.assertEqual(logical_plan.kwargs["columns"], ["name", "age"])

        self.assertEqual(len(logical_plan.children), 1)
        self.assertEqual(logical_plan.children[0].operation, "access")
        self.assertEqual(logical_plan.children[0].kwargs["table_name"], "people")

    def test_create_logical_plan_with_where(self):
        """Test creating a logical plan for a query with WHERE clause."""
        query = Query.generate_query("SELECT name FROM people WHERE age > 25;")
        logical_plan = self.planner.create_logical_plan(query)

        # Should have access -> filter -> project structure
        self.assertEqual(logical_plan.operation, "project")
        self.assertEqual(logical_plan.kwargs["columns"], ["name"])

        filter_node = logical_plan.children[0]
        self.assertEqual(filter_node.operation, "filter")
        filter_condition = filter_node.kwargs["filter_condition"]
        self.assertEqual(filter_condition.get_column(), "age")
        self.assertEqual(filter_condition.get_comparison(), Comparison.GREATER)
        self.assertEqual(filter_condition.get_value(), "25")

        access_node = filter_node.children[0]
        self.assertEqual(access_node.operation, "access")
        self.assertEqual(access_node.kwargs["table_name"], "people")

    def test_create_logical_plan_with_join(self):
        """Test creating a logical plan for a query with JOIN."""
        # Create another table for join
        schema2 = TupleDesc().add_integer("person_id").add_string("city")
        self.dbms.get_catalog().add_schema(schema2, "Addresses")

        query = Query.generate_query("SELECT name, city FROM People JOIN Addresses ON id = person_id;")
        logical_plan = self.planner.create_logical_plan(query)

        # Should have 2 x access -> join -> project structure
        self.assertEqual(logical_plan.operation, "project")
        self.assertEqual(logical_plan.kwargs["columns"], ["name", "city"])

        join_node = logical_plan.children[0]
        self.assertEqual(join_node.operation, "join")
        join_condition = join_node.kwargs["join_condition"]
        self.assertEqual(join_condition.get_left_column(), "id")
        self.assertEqual(join_condition.get_right_column(), "person_id")

        # input to join should be two table accesses
        self.assertEqual(join_node.children[0].operation, "access")
        self.assertEqual(join_node.children[0].kwargs["table_name"], "People")
        self.assertEqual(join_node.children[1].operation, "access")
        self.assertEqual(join_node.children[1].kwargs["table_name"], "Addresses")


    def test_create_logical_plan_complete(self):
        """Test creating a logical plan for a complete query."""
        query = Query.generate_query("SELECT name FROM people WHERE age >= 25 ORDER BY name LIMIT 2;")
        logical_plan = self.planner.create_logical_plan(query)

        # Should have access -> filter -> order_by -> project -> limit structure
        self.assertEqual(logical_plan.operation, "limit")
        self.assertEqual(logical_plan.kwargs["count"], 2)

        project_node = logical_plan.children[0]
        self.assertEqual(project_node.operation, "project")
        self.assertEqual(project_node.kwargs["columns"], ["name"])

        order_node = project_node.children[0]
        self.assertEqual(order_node.operation, "order_by")
        self.assertEqual(order_node.kwargs["columns"], ["name"])

        filter_node = order_node.children[0]
        self.assertEqual(filter_node.operation, "filter")
        filter_condition = filter_node.kwargs["filter_condition"]
        self.assertEqual(filter_condition.get_column(), "age")
        self.assertEqual(filter_condition.get_comparison(), Comparison.GEQ)
        self.assertEqual(filter_condition.get_value(), "25")

        access_node = filter_node.children[0]
        self.assertEqual(access_node.operation, "access")
        self.assertEqual(access_node.kwargs["table_name"], "people")

    def test_create_execution_plan_simple(self):
        """Test creating an execution plan from a logical plan."""
        query = Query.generate_query("SELECT name FROM people WHERE age > 25;")
        logical_plan = self.planner.create_logical_plan(query)
        execution_plan = self.planner.create_execution_plan(logical_plan)

        # Should be able to iterate over results
        results = list(execution_plan)
        self.assertEqual(len(results), 2)  # Alice (30) and Carol (35)
        execution_plan.close()


if __name__ == '__main__':
    unittest.main()
