"""
Query Planner - creates logical and execution plans for queries.
"""

from typing import List, Any
from simpledb.main.database_manager import DatabaseManager
from simpledb.parser.query import Query
from simpledb.access.read.access_iterator import AccessIterator
from simpledb.executor.projection.projection import Projection
from simpledb.executor.limit.limit import Limit
from simpledb.executor.ordering import InMemoryOrderBy
from simpledb.executor.filter.filter import Filter
from simpledb.executor.filter.equals import Equals
from simpledb.executor.filter.range import GreaterThanEquals, GreaterThan, LessThanEquals, LessThan
from simpledb.executor.filter.not_modifier import NotModifier
from simpledb.parser.filter_args import Comparison
from simpledb.executor.join.nested_loop_join import NestedLoopJoin
from simpledb.executor.join.hash_join import HashJoin


class LogicalPlanNode:
    """Represents a node in the logical query plan."""

    def __init__(self, operation: str, **kwargs):
        self.operation = operation
        self.kwargs = kwargs
        self.children: List['LogicalPlanNode'] = []

    def add_child(self, child: 'LogicalPlanNode'):
        self.children.append(child)

    def __str__(self):
        return f"{self.operation}({', '.join(f'{k}={v}' for k, v in self.kwargs.items())})"


class QueryPlanner:
    """Creates logical and execution plans for database queries."""

    def __init__(self, dbms: DatabaseManager):
        self.dbms = dbms

    def create_logical_plan(self, query: Query) -> LogicalPlanNode:
        """
        Create a logical plan tree from a parsed query.

        The logical plan represents the operations that need to be performed
        in a tree structure, starting from the leaves (data access) and working
        up to the final result (projection/limit).

        Note: so far, only a single join is supported...
        """
        # Start with the base table access
        root = LogicalPlanNode("access", table_name=query.get_table_name())

        # Add join if present
        if query.has_join_arguments():
            join_args = query.get_join_args()
            join_node = LogicalPlanNode("join", join_condition=join_args)
            join_node.add_child(root)
            join_node.add_child(LogicalPlanNode("access", table_name=join_args.get_join_table()))
            root = join_node

        # Add filters according to WHERE clause
        if query.has_filter_arguments():
            for filter_args in query.get_filter_args():
                filter_node = LogicalPlanNode("filter", filter_condition=filter_args)
                filter_node.add_child(root)
                root = filter_node

        # Add ordering
        if query.has_orderby_clause():
            order_node = LogicalPlanNode("order_by", columns=query.get_orderby_columns())
            order_node.add_child(root)
            root = order_node

        # Add projection
        if query.get_projected_columns():
            project_node = LogicalPlanNode("project", columns=query.get_projected_columns())
            project_node.add_child(root)
            root = project_node

        # Add limit
        if query.has_limit_clause():
            limit_node = LogicalPlanNode("limit", count=query.get_limit())
            limit_node.add_child(root)
            root = limit_node

        return root

    def create_execution_plan(self, logical_plan: LogicalPlanNode) -> AccessIterator:
        """
        Create an execution plan from a logical plan.

        This recursively walks the logical plan tree and creates the
        corresponding iterator pipeline.

        At this moment, we have no real query optimiser implemented, 
        so we will just follow the structure of the logical plan directly.
        If there are new physical operators added, this is where you would decide 
        which one to use based on the query and statistics about the data and any indexes.
        """
        return self._build_iterator(logical_plan)

    def _build_iterator(self, node: LogicalPlanNode) -> AccessIterator:
        """Recursively build the iterator pipeline from a logical plan node."""
        if node.operation == "access":
            # Base table access
            table = self.dbms.get_heap_file(node.kwargs["table_name"])
            return table.iterator()

        elif node.operation == "join":
            # Join operation
            left_iterator = self._build_iterator(node.children[0])
            right_iterator = self._build_iterator(node.children[1])
            # Prefer hash join for equi-joins; keep nested loop as a safe fallback.
            try:
                return HashJoin(left_iterator, right_iterator, node.kwargs["join_condition"])
            except Exception:
                return NestedLoopJoin(left_iterator, right_iterator, node.kwargs["join_condition"])

        elif node.operation == "filter":
            # Filter operation according to WHERE clause
            child_iterator = self._build_iterator(node.children[0])
            return QueryPlanner.filter_where(child_iterator, node.kwargs["filter_condition"])

        elif node.operation == "order_by":
            # Order by operation
            child_iterator = self._build_iterator(node.children[0])
            return InMemoryOrderBy(child_iterator, node.kwargs["columns"])

        elif node.operation == "project":
            # Projection operation
            child_iterator = self._build_iterator(node.children[0])
            return Projection(child_iterator, *node.kwargs["columns"])

        elif node.operation == "limit":
            # Limit operation
            child_iterator = self._build_iterator(node.children[0])
            return Limit(child_iterator, node.kwargs["count"])

        else:
            raise ValueError(f"Unknown operation: {node.operation}")

    @staticmethod
    def filter_where(query_iter, filter_args):
        """
        Applies the filter condition described by where_arg to the iterator rows, and returns an iterator over this
        pipelined view.

        You will need to implement the rest of the WHERE clause comparison signs. Currently we can only check if a column
        is equal. Have a look at the signs you need to implement in FilterArgs, and look at the filters that are available
        in Range and NotModifier classes
        """
        schema = query_iter.get_schema()
        # Gets the column name of the where clause
        column = filter_args.get_column()
        # Obtains the where value in the appropriate type to use for filtering (i.e. we need to convert the string
        # "10.9" to a (double)'10.9' so we can compare our records
        value = schema.get_field_type_by_name(column).parse_type(filter_args.get_value())
        # Applies the right where filter
        comparison = filter_args.get_comparison()
        if comparison == Comparison.EQUAL:
            return Filter(query_iter, column, Equals(value))
        elif comparison == Comparison.NOT_EQUAL:
            return Filter(query_iter, column, NotModifier(Equals(value)))
        elif comparison == Comparison.GEQ:
            return Filter(query_iter, column, GreaterThanEquals(value))
        elif comparison == Comparison.GREATER:
            return Filter(query_iter, column, GreaterThan(value))
        elif comparison == Comparison.LEQ:
            return Filter(query_iter, column, LessThanEquals(value))
        elif comparison == Comparison.LESS:
            return Filter(query_iter, column, LessThan(value))
        else:
            return query_iter