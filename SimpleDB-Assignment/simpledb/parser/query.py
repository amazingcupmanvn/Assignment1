"""
Query parser for database queries.
"""

import re
from typing import List, Tuple as PyTuple
from simpledb.parser.join_args import JoinArgs
from simpledb.parser.filter_args import FilterArgs


class Query:
    """Represents a database query."""

    def __init__(self, table_name: str, projected_columns: List[str], join_args: JoinArgs = None, where_args: FilterArgs = None, order_by: List[str] = None, limit: int = None):
        """Initialize a Query."""
        self.table_name = table_name
        self.projected_columns = projected_columns
        self.join_args  = join_args
        self.filter_args= where_args
        self.order_args = order_by
        self.limit_arg  = limit
 
    def get_projected_columns(self) -> List[str]:
        """Get the columns to project from the query."""
        return self.projected_columns

    def get_table_name(self) -> str:
        """Get the table name from the query."""
        return self.table_name

    def has_join_arguments(self) -> bool:
        """Check if the query has join arguments."""
        return self.join_args is not None

    def get_join_args(self) -> JoinArgs:
        """Get the join arguments."""
        return self.join_args

    def has_filter_arguments(self) -> bool:
        """Check if the query has where/filter arguments."""
        return self.filter_args is not None

    def get_filter_args(self) -> FilterArgs:
        """Get the where/filter arguments from the query."""
        return self.filter_args

    def has_orderby_clause(self) -> bool:
        """Check whether the query result needs to be sorted."""
        return self.order_args is not None and len(self.order_args) > 0

    def get_orderby_columns(self) -> List[str]:
        """Get the columns to order by."""
        return self.order_args

    def has_limit_clause(self) -> bool:
        """Check whether the query has a limit clause."""
        return self.limit_arg is not None

    def get_limit(self) -> int:
        """Get the number of rows to return."""
        return max(self.limit_arg, 0)
    
    def validate(self, catalog) -> str:
        """Validate the query against the database schema."""
        schema = catalog.read_schema(self.table_name)
        if schema is None:
            return f"Invalid Schema: {self.table_name}"
            
        schema_columns = schema.get_column_names()
        join_schema_columns = None

        if not self.has_join_arguments():
            # Single table query
            for column in self.projected_columns:
                if column not in schema_columns:
                    return f"Invalid Column: {column}"
        else:
            # Join query
            join_schema = catalog.read_schema(self.join_args.get_join_table())
            if join_schema is None:
                return f"Invalid Join-Table {self.join_args.get_join_table()}"
            
            join_schema_columns = join_schema.get_column_names()
            
            for column in self.projected_columns:
                in_left = column in schema_columns
                in_right = column in join_schema_columns
                
                if not in_left and not in_right:
                    return f"Invalid Join Column: {column}"
                if in_left and in_right:
                    return f"Ambiguous Join Column: {column}"
            
            if self.join_args.get_left_column() not in schema_columns:
                return "Join condition columns cannot be found in the schema"
            if self.join_args.get_right_column() not in join_schema_columns:
                return "Join condition columns cannot be found in the schema"
            
            left_type = schema.get_field_type_by_name(self.join_args.get_left_column())
            right_type = join_schema.get_field_type_by_name(self.join_args.get_right_column())
            
            if left_type != right_type:
                return "Join columns are of a different type"

        if self.has_filter_arguments():
            for args in self.filter_args:
                if args.get_column() not in schema_columns and (join_schema_columns is None or args.get_column() not in join_schema_columns):
                    return f"Invalid Where Column: {args.get_column()}"
                if args.get_comparison() is None:
                    return f"Invalid Where Comparison Operator: {args.get_comparison()}"                
                if args.get_column() in schema_columns and schema.get_field_type_by_name(args.get_column()).parse_type(args.get_value()) is False:
                    return f"Invalid Where Comparison Value: {args.get_value()} for column {args.get_column()} of type {schema.get_field_type_by_name(args.get_column())}"
                elif join_schema_columns is not None and args.get_column() in join_schema_columns and join_schema.get_field_type_by_name(args.get_column()).parse_type(args.get_value()) is False:
                    return f"Invalid Join Comparison Value: {args.get_value()} for column {args.get_column()} of type {schema.get_field_type_by_name(args.get_column())}"

        if self.has_orderby_clause():
            for column in self.order_args:
                if column not in schema_columns and (join_schema_columns is None or column not in join_schema_columns):
                    return f"Invalid Order-By Column: {column}"

        return None


    @staticmethod
    def generate_query(command: str):
        """Parse a query from a string."""
        pattern = r"SELECT ([\w, ]+)\s+"\
                  r"FROM (\w+)"\
                  r"(\s+JOIN (\w+)\s+ON (\w+)\s*=\s*(\w+))?"\
                  r"(\s+WHERE ((\w+\s*(>=|<>|!=|<=|=|>|<)\s*[\w\.]+)(\s+AND\s+\w+\s*(>=|<>|!=|<=|=|>|<)\s*[\w\.]+)*))?"\
                  r"(\s+ORDER BY (\w+(,\s*\w+)*))?"\
                  r"(\s+LIMIT (\d+))?"\
                  r";?"
        # Column names index 1
        # Table name index 2
        # Join Clause (optional, Index 3 to 6)
        # Where Clause (optional, Index 7+)
        # Order By Clause (optional, index 13 + order by columns: index 14)
        # Limit Clause (optional, index 16 + limit_value: index 17)
        match = re.match(pattern, command.strip(), re.IGNORECASE)
        
        if not match:
            return None
        
        table_name = match.group(2)
        projected_columns = [col.strip().lower() for col in match.group(1).split(',')]
        
        # parse Join clause
        join_args = None
        if match.group(3):  # Has JOIN
            join_args = JoinArgs(match.group(4), match.group(5).lower(), match.group(6).lower())
        
        # parse Where clause
        where_args = None
        if match.group(7): # has WHERE
            where_args = FilterArgs.parse(match.group(8))
        
        # parse OrderBy clause
        orderBy = None
        if match.group(13): # has ORDER BY
            orderBy = [col.strip().lower() for col in match.group(14).split(",")]
        
        # parse Limit clause
        limit = None;
        if match.group(16): # has LIMIT
            limit = int(match.group(17))

        return Query(table_name, projected_columns, join_args, where_args, orderBy, limit)


    def __str__(self) -> str:
        """String representation."""
        result = f"Running a query on: {self.table_name} projecting over columns: {self.projected_columns}"
        if self.has_join_arguments():
            result += f" joining {self.join_args}"
        if self.has_filter_arguments():
            for filter in self.filter_args:
                result += f" filtering on {filter}"
        if self.has_orderby_clause():
            result += f" order by {self.order_args}"
        if self.has_limit_clause():
            result += f" limit {self.limit_arg}"
        return result
