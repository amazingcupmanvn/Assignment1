"""
Class to simplify parsing and extracting of query data for WHERE clauses.
"""

import re
from enum import Enum
from typing import List


class Comparison(Enum):
    """Enum to represent the different types of comparisons in WHERE clauses."""
    """We can compare less than, less than or equal to, greater than or equal to, greater than, equal and not equal."""
    GEQ = ">="
    LEQ = "<="
    LESS = "<"
    GREATER = ">"
    EQUAL = "="
    NOT_EQUAL = "<>"  # alias: "!=" is also supported

    def get_symbol(self) -> str:
        """Get the symbol for this comparison."""
        return self.value

    @staticmethod
    def find(value: str) -> 'Comparison':
        """Find the Comparison enum from a string symbol."""
        for comp in Comparison:
            if comp.get_symbol() == value:
                return comp
        # we also support "!=" as a synonym for "<>"
        if value == "!=":
            return Comparison.NOT_EQUAL
        return None


class FilterArgs:
    """Class to represent the parsed arguments from a WHERE clause."""
    # Once a where clause has been parsed, it will be assigned here and you can access the values with the getters below.
    # Note: The get_value() method will return the comparison value AS A STRING. You will need to convert it into the
    #       appropriate object type first before you can use it. 

    def __init__(self, column: str, comp: Comparison, value: str):
        """Initialize a FilterArgs."""
        self.column = column
        self.comp = comp
        self.value = value

    def get_column(self) -> str:
        """Get the column name."""
        return self.column

    def get_comparison(self) -> Comparison:
        """Get the comparison operator."""
        return self.comp

    def get_value(self) -> str:
        """Get the comparison value as a string."""
        return self.value

    def __str__(self) -> str:
        """String representation."""
        return f"{self.column} {self.comp.get_symbol()} {self.value}"

    # The pattern that represents the format of a single where clause (<column> <comparison_operator> <value>)
    WHERE_CLAUSE_PATTERN = re.compile(r"(\w+)\s*(>=|<>|<=|=|>|<|!=)\s*([\w\.]+)")

    @staticmethod
    def parse(command: str):# -> List[FilterArgs]:
        """
        Parses a whole WHERE clause string, e.g. (age >= 10 AND speed < 20.0 AND name = john)
        into a list of FilterArgs. It splits the input string on their ANDs and attempts to
        match on the expected format of the where clause.
        """
        contents = re.split(" AND ", command.strip(), flags=re.IGNORECASE)
        filter_args = []
        for content in contents:
            matcher = FilterArgs.WHERE_CLAUSE_PATTERN.match(content.strip())
            if not matcher:
                raise ValueError(f"Invalid WHERE clause format: {content}")
            column = matcher.group(1)
            operator = matcher.group(2)
            value = matcher.group(3)
            comparison = Comparison.find(operator)
            if comparison is None:
                raise ValueError(f"Unknown comparison operator: {operator}")
            filter_args.append(FilterArgs(column, comparison, value))
        return filter_args