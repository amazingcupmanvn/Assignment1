"""
Comparator for ordering tuples by specified columns.
"""

from simpledb.main.catalog.type import Type
from simpledb.heap.tuple import Tuple


class ColumnComparator:
    """
    Compares two tuples based on specified columns.
    When used in conjunction with a sort method, the tuples will end up in sorted ascending order
    """

    def __init__(self, columns):
        if not columns:
            raise ValueError("No columns specified")
        self.columns = columns

    def compare(self, o1: Tuple, o2: Tuple) -> int:
        """Compare two tuples according to the configured columns."""
        for column in self.columns:
            left_value = o1.get_column(column)
            right_value = o2.get_column(column)
            column_type = o1.get_schema().get_field_type_by_name(column)

            if column_type == Type.INTEGER:
                compare = (left_value > right_value) - (left_value < right_value)
            elif column_type == Type.DOUBLE:
                compare = (left_value > right_value) - (left_value < right_value)
            elif column_type == Type.STRING:
                if left_value < right_value:
                    compare = -1
                elif left_value > right_value:
                    compare = 1
                else:
                    compare = 0
            elif column_type == Type.BOOLEAN:
                compare = (left_value > right_value) - (left_value < right_value)
            else:
                raise NotImplementedError(f"Unsupported type for ordering: {column_type}")

            if compare != 0:
                return compare

        return 0
