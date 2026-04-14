"""
Tuple representation - a physical row in the database.
"""

from typing import Any, List
from simpledb.main.catalog.type import Type
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.heap.page_id import PageId


class Tuple:
    """Represents a physical row in the database."""

    def __init__(self, schema: TupleDesc, values: List[Any] = None):
        """Create a new tuple with the given schema."""
        self.schema = schema
        self.row = [None] * schema.get_num_fields()
        self.slot_id = -1
        self.page_id = PageId()

        if values is not None:
            for i, value in enumerate(values):
                self.set_column(i, value)

    def get_schema(self) -> TupleDesc:
        """Get the schema of this tuple."""
        return self.schema

    def reset_with_schema(self, schema: TupleDesc) -> None:
        """Reset the tuple with the given schema."""
        self.schema = schema
        self.row = [None] * schema.get_num_fields()
        self.page_id = PageId()
        self.slot_id = -1

    def set_column(self, index_or_name, value: Any) -> None:
        """Set the value of a column."""
        if isinstance(index_or_name, str):
            index = self.schema.get_index_from_name(index_or_name)
        else:
            index = index_or_name

        column_type = self.schema.get_field_type(index)
        # Check that the value's type matches the column type
        if not isinstance(value, column_type.get_type_class()):
            raise TypeError(f"Invalid object type. Expected {column_type.get_type_class()}, got {type(value)}")
        self.row[index] = value

    def get_column(self, index_or_name) -> Any:
        """Get the value of a column."""
        if isinstance(index_or_name, str):
            index = self.schema.get_index_from_name(index_or_name)
        else:
            index = index_or_name

        if index < 0 or index >= len(self.row):
            raise IndexError("Invalid column index")
        return self.row[index]

    def get_slot_id(self) -> int:
        """Get the slot ID of this tuple on its page."""
        return self.slot_id

    def set_slot_id(self, slot_id: int) -> None:
        """Set the slot ID of this tuple."""
        self.slot_id = slot_id

    def get_page_id(self) -> PageId:
        """Get the page ID of this tuple."""
        return self.page_id

    def set_page_id(self, page_id: PageId) -> None:
        """Set the page ID of this tuple."""
        self.page_id = page_id

    def copy_values(self, other: 'Tuple') -> None:
        """Copy values from matching columns in another tuple."""
        for i in range(self.schema.get_num_fields()):
            try:
                field_name = self.schema.get_field_name(i)
                self.row[i] = other.get_column(field_name)
            except KeyError:
                pass

    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, Tuple):
            return False
        return self.row == other.row and self.schema == other.schema

    def __str__(self) -> str:
        """String representation."""
        return f"Tuple(schema={self.schema}, pageId={self.page_id}, slot={self.slot_id}, row={self.row})"

    def __repr__(self) -> str:
        """String representation for debugging."""
        return self.__str__()

    def to_row(self) -> str:
        """Convert row to string."""
        return str(self.row)
