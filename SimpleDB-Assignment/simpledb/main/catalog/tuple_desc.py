"""
Tuple Descriptor (Schema) Class for the Database.
Essentially a "schema" object that defines the types and field names of tuples.
"""

from typing import List
from simpledb.main.catalog.type import Type


class TupleDescItem:
    """Internal class to store name -> type pairs."""

    def __init__(self, field_type: Type, name: str):
        """Initialize a TupleDescItem."""
        self.type = field_type
        self.name = name.lower()  # Store names in lowercase for case-insensitive matching

    def get_name(self) -> str:
        """Get the field name."""
        return self.name

    def get_type(self) -> Type:
        """Get the field type."""
        return self.type


class TupleDesc:
    """TupleDescriptor class representing a database schema."""

    def __init__(self):
        """Create a new tuple descriptor."""
        self.columns: List[TupleDescItem] = []

    def add_string(self, name: str) -> 'TupleDesc':
        """Add a string column with the given name."""
        self.columns.append(TupleDescItem(Type.STRING, name))
        return self

    def add_integer(self, name: str) -> 'TupleDesc':
        """Add an integer column with the given name."""
        self.columns.append(TupleDescItem(Type.INTEGER, name))
        return self

    def add_double(self, name: str) -> 'TupleDesc':
        """Add a double column with the given name."""
        self.columns.append(TupleDescItem(Type.DOUBLE, name))
        return self

    def add_boolean(self, name: str) -> 'TupleDesc':
        """Add a boolean column with the given name."""
        self.columns.append(TupleDescItem(Type.BOOLEAN, name))
        return self

    def get_max_tuple_length(self) -> int:
        """Get the fixed length a tuple with this schema would have."""
        size = 0
        for item in self.columns:
            size += item.get_type().get_len()
        return size

    def get_num_fields(self) -> int:
        """Get the number of columns/fields in this TupleDesc."""
        return len(self.columns)

    def get_field_name(self, i: int) -> str:
        """Get the name of the field at position i."""
        if i < 0 or i >= self.get_num_fields():
            raise IndexError(f"Field index {i} out of range")
        return self.columns[i].get_name()

    def has_field(self, name: str) -> bool:
        """Check if a field with the given name exists."""
        return any(item.get_name() == name.lower() for item in self.columns)

    def get_field_type(self, i: int) -> Type:
        """Get the type of the field at position i."""
        if i < 0 or i >= self.get_num_fields():
            raise IndexError(f"Field index {i} out of range")
        return self.columns[i].get_type()

    def get_field_type_by_name(self, name: str) -> Type:
        """Get the type of the field with the given name."""
        return self.get_field_type(self.get_index_from_name(name.lower()))

    def get_index_from_name(self, field_name: str) -> int:
        """Get the position of the column with the given name."""
        for i, item in enumerate(self.columns):
            if item.get_name() == field_name.lower():
                return i
        raise KeyError(f"Field '{field_name}' not found in schema")

    def get_column_names(self) -> List[str]:
        """Get a list of column names in the schema."""
        return [item.get_name() for item in self.columns]
    
    def str(self) -> str:
        """return the tuple schema as a readable string."""
        retval = "("
        for item in self.columns:
            retval += (f"{item.get_name()}: {item.get_type()}")
            if item != self.columns[-1]:
                retval += ", "
        retval += ")"
        return retval

    @staticmethod
    def join(left: 'TupleDesc', right: 'TupleDesc') -> 'TupleDesc':
        """Compose a new schema for a join of two relations."""
        joined = TupleDesc()
        # Copy everything from the left relation
        for item in left.columns:
            joined.columns.append(item)
        # Copy new attributes from right (assumes they are of same type)
        for item in right.columns:
            if not joined.has_field(item.get_name()):
                joined.columns.append(item)
        return joined

    def project(self, *column_names: str) -> 'TupleDesc':
        """Return a new TupleDesc with only the given columns."""
        reduced = TupleDesc()
        for name in column_names:
            try:
                i = self.get_index_from_name(name)
                reduced.columns.append(self.columns[i])
            except KeyError:
                pass
        return reduced

    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, TupleDesc):
            return False
        if len(self.columns) != len(other.columns):
            return False
        for i in range(len(self.columns)):
            if self.columns[i].get_name() != other.columns[i].get_name():
                return False
            if self.columns[i].get_type() != other.columns[i].get_type():
                return False
        return True

    @property
    def columns_internal(self) -> List[TupleDescItem]:
        """Internal access to columns for join operations."""
        return self.columns
