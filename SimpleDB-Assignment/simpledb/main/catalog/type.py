"""
Type enum class.
Lists the supported types by the Database.
"""

from enum import Enum
from simpledb.main.database_constants import DatabaseConstants


class Type(Enum):
    """Enum for supported database types."""
    STRING = "STRING"
    DOUBLE = "DOUBLE"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"

    def get_len(self) -> int:
        """Get the byte length needed to store this type."""
        if self == Type.STRING:
            return 2 + DatabaseConstants.MAX_STRING_LENGTH
        elif self == Type.DOUBLE:
            return 8
        elif self == Type.INTEGER:
            return 4
        elif self == Type.BOOLEAN:
            return 1
        else:
            raise ValueError(f"Unknown type: {self}")

    def get_type_class(self):
        """Get the Python type class for this type."""
        if self == Type.STRING:
            return str
        elif self == Type.DOUBLE:
            return float
        elif self == Type.INTEGER:
            return int
        elif self == Type.BOOLEAN:
            return bool
        else:
            raise ValueError(f"Unknown type: {self}")

    def parse_type(self, value: str):
        """Convert a string to the appropriate Python type."""
        if self == Type.STRING:
            return value
        elif self == Type.DOUBLE:
            try:
                return float(value)
            except (ValueError, TypeError):
                return False
        elif self == Type.INTEGER:
            try:
                return int(value)
            except (ValueError, TypeError):
                return False
        elif self == Type.BOOLEAN:
            if value == "true" or value == "false":
                return value == "true"
            return None
        else:
            return None
