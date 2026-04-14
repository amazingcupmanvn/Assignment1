"""
Access Inserter base class.
"""

from abc import ABC, abstractmethod
from simpledb.main.catalog.tuple_desc import TupleDesc


class AccessInserter(ABC):
    """Generic Inserter for database access patterns."""

    @abstractmethod
    def get_schema(self) -> TupleDesc:
        """Get the schema of tuples produced by this iterator."""
        pass

    @abstractmethod
    def insert(self, row: list) -> None:
        """Insert a row."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the inserter."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
