"""
Page ID representation.
Identifies a page on disk.
"""

from simpledb.main.database_constants import DatabaseConstants


class PageId:
    """Represents a page ID that identifies a page on disk."""
    SIZE = 4  # Size in bytes to store a page ID (integer)
    
    def __init__(self, page_id: int = None):
        """Initialize a PageId."""
        if page_id is None:
            self.page_id = DatabaseConstants.INVALID_PAGE_ID
        else:
            self.page_id = page_id

    def is_valid(self) -> bool:
        """Check if this PageId is valid."""
        return self.page_id != DatabaseConstants.INVALID_PAGE_ID

    def get(self) -> int:
        """Get the integer associated with this PageId."""
        return self.page_id

    def set(self, value: int) -> None:
        """Set the integer value of this PageId."""
        self.page_id = value

    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, PageId):
            return False
        return self.page_id == other.page_id

    def __hash__(self) -> int:
        """Allow PageId to be used in dictionaries and sets."""
        return hash(self.page_id)

    def __str__(self) -> str:
        """String representation."""
        return f"PageId({self.page_id})"

    def __repr__(self) -> str:
        """String representation for debugging."""
        return self.__str__()
