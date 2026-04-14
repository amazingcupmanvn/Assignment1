"""
Join Arguments class.
"""


class JoinArgs:
    """Represents join arguments for a query."""

    def __init__(self, join_table: str, left_column: str, right_column: str):
        """Initialize JoinArgs."""
        self.join_table = join_table
        self.left_column = left_column
        self.right_column = right_column

    def get_join_table(self) -> str:
        """Get the table to join with."""
        return self.join_table

    def get_left_column(self) -> str:
        """Get the left join column."""
        return self.left_column

    def get_right_column(self) -> str:
        """Get the right join column."""
        return self.right_column

    def __str__(self) -> str:
        """String representation."""
        return f"JOIN {self.join_table} ON {self.left_column} = {self.right_column}"
