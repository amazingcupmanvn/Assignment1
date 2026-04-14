"""
Page class representing a PAGE_SIZE page in the database.
"""

import struct
from simpledb.main.database_constants import DatabaseConstants


class Page:
    """Represents a raw page in the database."""
    """self.data  stores the page's actual data as bytearray"""

    def __init__(self, data: bytes = None):
        """Create a new page, optionally with the given data."""
        if data is None:
            self.data = bytearray(DatabaseConstants.PAGE_SIZE)
        else:
            self.set_data(data)

    def set_data(self, data: bytearray) -> None:
        """SetterI for the data byte array."""
        if len(data) != DatabaseConstants.PAGE_SIZE:
            raise AssertionError(f"Data length {len(data)} != PAGE_SIZE {DatabaseConstants.PAGE_SIZE}")
        self.data = data

    def get_data(self) -> bytearray:
        """Get the data byte array."""
        return self.data

    def copy(self, other: 'Page') -> None:
        """Copy the contents of another page into this one."""
        self.data = bytearray(other.data)

    def get_boolean_value(self, offset: int) -> bool:
        """Get the boolean value at the offset."""
        return self.data[offset] != 0

    def set_boolean_value(self, value: bool, offset: int) -> None:
        """Set the boolean value at the offset."""
        self.data[offset] = 1 if value else 0

    def get_string_value(self, offset: int) -> str:
        """Get the string value at the offset."""
        length = struct.unpack_from('>H', self.data, offset)[0]
        start = offset + 2
        end = start + length
        return self.data[start:end].decode('ascii')

    def set_string_value(self, value: str, offset: int) -> None:
        """Set the string value at the offset."""
        value_bytes = value.encode('ascii')
        length = len(value_bytes)
        struct.pack_into('>H', self.data, offset, length)
        self.data[offset + 2:offset + 2 + length] = value_bytes

    def get_double_value(self, offset: int) -> float:
        """Get the double value at the offset."""
        return struct.unpack_from('>d', self.data, offset)[0]

    def set_double_value(self, value: float, offset: int) -> None:
        """Set the double value at the offset."""
        struct.pack_into('>d', self.data, offset, value)

    def get_integer_value(self, offset: int) -> int:
        """Get the integer value at the offset."""
        return struct.unpack_from('>i', self.data, offset)[0]

    def set_integer_value(self, value: int, offset: int) -> None:
        """Set the integer value at the offset."""
        struct.pack_into('>i', self.data, offset, value)

    def get_short_value(self, offset: int) -> int:
        """Get the short value at the offset."""
        return struct.unpack_from('>H', self.data, offset)[0]

    def set_short_value(self, value: int, offset: int) -> None:
        """Set the short value at the offset."""
        struct.pack_into('>H', self.data, offset, value)
