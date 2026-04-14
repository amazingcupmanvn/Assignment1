"""
Data Page class for storing records on disk.
"""

from simpledb.main.database_constants import DatabaseConstants
from simpledb.disk.slotted_page import SlottedPage
from simpledb.heap.page_id import PageId
from simpledb.heap.tuple import Tuple
from simpledb.main.catalog.type import Type
from simpledb.main.catalog.tuple_desc import TupleDesc


class DataPage(SlottedPage):
    """Abstract Data Page class for storing records."""

    def initialise(self, relation_name: str) -> None:
        """Initialize the data page with the given schema."""
        super().initialise(DatabaseConstants.DATA_PAGE_TYPE)
        self.set_schema_name(relation_name)

    def get_record_count(self) -> int:
        """Get the number of records currently stored on the page."""
        return self.get_num_slots()

    def insert_record(self, record: Tuple) -> bool:
        """Insert a (fixed size) record into the next available slot.
        Returns False if the page has no space remaining."""
        record_size = record.get_schema().get_max_tuple_length()
        try:
            slot_no = self.allocate_slot(record_size)
        except OverflowError:
            return False
        if slot_no == -1:
            return False

        offset = self.get_slot_offset(slot_no)
        self._write(record, offset)
        return True

    def get_record(self, slot_no: int, record: Tuple) -> None:
        """Read the record at position slot_no from the page."""
        offset = self.get_slot_offset(slot_no)
        self._read(record, offset)
        record.set_slot_id(slot_no)

    @staticmethod
    def get_max_records_on_page(record_or_schema) -> int:
        """Get the maximum number of records that can fit on the page."""
        if isinstance(record_or_schema, Tuple):
            schema = record_or_schema.get_schema()
        else:
            schema = record_or_schema
        return (DatabaseConstants.PAGE_SIZE - DatabaseConstants.PAGE_HEADER_SIZE) // (schema.get_max_tuple_length() + DatabaseConstants.SLOT_ENTRY_SIZE)

    def _write(self, tuple_obj: Tuple, offset: int) -> None:
        """Low-level: Write the tuple to the given offset in the page."""
        schema = tuple_obj.get_schema()
        length = schema.get_num_fields()
        for i in range(length):
            column_type = schema.get_field_type(i)
            value = tuple_obj.get_column(i)

            if column_type == Type.STRING:
                self.set_string_value(value, offset)
            elif column_type == Type.INTEGER:
                self.set_integer_value(value, offset)
            elif column_type == Type.DOUBLE:
                self.set_double_value(value, offset)
            elif column_type == Type.BOOLEAN:
                self.set_boolean_value(value, offset)
            else:
                raise AssertionError("Invalid column type")

            offset += column_type.get_len()

    def _read(self, tuple_obj: Tuple, offset: int) -> None:
        """Low-level: Read a tuple from the page starting at the offset."""
        schema = tuple_obj.get_schema()
        length = schema.get_num_fields()
        for i in range(length):
            column_type = schema.get_field_type(i)
            
            if column_type == Type.STRING:
                value = self.get_string_value(offset)
            elif column_type == Type.INTEGER:
                value = self.get_integer_value(offset)
            elif column_type == Type.DOUBLE:
                value = self.get_double_value(offset)
            elif column_type == Type.BOOLEAN:
                value = self.get_boolean_value(offset)
            else:
                raise AssertionError("Invalid column type")

            tuple_obj.set_column(i, value)
            offset += column_type.get_len()
