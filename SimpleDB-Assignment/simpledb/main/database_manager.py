"""
Database Components are initialised and stored in this class.
"""

import tempfile
import os
from simpledb.main.database_constants import DatabaseConstants
from simpledb.disk.disk_manager import DiskManager
from simpledb.buffer.buffer_manager import BufferManager, BufferAccessException
from simpledb.buffer.replacement.random_replacer import RandomReplacer
from simpledb.main.catalog.catalog import Catalog
from simpledb.heap.heap_file import HeapFile
from simpledb.main.catalog.tuple_desc import TupleDesc


class ComponentsNotInitialisedError(Exception):
    """Raised when database components are not initialized."""
    pass


class DatabaseManager:
    """Database Components are initialised and stored in this class."""

    def __init__(self, db_filename: str = DatabaseConstants.DEFAULT_DB_NAME, buffer_frames: int = DatabaseConstants.MAX_BUFFER_FRAMES):
        """Initialize the DatabaseManager."""
        self.is_initialised = False
        self.dm = None
        self.bm = None
        self.catalog = None
        self._temp_file = None
        self._initialise_components(db_filename, buffer_frames)

    def _initialise_components(self, db_filename: str, buffer_frames: int) -> None:
        """Initialize database components."""
        if not self.is_initialised:
            try:
                # Initialize DiskManager
                self.dm = DiskManager(db_filename, 1, self._temp_file)

                # Initialize BufferManager with Random replacer
                if ( buffer_frames < 3):
                    raise ValueError("Buffer frames must be at least 3.")
                self.bm = BufferManager(
                    buffer_frames,
                    RandomReplacer(),
                    self.dm
                )

                # Initialize Catalog
                self.catalog = Catalog()
                self.is_initialised = True

            except Exception as e:
                # Clean up on failure
                if self._temp_file and os.path.exists(self._temp_file.name):
                    os.unlink(self._temp_file.name)
                raise e

    def reset_components(self) -> None:
        """Reset database components."""
        self.close()
        self.is_initialised = False
        self._initialise_components()

    def get_catalog(self) -> Catalog:
        """Get the catalog component."""
        if self.catalog is None:
            raise ComponentsNotInitialisedError()
        return self.catalog

    def get_disk_manager(self) -> DiskManager:
        """Get the disk manager component."""
        if self.dm is None:
            raise ComponentsNotInitialisedError()
        return self.dm

    def get_buffer_manager(self) -> BufferManager:
        """Get the buffer manager component."""
        if self.bm is None:
            raise ComponentsNotInitialisedError()
        return self.bm

    def close(self) -> None:
        """Close the database and clean up resources."""
        try:
            if self.bm:
                self.bm.flush_dirty()
        except BufferAccessException as e:
            print(f"Error flushing buffer: {e}")

        try:
            if self.dm:
                self.dm.reset()
        except Exception as e:
            print(f"Error resetting disk manager: {e}")

        # Clean up temporary file
        if self._temp_file and os.path.exists(self._temp_file.name):
            try:
                os.unlink(self._temp_file.name)
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")

    def get_heap_file(self, name: str) -> HeapFile:
        """Get a heap file by name."""
        schema = self.catalog.read_schema(name)
        return HeapFile(schema, name, self.bm)

    def get_temp_heap_file(self, schema: TupleDesc) -> HeapFile:
        """Get a temporary heap file with the given schema."""
        return HeapFile(schema, buffer_manager=self.bm)
