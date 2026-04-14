"""
Query Engine - REPL for executing queries.
"""

import time
import readline  # just by importing improves input() to support editing and history
from simpledb.main.database_manager import DatabaseManager
from simpledb.parser.query import Query
from simpledb.executor.query_planner import QueryPlanner


class QueryEngine:
    """
    Query engine for executing database queries.

    This class parses, plans, and executes the query given to the program by the user.
 
    It loops waiting for input (run method), once a command has been entered, it creates a new query and then checks that
    it is valid. If it is, it will plan and execute the query. This involves pipelining the query based on the different components
    The actual query planning and execution plan generation is done in the QueryPlanner class.
    """

    def __init__(self, dbms: DatabaseManager):
        """Initialize the QueryEngine."""
        self.dbms = dbms
        self.planner = QueryPlanner(dbms)

    def run(self) -> None:
        """
        Run the query engine's read-eval-print loop (REPL):
         - Loops over input read in from the user, validating the correctness and calling execute on valid queries
         - Exits once "quit" or "exit" is called
         - Shows schema of available tables on command "schema" or "tables"
        """
        print("SimpleDB Query Engine")
        print("Type 'quit' to exit")
        print()
        
        while True:
            try:
                command = input("SQL> ").strip()
                if command.lower() == 'quit' or command.lower() == 'exit':
                    break
                if command.lower() == 'schema' or command.lower() == 'tables':
                    self.dbms.get_catalog().print_schemas()
                    continue
                if not command:
                    continue
                
                self._execute_query(command)
            except KeyboardInterrupt:
                print("\nInterrupted")
            except Exception as e:
                print(f"Error: {e}")

    def _execute_query(self, command: str) -> None:
        """Execute a single query."""

        # parse SQL query syntax
        query = Query.generate_query(command)
        if query is None:
            print("Invalid query syntax")
            return
        
        # SQL query syntax against catalog information (e.g. table and column names)
        error = query.validate(self.dbms.get_catalog())
        if error:
            print(f"Query Validation Error: {error}")
            return
        
        try:
            # Create logical plan
            logical_plan = self.planner.create_logical_plan(query)
            
            # Create execution plan from logical plan
            result_iterator = self.planner.create_execution_plan(logical_plan)
            
        except Exception as e:
            print(f"Query Planner Error: {e}")
            raise

        try:
            # Execute and display results
            start_time = time.time()

            row_count = 0
            print()
            for tuple_obj in result_iterator:
                print(tuple_obj.to_row())
                row_count += 1
            
            result_iterator.close()
            
            elapsed = time.time() - start_time
            print(f"\n{row_count} rows retrieved in {elapsed:.3f}s")
            
        except Exception as e:
            print(f"Execution Error: {e}")
            raise
