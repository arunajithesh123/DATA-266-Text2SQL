# bigquery_connector.py

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
from typing import List, Dict, Any, Optional
import logging
import json

class BigQueryConnector:
    """Class to handle interactions with Google BigQuery."""
    
    def __init__(self, project_id: Optional[str] = None, credentials_path: Optional[str] = None):
        """Initialize the BigQuery connector.
        
        Args:
            project_id: Google Cloud project ID
            credentials_path: Path to service account credentials JSON file
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        self._init_client()
    
    def _init_client(self):
        """Initialize the BigQuery client."""
        try:
            if self.credentials_path:
                self.client = bigquery.Client.from_service_account_json(
                    self.credentials_path, 
                    project=self.project_id
                )
            else:
                # Use default credentials
                self.client = bigquery.Client(project=self.project_id)
            
            logging.info(f"Successfully initialized BigQuery client for project: {self.project_id}")
        except Exception as e:
            logging.error(f"Failed to initialize BigQuery client: {str(e)}")
            raise
    
    def get_dataset_list(self) -> List[str]:
        """Get a list of available datasets in the project."""
        try:
            datasets = list(self.client.list_datasets())
            return [dataset.dataset_id for dataset in datasets]
        except GoogleAPIError as e:
            logging.error(f"Error fetching datasets: {str(e)}")
            raise
    
    def get_table_list(self, dataset_id: str) -> List[str]:
        """Get a list of tables in a specific dataset."""
        try:
            tables = list(self.client.list_tables(dataset_id))
            return [table.table_id for table in tables]
        except GoogleAPIError as e:
            logging.error(f"Error fetching tables for dataset {dataset_id}: {str(e)}")
            raise
    
    def get_table_schema(self, dataset_id: str, table_id: str) -> str:
        """Get schema information for a BigQuery table in CREATE TABLE format."""
        try:
            table_ref = self.client.dataset(dataset_id).table(table_id)
            table = self.client.get_table(table_ref)
            
            # Create the CREATE TABLE statement
            field_definitions = []
            for field in table.schema:
                field_type = field.field_type
                field_definitions.append(f"{field.name} {field_type}")
            
            create_table = f"CREATE TABLE {table_id} ({', '.join(field_definitions)})"
            return create_table
        except GoogleAPIError as e:
            logging.error(f"Error fetching schema for {dataset_id}.{table_id}: {str(e)}")
            raise
    
    def execute_query(self, sql_query: str, timeout_ms: int = 60000) -> Dict[str, Any]:
        """Execute a SQL query against BigQuery and return results.
        
        Args:
            sql_query: The SQL query to execute
            timeout_ms: Query timeout in milliseconds
            
        Returns:
            Dictionary containing query results and metadata
        """
        try:
            # Configure query job
            job_config = bigquery.QueryJobConfig(
                query_parameters=[],
                timeout_ms=timeout_ms
            )
            
            # Start the query job
            query_job = self.client.query(sql_query, job_config=job_config)
            
            # Wait for the query to complete
            results = query_job.result()
            
            # Process the results
            columns = [field.name for field in results.schema]
            rows = []
            
            for row in results:
                # Convert row to dictionary
                row_dict = {}
                for key, value in row.items():
                    # Handle non-serializable types
                    if isinstance(value, (bytes, bytearray)):
                        row_dict[key] = value.hex()
                    elif hasattr(value, 'isoformat'):  # For datetime objects
                        row_dict[key] = value.isoformat()
                    else:
                        row_dict[key] = value
                rows.append(row_dict)
            
            # Get query statistics
            stats = {
                "bytes_processed": query_job.total_bytes_processed,
                "execution_time_ms": (query_job.ended - query_job.started).total_seconds() * 1000 
                                     if query_job.ended and query_job.started else None,
                "cached": query_job.cache_hit,
                "row_count": len(rows)
            }
            
            return {
                "columns": columns,
                "rows": rows,
                "stats": stats,
                "success": True
            }
            
        except GoogleAPIError as e:
            logging.error(f"Error executing query: {str(e)}")
            return {
                "columns": [],
                "rows": [],
                "stats": {},
                "success": False,
                "error": str(e)
            }
    
    def validate_query(self, sql_query: str) -> Dict[str, Any]:
        """Validate a SQL query without executing it.
        
        Args:
            sql_query: The SQL query to validate
            
        Returns:
            Dictionary with validation result
        """
        try:
            # Configure dry run
            job_config = bigquery.QueryJobConfig(
                dry_run=True,
                use_query_cache=False
            )
            
            # Start dry run job
            query_job = self.client.query(sql_query, job_config=job_config)
            
            # Get estimated bytes processed
            bytes_processed = query_job.total_bytes_processed
            
            return {
                "valid": True,
                "bytes_processed": bytes_processed,
                "message": f"Query is valid. It would process approximately {bytes_processed} bytes."
            }
            
        except GoogleAPIError as e:
            logging.error(f"Query validation error: {str(e)}")
            return {
                "valid": False,
                "bytes_processed": None,
                "message": f"Query validation failed: {str(e)}"
            }
    
    def get_table_preview(self, dataset_id: str, table_id: str, max_rows: int = 10) -> Dict[str, Any]:
        """Get a preview of data in a table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            max_rows: Maximum number of rows to return
            
        Returns:
            Dictionary with preview data
        """
        try:
            # Construct query to fetch sample rows
            sql_query = f"SELECT * FROM `{dataset_id}.{table_id}` LIMIT {max_rows}"
            return self.execute_query(sql_query)
            
        except Exception as e:
            logging.error(f"Error fetching table preview for {dataset_id}.{table_id}: {str(e)}")
            return {
                "columns": [],
                "rows": [],
                "stats": {},
                "success": False,
                "error": str(e)
            }