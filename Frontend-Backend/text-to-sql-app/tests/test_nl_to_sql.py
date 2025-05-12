# test_nl_to_sql.py

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from google.cloud import bigquery
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the application modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import application modules
from app import generate_sql
from bigquery_connector import BigQueryConnector
from crewai_agents import SQLQueryAgent

class TestNLToSQL(unittest.TestCase):
    """Test cases for the Natural Language to SQL application."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all test methods."""
        # Load environment variables
        cls.model_path = os.getenv('MODEL_PATH', '/content/drive/MyDrive/mistral7b_sql_model')
        cls.use_cuda = os.getenv('USE_CUDA', 'True').lower() == 'true'
        cls.device = "cuda" if torch.cuda.is_available() and cls.use_cuda else "cpu"
        
        # Test data
        cls.test_queries = [
            {
                "question": "What is the total revenue by product category?",
                "context": "CREATE TABLE sales (product_id STRING, category STRING, revenue FLOAT, date DATE)",
                "expected_pattern": r"SELECT\s+category\s*,\s*SUM\s*\(\s*revenue\s*\)\s+.*\s+FROM\s+sales\s+.*\s+GROUP\s+BY\s+category"
            },
            {
                "question": "Find customers who spent more than $1000 last month",
                "context": "CREATE TABLE customers (id STRING, name STRING, spending FLOAT, purchase_date DATE)",
                "expected_pattern": r"SELECT\s+.*\s+FROM\s+customers\s+WHERE\s+.*spending\s*>\s*1000.*"
            }
        ]
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_generate_sql(self, mock_tokenizer, mock_model):
        """Test SQL generation function."""
        # Set up mocks
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value.decode.return_value = "Response: SELECT category, SUM(revenue) FROM sales GROUP BY category"
        mock_tokenizer.return_value.padding_side = "left"
        
        mock_model.return_value = MagicMock()
        mock_model.return_value.generate.return_value = torch.tensor([[4, 5, 6]])
        
        # Call the function
        for test_case in self.test_queries:
            sql_result = generate_sql(test_case["question"], test_case["context"])
            
            # Assert result matches expected pattern
            self.assertRegex(sql_result, test_case["expected_pattern"], 
                            f"Generated SQL doesn't match expected pattern for: {test_case['question']}")
    
    @patch('google.cloud.bigquery.Client')
    def test_bigquery_connector(self, mock_client):
        """Test BigQuery connector functionality."""
        # Set up mock
        mock_client.return_value = MagicMock()
        
        # Create sample dataset and table
        mock_dataset = MagicMock()
        mock_dataset.dataset_id = "test_dataset"
        
        mock_table = MagicMock()
        mock_table.table_id = "test_table"
        mock_table.schema = [
            MagicMock(name="product_id", field_type="STRING"),
            MagicMock(name="category", field_type="STRING"),
            MagicMock(name="revenue", field_type="FLOAT")
        ]
        
        # Configure mock client methods
        mock_client.return_value.list_datasets.return_value = [mock_dataset]
        mock_client.return_value.list_tables.return_value = [mock_table]
        mock_client.return_value.dataset.return_value.table.return_value = mock_table
        mock_client.return_value.get_table.return_value = mock_table
        
        # Create connector
        connector = BigQueryConnector(project_id="test-project")
        
        # Test getting schema
        schema = connector.get_table_schema("test_dataset", "test_table")
        expected_schema = "CREATE TABLE test_table (product_id STRING, category STRING, revenue FLOAT)"
        self.assertEqual(schema, expected_schema, "Schema doesn't match expected format")
        
        # Test query execution with mock results
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [
            {"category": "Electronics", "total_revenue": 10000.0},
            {"category": "Furniture", "total_revenue": 5000.0}
        ]
        mock_query_job.total_bytes_processed = 1024
        mock_query_job.cache_hit = False
        mock_query_job.started = 0
        mock_query_job.ended = 100
        
        mock_client.return_value.query.return_value = mock_query_job
        
        sql_query = "SELECT category, SUM(revenue) AS total_revenue FROM test_table GROUP BY category"
        result = connector.execute_query(sql_query)
        
        self.assertTrue(result["success"], "Query execution should succeed")
        self.assertEqual(len(result["rows"]), 2, "Should return 2 rows")
        self.assertEqual(result["rows"][0]["category"], "Electronics", "First row category should be Electronics")
    
    @patch('crewai.Agent')
    @patch('crewai.Task')
    @patch('crewai.Crew')
    def test_crewai_integration(self, mock_crew, mock_task, mock_agent):
        """Test CrewAI integration."""
        # Skip if no GPU available for model loading
        if self.device != "cuda":
            self.skipTest("Skipping CrewAI test as it requires GPU")
        
        # Set up mocks
        mock_schema_reasoner = MagicMock()
        mock_sql_optimizer = MagicMock()
        mock_ambiguity_resolver = MagicMock()
        
        mock_agent.side_effect = [mock_schema_reasoner, mock_sql_optimizer, mock_ambiguity_resolver]
        
        mock_schema_task = MagicMock()
        mock_optimize_task = MagicMock()
        mock_ambiguity_task = MagicMock()
        
        mock_task.side_effect = [mock_schema_task, mock_optimize_task, mock_ambiguity_task]
        
        mock_crew_instance = MagicMock()
        mock_crew.return_value = mock_crew_instance
        
        mock_crew_result = {
            "schema_task": {"analysis": "Table sales has relevant fields for this query"},
            "optimize_task": {"optimized_query": "SELECT category, SUM(revenue) AS total_revenue FROM sales GROUP BY category ORDER BY total_revenue DESC"},
            "ambiguity_task": {"interpretation": "The query is clear and specific"}
        }
        mock_crew_instance.kickoff.return_value = mock_crew_result
        
        # Create a simplified agent for testing
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                with patch('transformers.pipeline') as mock_pipeline:
                    # Set up pipeline mock
                    mock_pipeline.return_value = MagicMock()
                    
                    # Create the query agent
                    query_agent = SQLQueryAgent(model_path=self.model_path, device=self.device)
                    
                    # Test processing a query
                    question = "What is the total revenue by category?"
                    schema_context = "CREATE TABLE sales (product_id STRING, category STRING, revenue FLOAT)"
                    generated_sql = "SELECT category, SUM(revenue) FROM sales GROUP BY category"
                    
                    result = query_agent.process_query(question, schema_context, generated_sql)
                    
                    # Check if the result contains the expected keys
                    self.assertIn("optimized_sql", result, "Result should contain optimized_sql key")
                    self.assertIn("schema_analysis", result, "Result should contain schema_analysis key")
                    self.assertIn("ambiguity_check", result, "Result should contain ambiguity_check key")
                    
                    # Verify CrewAI was used properly
                    mock_crew.assert_called_once()
                    mock_crew_instance.kickoff.assert_called_once()

if __name__ == '__main__':
    unittest.main()