from flask import Flask, request, jsonify, render_template
from google.cloud import bigquery
import os
import re
import json
import requests
from dotenv import load_dotenv
import logging
import time

# Load environment variables
load_dotenv()

# Set Google Cloud credentials explicitly
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if credentials_path:
    # Make sure the path is absolute
    if not os.path.isabs(credentials_path):
        credentials_path = os.path.abspath(credentials_path)
    
    # Check if file exists
    if not os.path.isfile(credentials_path):
        print(f"WARNING: Credentials file not found at: {credentials_path}")
    else:
        print(f"Using credentials from: {credentials_path}")
    
    # Set environment variable
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
else:
    print("WARNING: GOOGLE_APPLICATION_CREDENTIALS not set in .env file")

app = Flask(__name__)

# Configuration
MODEL_API_URL = os.getenv('MODEL_API_URL', 'https://your-ngrok-url.ngrok-free.app/generate')
PROJECT_ID = os.getenv('PROJECT_ID')
DEFAULT_DATASET = os.getenv('DEFAULT_DATASET')

print(f"PROJECT_ID: {PROJECT_ID}")
print(f"DEFAULT_DATASET: {DEFAULT_DATASET}")
print(f"MODEL_API_URL: {MODEL_API_URL}")

# Remote Model Agent class for query enhancement
class RemoteAPIQueryAgent:
    """Agent that processes queries using a remote model API instead of a local model."""
    
    def __init__(self, api_url):
        """Initialize the agent with the remote API URL."""
        self.api_url = api_url
        print(f"Initialized RemoteAPIQueryAgent with API URL: {api_url}")
    
    def _schema_analysis(self, schema_context, question):
        """Analyze schema context in relation to the question."""
        try:
            # Format prompt for schema analysis
            prompt = f"""You are a Database Schema Expert. Analyze this schema in relation to the question.

    Schema:
    {schema_context}

    Question:
    {question}

    Provide a detailed analysis of which tables and fields are most relevant to this question:"""
            
            # Format payload for Hugging Face Inference Endpoints
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.1,
                    "do_sample": True
                }
            }
            
            # Add headers for Hugging Face token authentication
            headers = {
                "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
                "Content-Type": "application/json"
            }
            
            # Call Hugging Face Inference Endpoint
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return {"analysis": "Unable to perform schema analysis due to API error."}
            
            # Process the response - extract generated text
            result = response.json()
            
            # Extract the generated text
            analysis = ""
            if isinstance(result, list) and len(result) > 0:
                analysis = result[0].get('generated_text', '')
            else:
                analysis = result.get('generated_text', '')
                
            # Remove the prompt from the response if included
            if prompt in analysis:
                analysis = analysis[len(prompt):].strip()
            
            return {"analysis": analysis}
        except Exception as e:
            print(f"Error in schema analysis: {str(e)}")
            return {"analysis": f"Error during analysis: {str(e)}"}
    
    def _optimize_sql(self, sql_query):
        """Optimize the SQL query for better performance."""
        try:
            # Format prompt for SQL optimization
            prompt = f"""You are a SQL Query Optimizer. Enhance this SQL query for better performance and accuracy.

    Original SQL:
    {sql_query}

    Provide an optimized version of the query along with explanations of your improvements:"""
            
            # Format payload for Hugging Face Inference Endpoints
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }
            
            # Add headers for Hugging Face token authentication
            headers = {
                "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
                "Content-Type": "application/json"
            }
            
            # Call Hugging Face Inference Endpoint
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return {
                    "optimized_query": sql_query,
                    "explanation": "Unable to optimize query due to API error."
                }
            
            # Process the response - extract generated text
            result = response.json()
            
            # Extract the generated text
            optimization_text = ""
            if isinstance(result, list) and len(result) > 0:
                optimization_text = result[0].get('generated_text', '')
            else:
                optimization_text = result.get('generated_text', '')
                
            # Remove the prompt from the response if included
            if prompt in optimization_text:
                optimization_text = optimization_text[len(prompt):].strip()
            
            # Extract the optimized query and explanation
            lines = optimization_text.strip().split('\n')
            optimized_query = sql_query  # Default to original
            explanation = "No specific optimizations identified."
            
            # Simple parsing of the response
            query_started = False
            query_lines = []
            explanation_lines = []
            
            for line in lines:
                if line.strip().startswith("SELECT") or line.strip().startswith("WITH") or query_started:
                    query_started = True
                    if line.strip() and not line.lower().startswith("explanation"):
                        query_lines.append(line)
                elif query_lines and line.strip():  # After query, collect explanation
                    explanation_lines.append(line)
            
            if query_lines:
                optimized_query = "\n".join(query_lines).strip()
            if explanation_lines:
                explanation = "\n".join(explanation_lines).strip()
            
            return {
                "optimized_query": optimized_query,
                "explanation": explanation
            }
        except Exception as e:
            print(f"Error in SQL optimization: {str(e)}")
            return {
                "optimized_query": sql_query,
                "explanation": f"Error during optimization: {str(e)}"
            }
    
    def _check_ambiguities(self, question, schema_context):
        """Identify and analyze ambiguities in the user query."""
        try:
            # Format prompt for ambiguity detection
            prompt = f"""You are a Natural Language Query Analyzer. Identify ambiguities in this question.

    Question:
    {question}

    Database Schema:
    {schema_context}

    Identify any vague or ambiguous terms in the question that might lead to incorrect SQL generation.
    List specific clarification questions that would help resolve these ambiguities:"""
            
            # Format payload for Hugging Face Inference Endpoints
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 250,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }
            
            # Add headers for Hugging Face token authentication
            headers = {
                "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
                "Content-Type": "application/json"
            }
            
            # Call Hugging Face Inference Endpoint
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return {
                    "interpretation": "Unable to analyze ambiguities due to API error.",
                    "clarification_questions": []
                }
            
            # Process the response - extract generated text
            result = response.json()
            
            # Extract the generated text
            ambiguity_text = ""
            if isinstance(result, list) and len(result) > 0:
                ambiguity_text = result[0].get('generated_text', '')
            else:
                ambiguity_text = result.get('generated_text', '')
                
            # Remove the prompt from the response if included
            if prompt in ambiguity_text:
                ambiguity_text = ambiguity_text[len(prompt):].strip()
            
            # Parse the response
            lines = ambiguity_text.strip().split('\n')
            interpretation = "No significant ambiguities detected."
            questions = []
            
            in_questions = False
            for line in lines:
                if "clarification questions" in line.lower() or "questions:" in line.lower():
                    in_questions = True
                    continue
                
                if in_questions and line.strip():
                    # Check if line starts with a number or bullet point
                    if line.strip()[0].isdigit() or line.strip()[0] in ['-', '*', '•']:
                        questions.append(line.strip().lstrip('1234567890.*-• \t'))
                    else:
                        questions.append(line.strip())
                elif not in_questions and line.strip():
                    interpretation = line.strip()
            
            return {
                "interpretation": interpretation,
                "clarification_questions": questions
            }
        except Exception as e:
            print(f"Error in ambiguity analysis: {str(e)}")
            return {
                "interpretation": f"Error during ambiguity analysis: {str(e)}",
                "clarification_questions": []
            }
    
    def process_query(self, natural_language_query: str, schema_context: str, generated_sql: str) -> dict:
        """Process a query through all agents and return combined analysis."""
        start_time = time.time()
        print(f"Starting agent processing for query: {natural_language_query}")
        
        # Run all analyses in sequence
        schema_result = self._schema_analysis(schema_context, natural_language_query)
        print(f"Schema analysis completed in {time.time() - start_time:.2f}s")
        
        optimization_result = self._optimize_sql(generated_sql)
        print(f"SQL optimization completed in {time.time() - start_time:.2f}s")
        
        ambiguity_result = self._check_ambiguities(natural_language_query, schema_context)
        print(f"Ambiguity analysis completed in {time.time() - start_time:.2f}s")
        
        # Combine results
        return {
            "original_query": natural_language_query,
            "generated_sql": generated_sql,
            "optimized_sql": optimization_result.get("optimized_query", generated_sql),
            "schema_analysis": schema_result.get("analysis", "No schema analysis available"),
            "ambiguity_check": ambiguity_result.get("interpretation", "No ambiguity analysis available"),
            "clarification_questions": ambiguity_result.get("clarification_questions", []),
            "performance_suggestions": optimization_result.get("explanation", "No optimization suggestions available")
        }

# Global variable for the agent
query_agent = None

# Initialize agent function
def initialize_agent(api_url):
    """Initialize the RemoteAPIQueryAgent."""
    try:
        print("Initializing RemoteAPIQueryAgent...")
        start_time = time.time()
        agent = RemoteAPIQueryAgent(api_url=api_url)
        elapsed_time = time.time() - start_time
        print(f"RemoteAPIQueryAgent initialized in {elapsed_time:.2f} seconds")
        return agent
    except Exception as e:
        print(f"Error initializing RemoteAPIQueryAgent: {str(e)}")
        return None

# Prompt template for BigQuery SQL generation
BIGQUERY_SQL_PROMPT_TEMPLATE = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question. Follow these rules for Google BigQuery SQL dialect:

1. Use single quotes for string literals, not double quotes (e.g., WHERE name = 'value')
2. Make sure to select existing column names only (check the schema carefully)
3. For table aliases, use AS keyword (e.g., FROM table AS t)
4. For joins, use the ON keyword (e.g., JOIN other_table ON table.id = other_table.id)
5. For self-joins, use different aliases (e.g., FROM categories AS parent JOIN categories AS child)
6. Return only the SQL statement with no explanations
7. Do not use backticks or square brackets for identifiers
8. Only use tables that exist in the schema below

### Question:
```{question}```

### Database Schema:
```
{context}
```

### Important Database Information:
- This is a hierarchical product categories database
- The categories table contains both parent categories and subcategories
- The parent_category_id column references the category_id of another row in the same table
- When parent_category_id is NULL, it means it's a top-level category
- There is NO separate products table - all queries must use only the categories table
- Subcategories can be found by joining the categories table to itself 
- To find subcategories of a category, look for rows where parent_category_id equals the category_id of the parent

### Example Queries:
1. To find all top-level categories:
```
SELECT * FROM categories WHERE parent_category_id IS NULL
```

2. To find all subcategories of Electronics:
```
SELECT child.* 
FROM categories AS child
JOIN categories AS parent ON child.parent_category_id = parent.category_id
WHERE parent.category_name = 'Electronics'
```

### BigQuery SQL:
"""

# Initialize BigQuery client
try:
    client = bigquery.Client(project=PROJECT_ID)
    print("Successfully initialized BigQuery client")
except Exception as e:
    print(f"Error initializing BigQuery client: {str(e)}")

# Function to get all tables and their schemas in the dataset
def get_dataset_schema(dataset_id):
    """Get schemas for all tables in the dataset."""
    try:
        dataset_ref = client.dataset(dataset_id)
        tables = list(client.list_tables(dataset_ref))
        
        schema_info = []
        for table in tables:
            table_ref = client.dataset(dataset_id).table(table.table_id)
            table_obj = client.get_table(table_ref)
            
            # Create CREATE TABLE statement
            field_definitions = []
            for field in table_obj.schema:
                field_type = field.field_type
                field_definitions.append(f"{field.name} {field_type}")
            
            create_table = f"CREATE TABLE {table.table_id} ({', '.join(field_definitions)})"
            schema_info.append(create_table)
        
        return "\n".join(schema_info)
    except Exception as e:
        print(f"Error getting dataset schema: {str(e)}")
        return ""

# Function to check if model API is accessible
def check_model_api():
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
            "Content-Type": "application/json"
        }
        
        # Simple test request for Hugging Face Inference Endpoints
        test_payload = {
            "inputs": "Test",
            "parameters": {
                "max_new_tokens": 5,
                "do_sample": False
            }
        }
        
        response = requests.post(MODEL_API_URL, json=test_payload, headers=headers, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Error connecting to Hugging Face Inference Endpoint: {str(e)}")
        return False

# Function to get BigQuery schema information
def get_table_schema(dataset_id, table_id):
    """Get schema information for a BigQuery table in CREATE TABLE format."""
    try:
        table_ref = client.dataset(dataset_id).table(table_id)
        table = client.get_table(table_ref)
        
        # Create the CREATE TABLE statement
        field_definitions = []
        for field in table.schema:
            field_type = field.field_type
            field_definitions.append(f"{field.name} {field_type}")
        
        create_table = f"CREATE TABLE {table_id} ({', '.join(field_definitions)})"
        return create_table
    except Exception as e:
        print(f"Error getting schema: {str(e)}")
        return f"Error: {str(e)}"

# Function to validate and fix query logic
def validate_and_fix_query(sql, question, context):
    """
    Validates that the generated SQL only uses tables and columns from the schema
    and fixes common logical issues.
    """
    # Extract available tables and columns from the context
    tables = {}
    for statement in context.split(';'):
        table_match = re.search(r'CREATE TABLE (\w+) \((.*?)\)', statement, re.DOTALL)
        if table_match:
            table_name = table_match.group(1)
            columns_str = table_match.group(2)
            columns = []
            for column_def in columns_str.split(','):
                if column_def.strip():
                    column_name = column_def.strip().split(' ')[0]
                    columns.append(column_name)
            tables[table_name] = columns
    
    # Check if the query uses non-existent tables
    all_tables = list(tables.keys())
    sql_lower = sql.lower()
    
    # Check if query references tables not in the schema
    for table_ref in re.finditer(r'from\s+(\w+)|join\s+(\w+)', sql_lower):
        table_name = table_ref.group(1) or table_ref.group(2)
        if table_name and table_name not in [t.lower() for t in all_tables]:
            # Table doesn't exist - most likely this is a query assuming a products table
            # Determine if this is a products/categories type query
            is_product_query = "product" in table_name or any(word in question.lower() for word in ["product", "item", "goods"])
            
            if is_product_query and "categories" in all_tables:
                # Attempt to rewrite as a query just against categories
                category_name_match = re.search(r"category_name\s*=\s*'([^']+)'", sql)
                if category_name_match:
                    category_name = category_name_match.group(1)
                    # Create a better query using only the categories table
                    return f"""
                    SELECT * FROM categories 
                    WHERE category_name = '{category_name}' 
                    OR parent_category_id = (
                        SELECT category_id FROM categories 
                        WHERE category_name = '{category_name}'
                    )
                    """
    
    # Check if the query uses AND description condition unnecessarily
    is_subcategory_query = any(term in question.lower() for term in [
        "subcategories", "sub-categories", "belong to", "under", "child"
    ])
    
    has_and_description = "and description =" in sql_lower
    
    # For subcategory queries with restrictive AND conditions on description
    if is_subcategory_query and has_and_description:
        # Extract the condition on the category name
        name_match = re.search(r"category_name\s*=\s*'([^']+)'", sql)
        
        if name_match:
            category_name = name_match.group(1)
            
            # If it's a subquery
            if "where parent_category_id = (select" in sql_lower:
                # Replace with a version that only filters on name
                sql = f"""
                SELECT * FROM categories 
                WHERE parent_category_id = (
                    SELECT category_id FROM categories 
                    WHERE category_name = '{category_name}'
                )
                """
            # If it's a join
            elif "join" in sql_lower:
                # Replace with a version that only filters on name for the parent
                sql = f"""
                SELECT c1.* FROM categories AS c1
                JOIN categories AS c2 ON c1.parent_category_id = c2.category_id
                WHERE c2.category_name = '{category_name}'
                """
    
    return sql

# Function to fix common BigQuery SQL syntax issues
def fix_bigquery_sql_issues(sql, context):
    """Fix common BigQuery SQL syntax issues."""
    # Replace double quotes with single quotes for string literals
    # This regex tries to match double-quoted strings but not column identifiers
    
    # Replace double quotes with single quotes for string literals
    # This is a simplified approach, a more robust parser would be better
    pattern = r'(\s*=\s*)"([^"]*)"'
    replacement = r"\1'\2'"
    sql = re.sub(pattern, replacement, sql)
    
    # Also handle cases where the double quotes are at the end of a line or statement
    pattern = r'"([^"]*)"(\s*;?\s*$)'
    replacement = r"'\1'\2"
    sql = re.sub(pattern, replacement, sql)
    
    # Replace quotes in WHERE clauses
    pattern = r'(WHERE\s+\w+\s*=\s*)"([^"]*)"'
    replacement = r"\1'\2'"
    sql = re.sub(pattern, replacement, sql)
    
    # Fix subcategory_id references if they exist (check the schema)
    if 'subcategory_id' in sql and 'subcategory_id' not in context:
        sql = sql.replace('subcategory_id', 'category_id')
    
    # Remove any backticks that might have been added
    sql = sql.replace('`', '')
    
    return sql

# Function to generate SQL with remote model
def generate_sql_with_remote_model(question, context, dataset_id=None):
    """Generate SQL from natural language using the Hugging Face Inference Endpoint with complete schema context."""
    try:
        # Get the full dataset schema if available
        full_schema = context
        if dataset_id:
            dataset_schema = get_dataset_schema(dataset_id)
            if dataset_schema:
                full_schema = f"{context}\n\n# Other tables in the dataset:\n{dataset_schema}"
        
        # Format the prompt using the template
        prompt = BIGQUERY_SQL_PROMPT_TEMPLATE.format(
            question=question,
            context=full_schema
        )
        
        # Prepare payload for Hugging Face Inference Endpoint
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.1,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        # Add headers for Hugging Face token authentication
        headers = {
            "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
            "Content-Type": "application/json"
        }
        
        # Send request to the Hugging Face Inference Endpoint
        response = requests.post(MODEL_API_URL, json=payload, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"Error from model API: Status {response.status_code}")
            print(response.text)
            return fallback_generate_sql(question, context)
        
        # Process response from Hugging Face
        result = response.json()
        
        # Extract generated text from the response
        # Hugging Face typically returns [{"generated_text": "..."}]
        generated_text = ""
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '')
        else:
            generated_text = result.get('generated_text', '')
            
        print(f"Raw model output: {generated_text}")
        
        # Extract SQL from the model output
        if prompt in generated_text:
            # Remove the prompt from the result
            sql = generated_text[len(prompt):].strip()
        else:
            sql = generated_text.strip()
        
        if "BigQuery SQL:" in sql:
            sql = sql.split("BigQuery SQL:")[1].strip()
        
        if "Response:" in sql:
            sql = sql.split("Response:")[1].strip()
        
        # Remove markdown code block formatting first
        if "```" in sql:
            # Extract content between backticks if present
            parts = sql.split("```")
            if len(parts) >= 3:  # Has opening and closing backticks
                # Get the content between first set of backticks
                code_content = parts[1].strip()
                # Remove any language identifier (like 'sql')
                if code_content.lower().startswith(('sql', 'mysql', 'bigquery')):
                    code_content = code_content.split("\n", 1)[1] if "\n" in code_content else ""
                sql = code_content
            else:
                # Handle case with only opening backticks
                sql = sql.replace("```", "").strip()
        
        # Extract only the valid SQL part - VERY ROBUST APPROACH
        # Method 1: Extract just the first statement that ends with semicolon
        if ';' in sql:
            # Find all text up to the first semicolon
            semi_pos = sql.find(';')
            sql = sql[:semi_pos+1].strip()
        
        # Method 2: If there's still explanatory text, remove it
        for marker in ["###", "Explanation", "Note:", "--"]:
            if marker in sql:
                sql = sql.split(marker, 1)[0].strip()
        
        # Method 3: Remove any line starting with explanatory text
        lines = sql.split('\n')
        sql_lines = []
        for line in lines:
            if line.strip().startswith(('The ', 'To ', 'This ', 'Note:')):
                break
            sql_lines.append(line)
        
        if sql_lines:
            sql = '\n'.join(sql_lines).strip()
        
        # Add semicolon if missing after all the cleanup
        if not sql.endswith(';'):
            sql = sql + ';'
        
        # Final validation - ensure it's valid SQL
        sql = sql.lstrip()  # Remove leading whitespace
        valid_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
        is_valid = any(sql.upper().startswith(keyword) for keyword in valid_keywords)
        
        if not is_valid:
            print(f"Warning: Generated SQL doesn't start with a valid SQL keyword: {sql}")
            return fallback_generate_sql(question, context)
        
        print(f"Final cleaned SQL: {sql}")
        return sql
    except Exception as e:
        print(f"Error generating SQL with Hugging Face endpoint: {str(e)}")
        return fallback_generate_sql(question, context)

# Fallback function for SQL generation using rules
def fallback_generate_sql(question, context):
    """Generate SQL from natural language query using rules (fallback)."""
    question_lower = question.lower()
    
    # Extract table name from context
    table_match = re.search(r'CREATE TABLE (\w+)', context)
    table_name = table_match.group(1) if table_match else "unknown_table"
    
    # Extract fields from context
    fields = []
    if "(" in context and ")" in context:
        fields_text = context.split("(")[1].split(")")[0]
        for field_def in fields_text.split(","):
            if field_def.strip():
                field_name = field_def.strip().split(" ")[0]
                fields.append(field_name)
    
    # Check for special queries related to categories and subcategories
    if ("subcategories" in question_lower or "sub-categories" in question_lower) and any(category in question_lower for category in ["electronics", "sports", "home", "kitchen", "audio", "computers"]):
        # Extract the category name
        category_patterns = ["electronics", "sports", "home & kitchen", "home and kitchen", "audio", "computers"]
        found_category = None
        
        for pattern in category_patterns:
            if pattern in question_lower:
                found_category = pattern
                break
        
        # Format the category name correctly
        if found_category == "home & kitchen" or found_category == "home and kitchen":
            found_category = "Home & Kitchen"
        else:
            found_category = found_category.title()
        
        return f"""
        SELECT child.* 
        FROM {table_name} AS child
        JOIN {table_name} AS parent ON child.parent_category_id = parent.category_id
        WHERE parent.category_name = '{found_category}'
        """
    
    # Check for queries about products belonging to a category
    if any(word in question_lower for word in ["product", "item", "belong", "in"]) and any(category in question_lower for category in ["electronics", "sports", "home", "kitchen", "audio", "computers"]):
        # Extract the category name
        category_patterns = ["electronics", "sports", "home & kitchen", "home and kitchen", "audio", "computers"]
        found_category = None
        
        for pattern in category_patterns:
            if pattern in question_lower:
                found_category = pattern
                break
        
        # Format the category name correctly
        if found_category == "home & kitchen" or found_category == "home and kitchen":
            found_category = "Home & Kitchen"
        else:
            found_category = found_category.title()
        
        return f"""
        SELECT * 
        FROM {table_name}
        WHERE category_name = '{found_category}' 
        OR parent_category_id = (
            SELECT category_id FROM {table_name} 
            WHERE category_name = '{found_category}'
        )
        """
    
    if "don't have a parent" in question_lower or "without parent" in question_lower or "no parent" in question_lower:
        return f"SELECT * FROM {table_name} WHERE parent_category_id IS NULL"
    
    # Basic patterns
    if any(word in question_lower for word in ["count", "how many"]):
        return f"SELECT COUNT(*) AS count FROM {table_name}"
    
    # Default to SELECT *
    return f"SELECT * FROM {table_name}"

# Main function to generate SQL with validation
def generate_sql(question, context, dataset_id=None):
    """Generate SQL from natural language, with complete schema and validation."""
    if check_model_api():
        # Generate SQL using the remote model with full schema context
        sql = generate_sql_with_remote_model(question, context, dataset_id)
    else:
        # Fallback to rule-based generation
        print("Remote model API not accessible, using rule-based generation")
        sql = fallback_generate_sql(question, context)
    
    # Fix basic syntax issues (single vs double quotes, etc)
    sql = fix_bigquery_sql_issues(sql, context)
    
    # Validate and fix schema and logic issues
    sql = validate_and_fix_query(sql, question, context)
    
    return sql

# Function to execute SQL query
def execute_query(sql):
    """Execute SQL query on BigQuery."""
    try:
        # Add project and dataset information if needed
        if PROJECT_ID and DEFAULT_DATASET:
            # First, identify all table references in the query
            table_pattern = r'(?:FROM|JOIN)\s+(\w+)(?:\s+AS\s+\w+)?'
            table_refs = re.findall(table_pattern, sql, re.IGNORECASE)
            
            # Replace each table reference with a fully qualified reference
            for table_name in table_refs:
                if table_name and '.' not in table_name and '`' not in table_name:
                    # Create the fully qualified name
                    qualified_name = f"`{PROJECT_ID}.{DEFAULT_DATASET}.{table_name}`"
                    
                    # Replace FROM table_name (with optional AS alias)
                    sql = re.sub(
                        f"FROM\\s+{table_name}(\\s+AS\\s+\\w+)?", 
                        f"FROM {qualified_name}\\1", 
                        sql, 
                        flags=re.IGNORECASE
                    )
                    
                    # Replace JOIN table_name (with optional AS alias)
                    sql = re.sub(
                        f"JOIN\\s+{table_name}(\\s+AS\\s+\\w+)?", 
                        f"JOIN {qualified_name}\\1", 
                        sql, 
                        flags=re.IGNORECASE
                    )
            
            print(f"Modified query with qualified table names: {sql}")
        
        print(f"Executing query: {sql}")
        
        # Execute the query
        query_job = client.query(sql)
        results = query_job.result()
        
        # Convert results to list of dictionaries
        rows = []
        for row in results:
            rows.append(dict(row.items()))
        
        print(f"Query returned {len(rows)} rows")
        
        return {
            "success": True,
            "rows": rows
        }
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_model', methods=['GET'])
def check_model_route():
    is_available = check_model_api()
    return jsonify({"available": is_available, "url": MODEL_API_URL})

@app.route('/api/agent-status', methods=['GET'])
def check_agent_status():
    """Check if the agent is initialized."""
    global query_agent
    return jsonify({
        "agent_initialized": query_agent is not None,
        "available_agents": ["Schema Analyzer", "SQL Optimizer", "Ambiguity Detector"] if query_agent else []
    })

@app.route('/api/initialize-agent', methods=['POST'])
def init_agent_endpoint():
    """Initialize the query agent."""
    global query_agent
    
    if query_agent:
        return jsonify({"message": "Agent already initialized"}), 200
    
    try:
        api_url = request.json.get('api_url', MODEL_API_URL)
        query_agent = initialize_agent(api_url)
        
        if query_agent:
            return jsonify({
                "success": True,
                "message": "Agent initialized successfully",
                "available_agents": ["Schema Analyzer", "SQL Optimizer", "Ambiguity Detector"]
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to initialize agent"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/schema', methods=['GET'])
def get_schema():
    dataset_id = request.args.get('dataset_id', DEFAULT_DATASET)
    table_id = request.args.get('table_id')
    
    if not table_id:
        return jsonify({"error": "Table ID is required"}), 400
    
    schema = get_table_schema(dataset_id, table_id)
    return jsonify({"schema": schema})

@app.route('/api/query', methods=['POST'])
def process_query():
    global query_agent
    
    data = request.json
    
    # Extract data from request
    natural_language_query = data.get('query', '')
    dataset_id = data.get('dataset_id', DEFAULT_DATASET)
    table_id = data.get('table_id', '')
    
    if not natural_language_query or not table_id:
        return jsonify({"error": "Query and table ID are required"}), 400
    
    try:
        # Get table schema
        schema_context = get_table_schema(dataset_id, table_id)
        
        # Generate SQL from natural language with complete schema context
        generated_sql = generate_sql(natural_language_query, schema_context, dataset_id)
        
        # Process with agent if available
        agent_analysis = {}
        use_agent = request.args.get('use_agent', 'true').lower() == 'true'
        
        if use_agent and query_agent:
            try:
                print("Processing query with agent...")
                start_time = time.time()
                agent_analysis = query_agent.process_query(
                    natural_language_query=natural_language_query,
                    schema_context=schema_context,
                    generated_sql=generated_sql
                )
                elapsed_time = time.time() -start_time
                elapsed_time = time.time() - start_time
                print(f"Agent processing completed in {elapsed_time:.2f} seconds")
                
                # Use the optimized SQL if available
                if agent_analysis.get("optimized_sql"):
                    generated_sql = agent_analysis["optimized_sql"]
                    print(f"Using agent-optimized SQL: {generated_sql}")
            except Exception as e:
                print(f"Error in agent processing: {str(e)}")
                # Continue with the original SQL if agent processing fails
        
        # Execute the SQL query
        query_result = execute_query(generated_sql)
        
        if query_result["success"]:
            # Prepare response
            response = {
                'original_query': natural_language_query,
                'generated_sql': generated_sql,
                'results': query_result["rows"]
            }
            
            # Include agent analysis if available
            if agent_analysis:
                response['agent_analysis'] = {
                    'schema_analysis': agent_analysis.get('schema_analysis', ''),
                    'ambiguity_check': agent_analysis.get('ambiguity_check', ''),
                    'performance_suggestions': agent_analysis.get('performance_suggestions', ''),
                    'clarification_questions': agent_analysis.get('clarification_questions', [])
                }
        else:
            response = {
                'original_query': natural_language_query,
                'generated_sql': generated_sql,
                'error': query_result["error"]
            }
            
            # Include agent analysis even in case of error
            if agent_analysis:
                response['agent_analysis'] = {
                    'schema_analysis': agent_analysis.get('schema_analysis', ''),
                    'ambiguity_check': agent_analysis.get('ambiguity_check', ''),
                    'performance_suggestions': agent_analysis.get('performance_suggestions', ''),
                    'clarification_questions': agent_analysis.get('clarification_questions', [])
                }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_query')
def test_query():
    try:
        query = f"SELECT * FROM `{PROJECT_ID}.{DEFAULT_DATASET}.categories` LIMIT 10"
        query_job = client.query(query)
        results = list(query_job.result())
        
        # Convert results to something JSON-serializable
        rows = []
        for row in results:
            rows.append(dict(row.items()))
        
        return jsonify({
            "success": True,
            "query": query,
            "rows": rows,
            "row_count": len(rows)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "query": query if 'query' in locals() else None
        })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')
    
    # Initialize agent if specified in environment variables
    if os.getenv('INITIALIZE_AGENT', 'false').lower() == 'true':
        query_agent = initialize_agent(MODEL_API_URL)
    
    print(f"Starting server on {host}:{port}, debug={debug}")
    print(f"RemoteAPIQueryAgent initialized: {query_agent is not None}")
    app.run(debug=debug, host=host, port=port)