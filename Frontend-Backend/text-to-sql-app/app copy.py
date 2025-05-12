from flask import Flask, request, jsonify, render_template
from google.cloud import bigquery
import os
import re
import json
import requests
from dotenv import load_dotenv

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
        # Try to connect to the model API
        health_url = MODEL_API_URL.replace('/generate', '/health')
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
        except:
            # If the health endpoint doesn't exist, try the main endpoint with a simple request
            test_payload = {
                "prompt": "Test",
                "max_length": 5
            }
            response = requests.post(MODEL_API_URL, json=test_payload, timeout=5)
            return response.status_code == 200
    except Exception as e:
        print(f"Error connecting to model API: {str(e)}")
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
    """Generate SQL from natural language using the remote model API with complete schema context."""
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
        
        # Prepare payload for the model API
        payload = {
            "prompt": prompt,
            "max_length": 200,
            "temperature": 0.1,
            "top_p": 0.95
        }
        
        # Send request to the model API
        response = requests.post(MODEL_API_URL, json=payload, timeout=10)
        
        if response.status_code != 200:
            print(f"Error from model API: Status {response.status_code}")
            print(response.text)
            return fallback_generate_sql(question, context)
        
        # Process response
        result = response.json()
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
        
        # Clean up SQL
        if sql.endswith(';'):
            sql = sql[:-1]  # Remove trailing semicolon
            
        return sql
    except Exception as e:
        print(f"Error generating SQL with remote model: {str(e)}")
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
        
        # Execute the SQL query
        query_result = execute_query(generated_sql)
        
        if query_result["success"]:
            # Prepare response
            response = {
                'original_query': natural_language_query,
                'generated_sql': generated_sql,
                'results': query_result["rows"]
            }
        else:
            response = {
                'original_query': natural_language_query,
                'generated_sql': generated_sql,
                'error': query_result["error"]
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
    
    print(f"Starting server on {host}:{port}, debug={debug}")
    app.run(debug=debug, host=host, port=port)