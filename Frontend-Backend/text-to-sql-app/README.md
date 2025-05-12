# Natural Language to SQL Query System

This project is a full-stack application that translates natural language queries into SQL, executes them against a Google BigQuery database, and returns the results in a user-friendly format. The system leverages a fine-tuned Mistral-7B model for SQL generation and CrewAI for advanced query handling.

## Features

- **Natural Language Understanding**: Convert plain English questions into SQL queries
- **Schema-Aware Query Generation**: System understands your database schema for accurate query generation
- **Query Optimization**: Automatically enhance generated SQL for better performance
- **Ambiguity Resolution**: Identify and clarify ambiguities in user queries
- **Interactive UI**: User-friendly interface for querying your data
- **Real-time Execution**: Execute SQL directly against BigQuery and view results instantly

## System Architecture

The system consists of the following components:

1. **Web Frontend**: User interface for entering queries and viewing results
2. **Backend API**: Flask-based API that handles requests and coordinates components
3. **Fine-tuned Mistral-7B Model**: Specialized for SQL generation from natural language
4. **CrewAI Agents**: Advanced AI agents for query enhancement and optimization
5. **BigQuery Connector**: Interface to execute SQL queries against Google BigQuery

## Prerequisites

- Python 3.8+
- Google Cloud Platform account with BigQuery enabled
- Google Cloud service account credentials with BigQuery access
- CUDA-capable GPU (recommended for faster inference)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/nl-to-sql-system.git
   cd nl-to-sql-system
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root directory:
   ```
   # Application settings
   DEBUG=True
   PORT=8080
   HOST=0.0.0.0
   
   # Model settings
   MODEL_PATH=/content/drive/MyDrive/mistral7b_sql_model
   USE_CUDA=True
   USE_BF16=True
   
   # BigQuery settings
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
   PROJECT_ID=your-google-cloud-project-id
   DEFAULT_DATASET=your-default-dataset-id
   
   # CrewAI settings
   CREW_VERBOSE=True
   CREW_PROCESS=sequential
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:8080`

3. Enter your BigQuery dataset and table information

4. Type a natural language query, such as:
   - "What are the top 5 customers by total purchase amount?"
   - "Show me the average order value by month for 2024"
   - "Find all products with low inventory and high demand"

5. View the generated SQL, optimized SQL, and query results

## Using the API

The system provides a RESTful API that can be integrated with other applications:

### Process a Query

**Endpoint**: `POST /api/query`

**Request Body**:
```json
{
    "query": "What are the top 5 customers by total purchase amount?",
    "dataset_id": "sales_data",
    "table_id": "orders"
}
```

**Response**:
```json
{
    "original_query": "What are the top 5 customers by total purchase amount?",
    "generated_sql": "SELECT customer_id, SUM(amount) AS total_amount FROM orders GROUP BY customer_id ORDER BY total_amount DESC LIMIT 5",
    "optimized_sql": "SELECT customer_id, SUM(amount) AS total_amount FROM orders GROUP BY customer_id ORDER BY total_amount DESC LIMIT 5",
    "results": [
        {"customer_id": "C1001", "total_amount": 15000.00},
        {"customer_id": "C1002", "total_amount": 12500.50},
        ...
    ],
    "schema_analysis": "...",
    "ambiguity_check": "..."
}
```

### Get Table Schema

**Endpoint**: `GET /api/schema`

**Query Parameters**:
- `dataset_id`: BigQuery dataset ID
- `table_id`: BigQuery table ID

**Response**:
```json
{
    "schema": "CREATE TABLE orders (order_id STRING, customer_id STRING, amount FLOAT, order_date TIMESTAMP)"
}
```

## Fine-tuned Model Details

This system uses a Mistral-7B model fine-tuned specifically for SQL generation using the LoRA (Low-Rank Adaptation) approach. The model was trained on the [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset.

Performance metrics achieved:
- Exact Match Accuracy: 0.78
- SQL Component Accuracies:
  - Table: 0.92
  - Select: 0.86
  - Where: 0.81

## CrewAI Implementation

The system uses CrewAI to implement specialized agents:

1. **Schema Reasoner**: Analyzes database schemas to identify relevant tables and fields
2. **SQL Optimizer**: Improves generated SQL queries for better performance
3. **Ambiguity Resolver**: Identifies and clarifies ambiguous natural language queries

These agents work together to enhance the quality and reliability of SQL query generation.

## Project Structure

```
nl-to-sql-system/
├── app.py                    # Main application file
├── config.py                 # Configuration settings
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── bigquery_connector.py     # BigQuery interface
├── crewai_agents.py          # CrewAI agent implementation
├── templates/
│   └── index.html            # Web UI template
├── static/
│   ├── css/                  # Stylesheets
│   └── js/                   # JavaScript files
└── README.md                 # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses the Mistral-7B model by Mistral AI
- Fine-tuning methodology based on Hugging Face's transformers library
- CrewAI for agent-based system enhancement