# DATA-266-Text2SQL

# Natural Language to SQL (NL2SQL) System

This repository contains the code and documentation for our Natural Language to SQL (NL2SQL) system, which enables non-technical users to query relational databases using natural language.

## Project Overview

Our NL2SQL system democratizes data access by eliminating the technical barrier imposed by SQL. It combines a fine-tuned Mistral-7B language model (adapted using Low-Rank Adaptation) with a multi-agent framework to translate natural language questions into accurate and efficient SQL queries.

## Key Features

- **Natural Language Interface**: Query databases using plain English
- **High Accuracy**: 80% exact match accuracy, 100% execution accuracy
- **Schema Awareness**: Automatically understands database structure
- **Multi-Agent Architecture**: Specialized agents for schema reasoning, query optimization, and ambiguity resolution
- **Production-Ready**: Complete with authentication, monitoring, and error handling
- **BigQuery Integration**: Execution engine for enterprise-grade query processing
- **Responsive UI**: User-friendly interface with results visualization

## Technical Architecture

The system consists of these main components:

1. **Fine-tuned Mistral-7B**: The core language model adapted for SQL generation using LoRA
2. **CrewAI Agent Framework**: Three specialized agents that enhance the core model:
   - Schema Reasoner: Bridges natural language and database schema
   - SQL Optimizer: Improves query efficiency and correctness
   - Ambiguity Resolver: Clarifies vague user inputs
3. **Flask Backend**: Handles API requests, authentication, and orchestration
4. **BigQuery Connector**: Executes queries and processes results
5. **Web Frontend**: Bootstrap-based responsive interface

## Hugging Face Enpoints for the model
https://endpoints.huggingface.co/arunajithesh/endpoints/mistral-7b-sql-nkl
https://ys1youl289g9bhaw.us-east4.gcp.endpoints.huggingface.cloud

## Model Drive link
https://drive.google.com/drive/folders/12-usin2XqjgMuC02YddkXk2X14qLvIXg?usp=sharing

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Google Cloud account with BigQuery access
- Docker and Docker Compose (for containerized deployment)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/nl2sql-system.git
   cd nl2sql-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the application:
   ```
   python app.py
   ```

### Docker Deployment

```
cd deployment/docker
docker-compose up -d
```

## Usage Examples

### Basic Query
```
"Show me all customers from California who purchased more than $1000 last month"
```

### Join Query
```
"What is the average salary of employees in the Marketing department?"
```

### Analytical Query
```
"What is the total revenue by product category for the first quarter of 2023?"
```

## Evaluation Results

Our system achieves:
- 80% Exact Match Accuracy (from 0% baseline)
- 85% Execution Accuracy (from 54%)
- 98% Token Overlap (from 24%)
- 92% Component Accuracy (from 7%)

## Research Team

- Aruna Jithesh
- Pallav Mahajan
- Shanmukha Raj Siricilla
- Tanu Datt
- Venkata Nagasai Gautam Kasarabada

## Citation

If you use this code or system in your research, please cite our work:

```
@misc{nl2sql2025,
  author = {Jithesh, Aruna and Mahajan, Pallav and Siricilla, Shanmukha Raj and Datt, Tanu and Kasarabada, Venkata Nagasai Gautam},
  title = {Natural Language to SQL System with Multi-Agent Architecture},
  year = {2025},
  publisher = {San Jose State University},
  journal = {Data-266 Sec 21: Generative Model}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dr. Simon Shim (San Jose State University)
- The Mistral AI team for the Mistral-7B base model
- The CrewAI framework developers
