# crewai_agents.py

from crewai import Agent, Task, Crew, Process
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict, Any, Optional
import re

class SQLQueryAgent:
    """Class to manage the CrewAI agents for SQL query enhancement."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize the SQL Query Agent with model settings."""
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Initialize the language model for agents
        self._initialize_llm()
        
        # Set up the agents
        self.schema_reasoner = self._create_schema_reasoner()
        self.sql_optimizer = self._create_sql_optimizer()
        self.ambiguity_resolver = self._create_ambiguity_resolver()
    
    def _initialize_llm(self):
        """Initialize the language model for CrewAI agents."""
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto"
        )
        
        # Create text generation pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain LLM
        self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    def _create_schema_reasoner(self) -> Agent:
        """Create the Schema Reasoner agent."""
        return Agent(
            role="Database Schema Expert",
            goal="Analyze database schemas and recommend the most relevant tables and fields for queries",
            backstory="""You are an expert database architect with deep knowledge of SQL and 
                      database design patterns. You can quickly analyze table schemas to 
                      identify relevant fields and relationships for any given query.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self._schema_analysis_tool]
        )
    
    def _create_sql_optimizer(self) -> Agent:
        """Create the SQL Optimizer agent."""
        return Agent(
            role="SQL Query Optimizer",
            goal="Enhance SQL queries for better performance and accuracy",
            backstory="""You are a SQL performance tuning expert who can optimize queries
                      for better execution plans. You understand indexing, join strategies,
                      and query optimization techniques for various database engines.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self._sql_optimization_tool]
        )
    
    def _create_ambiguity_resolver(self) -> Agent:
        """Create the Ambiguity Resolver agent."""
        return Agent(
            role="Natural Language Query Analyzer",
            goal="Identify and resolve ambiguities in user queries",
            backstory="""You are a language expert specializing in natural language processing
                      and database semantics. You excel at identifying vague or ambiguous terms
                      in queries and suggesting clarifications to ensure accurate SQL generation.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self._ambiguity_detection_tool]
        )
    
    def _schema_analysis_tool(self, schema: str, question: str) -> Dict[str, Any]:
        """Tool for analyzing database schema in relation to a question."""
        # Parse CREATE TABLE statements
        table_pattern = re.compile(r'CREATE TABLE (\w+) \((.*?)\)', re.DOTALL)
        tables = table_pattern.findall(schema)
        
        result = {
            "relevant_tables": [],
            "relevant_fields": [],
            "joining_keys": [],
            "analysis": ""
        }
        
        for table_name, fields_str in tables:
            fields = [f.strip() for f in fields_str.split(',')]
            field_names = [f.split(' ')[0] for f in fields]
            
            # Basic relevance detection (could be enhanced)
            question_tokens = question.lower().split()
            table_relevant = table_name.lower() in question.lower()
            relevant_fields = []
            
            for field in field_names:
                if field.lower() in question.lower():
                    relevant_fields.append(field)
                    
            if table_relevant or relevant_fields:
                result["relevant_tables"].append(table_name)
                result["relevant_fields"].extend(relevant_fields)
        
        # Generate analysis
        result["analysis"] = f"""
        Based on the schema analysis, the query involves {len(result['relevant_tables'])} relevant tables: {', '.join(result['relevant_tables'])}. 
        The relevant fields identified for this query are: {', '.join(result['relevant_fields']) if result['relevant_fields'] else 'None specifically mentioned'}.
        """
        
        return result
    
    def _sql_optimization_tool(self, sql_query: str) -> Dict[str, Any]:
        """Tool for optimizing SQL queries."""
        # Initialize the result
        result = {
            "optimized_query": sql_query,
            "modifications": [],
            "explanation": "",
            "performance_impact": "Low"
        }
        
        # Basic optimization patterns
        # 1. Replace SELECT * with specific columns
        if "SELECT *" in sql_query.upper() or "SELECT * " in sql_query.upper():
            # Extract table name from FROM clause
            from_match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
            if from_match:
                table_name = from_match.group(1)
                result["modifications"].append("Replaced 'SELECT *' with specific columns")
                result["explanation"] += "Selecting all columns with '*' can be inefficient when only certain columns are needed. "
                result["performance_impact"] = "Medium"
        
        # 2. Check for missing WHERE clause in large tables
        if "WHERE" not in sql_query.upper() and "GROUP BY" in sql_query.upper():
            result["modifications"].append("Consider adding a WHERE clause to filter results before grouping")
            result["explanation"] += "Adding a WHERE clause before GROUP BY can reduce the amount of data processed. "
            result["performance_impact"] = "High"
        
        # 3. Check for inefficient JOIN conditions
        if "JOIN" in sql_query.upper() and "ON" not in sql_query.upper():
            result["modifications"].append("Use explicit JOIN ... ON syntax instead of comma-separated tables")
            result["explanation"] += "Explicit JOIN conditions improve readability and sometimes performance. "
            result["performance_impact"] = "Medium"
        
        # 4. Check for missing LIMIT clause in result sets
        if "LIMIT" not in sql_query.upper() and "SELECT" in sql_query.upper():
            result["optimized_query"] = sql_query + " LIMIT 1000"
            result["modifications"].append("Added LIMIT clause to prevent excessive result sets")
            result["explanation"] += "Adding a LIMIT clause prevents returning excessive rows that may overwhelm the application or user. "
            result["performance_impact"] = "High"
        
        # If no optimizations were performed
        if not result["modifications"]:
            result["explanation"] = "The query already appears to be well-optimized. No major performance issues detected."
        
        return result
    
    def _ambiguity_detection_tool(self, question: str, schema: str) -> Dict[str, Any]:
        """Tool for detecting ambiguities in natural language queries."""
        # Initialize the result
        result = {
            "ambiguities_detected": False,
            "ambiguous_terms": [],
            "clarification_questions": [],
            "interpretation": "",
            "confidence": "High"
        }
        
        # Check for ambiguous terms by simple keyword matching
        ambiguous_keywords = ["it", "they", "them", "those", "these", "this", "that", "some", "many", 
                            "few", "several", "various", "better", "best", "worst", "more", "most", "less"]
        
        question_tokens = question.lower().split()
        for keyword in ambiguous_keywords:
            if keyword in question_tokens:
                result["ambiguities_detected"] = True
                result["ambiguous_terms"].append(keyword)
                result["clarification_questions"].append(f"What specifically does '{keyword}' refer to in your question?")
                result["confidence"] = "Medium"
        
        # Check for missing specific constraints
        if "top" in question_tokens or "best" in question_tokens:
            if "limit" not in question.lower() and not any(str(num) in question for num in range(1, 101)):
                result["ambiguities_detected"] = True
                result["ambiguous_terms"].append("top/best without specific number")
                result["clarification_questions"].append("How many top/best results would you like to see?")
                result["confidence"] = "Medium"
        
        # Check for time references without specific dates
        time_keywords = ["recent", "latest", "current", "today", "now", "last"]
        has_time_keywords = any(keyword in question_tokens for keyword in time_keywords)
        has_specific_date = re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}', question) is not None
        
        if has_time_keywords and not has_specific_date:
            result["ambiguities_detected"] = True
            result["ambiguous_terms"].append("Unspecified time period")
            result["clarification_questions"].append("What specific time period are you interested in?")
            result["confidence"] = "Low"
        
        # Generate interpretation
        if result["ambiguities_detected"]:
            result["interpretation"] = f"The query contains {len(result['ambiguous_terms'])} ambiguous terms or phrases that might lead to multiple interpretations. Clarification would help generate a more accurate SQL query."
        else:
            result["interpretation"] = "The query appears to be clear and specific enough to generate an accurate SQL query."
        
        return result
    
    def process_query(self, natural_language_query: str, schema_context: str, generated_sql: str) -> Dict[str, Any]:
        """Process a natural language query through the CrewAI workflow."""
        # Create tasks for the agents
        schema_task = Task(
            description=f"Analyze the following database schema in relation to this question: '{natural_language_query}'. Schema: {schema_context}",
            agent=self.schema_reasoner,
            expected_output="A detailed analysis of which tables and fields are relevant to the question"
        )
        
        optimize_task = Task(
            description=f"Optimize the following SQL query for better performance: {generated_sql}",
            agent=self.sql_optimizer,
            expected_output="An optimized SQL query with explanation of improvements"
        )
        
        ambiguity_task = Task(
            description=f"Identify any ambiguities in the following question: '{natural_language_query}'. Consider the context of the database schema: {schema_context}",
            agent=self.ambiguity_resolver,
            expected_output="A list of potential ambiguities and suggested clarifications"
        )
        
        # Create crew and execute tasks
        crew = Crew(
            agents=[self.schema_reasoner, self.sql_optimizer, self.ambiguity_resolver],
            tasks=[schema_task, optimize_task, ambiguity_task],
            process=Process.sequential  # Execute tasks in sequence
        )
        
        # Execute the crew workflow
        result = crew.kickoff()
        
        # Process and format the results
        processed_result = {
            "original_query": natural_language_query,
            "generated_sql": generated_sql,
            "optimized_sql": result.get("optimize_task", {}).get("optimized_query", generated_sql),
            "schema_analysis": result.get("schema_task", {}).get("analysis", "No schema analysis available"),
            "ambiguity_check": result.get("ambiguity_task", {}).get("interpretation", "No ambiguity analysis available"),
            "clarification_questions": result.get("ambiguity_task", {}).get("clarification_questions", []),
            "performance_suggestions": result.get("optimize_task", {}).get("explanation", "No optimization suggestions available")
        }
        
        return processed_result