"""
Improved RLVR Training with Better SQL and Python Generation
Addresses poor training results with enhanced prompts, data, and verification
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random
import sqlite3
import tempfile
import subprocess
import re
from dataclasses import dataclass

from rlvr_core import VerificationResult, SQLVerifier, CodeVerifier, PatternVerifier


class ImprovedSQLGenerator:
    """Enhanced SQL prompt generation with better training data"""
    
    def __init__(self):
        self.schemas = self._create_comprehensive_schemas()
        self.question_templates = self._create_sql_templates()
    
    def _create_comprehensive_schemas(self):
        """Create diverse, realistic database schemas"""
        return [
            {
                "users": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "primary_key": True},
                        {"name": "name", "type": "TEXT"},
                        {"name": "email", "type": "TEXT"},
                        {"name": "age", "type": "INTEGER"},
                        {"name": "department", "type": "TEXT"},
                        {"name": "salary", "type": "REAL"},
                        {"name": "hire_date", "type": "TEXT"}
                    ],
                    "data": [
                        (1, "Alice Smith", "alice@company.com", 28, "Engineering", 85000.0, "2022-01-15"),
                        (2, "Bob Johnson", "bob@company.com", 34, "Marketing", 65000.0, "2021-03-22"),
                        (3, "Charlie Brown", "charlie@company.com", 29, "Engineering", 90000.0, "2022-06-01"),
                        (4, "Diana Wilson", "diana@company.com", 31, "Sales", 70000.0, "2021-11-10"),
                        (5, "Eve Davis", "eve@company.com", 26, "Engineering", 80000.0, "2023-02-14")
                    ]
                }
            },
            {
                "products": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "primary_key": True},
                        {"name": "name", "type": "TEXT"},
                        {"name": "price", "type": "REAL"},
                        {"name": "category", "type": "TEXT"},
                        {"name": "stock", "type": "INTEGER"},
                        {"name": "rating", "type": "REAL"}
                    ],
                    "data": [
                        (1, "Laptop Pro", 1299.99, "Electronics", 15, 4.5),
                        (2, "Wireless Mouse", 29.99, "Electronics", 50, 4.2),
                        (3, "Office Chair", 199.99, "Furniture", 8, 4.0),
                        (4, "Coffee Mug", 12.99, "Kitchen", 25, 4.8),
                        (5, "Desk Lamp", 45.99, "Furniture", 12, 4.3)
                    ]
                }
            },
            {
                "orders": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "primary_key": True},
                        {"name": "customer_id", "type": "INTEGER"},
                        {"name": "product_id", "type": "INTEGER"},
                        {"name": "quantity", "type": "INTEGER"},
                        {"name": "order_date", "type": "TEXT"},
                        {"name": "total_amount", "type": "REAL"}
                    ],
                    "data": [
                        (1, 1, 1, 1, "2023-12-01", 1299.99),
                        (2, 2, 2, 2, "2023-12-02", 59.98),
                        (3, 1, 4, 3, "2023-12-03", 38.97),
                        (4, 3, 3, 1, "2023-12-04", 199.99),
                        (5, 2, 5, 1, "2023-12-05", 45.99)
                    ]
                }
            }
        ]
    
    def _create_sql_templates(self):
        """Create comprehensive SQL question templates"""
        return {
            "basic_select": [
                "Select all records from {table}",
                "Show all {table} data",
                "Display everything in the {table} table",
                "Get all columns from {table}"
            ],
            "filtered_select": [
                "Find all {table} where {column} = '{value}'",
                "Select {table} with {column} greater than {numeric_value}",
                "Show {table} where {column} contains '{text_value}'",
                "Get {table} records where {column} is not null"
            ],
            "aggregation": [
                "Count the total number of {table}",
                "Find the average {column} in {table}",
                "Get the maximum {column} from {table}",
                "Calculate the sum of {column} in {table}",
                "Find the minimum {column} in {table}"
            ],
            "grouping": [
                "Count {table} by {column}",
                "Group {table} by {column} and show counts",
                "Find average {numeric_column} by {group_column}",
                "Show {table} grouped by {column} with totals"
            ],
            "ordering": [
                "Show {table} ordered by {column}",
                "List {table} sorted by {column} descending",
                "Display {table} arranged by {column} ascending"
            ],
            "complex": [
                "Find top 3 {table} with highest {column}",
                "Show {table} where {column1} > {value} and {column2} = '{text}'",
                "Get distinct {column} values from {table}"
            ]
        }
    
    def generate_sql_prompt(self) -> Tuple[str, Dict]:
        """Generate improved SQL prompt with context"""
        schema = random.choice(self.schemas)
        table_name = list(schema.keys())[0]
        table_info = schema[table_name]
        columns = [col["name"] for col in table_info["columns"]]
        
        # Select question type and template
        question_type = random.choice(list(self.question_templates.keys()))
        template = random.choice(self.question_templates[question_type])
        
        # Fill in template based on type
        if question_type == "basic_select":
            question = template.format(table=table_name)
            expected_query = f"SELECT * FROM {table_name};"
            
        elif question_type == "filtered_select":
            column = random.choice(columns)
            col_info = next(col for col in table_info["columns"] if col["name"] == column)
            
            if col_info["type"] == "INTEGER" or col_info["type"] == "REAL":
                numeric_value = random.randint(1, 100)
                question = template.format(
                    table=table_name, 
                    column=column, 
                    numeric_value=numeric_value,
                    value=numeric_value,
                    text_value="sample"
                )
                expected_query = f"SELECT * FROM {table_name} WHERE {column} > {numeric_value};"
            else:
                # Text column
                sample_data = table_info.get("data", [])
                if sample_data:
                    sample_value = str(sample_data[0][columns.index(column)])
                    question = template.format(
                        table=table_name,
                        column=column,
                        value=sample_value,
                        text_value=sample_value,
                        numeric_value=10
                    )
                    expected_query = f"SELECT * FROM {table_name} WHERE {column} = '{sample_value}';"
                else:
                    question = template.format(
                        table=table_name,
                        column=column,
                        value="sample",
                        text_value="sample", 
                        numeric_value=10
                    )
                    expected_query = f"SELECT * FROM {table_name} WHERE {column} = 'sample';"
        
        elif question_type == "aggregation":
            if "Count" in template:
                question = template.format(table=table_name, column="*")
                expected_query = f"SELECT COUNT(*) FROM {table_name};"
            else:
                numeric_cols = [col["name"] for col in table_info["columns"] 
                              if col["type"] in ["INTEGER", "REAL"]]
                if numeric_cols:
                    column = random.choice(numeric_cols)
                    if "average" in template.lower():
                        question = template.format(table=table_name, column=column)
                        expected_query = f"SELECT AVG({column}) FROM {table_name};"
                    elif "maximum" in template.lower():
                        question = template.format(table=table_name, column=column)
                        expected_query = f"SELECT MAX({column}) FROM {table_name};"
                    elif "minimum" in template.lower():
                        question = template.format(table=table_name, column=column)
                        expected_query = f"SELECT MIN({column}) FROM {table_name};"
                    else:  # sum
                        question = template.format(table=table_name, column=column)
                        expected_query = f"SELECT SUM({column}) FROM {table_name};"
                else:
                    question = f"Count the total number of {table_name}"
                    expected_query = f"SELECT COUNT(*) FROM {table_name};"
        
        else:
            # Fallback for other types
            question = f"Select all {table_name}"
            expected_query = f"SELECT * FROM {table_name};"
        
        # Create comprehensive prompt
        schema_description = self._format_schema_description(table_name, table_info)
        
        prompt = f"""You are an expert SQL developer. Write a SQL query for the following request.

Database Schema:
{schema_description}

Task: {question}

Write only the SQL query without any explanation or additional text:"""

        context = {
            "schema": schema,
            "question": question,
            "table": table_name,
            "expected_query": expected_query,
            "columns": columns,
            "question_type": question_type
        }
        
        return prompt, context
    
    def _format_schema_description(self, table_name: str, table_info: Dict) -> str:
        """Format schema for better LLM understanding"""
        lines = [f"Table: {table_name}"]
        lines.append("Columns:")
        
        for col in table_info["columns"]:
            col_desc = f"  - {col['name']} ({col['type']}"
            if col.get('primary_key'):
                col_desc += ", PRIMARY KEY"
            col_desc += ")"
            lines.append(col_desc)
        
        # Add sample data if available
        if "data" in table_info and table_info["data"]:
            lines.append("\nSample Data:")
            column_names = [col["name"] for col in table_info["columns"]]
            lines.append("  " + " | ".join(column_names))
            lines.append("  " + "-" * (len(" | ".join(column_names))))
            
            for row in table_info["data"][:3]:  # Show first 3 rows
                lines.append("  " + " | ".join(str(val) for val in row))
        
        return "\n".join(lines)


class ImprovedPythonGenerator:
    """Enhanced Python code generation with better prompts and examples"""
    
    def __init__(self):
        self.function_templates = self._create_python_templates()
    
    def _create_python_templates(self):
        """Create comprehensive Python function templates"""
        return {
            "basic_math": [
                {
                    "signature": "def add(a, b):",
                    "description": "Add two numbers and return the result",
                    "test_cases": [
                        {"inputs": {"a": 5, "b": 3}, "expected_output": 8, "output_var": "result"},
                        {"inputs": {"a": -2, "b": 7}, "expected_output": 5, "output_var": "result"},
                        {"inputs": {"a": 0, "b": 0}, "expected_output": 0, "output_var": "result"}
                    ],
                    "expected_code": "return a + b"
                },
                {
                    "signature": "def multiply(x, y):",
                    "description": "Multiply two numbers and return the result",
                    "test_cases": [
                        {"inputs": {"x": 4, "y": 3}, "expected_output": 12, "output_var": "result"},
                        {"inputs": {"x": -2, "y": 5}, "expected_output": -10, "output_var": "result"},
                        {"inputs": {"x": 0, "y": 100}, "expected_output": 0, "output_var": "result"}
                    ],
                    "expected_code": "return x * y"
                },
                {
                    "signature": "def subtract(a, b):",
                    "description": "Subtract b from a and return the result",
                    "test_cases": [
                        {"inputs": {"a": 10, "b": 3}, "expected_output": 7, "output_var": "result"},
                        {"inputs": {"a": 5, "b": 8}, "expected_output": -3, "output_var": "result"}
                    ],
                    "expected_code": "return a - b"
                }
            ],
            "string_operations": [
                {
                    "signature": "def reverse_string(s):",
                    "description": "Reverse a string and return it",
                    "test_cases": [
                        {"inputs": {"s": "hello"}, "expected_output": "olleh", "output_var": "result"},
                        {"inputs": {"s": "world"}, "expected_output": "dlrow", "output_var": "result"},
                        {"inputs": {"s": ""}, "expected_output": "", "output_var": "result"}
                    ],
                    "expected_code": "return s[::-1]"
                },
                {
                    "signature": "def count_vowels(text):",
                    "description": "Count the number of vowels in a string",
                    "test_cases": [
                        {"inputs": {"text": "hello"}, "expected_output": 2, "output_var": "result"},
                        {"inputs": {"text": "programming"}, "expected_output": 3, "output_var": "result"}
                    ],
                    "expected_code": "return sum(1 for c in text.lower() if c in 'aeiou')"
                }
            ],
            "list_operations": [
                {
                    "signature": "def find_maximum(numbers):",
                    "description": "Find the maximum value in a list of numbers",
                    "test_cases": [
                        {"inputs": {"numbers": [1, 5, 3, 9, 2]}, "expected_output": 9, "output_var": "result"},
                        {"inputs": {"numbers": [-1, -5, -2]}, "expected_output": -1, "output_var": "result"}
                    ],
                    "expected_code": "return max(numbers)"
                },
                {
                    "signature": "def calculate_average(numbers):",
                    "description": "Calculate the average of a list of numbers",
                    "test_cases": [
                        {"inputs": {"numbers": [2, 4, 6, 8]}, "expected_output": 5.0, "output_var": "result"},
                        {"inputs": {"numbers": [1, 2, 3]}, "expected_output": 2.0, "output_var": "result"}
                    ],
                    "expected_code": "return sum(numbers) / len(numbers)"
                }
            ],
            "algorithms": [
                {
                    "signature": "def factorial(n):",
                    "description": "Calculate the factorial of n",
                    "test_cases": [
                        {"inputs": {"n": 5}, "expected_output": 120, "output_var": "result"},
                        {"inputs": {"n": 0}, "expected_output": 1, "output_var": "result"},
                        {"inputs": {"n": 3}, "expected_output": 6, "output_var": "result"}
                    ],
                    "expected_code": "if n <= 1: return 1\nreturn n * factorial(n-1)"
                },
                {
                    "signature": "def is_prime(n):",
                    "description": "Check if a number is prime",
                    "test_cases": [
                        {"inputs": {"n": 7}, "expected_output": True, "output_var": "result"},
                        {"inputs": {"n": 4}, "expected_output": False, "output_var": "result"},
                        {"inputs": {"n": 2}, "expected_output": True, "output_var": "result"}
                    ],
                    "expected_code": "if n < 2: return False\nfor i in range(2, int(n**0.5) + 1):\n    if n % i == 0: return False\nreturn True"
                }
            ]
        }
    
    def generate_python_prompt(self) -> Tuple[str, Dict]:
        """Generate improved Python prompt with context"""
        category = random.choice(list(self.function_templates.keys()))
        template = random.choice(self.function_templates[category])
        
        # Create comprehensive prompt
        prompt = f"""You are an expert Python programmer. Complete the following function.

Function: {template['signature']}
Task: {template['description']}

Test Cases:
"""
        
        # Add test cases to prompt for better context
        for i, test_case in enumerate(template['test_cases'], 1):
            inputs_str = ", ".join(f"{k}={repr(v)}" for k, v in test_case['inputs'].items())
            prompt += f"  Test {i}: {template['signature'].split('(')[0].split()[-1]}({inputs_str}) should return {test_case['expected_output']}\n"
        
        prompt += f"\nComplete the function implementation:\n{template['signature']}\n    # Your code here"
        
        context = {
            "signature": template['signature'],
            "description": template['description'],
            "test_cases": template['test_cases'],
            "expected_code": template['expected_code'],
            "category": category
        }
        
        return prompt, context


class ImprovedRLVRDataGenerator:
    """Enhanced data generator with better SQL and Python prompts"""
    
    def __init__(self, tokenizer=None, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sql_generator = ImprovedSQLGenerator()
        self.python_generator = ImprovedPythonGenerator()
        
        # Special tokens
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.sep_token_id = 3
    
    def generate_prompt(self, task_type: str) -> Tuple[str, Dict]:
        """Generate improved prompts based on task type"""
        if task_type == "sql":
            return self.sql_generator.generate_sql_prompt()
        elif task_type == "python":
            return self.python_generator.generate_python_prompt()
        elif task_type == "arithmetic":
            return self._generate_arithmetic_prompt()
        elif task_type == "fibonacci":
            return self._generate_fibonacci_prompt()
        else:
            return "Generate text:", {}
    
    def _generate_arithmetic_prompt(self) -> Tuple[str, Dict]:
        """Generate arithmetic sequence prompt"""
        start = random.randint(1, 20)
        step = random.randint(2, 8)
        length = random.randint(4, 6)
        sequence = [start + i * step for i in range(length)]
        
        prompt = f"""Continue the arithmetic sequence. The pattern shows a constant difference between consecutive terms.

Sequence: {', '.join(map(str, sequence))}, ?

What is the next number?"""
        
        context = {
            "start": start,
            "step": step,
            "difference": step,
            "sequence": sequence,
            "next_number": sequence[-1] + step
        }
        
        return prompt, context
    
    def _generate_fibonacci_prompt(self) -> Tuple[str, Dict]:
        """Generate Fibonacci sequence prompt"""
        sequence = [1, 1]
        length = random.randint(3, 6)
        for i in range(length):
            sequence.append(sequence[-1] + sequence[-2])
        
        prompt = f"""Continue the Fibonacci sequence. Each number is the sum of the two preceding numbers.

Sequence: {', '.join(map(str, sequence))}, ?

What is the next number?"""
        
        context = {
            "sequence": sequence,
            "next_number": sequence[-1] + sequence[-2]
        }
        
        return prompt, context
    
    def tokenize(self, text: str) -> mx.array:
        """Tokenize text to token ids"""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            while len(tokens) < self.max_seq_len:
                tokens.append(self.pad_token_id)
            return mx.array(tokens)
        else:
            # Simple character-level tokenization as fallback
            tokens = [ord(c) % 1000 for c in text[:self.max_seq_len]]
            while len(tokens) < self.max_seq_len:
                tokens.append(self.pad_token_id)
            return mx.array(tokens)
    
    def decode(self, token_ids: mx.array) -> str:
        """Decode token ids to text"""
        if self.tokenizer:
            return self.tokenizer.decode(token_ids.tolist())
        else:
            # Simple character-level decoding as fallback
            chars = []
            for token_id in token_ids.tolist():
                if token_id == self.pad_token_id:
                    break
                if token_id < 128:
                    chars.append(chr(token_id))
            return ''.join(chars)


class ImprovedRLVRRewardComputer:
    """Enhanced reward computer with better verification"""
    
    def __init__(self):
        self.sql_verifier = SQLVerifier()
        self.code_verifier = CodeVerifier()
        self.pattern_verifier = PatternVerifier()
    
    def compute_reward(self, 
                      generated_text: str, 
                      task_type: str, 
                      context: Optional[Dict] = None) -> Tuple[float, VerificationResult]:
        """Compute reward with improved verification"""
        context = context or {}
        
        if task_type == "sql":
            return self._compute_sql_reward(generated_text, context)
        elif task_type == "python":
            return self._compute_python_reward(generated_text, context)
        elif task_type == "arithmetic":
            return self._compute_arithmetic_reward(generated_text, context)
        elif task_type == "fibonacci":
            return self._compute_fibonacci_reward(generated_text, context)
        else:
            return 0.0, VerificationResult(
                success=False,
                reward=0.0,
                feedback="Unknown task type",
                details={}
            )
    
    def _compute_sql_reward(self, generated_text: str, context: Dict) -> Tuple[float, VerificationResult]:
        """Compute SQL reward with enhanced verification"""
        # Extract SQL query from generated text
        sql_query = self._extract_sql_query(generated_text)
        
        if not sql_query:
            return -1.0, VerificationResult(
                success=False,
                reward=-1.0,
                feedback="No valid SQL query found in generated text",
                details={"generated_text": generated_text}
            )
        
        # Set up database from context
        if "schema" in context:
            self.sql_verifier.setup_test_database(context["schema"])
        
        # Verify the SQL query
        result = self.sql_verifier.verify_sql_query(sql_query)
        
        # Bonus points for query complexity and correctness
        bonus = 0.0
        if result.success:
            # Bonus for using appropriate SQL keywords
            query_upper = sql_query.upper()
            if "WHERE" in query_upper:
                bonus += 0.2
            if any(agg in query_upper for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"]):
                bonus += 0.3
            if "GROUP BY" in query_upper:
                bonus += 0.2
            if "ORDER BY" in query_upper:
                bonus += 0.1
        
        final_reward = result.reward + bonus
        result.reward = final_reward
        
        return final_reward, result
    
    def _compute_python_reward(self, generated_text: str, context: Dict) -> Tuple[float, VerificationResult]:
        """Compute Python reward with enhanced verification"""
        # Extract Python code from generated text
        python_code = self._extract_python_code(generated_text, context)
        
        if not python_code:
            return -1.0, VerificationResult(
                success=False,
                reward=-1.0,
                feedback="No valid Python code found in generated text",
                details={"generated_text": generated_text}
            )
        
        # Get test cases from context
        test_cases = context.get("test_cases", [])
        
        if not test_cases:
            # Simple syntax check
            try:
                compile(python_code, "<string>", "exec")
                return 0.5, VerificationResult(
                    success=True,
                    reward=0.5,
                    feedback="Valid Python syntax, but no test cases to verify",
                    details={"code": python_code}
                )
            except SyntaxError as e:
                return -0.5, VerificationResult(
                    success=False,
                    reward=-0.5,
                    feedback=f"Python syntax error: {str(e)}",
                    details={"code": python_code, "error": str(e)}
                )
        
        # Verify with test cases
        result = self.code_verifier.verify_python_code(python_code, test_cases)
        
        # Bonus for code quality
        bonus = 0.0
        if result.success:
            # Bonus for concise, readable code
            lines = python_code.strip().split('\n')
            if len(lines) <= 3:  # Concise solution
                bonus += 0.2
            # Bonus for using appropriate Python constructs
            if any(keyword in python_code for keyword in ['return', 'if', 'for', 'while']):
                bonus += 0.1
        
        final_reward = result.reward + bonus
        result.reward = final_reward
        
        return final_reward, result
    
    def _compute_arithmetic_reward(self, generated_text: str, context: Dict) -> Tuple[float, VerificationResult]:
        """Compute arithmetic sequence reward"""
        # Extract number from generated text
        numbers = re.findall(r'-?\d+', generated_text)
        
        if not numbers:
            return -1.0, VerificationResult(
                success=False,
                reward=-1.0,
                feedback="No number found in generated text",
                details={"generated_text": generated_text}
            )
        
        try:
            predicted_number = int(numbers[0])
            expected_number = context.get("next_number")
            
            if expected_number is None:
                # Calculate expected from sequence
                sequence = context.get("sequence", [])
                if len(sequence) >= 2:
                    diff = sequence[-1] - sequence[-2]
                    expected_number = sequence[-1] + diff
                else:
                    step = context.get("step", 1)
                    start = context.get("start", 1)
                    length = len(sequence) if sequence else 5
                    expected_number = start + length * step
            
            if predicted_number == expected_number:
                return 2.0, VerificationResult(
                    success=True,
                    reward=2.0,
                    feedback=f"Correct! The next number is {predicted_number}",
                    details={
                        "predicted": predicted_number,
                        "expected": expected_number,
                        "context": context
                    }
                )
            else:
                # Partial credit for being close
                diff = abs(predicted_number - expected_number)
                if diff <= 2:
                    reward = 0.5
                    feedback = f"Close but not exact. Got {predicted_number}, expected {expected_number}"
                else:
                    reward = -0.5
                    feedback = f"Incorrect. Got {predicted_number}, expected {expected_number}"
                
                return reward, VerificationResult(
                    success=False,
                    reward=reward,
                    feedback=feedback,
                    details={
                        "predicted": predicted_number,
                        "expected": expected_number,
                        "difference": diff
                    }
                )
        
        except ValueError:
            return -1.0, VerificationResult(
                success=False,
                reward=-1.0,
                feedback=f"Invalid number format: {numbers[0]}",
                details={"extracted_text": numbers[0]}
            )
    
    def _compute_fibonacci_reward(self, generated_text: str, context: Dict) -> Tuple[float, VerificationResult]:
        """Compute Fibonacci sequence reward"""
        return self._compute_arithmetic_reward(generated_text, context)  # Same logic
    
    def _extract_sql_query(self, text: str) -> Optional[str]:
        """Extract SQL query from generated text"""
        text = text.strip()
        
        # Look for SQL keywords
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        
        # Try to find a line that starts with SQL keyword
        for line in text.split('\n'):
            line = line.strip()
            if any(line.upper().startswith(keyword) for keyword in sql_keywords):
                # Clean up the query
                query = line.rstrip(';') + ';' if not line.endswith(';') else line
                return query
        
        # Fallback: if the entire text looks like SQL
        if any(keyword in text.upper() for keyword in sql_keywords):
            return text.rstrip(';') + ';' if not text.endswith(';') else text
        
        return None
    
    def _extract_python_code(self, text: str, context: Dict) -> Optional[str]:
        """Extract Python code from generated text"""
        # Get the function signature from context
        signature = context.get("signature", "")
        
        if signature:
            # Try to construct the complete function
            lines = text.strip().split('\n')
            
            # Look for return statement or function body
            for line in lines:
                line = line.strip()
                if line.startswith('return ') or 'return' in line:
                    # Simple return statement
                    if line.startswith('return '):
                        return f"{signature}\n    {line}"
                    else:
                        return f"{signature}\n    {line}"
            
            # Look for code without 'return' keyword
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('def'):
                    # Assume it's the function body
                    if not line.startswith('return'):
                        line = f"return {line}"
                    return f"{signature}\n    {line}"
        
        # Fallback: return the text as is if it looks like Python
        if 'def ' in text or 'return' in text:
            return text
        
        return None


def test_improved_rlvr():
    """Test the improved RLVR system"""
    print("Testing Improved RLVR System")
    print("=" * 50)
    
    # Create improved generator
    generator = ImprovedRLVRDataGenerator()
    reward_computer = ImprovedRLVRRewardComputer()
    
    # Test SQL generation
    print("\n1. Testing SQL Generation:")
    for i in range(3):
        prompt, context = generator.generate_prompt("sql")
        print(f"\nSQL Test {i+1}:")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Expected: {context.get('expected_query', 'N/A')}")
        
        # Test with expected query
        reward, result = reward_computer.compute_reward(
            context.get('expected_query', ''), 
            "sql", 
            context
        )
        print(f"Reward: {reward:.2f}, Success: {result.success}")
    
    # Test Python generation
    print("\n2. Testing Python Generation:")
    for i in range(3):
        prompt, context = generator.generate_prompt("python")
        print(f"\nPython Test {i+1}:")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Expected: {context.get('expected_code', 'N/A')}")
        
        # Test with expected code
        if context.get('expected_code'):
            full_code = f"{context['signature']}\n    {context['expected_code']}"
            reward, result = reward_computer.compute_reward(
                full_code, 
                "python", 
                context
            )
            print(f"Reward: {reward:.2f}, Success: {result.success}")
    
    print("\n✅ Improved RLVR system tested!")


if __name__ == "__main__":
    test_improved_rlvr()