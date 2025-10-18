"""
Improved SQL Training Script with Better Data Handling
"""

import json
from mlx_lm import load, generate
from mlx_lm.tuner import TrainingArgs

def analyze_wikisql_format():
    """Analyze the WikiSQL dataset format to understand the problem"""
    from datasets import load_dataset
    
    dataset = load_dataset("mlx-community/wikisql")
    
    print("Dataset structure:")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['valid'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Examine first few samples
    print("\nFirst 3 training samples:")
    for i in range(3):
        sample = dataset['train'][i]
        print(f"\nSample {i+1}:")
        print(json.dumps(sample, indent=2))
        
    return dataset

def create_sql_instruction_prompt(question: str, table_info: str) -> str:
    """Create a better formatted prompt for SQL generation"""
    
    instruction = """You are a SQL expert. Given a table schema and a question, write the corresponding SQL query.

### Table Schema:
{table_info}

### Question:
{question}

### SQL Query:
""".format(table_info=table_info, question=question)
    
    return instruction

def preprocess_wikisql_for_training(dataset):
    """Preprocess WikiSQL dataset for better training"""
    
    processed_samples = []
    
    for sample in dataset:
        # Extract table information
        table_name = sample.get('table', {}).get('name', 'unknown_table')
        columns = sample.get('table', {}).get('header', [])
        
        # Format table info
        table_info = f"Table: {table_name}\nColumns: {', '.join(columns)}"
        
        # Get question and SQL
        question = sample.get('question', '')
        sql = sample.get('sql', {})
        
        # Create instruction-following format
        prompt = create_sql_instruction_prompt(question, table_info)
        
        # Format the SQL answer (simplified)
        sql_query = f"SELECT {sql.get('sel', '*')} FROM {table_name}"
        if sql.get('conds'):
            conditions = []
            for cond in sql['conds']:
                col_idx = cond[0]
                op = cond[1]
                val = cond[2]
                col_name = columns[col_idx] if col_idx < len(columns) else 'unknown'
                conditions.append(f"{col_name} = '{val}'")
            sql_query += " WHERE " + " AND ".join(conditions)
        
        # Create training sample
        full_text = prompt + sql_query
        processed_samples.append({
            'text': full_text,
            'prompt': prompt,
            'completion': sql_query
        })
    
    return processed_samples

# Recommendations for better training
print("""
RECOMMENDATIONS FOR BETTER SQL FINE-TUNING:

1. Use a model pre-trained on code/SQL:
   - CodeLlama-7B-Instruct
   - SQLCoder-7B
   - StarCoder-7B

2. Preprocess data properly:
   - Use instruction-following format
   - Add system prompts for SQL generation
   - Include table schema in every prompt

3. Training improvements:
   - Use larger dataset (full WikiSQL has 80k+ samples)
   - Train for more iterations (1000-2000)
   - Use smaller learning rate (5e-5)
   - Add SQL-specific tokens to vocabulary

4. Alternative datasets:
   - Spider (more complex SQL)
   - Text2SQL benchmarks
   - Custom SQL instruction datasets

5. Evaluation metrics:
   - Exact match accuracy
   - Execution accuracy
   - SQL syntax validation
""")

if __name__ == "__main__":
    # Analyze the dataset
    dataset = analyze_wikisql_format()
    
    # Show preprocessing example
    print("\n" + "="*50)
    print("PREPROCESSING EXAMPLE")
    print("="*50)
    
    train_samples = list(dataset['train'])[:3]
    processed = preprocess_wikisql_for_training(train_samples)
    
    for i, sample in enumerate(processed):
        print(f"\nProcessed Sample {i+1}:")
        print(sample['text'])
        print("-" * 30)