#!/usr/bin/env python
"""Script to run the MLX notebook and validate for errors"""

import subprocess
import json
import sys

def run_notebook(notebook_path):
    """Run a Jupyter notebook and return any errors"""
    print(f"Running notebook: {notebook_path}")
    
    # Convert notebook to Python script and execute
    cmd = [
        "jupyter", "nbconvert", 
        "--to", "script", 
        "--execute",
        "--ExecutePreprocessor.timeout=600",
        "--stdout",
        notebook_path
    ]
    
    try:
        # Run the notebook
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print("\n❌ Notebook execution failed!")
            print("\nSTDERR:")
            print(result.stderr)
            return False
        else:
            print("\n✅ Notebook executed successfully!")
            return True
            
    except Exception as e:
        print(f"\n❌ Error running notebook: {e}")
        return False

def run_notebook_papermill(notebook_path):
    """Alternative method using papermill if available"""
    try:
        import papermill as pm
        
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        print(f"Running notebook with papermill: {notebook_path}")
        
        pm.execute_notebook(
            notebook_path,
            output_path,
            kernel_name='mlx'
        )
        
        print("\n✅ Notebook executed successfully with papermill!")
        return True
        
    except ImportError:
        print("Papermill not installed, falling back to nbconvert")
        return run_notebook(notebook_path)
    except Exception as e:
        print(f"\n❌ Error with papermill: {e}")
        return False

if __name__ == "__main__":
    notebook = "MLXTraining_ordered.ipynb"
    
    # First ensure we're in the MLX environment
    print("Make sure you're running this in the MLX environment:")
    print("source mlx_env/bin/activate")
    print()
    
    # Try to run the notebook
    success = run_notebook_papermill(notebook)
    
    if not success:
        print("\nTrying alternative method...")
        success = run_notebook(notebook)
    
    sys.exit(0 if success else 1)