"""
Code Editor Module
-----------------
Provides capabilities for code generation, analysis, and self-modification
"""

import os
import re
import ast
import inspect
import importlib
import importlib.util
import tempfile
import traceback
import subprocess
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

# Local imports
from AgentSystem.utils.logger import get_logger

# Get module logger
logger = get_logger("modules.code_editor")


class CodeEditor:
    """Module for code editing, generation, and self-modification"""
    
    def __init__(self, workspace_dir: str = None):
        """
        Initialize the code editor module
        
        Args:
            workspace_dir: Directory to use as workspace for code operations
                           (default: use system temp directory)
        """
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="agentsystem_code_")
        logger.info(f"Initialized CodeEditor with workspace: {self.workspace_dir}")
        
        # Track all code modifications for safety and auditing
        self.modifications = []
        
    def get_tools(self) -> Dict[str, Any]:
        """Get tools provided by this module"""
        return {
            "read_file": {
                "description": "Read the contents of a file",
                "function": self.read_file,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        }
                    },
                    "required": ["path"]
                }
            },
            "write_file": {
                "description": "Write content to a file",
                "function": self.write_file,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        },
                        "mode": {
                            "type": "string",
                            "description": "Write mode: 'w' for write, 'a' for append",
                            "enum": ["w", "a"],
                            "default": "w"
                        }
                    },
                    "required": ["path", "content"]
                }
            },
            "modify_file": {
                "description": "Make targeted modifications to a file",
                "function": self.modify_file,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to modify"
                        },
                        "changes": {
                            "type": "array",
                            "description": "List of changes to make",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "find": {
                                        "type": "string",
                                        "description": "Text to find (can be regex)"
                                    },
                                    "replace": {
                                        "type": "string",
                                        "description": "Text to replace with"
                                    },
                                    "use_regex": {
                                        "type": "boolean",
                                        "description": "Whether to treat find as regex",
                                        "default": False
                                    }
                                },
                                "required": ["find", "replace"]
                            }
                        }
                    },
                    "required": ["path", "changes"]
                }
            },
            "analyze_code": {
                "description": "Analyze a Python code file",
                "function": self.analyze_code,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Python file to analyze"
                        }
                    },
                    "required": ["path"]
                }
            },
            "execute_code": {
                "description": "Execute Python code and return the result",
                "function": self.execute_code,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        },
                        "safe_mode": {
                            "type": "boolean",
                            "description": "Whether to execute in safe mode with limited imports",
                            "default": True
                        }
                    },
                    "required": ["code"]
                }
            },
            "generate_code": {
                "description": "Generate code based on a specification",
                "function": self.generate_code,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "specification": {
                            "type": "string",
                            "description": "Detailed specification of what the code should do"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language to generate code in",
                            "default": "python"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save the generated code (optional)"
                        }
                    },
                    "required": ["specification"]
                }
            },
            "test_code": {
                "description": "Run tests for a Python module",
                "function": self.test_code,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Python file to test"
                        },
                        "test_code": {
                            "type": "string",
                            "description": "Test code to run (optional)"
                        }
                    },
                    "required": ["path"]
                }
            },
            "refactor_code": {
                "description": "Refactor a Python code file",
                "function": self.refactor_code,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Python file to refactor"
                        },
                        "refactoring_type": {
                            "type": "string",
                            "description": "Type of refactoring to perform",
                            "enum": ["extract_function", "rename", "optimize", "improve_readability"],
                            "default": "improve_readability"
                        },
                        "details": {
                            "type": "object",
                            "description": "Details specific to the refactoring type"
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    
    def read_file(self, path: str) -> Dict[str, Any]:
        """
        Read the contents of a file
        
        Args:
            path: Path to the file to read
            
        Returns:
            Dictionary with file content and metadata
        """
        try:
            # Ensure the path is not trying to access protected areas
            abs_path = os.path.abspath(path)
            
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }
            
            # Read the file content
            with open(abs_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Get file metadata
            stat = os.stat(abs_path)
            ext = os.path.splitext(abs_path)[1].lower()
            
            return {
                "success": True,
                "path": abs_path,
                "content": content,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "extension": ext,
                "is_python": ext == ".py"
            }
            
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def write_file(self, path: str, content: str, mode: str = "w") -> Dict[str, Any]:
        """
        Write content to a file
        
        Args:
            path: Path to the file to write
            content: Content to write to the file
            mode: Write mode ('w' for write, 'a' for append)
            
        Returns:
            Dictionary with result information
        """
        try:
            # Ensure the path is not trying to access protected areas
            abs_path = os.path.abspath(path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            
            # Write the content
            with open(abs_path, mode, encoding='utf-8') as file:
                file.write(content)
            
            # Record the modification
            self.modifications.append({
                "action": "write",
                "path": abs_path,
                "mode": mode,
                "timestamp": os.path.getmtime(abs_path)
            })
            
            return {
                "success": True,
                "path": abs_path,
                "size": len(content),
                "mode": mode
            }
            
        except Exception as e:
            logger.error(f"Error writing to file {path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def modify_file(self, path: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make targeted modifications to a file
        
        Args:
            path: Path to the file to modify
            changes: List of changes to make, each with 'find' and 'replace' keys
            
        Returns:
            Dictionary with result information
        """
        try:
            # Read the current content
            read_result = self.read_file(path)
            if not read_result.get("success", False):
                return read_result
            
            content = read_result["content"]
            original_content = content
            
            # Apply each change
            applied_changes = []
            for i, change in enumerate(changes):
                find_text = change["find"]
                replace_text = change["replace"]
                use_regex = change.get("use_regex", False)
                
                if use_regex:
                    # Use regex for replacement
                    pattern = re.compile(find_text, re.DOTALL | re.MULTILINE)
                    new_content = pattern.sub(replace_text, content)
                    matches = len(pattern.findall(content))
                else:
                    # Use string replacement
                    new_content = content.replace(find_text, replace_text)
                    matches = content.count(find_text)
                
                if new_content != content:
                    applied_changes.append({
                        "index": i,
                        "matches": matches,
                        "use_regex": use_regex
                    })
                
                content = new_content
            
            # Write the modified content if changes were made
            if content != original_content:
                write_result = self.write_file(path, content)
                if not write_result.get("success", False):
                    return write_result
                
                # Record the modification
                self.modifications.append({
                    "action": "modify",
                    "path": os.path.abspath(path),
                    "changes": applied_changes,
                    "timestamp": os.path.getmtime(os.path.abspath(path))
                })
                
                return {
                    "success": True,
                    "path": os.path.abspath(path),
                    "changes_applied": len(applied_changes),
                    "applied_changes": applied_changes
                }
            else:
                return {
                    "success": True,
                    "path": os.path.abspath(path),
                    "changes_applied": 0,
                    "message": "No changes were made to the file"
                }
            
        except Exception as e:
            logger.error(f"Error modifying file {path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def analyze_code(self, path: str) -> Dict[str, Any]:
        """
        Analyze a Python code file
        
        Args:
            path: Path to the Python file to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Read the file
            read_result = self.read_file(path)
            if not read_result.get("success", False):
                return read_result
            
            content = read_result["content"]
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Extract information
            classes = []
            functions = []
            imports = []
            variables = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    class_vars = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append({
                                "name": item.name,
                                "args": [arg.arg for arg in item.args.args],
                                "line": item.lineno
                            })
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    class_vars.append({
                                        "name": target.id,
                                        "line": item.lineno
                                    })
                    
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": methods,
                        "variables": class_vars
                    })
                elif isinstance(node, ast.FunctionDef) and node.parent_node is tree:
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "line": node.lineno
                    })
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append({
                            "name": name.name,
                            "alias": name.asname,
                            "line": node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        imports.append({
                            "name": f"{node.module}.{name.name}" if node.module else name.name,
                            "alias": name.asname,
                            "line": node.lineno
                        })
                elif isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
                    for target in node.targets:
                        variables.append({
                            "name": target.id,
                            "line": node.lineno
                        })
            
            # Calculate complexity metrics
            lines_of_code = len(content.split('\n'))
            docstring_lines = sum(len(ast.get_docstring(node).split('\n')) if ast.get_docstring(node) else 0 
                                 for node in ast.walk(tree) 
                                 if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Module)))
            
            return {
                "success": True,
                "path": os.path.abspath(path),
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "variables": variables,
                "metrics": {
                    "lines_of_code": lines_of_code,
                    "docstring_lines": docstring_lines,
                    "num_classes": len(classes),
                    "num_functions": len(functions),
                    "num_imports": len(imports)
                }
            }
            
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Syntax error: {e}",
                "line": e.lineno,
                "offset": e.offset,
                "text": e.text
            }
        except Exception as e:
            logger.error(f"Error analyzing code {path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def execute_code(self, code: str, safe_mode: bool = True) -> Dict[str, Any]:
        """
        Execute Python code and return the result
        
        Args:
            code: Python code to execute
            safe_mode: Whether to execute in safe mode with limited imports
            
        Returns:
            Dictionary with execution results
        """
        if safe_mode:
            # Create a restricted globals dictionary
            restricted_globals = {
                "__builtins__": {
                    name: getattr(__builtins__, name)
                    for name in [
                        "abs", "all", "any", "ascii", "bin", "bool", "bytes", 
                        "callable", "chr", "complex", "dict", "dir", "divmod", 
                        "enumerate", "filter", "float", "format", "frozenset", 
                        "getattr", "hasattr", "hash", "hex", "id", "int", 
                        "isinstance", "issubclass", "iter", "len", "list", 
                        "map", "max", "min", "next", "object", "oct", "ord", 
                        "pow", "print", "range", "repr", "reversed", "round", 
                        "set", "slice", "sorted", "str", "sum", "tuple", "type", 
                        "zip"
                    ]
                }
            }
            
            # Create a locals dictionary to capture output
            locals_dict = {}
            
            try:
                # Execute the code in a restricted environment
                exec(code, restricted_globals, locals_dict)
                
                return {
                    "success": True,
                    "variables": {k: repr(v) for k, v in locals_dict.items() if not k.startswith('_')},
                    "output": "Code executed successfully in safe mode"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        else:
            # Execute without restrictions (more dangerous)
            try:
                # Create a dictionary to capture output
                result = {}
                
                # Create a temporary file for execution
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as temp:
                    temp.write(code)
                    temp_path = temp.name
                
                try:
                    # Execute as a subprocess to capture stdout/stderr
                    process = subprocess.run(
                        [sys.executable, temp_path],
                        capture_output=True,
                        text=True,
                        timeout=30  # Timeout after 30 seconds
                    )
                    
                    return {
                        "success": process.returncode == 0,
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                        "returncode": process.returncode
                    }
                    
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_path)
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": "Execution timed out after 30 seconds"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
    
    def generate_code(self, specification: str, language: str = "python", 
                     output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate code based on a specification
        
        Args:
            specification: Detailed specification of what the code should do
            language: Programming language to generate code in
            output_path: Path to save the generated code (optional)
            
        Returns:
            Dictionary with generated code
        """
        try:
            # Use the AI service to generate code
            from AgentSystem.services.ai import ai_service, AIMessage, AIRequestOptions
            
            # Format the prompt
            messages = [
                AIMessage(role="system", content=f"""You are an expert {language} programmer. 
Your task is to generate high-quality, well-documented, and efficient code based on the specification provided.
Include appropriate error handling, comments, and follow best practices for {language}.
Only output the code with no additional explanation or markdown formatting."""),
                AIMessage(role="user", content=f"Please generate {language} code for the following specification:\n\n{specification}")
            ]
            
            # Generate the code
            options = AIRequestOptions(temperature=0.2)
            response = ai_service.complete(messages=messages, options=options)
            
            # Extract the code
            generated_code = response.content.strip()
            
            # Remove markdown code blocks if present
            if generated_code.startswith("```") and generated_code.endswith("```"):
                generated_code = "\n".join(generated_code.split("\n")[1:-1])
            
            # Save to file if output path is provided
            if output_path:
                write_result = self.write_file(output_path, generated_code)
                if not write_result.get("success", False):
                    return write_result
                
                # Record the generation
                self.modifications.append({
                    "action": "generate",
                    "path": os.path.abspath(output_path),
                    "language": language,
                    "specification": specification[:200] + "..." if len(specification) > 200 else specification,
                    "timestamp": os.path.getmtime(os.path.abspath(output_path))
                })
                
                return {
                    "success": True,
                    "code": generated_code,
                    "language": language,
                    "path": os.path.abspath(output_path)
                }
            else:
                return {
                    "success": True,
                    "code": generated_code,
                    "language": language
                }
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def test_code(self, path: str, test_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Run tests for a Python module
        
        Args:
            path: Path to the Python file to test
            test_code: Test code to run (optional)
            
        Returns:
            Dictionary with test results
        """
        try:
            # Get the absolute path
            abs_path = os.path.abspath(path)
            
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }
            
            # Create a temporary directory for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy the file to test
                file_name = os.path.basename(abs_path)
                test_file_path = os.path.join(temp_dir, file_name)
                
                # Read the file content
                with open(abs_path, 'r', encoding='utf-8') as src_file:
                    content = src_file.read()
                
                # Write to the test location
                with open(test_file_path, 'w', encoding='utf-8') as dest_file:
                    dest_file.write(content)
                
                # Create test file
                if test_code:
                    test_path = os.path.join(temp_dir, f"test_{file_name}")
                    with open(test_path, 'w', encoding='utf-8') as test_file:
                        test_file.write(test_code)
                else:
                    # Generate simple tests if none provided
                    module_name = os.path.splitext(file_name)[0]
                    test_path = os.path.join(temp_dir, f"test_{file_name}")
                    
                    # Simple test template
                    auto_test_code = f"""
import unittest
import {module_name}

class Test{module_name.capitalize()}(unittest.TestCase):
    def test_module_imports(self):
        # Basic test to ensure the module can be imported
        self.assertIsNotNone({module_name})
        
    # Add more tests here

if __name__ == '__main__':
    unittest.main()
"""
                    with open(test_path, 'w', encoding='utf-8') as test_file:
                        test_file.write(auto_test_code)
                
                # Run the tests
                process = subprocess.run(
                    [sys.executable, "-m", "unittest", f"test_{module_name}"],
                    capture_output=True,
                    text=True,
                    cwd=temp_dir
                )
                
                return {
                    "success": process.returncode == 0,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "returncode": process.returncode,
                    "test_file": test_path if test_code else "auto-generated"
                }
            
        except Exception as e:
            logger.error(f"Error testing code {path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def refactor_code(self, path: str, refactoring_type: str = "improve_readability", 
                     details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Refactor a Python code file
        
        Args:
            path: Path to the Python file to refactor
            refactoring_type: Type of refactoring to perform
            details: Details specific to the refactoring type
            
        Returns:
            Dictionary with refactoring results
        """
        try:
            # Get the absolute path
            abs_path = os.path.abspath(path)
            
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }
            
            # Read the current code
            with open(abs_path, 'r', encoding='utf-8') as file:
                original_code = file.read()
            
            # Use the AI service to refactor the code
            from AgentSystem.services.ai import ai_service, AIMessage, AIRequestOptions
            
            details_str = ""
            if details:
                details_str = "\n\nAdditional details:\n"
                for key, value in details.items():
                    details_str += f"- {key}: {value}\n"
            
            # Format the prompt based on refactoring type
            if refactoring_type == "extract_function":
                prompt = f"""Refactor the following Python code by extracting repeated or complex logic into separate functions.
Focus on improving modularity and reusability without changing the functionality.{details_str}

Original code:
```python
{original_code}
```

Provide the refactored code only with no additional explanation."""
                
            elif refactoring_type == "rename":
                prompt = f"""Refactor the following Python code by improving variable, function, and class names.
Focus on making the names more descriptive and following Python naming conventions.{details_str}

Original code:
```python
{original_code}
```

Provide the refactored code only with no additional explanation."""
                
            elif refactoring_type == "optimize":
                prompt = f"""Refactor the following Python code to optimize its performance.
Focus on improving algorithm efficiency, reducing unnecessary operations, and optimizing resource usage.{details_str}

Original code:
```python
{original_code}
```

Provide the refactored code only with no additional explanation."""
                
            else:  # improve_readability
                prompt = f"""Refactor the following Python code to improve readability and maintainability.
Focus on code structure, comments, docstrings, and adherence to PEP 8 standards.{details_str}

Original code:
```python
{original_code}
```

Provide the refactored code only with no additional explanation."""
            
            # Generate the refactored code
            messages = [
                AIMessage(role="system", content="You are an expert Python programmer specializing in code refactoring. Your task is to improve code quality while preserving functionality."),
                AIMessage(role="user", content=prompt)
            ]
            
            options = AIRequestOptions(temperature=0.2)
            response = ai_service.complete(messages=messages, options=options)
            
            # Extract the code
            refactored_code = response.content.strip()
            
            # Remove markdown code blocks if present
            if refactored_code.startswith("```python") and refactored_code.endswith("```"):
                refactored_code = "\n".join(refactored_code.split("\n")[1:-1])
            elif refactored_code.startswith("```") and refactored_code.endswith("```"):
                refactored_code = "\n".join(refactored_code.split("\n")[1:-1])
            
            # Create a backup of the original file
            backup_path = f"{abs_path}.bak"
            with open(backup_path, 'w', encoding='utf-8') as file:
                file.write(original_code)
            
            # Write the refactored code
            with open(abs_path, 'w', encoding='utf-8') as file:
                file.write(refactored_code)
            
            # Record the refactoring
            self.modifications.append({
                "action": "refactor",
                "path": abs_path,
                "type": refactoring_type,
                "backup": backup_path,
                "timestamp": os.path.getmtime(abs_path)
            })
            
            # Return the result
            return {
                "success": True,
                "path": abs_path,
                "refactoring_type": refactoring_type,
                "backup": backup_path,
                "changes": self._diff_summary(original_code, refactored_code)
            }
            
        except Exception as e:
            logger.error(f"Error refactoring code {path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _diff_summary(self, original: str, modified: str) -> Dict[str, Any]:
        """Generate a summary of changes between original and modified code"""
        import difflib
        import sys
        
        # Split into lines
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()
        
        # Calculate diff
        diff = list(difflib.unified_diff(
            original_lines, modified_lines, 
            fromfile='original', tofile='modified', 
            lineterm=''
        ))
        
        # Count additions and deletions
        additions = len([line for line in diff if line.startswith('+')])
        deletions = len([line for line in diff if line.startswith('-')])
        
        # Format diff for display (limited to 50 lines)
        formatted_diff = '\n'.join(diff[:50])
        if len(diff) > 50:
            formatted_diff += '\n... (diff truncated)'
        
        return {
            "lines_changed": additions + deletions,
            "additions": additions,
            "deletions": deletions,
            "diff_preview": formatted_diff
        }
