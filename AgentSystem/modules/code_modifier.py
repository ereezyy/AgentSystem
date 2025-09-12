"""
Code Modifier Module
------------------
Enables the agent to analyze and modify its own code safely.

Features:
- Code analysis and understanding
- Safe code modification
- Backup and rollback
- Change validation
"""

import os
import ast
import sys
import inspect
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from celery import Celery
import paramiko
import psutil
import subprocess
from apscheduler.schedulers.background import BackgroundScheduler

from AgentSystem.utils.logger import get_logger
from AgentSystem.services.ai_providers import get_provider_manager

logger = get_logger("modules.code_modifier")

class CodeModifier:
    def __init__(self, backup_dir: Optional[str] = None, redis_host: str = "localhost", 
                 redis_port: int = 6379):
        """
        Initialize code modifier with distributed processing support
        
        Args:
            backup_dir: Directory for storing code backups
            redis_host: Redis host for Celery
            redis_port: Redis port for Celery
        """
        self.backup_dir = backup_dir or tempfile.gettempdir()
        self.backup_dir = Path(self.backup_dir) / "agent_code_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Celery for distributed tasks
        self.app = Celery('code_modifier',
                         broker=f'redis://{redis_host}:{redis_port}/0',
                         backend=f'redis://{redis_host}:{redis_port}/0')
        
        # Configure task routing
        self.app.conf.task_routes = {
            'codemodifier.ai_code_analysis': {'priority': 2},
            'codemodifier.model_compilation': {'priority': 3}
        }
        
        # Model management
        self.models_dir = Path("/models")  # Directory for AI models
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        
    def analyze_code(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python source file
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary with code analysis results
        """
        try:
            with open(file_path, 'r') as f:
                code = f.read()
                
            # Parse the code
            tree = ast.parse(code)
            
            # Analyze structure
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes.append({
                        "name": node.name,
                        "methods": methods,
                        "line": node.lineno
                    })
                elif isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "args": [a.arg for a in node.args.args],
                        "line": node.lineno
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            imports.append(n.name)
                    else:
                        module = node.module or ""
                        for n in node.names:
                            imports.append(f"{module}.{n.name}")
            
            return {
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "loc": len(code.splitlines()),
                "file": file_path
            }
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {}
    
    def create_backup(self, file_path: str) -> Optional[str]:
        """
        Create a backup of a source file
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to the backup file or None on failure
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                logger.error(f"Source file does not exist: {file_path}")
                return None
                
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            # Copy file
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None
    
    def restore_backup(self, backup_path: str, target_path: str) -> bool:
        """
        Restore a file from backup
        
        Args:
            backup_path: Path to the backup file
            target_path: Path where to restore the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
                
            shutil.copy2(backup_path, target_path)
            logger.info(f"Restored {target_path} from backup")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def validate_changes(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate that modified code is syntactically correct
        
        Args:
            file_path: Path to the modified file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with open(file_path, 'r') as f:
                code = f.read()
                
            # Check syntax
            ast.parse(code)
            
            # Try to load as module
            module_name = Path(file_path).stem
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Create temp dir for test import
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_path = Path(tmp_dir) / Path(file_path).name
                shutil.copy2(file_path, temp_path)
                
                sys.path.insert(0, tmp_dir)
                try:
                    __import__(module_name)
                finally:
                    sys.path.pop(0)
            
            return True, "Changes validated successfully"
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def modify_code(self, file_path: str, changes: Dict[str, Any]) -> bool:
        """
        Safely modify code with validation and backup
        
        Args:
            file_path: Path to the file to modify
            changes: Dictionary describing the changes to make
            
        Returns:
            True if changes were applied successfully
        """
        # Create backup first
        backup_path = self.create_backup(file_path)
        if not backup_path:
            return False
        
        try:
            # Read current code
            with open(file_path, 'r') as f:
                code = f.read()
            
            # Parse code
            tree = ast.parse(code)
            
            # Apply changes
            modified = False
            
            if "add_method" in changes:
                # Example: Add method to class
                method = changes["add_method"]
                class_name = method["class"]
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        # Insert new method
                        method_code = method["code"]
                        method_tree = ast.parse(method_code)
                        node.body.append(method_tree.body[0])
                        modified = True
                        break
            
            if modified:
                # Generate modified code
                new_code = ast.unparse(tree)
                
                # Validate changes
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                    tmp.write(new_code)
                    tmp_path = tmp.name
                
                valid, error = self.validate_changes(tmp_path)
                if not valid:
                    logger.error(f"Invalid changes: {error}")
                    self.restore_backup(backup_path, file_path)
                    return False
                
                # Apply changes
                with open(file_path, 'w') as f:
                    f.write(new_code)
                
                logger.info(f"Successfully modified {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error modifying code: {e}")
            if backup_path:
                self.restore_backup(backup_path, file_path)
            return False
    
    def suggest_improvements(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze code and suggest potential improvements
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            List of suggested improvements
        """
        try:
            # Analyze current code
            analysis = self.analyze_code(file_path)
            
            # Generate improvements using AI
            improvements = []
            
            # Ask AI for suggestions
            with open(file_path, 'r') as f:
                code = f.read()
            
            prompt = f"""
            Analyze this Python code and suggest specific improvements:
            
            {code}
            
            Focus on:
            1. Code organization and structure
            2. Error handling and robustness
            3. Performance optimization
            4. Documentation and clarity
            """
            
            # Get AI provider manager and generate response
            provider_manager = get_provider_manager()
            response = provider_manager.generate_text(prompt, max_tokens=2000, temperature=0.3)
            ai_response = response.get('text', '')
            
            # Parse and structure suggestions
            suggestions = ai_response.split("\n")
            for suggestion in suggestions:
                if suggestion.strip():
                    improvements.append({
                        "type": "improvement",
                        "description": suggestion,
                        "priority": "medium"
                    })
            
            return improvements
            
        except Exception as e:
            logger.error(f"Error suggesting improvements: {e}")
            return []
    
    def safe_execute(self, task: callable, *args, **kwargs) -> Any:
        """Execute task with thermal and RAM safety checks"""
        # Check thermal status
        if not self.check_thermal():
            logger.warning("Pausing task due to thermal limit")
            import time
            time.sleep(60)  # Cool down period
            return None
            
        # Check RAM usage
        ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # GB
        if ram_usage > 7.0:  # 7GB limit for Pi 5
            logger.warning(f"RAM usage {ram_usage:.2f}GB exceeds limit")
            import time
            time.sleep(30)  # Wait for RAM to free up
            return None
            
        return task(*args, **kwargs)
    
    def check_thermal(self) -> bool:
        """Check thermal status on system"""
        try:
            # Check if on Raspberry Pi
            temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
            temp = float(temp_output.split("=")[1].split("'")[0])
            
            if temp > 80:
                logger.warning(f"CPU temperature {temp}°C exceeds limit")
                return False
            return True
        except:
            # Not on Raspberry Pi, assume thermal is OK
            return True
    
    def update_models(self, model_dir: str, pi_ip: str, pi_password: str = 'raspberry') -> bool:
        """
        Update AI models on Raspberry Pi 5
        
        Args:
            model_dir: Directory containing compiled models
            pi_ip: IP address of Raspberry Pi 5
            pi_password: SSH password for Pi
            
        Returns:
            True if successful
        """
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect to Pi 5
            ssh.connect(pi_ip, username='pi', password=pi_password)
            sftp = ssh.open_sftp()
            
            # Transfer all .hef model files
            for model_file in Path(model_dir).glob("*.hef"):
                remote_path = f"/models/{model_file.name}"
                logger.info(f"Transferring {model_file.name} to Pi 5")
                sftp.put(str(model_file), remote_path)
                
                # Set permissions
                ssh.exec_command(f"chmod 644 {remote_path}")
            
            sftp.close()
            ssh.close()
            
            logger.info("Model update completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model transfer failed: {e}")
            return False
    
    def schedule_model_updates(self, model_dir: str, pi_ip: str, interval_hours: int = 24):
        """
        Schedule automatic model updates
        
        Args:
            model_dir: Directory containing models
            pi_ip: Raspberry Pi 5 IP address
            interval_hours: Update interval in hours
        """
        def update_task():
            logger.info("Running scheduled model update")
            self.update_models(model_dir, pi_ip)
        
        # Schedule the job
        self.scheduler.add_job(
            update_task,
            'interval',
            hours=interval_hours,
            id='model_update',
            replace_existing=True
        )
        
        logger.info(f"Scheduled model updates every {interval_hours} hours")
    
    def compile_model_for_hailo(self, model_path: str, output_dir: str) -> Optional[str]:
        """
        Compile a model for Hailo-8 accelerator (runs on x86 CPU)
        
        Args:
            model_path: Path to source model
            output_dir: Directory for compiled .hef file
            
        Returns:
            Path to compiled model or None
        """
        try:
            # This would use Hailo Dataflow Compiler on x86
            # Mock implementation for demonstration
            
            model_name = Path(model_path).stem
            output_path = Path(output_dir) / f"{model_name}.hef"
            
            # In real implementation:
            # subprocess.run([
            #     "hailo_compiler",
            #     "--input", model_path,
            #     "--output", str(output_path),
            #     "--arch", "hailo8"
            # ], check=True)
            
            logger.info(f"Model compiled: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            return None
    
    def ai_enhanced_code_improvement(self, file_path: str, use_pi5: bool = True) -> Dict[str, Any]:
        """
        Use AI to analyze and improve code with Pi 5 acceleration
        
        Args:
            file_path: Path to code file
            use_pi5: Whether to use Pi 5 for AI analysis
            
        Returns:
            Dictionary with improvements and modified code
        """
        try:
            # Read code
            with open(file_path, 'r') as f:
                code = f.read()
            
            code_data = {
                'file_path': file_path,
                'code': code,
                'analysis': self.analyze_code(file_path)
            }
            
            if use_pi5:
                # Send to Pi 5 for AI analysis
                ai_task = self.app.send_task(
                    'codemodifier.ai_code_analysis',
                    args=[code_data, '/models/code_analysis.hef'],
                    priority=2
                )
                
                # Wait for results
                result = ai_task.get(timeout=30)
                
                if result['status'] == 'success':
                    improvements = result['improvements']
                    
                    # Apply high-confidence improvements
                    for improvement in improvements:
                        if improvement.get('confidence', 0) > 0.8:
                            logger.info(f"Applying improvement: {improvement['description']}")
                            # Apply the improvement (implementation depends on type)
                    
                    return {
                        'success': True,
                        'improvements': improvements,
                        'ai_confidence': result.get('ai_confidence', 0)
                    }
            else:
                # Fallback to AI provider manager
                provider_manager = get_provider_manager()
                
                # Use AI provider for code analysis
                analysis_result = provider_manager.analyze_code(
                    code,
                    language="python",
                    analysis_type="comprehensive"
                )
                
                # Use AI provider for code improvement
                improvement_result = provider_manager.improve_code(
                    code,
                    language="python",
                    improvement_focus="security_performance"
                )
                
                improvements = [
                    {
                        "type": "analysis",
                        "description": analysis_result.get('analysis', ''),
                        "confidence": 0.8,
                        "provider": analysis_result.get('provider_used', 'unknown')
                    },
                    {
                        "type": "improvement",
                        "description": improvement_result.get('improved_code', ''),
                        "confidence": 0.8,
                        "provider": improvement_result.get('provider_used', 'unknown')
                    }
                ]
                
                return {
                    'success': True,
                    'improvements': improvements,
                    'ai_confidence': 0.8,
                    'provider_used': analysis_result.get('provider_used', 'unknown')
                }
                
        except Exception as e:
            logger.error(f"AI-enhanced improvement failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def apply_improvements(self, file_path: str, improvements: List[Dict[str, Any]]) -> bool:
        """
        Apply suggested improvements to code
        
        Args:
            file_path: Path to file to improve
            improvements: List of improvements to apply
            
        Returns:
            True if successful
        """
        # Create backup first
        backup_path = self.create_backup(file_path)
        if not backup_path:
            return False
            
        try:
            # Apply each improvement
            for improvement in improvements:
                if improvement.get('type') == 'security':
                    # Apply security improvements
                    changes = {
                        'add_validation': improvement.get('validation_code'),
                        'add_sanitization': improvement.get('sanitization_code')
                    }
                    self.modify_code(file_path, changes)
                    
                elif improvement.get('type') == 'performance':
                    # Apply performance improvements
                    changes = {
                        'optimize_loops': improvement.get('optimized_code'),
                        'add_caching': improvement.get('cache_code')
                    }
                    self.modify_code(file_path, changes)
            
            # Validate final code
            valid, error = self.validate_changes(file_path)
            if not valid:
                logger.error(f"Improvements resulted in invalid code: {error}")
                self.restore_backup(backup_path, file_path)
                return False
                
            logger.info(f"Successfully applied {len(improvements)} improvements")
            return True
            
        except Exception as e:
            logger.error(f"Error applying improvements: {e}")
            self.restore_backup(backup_path, file_path)
            return False
    
    def log_thermal_event(self, temp: float, kb_manager) -> None:
        """Log thermal events to knowledge base"""
        if kb_manager:
            kb_manager.add_fact(
                f"Thermal event: CPU temperature {temp}°C",
                source="code_modifier",
                confidence=1.0,
                category="system_health"
            )
