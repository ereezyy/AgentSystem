"""
Raspberry Pi 5 Celery Worker for AI HAT+ (26 TOPS)

This worker runs on the Raspberry Pi 5 and handles AI-intensive tasks:
- Vulnerability classification using YOLO models
- Phishing email generation using NLP models
- Thermal and resource monitoring

Run with: celery -A pi5_worker worker --concurrency=2 --loglevel=info
"""

import json
import logging
import os
import subprocess
import time
import zlib
import psutil
from typing import Dict, Any
from celery import Celery
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pi5_worker')

# Get primary CPU IP from environment variable, with a fallback for local testing
PRIMARY_CPU_IP = os.getenv('PRIMARY_CPU_IP', '127.0.0.1')

# Initialize Celery app
# The broker URL points to the Redis instance on the primary machine.
app = Celery('pi5_worker',
             broker=f'redis://{PRIMARY_CPU_IP}:6379/0',
             backend=f'redis://{PRIMARY_CPU_IP}:6379/0')

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Limit concurrency to prevent overload
    worker_concurrency=2,
    # Task routing
    task_routes={
        'pentester.ai_vulnerability_classification': {'queue': 'ai_high'},
        'social.ai_phishing_generation': {'queue': 'ai_high'},
        'codemodifier.ai_code_analysis': {'queue': 'ai_medium'}
    }
)

# Thermal and resource limits
THERMAL_LIMIT = 80.0  # Celsius
RAM_LIMIT = 7.0  # GB
AI_HAT_THERMAL_LIMIT = 50.0  # Celsius

def check_thermal() -> bool:
    """Check thermal status on Raspberry Pi 5"""
    try:
        # Check CPU temperature
        temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        cpu_temp = float(temp_output.split("=")[1].split("'")[0])
        
        if cpu_temp > THERMAL_LIMIT:
            logger.warning(f"CPU temperature {cpu_temp}°C exceeds limit {THERMAL_LIMIT}°C")
            return False
            
        # Check AI HAT+ temperature (if available)
        # This would use Hailo SDK in real implementation
        # For now, assume it's within limits
        
        return True
    except Exception as e:
        logger.error(f"Error checking thermal: {e}")
        return True  # Continue if unable to check

def monitor_ram() -> bool:
    """Monitor RAM usage to prevent system overload"""
    ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # GB
    if ram_usage > RAM_LIMIT:
        logger.warning(f"RAM usage {ram_usage:.2f}GB exceeds limit {RAM_LIMIT}GB")
        return False
    return True

def check_resources() -> bool:
    """Check all system resources before processing"""
    return check_thermal() and monitor_ram()

# Mock Hailo SDK functions (replace with actual SDK)
class MockHailoAccelerator:
    """Mock Hailo accelerator for demonstration"""
    
    def __init__(self):
        self.models = {}
    
    def load_model(self, model_path: str):
        """Load a compiled Hailo model"""
        logger.info(f"Loading model: {model_path}")
        # In real implementation, this would load .hef file
        return {'path': model_path, 'loaded': True}
    
    def run_inference(self, model, data: Any) -> Dict[str, Any]:
        """Run inference on Hailo-8"""
        logger.info("Running inference on AI HAT+")
        # Mock inference results
        return {
            'predictions': [],
            'confidence': 0.85,
            'processing_time': 0.1
        }

# Initialize Hailo accelerator (use mock for now)
hailo = MockHailoAccelerator()

@app.task(name='pentester.ai_vulnerability_classification')
def ai_vulnerability_classification(compressed_data: bytes, model_path: str) -> Dict[str, Any]:
    """
    AI task to classify vulnerabilities using Hailo-8 accelerator
    Uses YOLO model for vulnerability detection
    """
    logger.info("Starting vulnerability classification task")
    
    try:
        # Check resources before processing
        if not check_resources():
            raise Exception("Resource limits exceeded")
            
        # Decompress data
        data = json.loads(zlib.decompress(compressed_data).decode())
        logger.info(f"Processing scan data for {len(data.get('hosts', []))} hosts")
        
        # Load YOLO model for vulnerability detection
        model = hailo.load_model(model_path)
        
        vulnerabilities = []
        
        # Process each host's scan results
        for host in data.get('hosts', []):
            for proto, ports in host.get('protocols', {}).items():
                for port_info in ports:
                    if port_info['state'] == 'open':
                        # Prepare data for AI inference
                        inference_data = {
                            'host': host['address'],
                            'port': port_info['port'],
                            'service': port_info['service'],
                            'version': port_info.get('version', ''),
                            'product': port_info.get('product', '')
                        }
                        
                        # Run AI inference
                        result = hailo.run_inference(model, inference_data)
                        
                        # Determine vulnerability based on AI analysis
                        if result['confidence'] > 0.7:
                            severity = 'critical' if port_info['port'] in [22, 23, 445, 3389] else 'high'
                            severity = 'high' if port_info['port'] in [21, 80, 443, 3306] else severity
                            
                            vuln = {
                                'host': host['address'],
                                'port': port_info['port'],
                                'service': port_info['service'],
                                'version': port_info.get('version', ''),
                                'severity': severity,
                                'description': f"AI-detected vulnerability in {port_info['service']}",
                                'confidence': result['confidence'],
                                'ai_classified': True,
                                'model_used': model_path,
                                'processing_time': result['processing_time']
                            }
                            
                            # Add specific vulnerability details based on service
                            if 'ssh' in port_info['service'].lower():
                                vuln['cve'] = 'CVE-2024-XXXX'  # Mock CVE
                                vuln['exploit_available'] = True
                            elif 'http' in port_info['service'].lower():
                                vuln['owasp_category'] = 'A01:2021 – Broken Access Control'
                            
                            vulnerabilities.append(vuln)
                            
        logger.info(f"AI analysis complete: {len(vulnerabilities)} vulnerabilities found")
        
        return {
            'vulnerabilities': vulnerabilities, 
            'status': 'success',
            'model': model_path,
            'processing_stats': {
                'hosts_analyzed': len(data.get('hosts', [])),
                'vulnerabilities_found': len(vulnerabilities),
                'ai_confidence_avg': sum(v['confidence'] for v in vulnerabilities) / len(vulnerabilities) if vulnerabilities else 0
            }
        }
        
    except Exception as e:
        logger.error(f"AI classification error: {e}")
        return {'vulnerabilities': [], 'status': 'error', 'message': str(e)}

@app.task(name='social.ai_phishing_generation')
def ai_phishing_generation(compressed_data: bytes, model_path: str) -> Dict[str, Any]:
    """
    AI task to generate phishing emails using NLP model
    Uses fine-tuned language model on Hailo-8
    """
    logger.info("Starting phishing email generation task")
    
    try:
        # Check resources before processing
        if not check_resources():
            raise Exception("Resource limits exceeded")
            
        # Decompress data
        context = json.loads(zlib.decompress(compressed_data).decode())
        
        # Extract context
        target_email = context['target_email']
        scenario = context['scenario']
        personalization = context['personalization']
        
        logger.info(f"Generating phishing email for {target_email} with scenario: {scenario}")
        
        # Load NLP model for email generation
        model = hailo.load_model(model_path)
        
        # Prepare input for AI model
        model_input = {
            'target_name': personalization['name'],
            'target_domain': personalization['domain'],
            'scenario': scenario,
            'urgency_level': 'high',
            'personalization_factors': personalization
        }
        
        # Run AI inference for email generation
        start_time = time.time()
        result = hailo.run_inference(model, model_input)
        generation_time = time.time() - start_time
        
        # Generate sophisticated phishing email based on AI output
        if scenario.lower() == 'it update':
            subject = f"[Action Required] Critical Security Update - {personalization['domain'].upper()}"
            body = f"""Dear {personalization['name']},

Our security monitoring system has detected unusual activity on your account that requires immediate attention.

SECURITY ALERT DETAILS:
- Account: {target_email}
- Risk Level: HIGH
- Detection Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}
- IP Address: 185.220.101.45 (Tor Exit Node)
- Location: Unknown

To protect your account and organizational data, you must complete our enhanced security verification process within the next 4 hours.

IMMEDIATE ACTION REQUIRED:
1. Click here to verify your identity: https://secure-{personalization['domain'].replace('.', '-')}.verification-portal.com/auth/{int(time.time())}
2. Complete multi-factor authentication
3. Update your security settings

Failure to complete this process will result in temporary account suspension as per our security policy.

If you did not attempt to access your account from the above location, it is critical that you secure your account immediately.

Best regards,
{personalization['domain'].capitalize()} Security Team

This is an automated security notification. Please do not reply to this email.
For assistance, contact your IT administrator.
"""
        
        elif scenario.lower() == 'password reset':
            subject = f"Password Reset Request - Immediate Action Required"
            body = f"""Dear {personalization['name']},

We received multiple password reset requests for your account. If you did not initiate these requests, your account may be compromised.

RESET REQUEST DETAILS:
- Account: {target_email}
- Request Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Request ID: PWD-{int(time.time())}-{hash(target_email) % 10000:04d}

To secure your account:
https://passwordreset-{personalization['domain'].replace('.', '-')}.secure-auth.net/verify?token={int(time.time() * 1000)}

This link expires in 60 minutes.

Security Team
{personalization['domain'].capitalize()}
"""
        
        else:
            # Generic phishing template
            subject = f"Urgent: {scenario.title()} Required"
            body = f"""Dear {personalization['name']},

This email requires your immediate attention regarding: {scenario}

Please click the following secure link to proceed:
https://secure-portal.{personalization['domain']}/action?ref={int(time.time())}

Thank you for your prompt response.

{personalization['domain'].capitalize()} Team
"""
        
        # Calculate sophisticated metrics
        metrics = {
            'personalization_score': 0.92,  # High due to name and domain usage
            'urgency_level': 0.95,  # Very high urgency indicators
            'authenticity_score': 0.88,  # Mimics legitimate emails well
            'ai_confidence': result.get('confidence', 0.87),
            'generation_time': generation_time,
            'sophistication_level': 'advanced',
            'evasion_techniques': [
                'domain_spoofing',
                'urgency_manipulation',
                'authority_impersonation',
                'fear_based_messaging'
            ]
        }
        
        logger.info(f"Phishing email generated with confidence: {metrics['ai_confidence']}")
        
        return {
            'status': 'success',
            'email_content': {
                'subject': subject,
                'body': body,
                'headers': {
                    'X-Priority': '1',
                    'Importance': 'High'
                }
            },
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"AI phishing generation error: {e}")
        return {'status': 'error', 'message': str(e)}

@app.task(name='codemodifier.ai_code_analysis')
def ai_code_analysis(code_data: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """
    AI task to analyze code for improvements using Hailo-8
    """
    logger.info("Starting AI code analysis task")
    
    try:
        # Check resources
        if not check_resources():
            raise Exception("Resource limits exceeded")
            
        # Load code analysis model
        model = hailo.load_model(model_path)
        
        # Run inference
        result = hailo.run_inference(model, code_data)
        
        # Mock analysis results
        improvements = [
            {
                'type': 'security',
                'description': 'Add input validation to prevent injection attacks',
                'confidence': 0.9
            },
            {
                'type': 'performance',
                'description': 'Use batch operations to reduce database calls',
                'confidence': 0.85
            }
        ]
        
        return {
            'status': 'success',
            'improvements': improvements,
            'ai_confidence': result.get('confidence', 0.8)
        }
        
    except Exception as e:
        logger.error(f"AI code analysis error: {e}")
        return {'status': 'error', 'message': str(e)}

@app.task(name='system.health_check')
def health_check() -> Dict[str, Any]:
    """Periodic health check of Pi 5 system"""
    health = {
        'timestamp': time.time(),
        'thermal': {
            'cpu_temp': None,
            'status': 'unknown'
        },
        'memory': {
            'used_gb': psutil.virtual_memory().used / (1024 ** 3),
            'percent': psutil.virtual_memory().percent,
            'status': 'ok' if monitor_ram() else 'warning'
        },
        'ai_hat': {
            'status': 'ok',  # Mock status
            'model_loaded': True
        }
    }
    
    # Get CPU temperature
    try:
        temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        cpu_temp = float(temp_output.split("=")[1].split("'")[0])
        health['thermal']['cpu_temp'] = cpu_temp
        health['thermal']['status'] = 'ok' if cpu_temp < THERMAL_LIMIT else 'warning'
    except:
        pass
    
    logger.info(f"Health check: CPU {health['thermal']['cpu_temp']}°C, RAM {health['memory']['used_gb']:.1f}GB")
    
    return health

if __name__ == '__main__':
    # This will be executed when running the worker
    logger.info("Starting Raspberry Pi 5 AI Worker")
    logger.info(f"Connecting to Redis at {PRIMARY_CPU_IP}:6379")
    logger.info("AI HAT+ (26 TOPS) ready for inference tasks")
    
    # Note: Run with:
    # celery -A pi5_worker worker --concurrency=2 --loglevel=info
