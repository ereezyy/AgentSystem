"""
Social Engineering Tester Module for ethical penetration testing

This module handles social engineering tests with distributed processing:
- Primary CPU: Orchestration and non-AI tasks
- Raspberry Pi 5: AI-intensive NLP tasks using Hailo-8 accelerator
"""

import json
import logging
import time
import zlib
import smtplib
import re
from typing import Dict, List, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from celery import Celery
import psutil
import subprocess

from AgentSystem.utils.logger import setup_logger

logger = setup_logger('social_engineering_tester')


class SocialEngineeringTester:
    """
    Social engineering testing module with AI-enhanced capabilities
    Leverages Raspberry Pi 5 with AI HAT+ for phishing email generation
    """
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.app = Celery('social_engineering', 
                         broker=f'redis://{redis_host}:{redis_port}/0',
                         backend=f'redis://{redis_host}:{redis_port}/0')
        
        # Configure task priorities
        self.app.conf.task_routes = {
            'social.ai_phishing_generation': {'priority': 1},
            'social.template_processing': {'priority': 2}
        }
        
        # Thermal and RAM monitoring
        self.thermal_limit = 80.0  # Celsius
        self.ram_limit = 7.0  # GB
        
        # Email templates for fallback
        self.email_templates = {
            'it_update': {
                'subject': 'Important: System Security Update Required',
                'body': """Dear {name},

Our IT department has detected a critical security vulnerability in your account. 
To protect your data, please click the link below to update your security settings:

{link}

This update must be completed within 24 hours to avoid account suspension.

Best regards,
IT Security Team"""
            },
            'password_reset': {
                'subject': 'Password Reset Request',
                'body': """Dear {name},

We received a request to reset your password. If you did not make this request, 
please click the link below immediately to secure your account:

{link}

This link will expire in 1 hour.

Security Team"""
            }
        }
        
        logger.info("SocialEngineeringTester initialized with distributed processing")
    
    def compress_task_data(self, data: str) -> bytes:
        """Compress data for efficient network transfer"""
        return zlib.compress(data.encode())
    
    def decompress_task_data(self, data: bytes) -> str:
        """Decompress received data"""
        return zlib.decompress(data).decode()
    
    def monitor_ram(self, max_usage: float = None) -> bool:
        """Monitor RAM usage to prevent system overload"""
        if max_usage is None:
            max_usage = self.ram_limit
            
        ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # GB
        if ram_usage > max_usage:
            logger.warning(f"RAM usage {ram_usage:.2f}GB exceeds limit {max_usage}GB")
            return False
        return True
    
    def check_thermal(self) -> bool:
        """Check thermal status on Raspberry Pi 5"""
        try:
            # This works on Raspberry Pi
            temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
            temp = float(temp_output.split("=")[1].split("'")[0])
            
            if temp > self.thermal_limit:
                logger.warning(f"CPU temperature {temp}°C exceeds limit {self.thermal_limit}°C")
                return False
            return True
        except:
            # If not on Raspberry Pi, assume thermal is OK
            return True
    
    def safe_execute(self, task: callable, *args, **kwargs) -> Any:
        """Execute task with thermal and RAM safety checks"""
        if not self.check_thermal():
            logger.warning("Pausing task due to thermal limit")
            time.sleep(60)  # Cool down period
            return None
            
        if not self.monitor_ram():
            logger.warning("Pausing task due to RAM limit")
            time.sleep(30)  # Wait for RAM to free up
            return None
            
        return task(*args, **kwargs)
    
    def validate_input(self, data: str) -> bool:
        """Validate input data for security"""
        # Check for potential injection attempts
        dangerous_patterns = [
            r'[<>{}]',  # HTML/script tags
            r';\s*rm\s+-rf',  # Shell injection
            r'DROP\s+TABLE',  # SQL injection
            r'eval\s*\(',  # Code execution
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                return False
        return True
    
    def ai_phishing_simulation(self, target_email: str, scenario: str, 
                             model_path: str = "/models/nlp_phishing.hef",
                             test_mode: bool = True) -> Dict[str, Any]:
        """
        Generate and optionally send AI-enhanced phishing email
        1. Validate target and scenario
        2. Send to Pi 5 for AI generation
        3. Return phishing email and metrics
        """
        result = {
            'success': False,
            'target': target_email,
            'scenario': scenario,
            'email_content': None,
            'metrics': {}
        }
        
        try:
            # Step 1: Validate inputs
            if not self.validate_input(target_email) or not self.validate_input(scenario):
                result['error'] = 'Invalid input detected'
                return result
            
            # Prepare context for AI generation
            context = {
                'target_email': target_email,
                'scenario': scenario,
                'personalization': self._gather_target_info(target_email),
                'timestamp': time.time()
            }
            
            # Step 2: Compress and send to Pi 5 for AI generation
            compressed_data = self.compress_task_data(json.dumps(context))
            
            # Queue AI task for Pi 5
            ai_task = self.app.send_task(
                'social.ai_phishing_generation',
                args=[compressed_data, model_path],
                priority=1
            )
            
            # Wait for AI generation with timeout
            try:
                ai_result = ai_task.get(timeout=30)
                
                if ai_result['status'] == 'success':
                    email_content = ai_result['email_content']
                    result['email_content'] = email_content
                    result['metrics'] = ai_result.get('metrics', {})
                    
                    # Step 3: Send email if not in test mode
                    if not test_mode:
                        send_result = self._send_phishing_email(
                            target_email, 
                            email_content['subject'],
                            email_content['body']
                        )
                        result['sent'] = send_result
                    else:
                        result['sent'] = False
                        result['test_mode'] = True
                    
                    result['success'] = True
                    logger.info(f"Phishing simulation complete for {target_email}")
                    
                else:
                    # Fallback to template
                    result = self._fallback_phishing_generation(target_email, scenario)
                    
            except Exception as e:
                logger.error(f"AI generation failed: {e}")
                # Fallback to template
                result = self._fallback_phishing_generation(target_email, scenario)
                
        except Exception as e:
            logger.error(f"Phishing simulation failed: {e}")
            result['error'] = str(e)
            
        return result
    
    def _gather_target_info(self, email: str) -> Dict[str, str]:
        """Gather information about target for personalization"""
        info = {
            'email': email,
            'name': email.split('@')[0].replace('.', ' ').title(),
            'domain': email.split('@')[1] if '@' in email else '',
            'timestamp': time.strftime('%Y-%m-%d')
        }
        
        # In a real scenario, this could query OSINT sources
        # For now, we'll use basic extraction
        
        return info
    
    def _fallback_phishing_generation(self, target_email: str, scenario: str) -> Dict[str, Any]:
        """Generate phishing email using templates when AI is unavailable"""
        template_key = 'it_update'  # Default
        
        # Map scenarios to templates
        scenario_mapping = {
            'it update': 'it_update',
            'password': 'password_reset',
            'security': 'it_update',
            'reset': 'password_reset'
        }
        
        for key, value in scenario_mapping.items():
            if key in scenario.lower():
                template_key = value
                break
                
        template = self.email_templates[template_key]
        target_info = self._gather_target_info(target_email)
        
        # Generate phishing link (in real scenario, this would be a controlled domain)
        phishing_link = f"https://secure-update.example.com/verify?user={target_info['email']}&token=DEMO"
        
        email_content = {
            'subject': template['subject'],
            'body': template['body'].format(
                name=target_info['name'],
                link=phishing_link
            )
        }
        
        return {
            'success': True,
            'target': target_email,
            'scenario': scenario,
            'email_content': email_content,
            'metrics': {
                'method': 'template',
                'confidence': 0.6
            },
            'test_mode': True
        }
    
    def _send_phishing_email(self, to_email: str, subject: str, body: str,
                           smtp_server: str = "localhost", smtp_port: int = 587) -> bool:
        """
        Send phishing email (for authorized testing only)
        Note: This should only be used with proper authorization and in controlled environments
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = "security@example.com"  # Spoofed sender
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Note: In a real implementation, this would use proper SMTP configuration
            # and only send to authorized test accounts
            logger.info(f"Email would be sent to {to_email} (simulated)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def spear_phishing_campaign(self, targets: List[Dict[str, str]], 
                              campaign_name: str) -> Dict[str, Any]:
        """
        Run a spear phishing campaign against multiple targets
        Each target can have personalized content
        """
        campaign_results = {
            'campaign_name': campaign_name,
            'start_time': time.time(),
            'targets': len(targets),
            'successful': 0,
            'failed': 0,
            'results': []
        }
        
        for target in targets:
            email = target.get('email')
            scenario = target.get('scenario', 'it update')
            
            # Run phishing simulation for each target
            result = self.ai_phishing_simulation(email, scenario, test_mode=True)
            
            if result['success']:
                campaign_results['successful'] += 1
            else:
                campaign_results['failed'] += 1
                
            campaign_results['results'].append(result)
            
            # Add delay between emails to avoid detection
            time.sleep(2)
            
        campaign_results['end_time'] = time.time()
        campaign_results['duration'] = campaign_results['end_time'] - campaign_results['start_time']
        
        return campaign_results
    
    def voice_phishing_script(self, target_info: Dict[str, str], 
                            scenario: str = "tech support") -> Dict[str, Any]:
        """
        Generate voice phishing (vishing) script using AI
        """
        script_data = {
            'target': target_info,
            'scenario': scenario,
            'script': None,
            'talking_points': []
        }
        
        # This would use AI to generate realistic conversation scripts
        # For now, provide a template
        
        if scenario == "tech support":
            script_data['script'] = f"""
Opening: "Hello, this is [Name] from the IT Security Department. 
I'm calling about a security alert on your account {target_info.get('email', 'account')}.

Establish Trust: "I see here that there have been several failed login attempts 
from an IP address in [Foreign Country]. Were you trying to access your account from there?"

Create Urgency: "To prevent unauthorized access, I need to verify your identity 
and help you secure your account immediately."

Information Gathering: "Can you please verify the last four digits of your 
employee ID and your department?"

Payload: "I'm going to send you a secure verification code to your email. 
Please read it back to me so I can confirm your identity."
"""
            
            script_data['talking_points'] = [
                "Remain calm and professional",
                "Use technical jargon sparingly",
                "Mirror the target's speech patterns",
                "Have responses ready for common objections",
                "Never break character"
            ]
            
        return script_data
    
    def generate_report(self, results: List[Dict[str, Any]], 
                       format: str = 'json') -> str:
        """Generate social engineering test report"""
        report_data = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(results),
            'successful': sum(1 for r in results if r.get('success', False)),
            'failed': sum(1 for r in results if not r.get('success', False)),
            'techniques_used': {
                'phishing_emails': 0,
                'spear_phishing': 0,
                'vishing': 0
            },
            'results': results
        }
        
        # Analyze techniques used
        for result in results:
            if 'email_content' in result:
                if result.get('personalized', False):
                    report_data['techniques_used']['spear_phishing'] += 1
                else:
                    report_data['techniques_used']['phishing_emails'] += 1
            elif 'script' in result:
                report_data['techniques_used']['vishing'] += 1
                
        if format == 'json':
            return json.dumps(report_data, indent=2)
        elif format == 'text':
            report = f"Social Engineering Test Report\n"
            report += f"==============================\n"
            report += f"Date: {report_data['test_date']}\n"
            report += f"Total Tests: {report_data['total_tests']}\n"
            report += f"Successful: {report_data['successful']}\n"
            report += f"Failed: {report_data['failed']}\n\n"
            report += f"Techniques Used:\n"
            for technique, count in report_data['techniques_used'].items():
                report += f"  {technique.replace('_', ' ').title()}: {count}\n"
                
            return report
        else:
            raise ValueError(f"Unsupported report format: {format}")


# Celery worker tasks for Raspberry Pi 5
if __name__ == "__main__":
    # This section runs on the Raspberry Pi 5
    from celery import Celery
    import subprocess
    
    app = Celery('social', broker='redis://primary_cpu_ip:6379/0')
    
    @app.task(name='social.ai_phishing_generation')
    def ai_phishing_generation(compressed_data: bytes, model_path: str) -> Dict[str, Any]:
        """
        AI task to generate phishing emails using NLP model
        Runs on Raspberry Pi 5 with AI HAT+
        """
        try:
            # Check thermal before processing
            temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
            temp = float(temp_output.split("=")[1].split("'")[0])
            if temp > 80:
                raise Exception(f"Thermal limit exceeded: {temp}°C")
                
            # Decompress data
            context = json.loads(zlib.decompress(compressed_data).decode())
            
            # Initialize Hailo accelerator (mock for now)
            # In real implementation, use hailo_sdk with NLP model
            
            # Extract context
            target_email = context['target_email']
            scenario = context['scenario']
            personalization = context['personalization']
            
            # Generate AI-powered phishing email
            # This would use a fine-tuned language model
            email_content = {
                'subject': f"Urgent: {scenario.title()} Required",
                'body': f"""Dear {personalization['name']},

We have detected an issue with your account that requires immediate attention. 
Our {scenario} process has identified your account for mandatory verification.

Please click the following link to complete the {scenario} process:
https://secure-verify.example.com/auth?user={target_email}&ref={int(time.time())}

This action must be completed within 24 hours to avoid service interruption.

Thank you for your prompt attention to this matter.

Best regards,
Security Team
{personalization['domain'].capitalize()} IT Department"""
            }
            
            # Calculate metrics
            metrics = {
                'personalization_score': 0.85,
                'urgency_level': 0.9,
                'authenticity_score': 0.8,
                'ai_confidence': 0.87,
                'generation_time': 0.15  # seconds
            }
            
            return {
                'status': 'success',
                'email_content': email_content,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"AI phishing generation error: {e}")
            return {'status': 'error', 'message': str(e)}
