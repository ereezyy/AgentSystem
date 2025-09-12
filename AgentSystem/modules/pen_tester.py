"""
PenTester Module for ethical hacking with Raspberry Pi 5 AI HAT+ support

This module handles penetration testing tasks with distributed processing:
- Primary CPU: Orchestration and non-AI tasks
- Raspberry Pi 5: AI-intensive tasks using Hailo-8 accelerator
"""

import subprocess
import json
import logging
import time
import zlib
from typing import Dict, List, Any, Optional
from pathlib import Path
from celery import Celery
import nmap
import psutil

from AgentSystem.utils.logger import setup_logger

logger = setup_logger('pen_tester')


class PenTester:
    """
    Penetration testing module with AI-enhanced capabilities
    Leverages Raspberry Pi 5 with AI HAT+ for vulnerability classification
    """
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.app = Celery('pen_tester', 
                         broker=f'redis://{redis_host}:{redis_port}/0',
                         backend=f'redis://{redis_host}:{redis_port}/0')
        
        # Configure task priorities
        self.app.conf.task_routes = {
            'pentester.ai_vulnerability_classification': {'priority': 1},
            'pentester.nmap_scan': {'priority': 2}
        }
        
        # Initialize nmap scanner
        self.nm = nmap.PortScanner()
        
        # Thermal and RAM monitoring
        self.thermal_limit = 80.0  # Celsius
        self.ram_limit = 7.0  # GB
        
        logger.info("PenTester initialized with distributed processing")
    
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
    
    def ai_enhanced_scan(self, target: str, model_path: str = "/models/yolo_vuln.hef",
                        scan_type: str = "-sV") -> List[Dict[str, Any]]:
        """
        Perform AI-enhanced vulnerability scan
        1. Run Nmap scan on primary CPU
        2. Send results to Pi 5 for AI classification
        3. Return enhanced vulnerability report
        """
        vulnerabilities = []
        
        try:
            # Step 1: Run Nmap scan locally (primary CPU)
            logger.info(f"Starting Nmap scan on target: {target}")
            scan_result = self.safe_execute(self._perform_nmap_scan, target, scan_type)
            
            if not scan_result:
                return vulnerabilities
            
            # Step 2: Compress and send to Pi 5 for AI analysis
            compressed_data = self.compress_task_data(json.dumps(scan_result))
            
            # Queue AI task for Pi 5
            ai_task = self.app.send_task(
                'pentester.ai_vulnerability_classification',
                args=[compressed_data, model_path],
                priority=1
            )
            
            # Wait for AI analysis with timeout
            try:
                ai_result = ai_task.get(timeout=30)
                vulnerabilities = ai_result.get('vulnerabilities', [])
                
                # Step 3: Enhance scan results with AI insights
                for vuln in vulnerabilities:
                    vuln['scan_time'] = time.time()
                    vuln['source'] = 'ai_enhanced_scan'
                    vuln['model'] = model_path
                    
                logger.info(f"AI analysis complete: {len(vulnerabilities)} vulnerabilities found")
                
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
                # Fallback to basic scan results
                vulnerabilities = self._parse_basic_vulnerabilities(scan_result)
                
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            
        return vulnerabilities
    
    def _perform_nmap_scan(self, target: str, scan_type: str) -> Dict[str, Any]:
        """Execute Nmap scan and return results"""
        try:
            self.nm.scan(hosts=target, arguments=scan_type)
            
            scan_data = {
                'target': target,
                'scan_type': scan_type,
                'hosts': []
            }
            
            for host in self.nm.all_hosts():
                host_data = {
                    'address': host,
                    'state': self.nm[host].state(),
                    'protocols': {}
                }
                
                for proto in self.nm[host].all_protocols():
                    ports = self.nm[host][proto].keys()
                    host_data['protocols'][proto] = []
                    
                    for port in ports:
                        port_info = {
                            'port': port,
                            'state': self.nm[host][proto][port]['state'],
                            'service': self.nm[host][proto][port].get('name', ''),
                            'version': self.nm[host][proto][port].get('version', ''),
                            'product': self.nm[host][proto][port].get('product', '')
                        }
                        host_data['protocols'][proto].append(port_info)
                
                scan_data['hosts'].append(host_data)
                
            return scan_data
            
        except Exception as e:
            logger.error(f"Nmap scan error: {e}")
            raise
    
    def _parse_basic_vulnerabilities(self, scan_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse basic vulnerabilities from scan results without AI"""
        vulnerabilities = []
        
        # Common vulnerable services/versions
        vulnerable_services = {
            'ftp': ['vsftpd 2.3.4', 'ProFTPD 1.3.3'],
            'ssh': ['OpenSSH 4.', 'OpenSSH 5.'],
            'http': ['Apache/2.2.', 'nginx/1.0.'],
            'mysql': ['5.0.', '5.1.'],
            'smb': ['Samba 3.', 'Samba 4.0.']
        }
        
        for host in scan_result.get('hosts', []):
            for proto, ports in host.get('protocols', {}).items():
                for port_info in ports:
                    service = port_info.get('service', '')
                    version = port_info.get('version', '')
                    
                    # Check for known vulnerable versions
                    for vuln_service, vuln_versions in vulnerable_services.items():
                        if service.lower() == vuln_service:
                            for vuln_version in vuln_versions:
                                if vuln_version in version:
                                    vulnerabilities.append({
                                        'host': host['address'],
                                        'port': port_info['port'],
                                        'service': service,
                                        'version': version,
                                        'severity': 'high',
                                        'description': f'Potentially vulnerable {service} version detected',
                                        'confidence': 0.7
                                    })
                                    
        return vulnerabilities
    
    def exploit_vulnerability(self, vulnerability: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to exploit a discovered vulnerability (ethical testing only)
        This is a placeholder - real exploitation should be done with proper authorization
        """
        result = {
            'vulnerability': vulnerability,
            'exploited': False,
            'message': 'Exploitation simulation only - requires proper authorization',
            'recommendations': []
        }
        
        # Add recommendations based on vulnerability type
        service = vulnerability.get('service', '').lower()
        
        if 'ftp' in service:
            result['recommendations'].extend([
                'Update FTP server to latest version',
                'Disable anonymous FTP access',
                'Use SFTP/FTPS instead of plain FTP'
            ])
        elif 'ssh' in service:
            result['recommendations'].extend([
                'Update OpenSSH to latest version',
                'Disable password authentication',
                'Use key-based authentication only'
            ])
        elif 'http' in service:
            result['recommendations'].extend([
                'Update web server to latest version',
                'Enable HTTPS with strong ciphers',
                'Implement security headers (CSP, HSTS, etc.)'
            ])
            
        return result
    
    def generate_report(self, vulnerabilities: List[Dict[str, Any]], 
                       format: str = 'json') -> str:
        """Generate vulnerability report in specified format"""
        report_data = {
            'scan_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_vulnerabilities': len(vulnerabilities),
            'severity_breakdown': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'vulnerabilities': vulnerabilities
        }
        
        # Count vulnerabilities by severity
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'low').lower()
            if severity in report_data['severity_breakdown']:
                report_data['severity_breakdown'][severity] += 1
                
        if format == 'json':
            return json.dumps(report_data, indent=2)
        elif format == 'text':
            report = f"Vulnerability Scan Report\n"
            report += f"========================\n"
            report += f"Date: {report_data['scan_date']}\n"
            report += f"Total Vulnerabilities: {report_data['total_vulnerabilities']}\n\n"
            report += f"Severity Breakdown:\n"
            for severity, count in report_data['severity_breakdown'].items():
                report += f"  {severity.capitalize()}: {count}\n"
            report += f"\nDetailed Findings:\n"
            for i, vuln in enumerate(vulnerabilities, 1):
                report += f"\n{i}. {vuln.get('description', 'Unknown vulnerability')}\n"
                report += f"   Host: {vuln.get('host', 'N/A')}\n"
                report += f"   Port: {vuln.get('port', 'N/A')}\n"
                report += f"   Service: {vuln.get('service', 'N/A')}\n"
                report += f"   Severity: {vuln.get('severity', 'N/A')}\n"
                
            return report
        else:
            raise ValueError(f"Unsupported report format: {format}")


# Celery worker tasks for Raspberry Pi 5
if __name__ == "__main__":
    # This section runs on the Raspberry Pi 5
    from celery import Celery
    import subprocess
    
    app = Celery('pentester', broker='redis://primary_cpu_ip:6379/0')
    
    @app.task(name='pentester.ai_vulnerability_classification')
    def ai_vulnerability_classification(compressed_data: bytes, model_path: str) -> Dict[str, Any]:
        """
        AI task to classify vulnerabilities using Hailo-8 accelerator
        Runs on Raspberry Pi 5 with AI HAT+
        """
        try:
            # Check thermal before processing
            temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
            temp = float(temp_output.split("=")[1].split("'")[0])
            if temp > 80:
                raise Exception(f"Thermal limit exceeded: {temp}°C")
                
            # Decompress data
            data = json.loads(zlib.decompress(compressed_data).decode())
            
            # Initialize Hailo accelerator (mock for now)
            # In real implementation, use hailo_sdk
            vulnerabilities = []
            
            # Simulate AI analysis
            for host in data.get('hosts', []):
                for proto, ports in host.get('protocols', {}).items():
                    for port_info in ports:
                        # AI model would analyze port/service combinations
                        confidence = 0.85  # Mock confidence score
                        
                        if port_info['state'] == 'open':
                            vuln = {
                                'host': host['address'],
                                'port': port_info['port'],
                                'service': port_info['service'],
                                'version': port_info.get('version', ''),
                                'severity': 'high' if port_info['port'] in [21, 22, 23, 445] else 'medium',
                                'description': f"Potential vulnerability in {port_info['service']}",
                                'confidence': confidence,
                                'ai_classified': True
                            }
                            vulnerabilities.append(vuln)
                            
            return {'vulnerabilities': vulnerabilities, 'status': 'success'}
            
        except Exception as e:
            logger.error(f"AI classification error: {e}")
            return {'vulnerabilities': [], 'status': 'error', 'message': str(e)}
