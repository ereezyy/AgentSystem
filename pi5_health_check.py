#!/usr/bin/env python3
"""
AgentSystem Pi5 Worker Health Check Script

This script monitors the health of the Pi5 Celery worker and system resources.
It can be run as a standalone script or integrated with monitoring systems.

Usage:
    python3 pi5_health_check.py [--verbose] [--json] [--alert-threshold]
    
Returns:
    Exit code 0: All systems healthy
    Exit code 1: Warning conditions detected
    Exit code 2: Critical conditions detected
"""

import os
import sys
import json
import time
import psutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Add the AgentSystem directory to Python path
sys.path.insert(0, '/opt/agentsystem')

try:
    import redis
    from celery import Celery
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(2)


class Pi5HealthChecker:
    """Health checker for Pi5 AgentSystem worker"""
    
    def __init__(self, config_path="/opt/agentsystem/.env"):
        self.config_path = config_path
        self.load_config()
        self.health_data = {
            "timestamp": datetime.now().isoformat(),
            "hostname": psutil.boot_time(),
            "status": "unknown",
            "checks": {},
            "alerts": [],
            "system_info": {}
        }
        
    def load_config(self):
        """Load configuration from .env file"""
        self.config = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        self.config[key] = value
        
        # Set defaults
        self.redis_url = self.config.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
        self.cpu_threshold = float(self.config.get('CPU_TEMP_THRESHOLD', '75'))
        self.memory_threshold = float(self.config.get('MEMORY_THRESHOLD', '85'))
        self.disk_threshold = float(self.config.get('DISK_THRESHOLD', '90'))
        
    def check_system_resources(self):
        """Check CPU, memory, disk, and temperature"""
        checks = {}
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        checks['cpu_usage'] = {
            'value': cpu_percent,
            'threshold': 90,
            'status': 'ok' if cpu_percent < 90 else 'warning' if cpu_percent < 95 else 'critical',
            'unit': '%'
        }
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        checks['memory_usage'] = {
            'value': memory_percent,
            'threshold': self.memory_threshold,
            'status': 'ok' if memory_percent < self.memory_threshold else 'warning' if memory_percent < 95 else 'critical',
            'unit': '%',
            'available_mb': memory.available // (1024*1024)
        }
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        checks['disk_usage'] = {
            'value': disk_percent,
            'threshold': self.disk_threshold,
            'status': 'ok' if disk_percent < self.disk_threshold else 'warning' if disk_percent < 95 else 'critical',
            'unit': '%',
            'free_gb': disk.free // (1024*1024*1024)
        }
        
        # CPU Temperature (Pi5 specific)
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_raw = int(f.read().strip())
                cpu_temp = temp_raw / 1000.0
                checks['cpu_temperature'] = {
                    'value': cpu_temp,
                    'threshold': self.cpu_threshold,
                    'status': 'ok' if cpu_temp < self.cpu_threshold else 'warning' if cpu_temp < 80 else 'critical',
                    'unit': '°C'
                }
        except Exception as e:
            checks['cpu_temperature'] = {
                'value': None,
                'status': 'error',
                'error': str(e)
            }
        
        # Load average
        load_avg = os.getloadavg()
        cpu_count = psutil.cpu_count()
        load_percent = (load_avg[0] / cpu_count) * 100
        checks['load_average'] = {
            'value': load_avg[0],
            'load_percent': load_percent,
            'cpu_count': cpu_count,
            'status': 'ok' if load_percent < 80 else 'warning' if load_percent < 100 else 'critical'
        }
        
        return checks
    
    def check_redis_connection(self):
        """Check Redis broker connection"""
        try:
            r = redis.from_url(self.redis_url)
            r.ping()
            return {
                'status': 'ok',
                'connected': True,
                'redis_info': {
                    'used_memory_human': r.info().get('used_memory_human', 'unknown'),
                    'connected_clients': r.info().get('connected_clients', 0)
                }
            }
        except Exception as e:
            return {
                'status': 'critical',
                'connected': False,
                'error': str(e)
            }
    
    def check_celery_worker(self):
        """Check Celery worker status"""
        try:
            # Create Celery app instance
            app = Celery('AgentSystem')
            app.config_from_object({
                'broker_url': self.redis_url,
                'result_backend': self.redis_url,
            })
            
            # Get worker stats
            inspect = app.control.inspect()
            stats = inspect.stats()
            active = inspect.active()
            
            if stats:
                worker_name = list(stats.keys())[0] if stats else None
                worker_stats = stats.get(worker_name, {}) if worker_name else {}
                active_tasks = active.get(worker_name, []) if active and worker_name else []
                
                return {
                    'status': 'ok',
                    'worker_online': True,
                    'worker_name': worker_name,
                    'active_tasks': len(active_tasks),
                    'pool_processes': worker_stats.get('pool', {}).get('processes', 0),
                    'total_tasks': worker_stats.get('total', 0)
                }
            else:
                return {
                    'status': 'critical',
                    'worker_online': False,
                    'error': 'No workers found'
                }
                
        except Exception as e:
            return {
                'status': 'critical',
                'worker_online': False,
                'error': str(e)
            }
    
    def check_agentsystem_process(self):
        """Check if AgentSystem worker process is running"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'celery' in cmdline and 'AgentSystem.pi5_worker' in cmdline:
                    return {
                        'status': 'ok',
                        'process_running': True,
                        'pid': proc.info['pid'],
                        'process_name': proc.info['name']
                    }
            
            return {
                'status': 'critical',
                'process_running': False,
                'error': 'AgentSystem worker process not found'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'process_running': False,
                'error': str(e)
            }
    
    def check_log_files(self):
        """Check log file status and recent errors"""
        log_dir = Path('/opt/agentsystem/logs')
        checks = {}
        
        if log_dir.exists():
            log_files = list(log_dir.glob('*.log'))
            checks['log_files_found'] = len(log_files)
            
            # Check most recent log file
            if log_files:
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                checks['latest_log_file'] = str(latest_log)
                checks['latest_log_size'] = latest_log.stat().st_size
                checks['latest_log_modified'] = datetime.fromtimestamp(
                    latest_log.stat().st_mtime
                ).isoformat()
                
                # Quick scan for recent errors
                try:
                    with open(latest_log, 'r') as f:
                        # Read last 100 lines
                        lines = f.readlines()[-100:]
                        error_lines = [line for line in lines if 'ERROR' in line or 'CRITICAL' in line]
                        checks['recent_errors'] = len(error_lines)
                        if error_lines:
                            checks['latest_error'] = error_lines[-1].strip()
                except Exception as e:
                    checks['log_read_error'] = str(e)
        else:
            checks['log_directory_exists'] = False
            
        return checks
    
    def run_health_check(self):
        """Run all health checks"""
        self.health_data['checks']['system_resources'] = self.check_system_resources()
        self.health_data['checks']['redis'] = self.check_redis_connection()
        self.health_data['checks']['celery_worker'] = self.check_celery_worker()
        self.health_data['checks']['process'] = self.check_agentsystem_process()
        self.health_data['checks']['logs'] = self.check_log_files()
        
        # Determine overall status
        critical_issues = []
        warning_issues = []
        
        for check_name, check_data in self.health_data['checks'].items():
            if isinstance(check_data, dict):
                if check_data.get('status') == 'critical':
                    critical_issues.append(f"{check_name}: {check_data.get('error', 'Critical issue')}")
                elif check_data.get('status') == 'warning':
                    warning_issues.append(f"{check_name}: Warning condition")
                    
                # Check nested checks (like system resources)
                for subcheck_name, subcheck_data in check_data.items():
                    if isinstance(subcheck_data, dict) and subcheck_data.get('status'):
                        if subcheck_data.get('status') == 'critical':
                            critical_issues.append(f"{check_name}.{subcheck_name}: {subcheck_data.get('error', 'Critical')}")
                        elif subcheck_data.get('status') == 'warning':
                            warning_issues.append(f"{check_name}.{subcheck_name}: Warning")
        
        self.health_data['alerts'] = critical_issues + warning_issues
        
        if critical_issues:
            self.health_data['status'] = 'critical'
            return 2
        elif warning_issues:
            self.health_data['status'] = 'warning'
            return 1
        else:
            self.health_data['status'] = 'healthy'
            return 0
    
    def get_system_info(self):
        """Get basic system information"""
        return {
            'hostname': os.uname().nodename,
            'platform': os.uname().sysname,
            'architecture': os.uname().machine,
            'kernel': os.uname().release,
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            'uptime_seconds': time.time() - psutil.boot_time()
        }


def main():
    parser = argparse.ArgumentParser(description='AgentSystem Pi5 Worker Health Check')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--json', '-j', action='store_true', help='JSON output')
    parser.add_argument('--config', '-c', default='/opt/agentsystem/.env', help='Config file path')
    
    args = parser.parse_args()
    
    checker = Pi5HealthChecker(args.config)
    checker.health_data['system_info'] = checker.get_system_info()
    
    exit_code = checker.run_health_check()
    
    if args.json:
        print(json.dumps(checker.health_data, indent=2))
    else:
        # Human-readable output
        status_symbols = {'healthy': '✅', 'warning': '⚠️', 'critical': '❌', 'error': '💥'}
        symbol = status_symbols.get(checker.health_data['status'], '❓')
        
        print(f"{symbol} AgentSystem Pi5 Worker Health: {checker.health_data['status'].upper()}")
        print(f"Timestamp: {checker.health_data['timestamp']}")
        print(f"Hostname: {checker.health_data['system_info']['hostname']}")
        
        if args.verbose or checker.health_data['status'] != 'healthy':
            print("\nDetailed Checks:")
            for check_name, check_data in checker.health_data['checks'].items():
                print(f"  {check_name}: {check_data.get('status', 'unknown')}")
                
        if checker.health_data['alerts']:
            print(f"\nAlerts ({len(checker.health_data['alerts'])}):")
            for alert in checker.health_data['alerts']:
                print(f"  - {alert}")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()