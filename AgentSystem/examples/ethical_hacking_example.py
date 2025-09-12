"""
Example: Ethical Hacking with Raspberry Pi 5 AI HAT+ Integration

This example demonstrates the distributed architecture with:
- Primary CPU: Orchestration and database management
- Raspberry Pi 5: AI-intensive tasks using 26 TOPS accelerator

Requirements:
- Redis server running on primary CPU
- PostgreSQL for scalable knowledge storage
- Raspberry Pi 5 with AI HAT+ configured
"""

import time
import logging
from pathlib import Path

from AgentSystem.modules.pen_tester import PenTester
from AgentSystem.modules.social_engineering_tester import SocialEngineeringTester
from AgentSystem.modules.knowledge_manager import KnowledgeManager
from AgentSystem.modules.code_modifier import CodeModifier
from AgentSystem.modules.learning_agent import LearningAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PRIMARY_CPU_IP = "192.168.1.100"  # Replace with actual IP
PI5_IP = "192.168.1.150"  # Replace with Pi 5 IP
REDIS_PORT = 6379

def main():
    """
    Main function demonstrating ethical hacking with Pi 5 integration
    """
    logger.info("Starting Ethical Hacking System with Raspberry Pi 5 AI HAT+")
    
    # 1. Initialize Knowledge Manager with PostgreSQL
    logger.info("Initializing Knowledge Manager with PostgreSQL...")
    kb_manager = KnowledgeManager(
        use_postgres=True,
        pg_host=PRIMARY_CPU_IP,
        pg_port=5432,
        pg_dbname="ethical_hacking_kb",
        pg_user="agent",
        pg_password="secure_password"
    )
    
    # 2. Initialize PenTester with distributed processing
    logger.info("Initializing PenTester module...")
    pen_tester = PenTester(
        redis_host=PRIMARY_CPU_IP,
        redis_port=REDIS_PORT
    )
    
    # 3. Initialize Social Engineering Tester
    logger.info("Initializing Social Engineering Tester...")
    social_tester = SocialEngineeringTester(
        redis_host=PRIMARY_CPU_IP,
        redis_port=REDIS_PORT
    )
    
    # 4. Initialize Code Modifier for model management
    logger.info("Initializing Code Modifier...")
    code_modifier = CodeModifier(
        redis_host=PRIMARY_CPU_IP,
        redis_port=REDIS_PORT
    )
    
    # 5. Schedule automatic model updates to Pi 5
    logger.info("Scheduling model updates to Pi 5...")
    code_modifier.schedule_model_updates(
        model_dir="/models/compiled",
        pi_ip=PI5_IP,
        interval_hours=24
    )
    
    # 6. Demonstrate AI-Enhanced Vulnerability Scanning
    logger.info("\n=== PHASE 1: AI-Enhanced Vulnerability Scanning ===")
    
    # Target for ethical testing (authorized only!)
    target_ip = "192.168.1.200"  # Replace with authorized test target
    
    # Run AI-enhanced scan using Pi 5's YOLO model
    logger.info(f"Scanning target: {target_ip}")
    vulnerabilities = pen_tester.ai_enhanced_scan(
        target=target_ip,
        model_path="/models/yolo_vuln.hef",
        scan_type="-sV -O"  # Version and OS detection
    )
    
    # Store vulnerabilities in knowledge base
    for vuln in vulnerabilities:
        fact = f"Vulnerability found on {vuln['host']}:{vuln['port']} - {vuln['description']}"
        kb_manager.add_fact(
            content=fact,
            source="pen_tester",
            confidence=vuln.get('confidence', 0.8),
            category="vulnerabilities"
        )
    
    # Generate vulnerability report
    report = pen_tester.generate_report(vulnerabilities, format='text')
    logger.info(f"\nVulnerability Report:\n{report}")
    
    # 7. Demonstrate Social Engineering Testing
    logger.info("\n=== PHASE 2: AI-Powered Social Engineering Testing ===")
    
    # Test phishing simulation (authorized targets only!)
    test_targets = [
        {"email": "test1@example.com", "scenario": "IT update"},
        {"email": "test2@example.com", "scenario": "password reset"}
    ]
    
    # Run spear phishing campaign
    campaign_results = social_tester.spear_phishing_campaign(
        targets=test_targets,
        campaign_name="Q1_2025_Security_Test"
    )
    
    # Store results in knowledge base
    kb_manager.add_fact(
        content=f"Phishing campaign '{campaign_results['campaign_name']}' completed: "
                f"{campaign_results['successful']}/{campaign_results['targets']} successful",
        source="social_engineering",
        confidence=1.0,
        category="security_testing"
    )
    
    # Generate social engineering report
    se_report = social_tester.generate_report(
        campaign_results['results'],
        format='text'
    )
    logger.info(f"\nSocial Engineering Report:\n{se_report}")
    
    # 8. Demonstrate Code Self-Improvement
    logger.info("\n=== PHASE 3: AI-Powered Code Self-Improvement ===")
    
    # Analyze and improve a module using Pi 5
    target_module = "AgentSystem/modules/pen_tester.py"
    improvement_result = code_modifier.ai_enhanced_code_improvement(
        file_path=target_module,
        use_pi5=True
    )
    
    if improvement_result['success']:
        logger.info(f"Found {len(improvement_result['improvements'])} improvements")
        
        # Apply high-confidence improvements
        high_conf_improvements = [
            imp for imp in improvement_result['improvements']
            if imp.get('confidence', 0) > 0.85
        ]
        
        if high_conf_improvements:
            success = code_modifier.apply_improvements(
                target_module,
                high_conf_improvements
            )
            if success:
                logger.info("Successfully applied code improvements")
    
    # 9. Monitor System Health
    logger.info("\n=== PHASE 4: System Health Monitoring ===")
    
    # Check Pi 5 health via Celery
    from celery import Celery
    app = Celery('monitor', broker=f'redis://{PRIMARY_CPU_IP}:{REDIS_PORT}/0')
    
    health_task = app.send_task('system.health_check')
    try:
        health = health_task.get(timeout=10)
        logger.info(f"Pi 5 Health Status:")
        logger.info(f"  CPU Temperature: {health['thermal']['cpu_temp']}°C")
        logger.info(f"  RAM Usage: {health['memory']['used_gb']:.1f}GB ({health['memory']['percent']}%)")
        logger.info(f"  AI HAT+ Status: {health['ai_hat']['status']}")
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
    
    # 10. Knowledge Base Statistics
    logger.info("\n=== Knowledge Base Statistics ===")
    
    # Get categories of learned facts
    categories = kb_manager.get_fact_categories() if hasattr(kb_manager, 'get_fact_categories') else []
    
    # Search for security-related facts
    security_facts = kb_manager.search_facts("security", limit=5)
    vuln_facts = kb_manager.search_facts("vulnerability", limit=5)
    
    logger.info(f"Security-related facts: {len(security_facts)}")
    logger.info(f"Vulnerability facts: {len(vuln_facts)}")
    
    # 11. Demonstrate Continuous Learning Integration
    logger.info("\n=== PHASE 5: Continuous Learning Integration ===")
    
    # Create learning agent that coordinates all modules
    learning_agent = LearningAgent()
    
    # Queue automated security research
    security_topics = [
        "latest CVE vulnerabilities 2025",
        "AI-powered penetration testing techniques",
        "social engineering defense strategies",
        "zero-day exploit detection methods"
    ]
    
    for topic in security_topics:
        learning_agent.queue_research(topic)
        logger.info(f"Queued research: {topic}")
    
    # Queue code improvements for security modules
    security_modules = [
        "AgentSystem/modules/pen_tester.py",
        "AgentSystem/modules/social_engineering_tester.py"
    ]
    
    for module in security_modules:
        if Path(module).exists():
            learning_agent.queue_code_improvement(module)
            logger.info(f"Queued improvement: {module}")
    
    # 12. Summary and Recommendations
    logger.info("\n=== Summary and Recommendations ===")
    
    # Generate AI-powered recommendations
    recommendations = []
    
    # Based on vulnerabilities found
    critical_vulns = [v for v in vulnerabilities if v.get('severity') == 'critical']
    if critical_vulns:
        recommendations.append({
            'priority': 'CRITICAL',
            'action': f'Address {len(critical_vulns)} critical vulnerabilities immediately',
            'details': 'Apply patches, update services, or implement compensating controls'
        })
    
    # Based on phishing results
    phishing_success_rate = campaign_results['successful'] / campaign_results['targets']
    if phishing_success_rate > 0.3:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Implement security awareness training',
            'details': f'{phishing_success_rate*100:.0f}% phishing success rate indicates training needed'
        })
    
    # Log recommendations
    for rec in recommendations:
        logger.info(f"\n[{rec['priority']}] {rec['action']}")
        logger.info(f"  Details: {rec['details']}")
        
        # Store in knowledge base
        kb_manager.add_fact(
            content=f"Security recommendation: {rec['action']} - {rec['details']}",
            source="security_analysis",
            confidence=0.9,
            category="recommendations"
        )
    
    logger.info("\n=== Ethical Hacking System Demo Complete ===")
    logger.info("All tasks distributed between Primary CPU and Raspberry Pi 5")
    logger.info("AI HAT+ utilized for vulnerability classification and phishing generation")
    logger.info("Knowledge base updated with findings and recommendations")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise
