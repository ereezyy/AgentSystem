#!/usr/bin/env python
"""
AgentSystem Main Entry Point
---------------------------
Starts the autonomous agent system
"""

import os
import sys
import time
import argparse
import json
import logging
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from AgentSystem.utils.logger import get_logger, setup_logging
from AgentSystem.utils.env_loader import get_env, env
from AgentSystem.core.agent import Agent, AgentConfig
from AgentSystem.core.state import AgentState
from AgentSystem.core.memory import Memory
from AgentSystem.services.ai import ai_service
from AgentSystem.modules.browser import BrowserModule
from AgentSystem.modules.email import EmailModule
from AgentSystem.services.ai_providers.multimodal_provider import multimodal_provider
from AgentSystem.core.agent_swarm import swarm_coordinator
from AgentSystem.services.streaming_service import streaming_service
from AgentSystem.monitoring.realtime_dashboard import dashboard_service
from AgentSystem.analytics.predictive_analytics import analytics_engine
from AgentSystem.scaling.auto_scaler import auto_scaler
from AgentSystem.scaling.advanced_load_balancer import advanced_load_balancer
from AgentSystem.usage.overage_billing import OverageBilling

# Set up logger
logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Autonomous Agent System")

    # General options
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--log-level", type=str, choices=["debug", "info", "warning", "error", "critical"],
                        default="info", help="Logging level")
    parser.add_argument("--log-file", type=str, help="Log file path")

    # Mode options
    parser.add_argument("--mode", type=str, choices=["interactive", "server", "task"],
                        default="interactive", help="Operating mode")
    parser.add_argument("--task", type=str, help="Task to run (for task mode)")

    # Server options
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (for server mode)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (for server mode)")

    # Agent options
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--no-memory", action="store_true", help="Disable persistent memory")
    parser.add_argument("--no-browser", action="store_true", help="Disable browser module")
    parser.add_argument("--no-email", action="store_true", help="Disable email module")

    return parser.parse_args()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    # Default config
    config = {
        "agent": {
            "name": "AutoAgent",
            "description": "Autonomous agent system",
            "capabilities": ["browser", "email", "coding", "ai"],
            "max_iterations": 100,
            "timeout_seconds": 300
        },
        "pricing": {
            "tiers": {
                "Starter": {
                    "base_price": 49,
                    "discount": 0.10,  # 10% discount for acquisition
                    "features": ["Basic AI Agents", "Limited API Calls"]
                },
                "Pro": {
                    "base_price": 199,
                    "discount": 0.0,
                    "features": ["Advanced AI Agents", "Increased API Calls", "Basic Integrations"],
                    "overage_model": {
                        "type": "credits-per-task",
                        "cost_per_credit": 0.99,  # Cost per task resolution
                        "description": "Additional tasks beyond plan limits charged per resolution"
                    }
                },
                "Enterprise": {
                    "base_price": 999,
                    "discount": 0.0,
                    "features": ["Full AI Agent Suite", "Unlimited API Calls", "Enterprise Integrations", "White-Label Branding"]
                },
                "Custom": {
                    "base_price": 0,  # Quote-based
                    "discount": 0.0,
                    "features": ["Custom Solutions", "Dedicated Support"]
                }
            }
        },
        "browser": {
            "headless": True,
            "browser_type": "chromium",
            "viewport_size": [1280, 720],
            "slow_mo": 50,
            "timeout": 30000
        },
        "email": {
            "use_ssl": True
        },
        "memory": {
            "max_working_items": 100
        }
    }

    # Load from file if provided
    if config_path:
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)

            # Merge configs
            for section, values in file_config.items():
                if section in config:
                    config[section].update(values)
                else:
                    config[section] = values

            logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")

    return config


def setup_agent(config: Dict[str, Any], args) -> Agent:
    """
    Set up the agent

    Args:
        config: Configuration dictionary
        args: Command-line arguments

    Returns:
        Agent instance
    """
    # Create agent config
    agent_config = AgentConfig(
        name=config["agent"].get("name", "AutoAgent"),
        description=config["agent"].get("description", "Autonomous agent system"),
        capabilities=config["agent"].get("capabilities", []),
        max_iterations=config["agent"].get("max_iterations", 100),
        timeout_seconds=config["agent"].get("timeout_seconds", 300)
    )

    # Create state manager
    state_manager = AgentState()

    # Create memory manager (unless disabled)
    memory_manager = None
    if not args.no_memory:
        memory_manager = Memory(
            max_working_items=config["memory"].get("max_working_items", 100)
        )

    # Create agent
    agent = Agent(
        config=agent_config,
        state_manager=state_manager,
        memory_manager=memory_manager
    )

    # Set up modules

    # Browser module (unless disabled)
    if not args.no_browser:
        browser_config = config.get("browser", {})
        browser_module = BrowserModule(
            headless=args.headless if args.headless is not None else browser_config.get("headless", True),
            browser_type=browser_config.get("browser_type", "chromium"),
            viewport_size=tuple(browser_config.get("viewport_size", [1280, 720])),
            slow_mo=browser_config.get("slow_mo", 50),
            timeout=browser_config.get("timeout", 30000),
            memory_manager=memory_manager
        )
        agent.register_module("browser", browser_module)
        logger.info("Registered browser module")

    # Email module (unless disabled)
    if not args.no_email:
        email_config = config.get("email", {})
        email_module = EmailModule(
            smtp_server=get_env("SMTP_SERVER"),
            smtp_port=int(get_env("SMTP_PORT") or 0),
            imap_server=get_env("IMAP_SERVER"),
            imap_port=int(get_env("IMAP_PORT") or 0),
            username=get_env("EMAIL_USERNAME"),
            password=get_env("EMAIL_PASSWORD"),
            use_ssl=email_config.get("use_ssl", True),
            memory_manager=memory_manager
        )
        agent.register_module("email", email_module)
        logger.info("Registered email module")

    return agent


def run_interactive_mode(agent: Agent):
    """
    Run in interactive mode

    Args:
        agent: Agent instance
    """
    print(f"\n=== {agent.config.name} Interactive Mode ===")
    print(f"{agent.config.description}")
    print("Type 'help' for available commands, 'exit' to quit")

    while True:
        try:
            command = input("\n> ").strip()

            if command.lower() in ['exit', 'quit', 'q']:
                break

            elif command.lower() in ['help', '?', 'h']:
                print("\nAvailable commands:")
                print("  help         - Show this help message")
                print("  exit         - Exit the program")
                print("  status       - Show agent status")
                print("  modules      - List loaded modules")
                print("  capabilities - List agent capabilities")
                print("  task <task>  - Run a task")
                print("  browser <url> - Open browser and navigate to URL")
                print("  email <to> <subject> - Send an email")

            elif command.lower() == 'status':
                status = agent.state_manager.status.name
                current_task = agent.state_manager.current_task_id

                print(f"\nAgent Status: {status}")
                print(f"Current Task: {current_task or 'None'}")

                # If the agent has a memory manager, show memory stats
                if agent.memory_manager:
                    working_count = len(agent.memory_manager.working_memory)
                    print(f"Working Memory Items: {working_count}")

            elif command.lower() == 'modules':
                modules = list(agent._modules.keys())
                print(f"\nLoaded Modules: {', '.join(modules) if modules else 'None'}")

            elif command.lower() == 'capabilities':
                capabilities = agent.config.capabilities
                print(f"\nCapabilities: {', '.join(capabilities) if capabilities else 'None'}")

            elif command.lower().startswith('task '):
                task_text = command[5:].strip()
                if not task_text:
                    print("Please specify a task")
                    continue

                print(f"\nRunning task: {task_text}")

                # Create a task
                task_id = agent.state_manager.create_task(
                    name="User Task",
                    description=task_text,
                    priority=5
                )

                # Start the task
                agent.state_manager.start_task(task_id)

                # Run the agent
                result = agent.run(task=task_id)

                print(f"\nTask completed. Result: {result}")

            elif command.lower().startswith('browser '):
                url = command[8:].strip()
                if not url:
                    print("Please specify a URL")
                    continue

                # Add http if missing
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url

                browser_module = agent.get_module("browser")
                if not browser_module:
                    print("Browser module not available")
                    continue

                print(f"Opening {url} in browser...")

                # Start browser if not started
                if not browser_module._page:
                    browser_module.start()

                # Navigate to URL
                result = browser_module.navigate(url)

                if result["success"]:
                    print(f"Navigated to {url}")
                else:
                    print(f"Error navigating to {url}: {result.get('error')}")

            elif command.lower().startswith('email '):
                parts = command[6:].strip().split(' ', 1)
                if len(parts) < 2:
                    print("Usage: email <to> <subject>")
                    continue

                to_email = parts[0]
                subject = parts[1]

                email_module = agent.get_module("email")
                if not email_module:
                    print("Email module not available")
                    continue

                body = input("Enter email body (press Enter, then Ctrl+D when done):\n")

                # Create and send email
                email_obj = email_module.create_email(
                    subject=subject,
                    body=body,
                    recipients=[to_email]
                )

                print(f"Sending email to {to_email}...")
                result = email_module.send_email(email_obj)

                if result:
                    print("Email sent successfully")
                else:
                    print("Error sending email")

            else:
                print("Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        except Exception as e:
            print(f"Error: {e}")


def run_server_mode(agent: Agent, host: str, port: int):
    """
    Run in server mode

    Args:
        agent: Agent instance
        host: Server host
        port: Server port
    """
    try:
        from bottle import Bottle, request, response, static_file, run
    except ImportError:
        logger.error("Bottle required for server mode. Install with: pip install bottle")
        return

    # Create Bottle app
    app = Bottle()

    # Serve static files
    @app.route('/static/<filepath:path>')
    def serve_static(filepath):
        return static_file(filepath, root='static')

    @app.route('/')
    def root():
        return static_file('index.html', root='static')

    @app.route('/health')
    def health():
        """Health check endpoint"""
        response.content_type = 'application/json'
        return {
            "status": "healthy",
            "service": "agentsystem",
            "timestamp": time.time(),
            "version": "2.0.0"
        }

    @app.route('/task', method='POST')
    def run_task():
        data = request.json
        if not data or 'task' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Task description required"}

        task_id = agent.state_manager.create_task(
            name="API Task",
            description=data['task'],
            priority=data.get('priority', 5)
        )

        # Run in background (simplified, as Bottle doesn't have built-in background tasks)
        import threading
        threading.Thread(target=agent.run, args=(task_id,)).start()

        response.content_type = 'application/json'
        return {"task_id": task_id, "status": "started"}

    @app.route('/task/<task_id>')
    def get_task(task_id):
        task = agent.state_manager.get_task(task_id)
        if not task:
            response.status = 404
            response.content_type = 'application/json'
            return {"detail": f"Task {task_id} not found"}

        response.content_type = 'application/json'
        return {
            "task_id": task_id,
            "name": task.name,
            "description": task.description,
            "status": task.status,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at
        }

    @app.route('/task/<task_id>/start', method='POST')
    def start_task(task_id):
        task = agent.state_manager.get_task(task_id)
        if not task:
            response.status = 404
            response.content_type = 'application/json'
            return {"detail": f"Task {task_id} not found"}
        if task.status in ["RUNNING", "COMPLETED"]:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": f"Task {task_id} is already {task.status.lower()}"}
        agent.state_manager.start_task(task_id)
        response.content_type = 'application/json'
        return {"task_id": task_id, "status": "started"}

    @app.route('/task/<task_id>/stop', method='POST')
    def stop_task(task_id):
        task = agent.state_manager.get_task(task_id)
        if not task:
            response.status = 404
            response.content_type = 'application/json'
            return {"detail": f"Task {task_id} not found"}
        if task.status != "RUNNING":
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": f"Task {task_id} is not running"}
        agent.state_manager.update_task_status(task_id, "STOPPED")
        response.content_type = 'application/json'
        return {"task_id": task_id, "status": "stopped"}

    @app.route('/task/<task_id>/delete', method='POST')
    def delete_task(task_id):
        task = agent.state_manager.get_task(task_id)
        if not task:
            response.status = 404
            response.content_type = 'application/json'
            return {"detail": f"Task {task_id} not found"}
        agent.state_manager.delete_task(task_id)
        response.content_type = 'application/json'
        return {"task_id": task_id, "status": "deleted"}

    @app.route('/tasks')
    def get_all_tasks():
        logger.info("Received request for /tasks endpoint")
        tasks = agent.state_manager.get_all_tasks()
        logger.info(f"Returning {len(tasks)} tasks from state manager")
        response.content_type = 'application/json'
        return [
            {
                "task_id": task_id,
                "name": task.name,
                "description": task.description,
                "status": task.status,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at
            }
            for task_id, task in tasks.items()
        ]

    @app.route('/browser/navigate', method='POST')
    def browser_navigate():
        data = request.json
        if not data or 'url' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "URL required"}

        browser_module = agent.get_module("browser")
        if not browser_module:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Browser module not available"}

        # Add http if missing
        url = data['url']
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Start browser if not started
        if not browser_module._page:
            browser_module.start()

        # Navigate to URL
        result = browser_module.navigate(url)

        response.content_type = 'application/json'
        return {
            "success": result["success"],
            "url": url,
            "error": result.get("error")
        }

    @app.route('/email/send', method='POST')
    def send_email():
        data = request.json
        if not data or 'to' not in data or 'subject' not in data or 'body' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "To, subject, and body required"}

        email_module = agent.get_module("email")
        if not email_module:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Email module not available"}

        # Create and send email
        email_obj = email_module.create_email(
            subject=data['subject'],
            body=data['body'],
            recipients=[data['to']]
        )

        result = email_module.send_email(email_obj)

        response.content_type = 'application/json'
        return {"success": result}

    # MULTI-MODAL AI ENDPOINTS

    @app.route('/ai/vision/analyze', method='POST')
    def analyze_image():
        data = request.json
        if not data or 'image_data' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Image data required"}

        import asyncio
        result = asyncio.run(multimodal_provider.analyze_image(
            data['image_data'],
            data.get('prompt', 'Describe this image in detail')
        ))

        response.content_type = 'application/json'
        return result

    @app.route('/ai/vision/generate', method='POST')
    def generate_image():
        data = request.json
        if not data or 'prompt' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Prompt required"}

        import asyncio
        result = asyncio.run(multimodal_provider.generate_image(
            data['prompt'],
            data.get('size', '1024x1024'),
            data.get('quality', 'standard')
        ))

        response.content_type = 'application/json'
        return result

    @app.route('/ai/audio/transcribe', method='POST')
    def transcribe_audio():
        # Handle file upload for audio transcription
        upload = request.files.get('audio')
        if not upload:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Audio file required"}

        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            upload.save(temp_file.name)

            import asyncio
            result = asyncio.run(multimodal_provider.transcribe_audio(temp_file.name))

        # Clean up temp file
        import os
        os.unlink(temp_file.name)

        response.content_type = 'application/json'
        return result

    @app.route('/ai/audio/tts', method='POST')
    def text_to_speech():
        data = request.json
        if not data or 'text' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Text required"}

        import asyncio
        result = asyncio.run(multimodal_provider.text_to_speech(
            data['text'],
            data.get('voice', 'alloy')
        ))

        response.content_type = 'application/json'
        return result

    @app.route('/ai/code/generate', method='POST')
    def generate_code():
        data = request.json
        if not data or 'prompt' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Code prompt required"}

        import asyncio
        result = asyncio.run(multimodal_provider.generate_code(
            data['prompt'],
            data.get('language', 'python')
        ))

        response.content_type = 'application/json'
        return result

    @app.route('/ai/code/review', method='POST')
    def review_code():
        data = request.json
        if not data or 'code' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Code required"}

        import asyncio
        result = asyncio.run(multimodal_provider.review_code(
            data['code'],
            data.get('language', 'python')
        ))

        response.content_type = 'application/json'
        return result

    # AGENT SWARM ENDPOINTS

    @app.route('/swarm/task', method='POST')
    def submit_swarm_task():
        data = request.json
        if not data or 'description' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Task description required"}

        import asyncio
        task_id = asyncio.run(swarm_coordinator.submit_task(
            data['description'],
            data.get('complexity', 5),
            data.get('priority', 5)
        ))

        response.content_type = 'application/json'
        return {"task_id": task_id, "status": "submitted"}

    @app.route('/swarm/status')
    def get_swarm_status():
        import asyncio
        status = asyncio.run(swarm_coordinator.get_swarm_status())

        response.content_type = 'application/json'
        return status

    # STREAMING ENDPOINTS

    @app.route('/stream/start', method='POST')
    def start_stream():
        data = request.json
        if not data or 'messages' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Messages required"}

        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4')
        messages = data['messages']

        import asyncio
        if provider == 'openai':
            stream_id = asyncio.run(streaming_service.start_openai_stream(messages, model))
        elif provider == 'anthropic':
            stream_id = asyncio.run(streaming_service.start_anthropic_stream(messages, model))
        else:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Unsupported provider"}

        response.content_type = 'application/json'
        return {"stream_id": stream_id, "provider": provider, "model": model}

    @app.route('/stream/<stream_id>/chunks')
    def get_stream_chunks(stream_id):
        since_chunk_id = request.query.get('since_chunk_id')

        import asyncio
        chunks = asyncio.run(streaming_service.get_stream_chunks(stream_id, since_chunk_id))

        response.content_type = 'application/json'
        return {"chunks": [{"id": c.id, "content": c.content, "timestamp": c.timestamp, "is_final": c.is_final} for c in chunks]}

    @app.route('/stream/<stream_id>/progress')
    def get_stream_progress(stream_id):
        import asyncio
        progress = asyncio.run(streaming_service.get_stream_progress(stream_id))

        response.content_type = 'application/json'
        if progress:
            return {
                "stream_id": progress.stream_id,
                "status": progress.status.value,
                "processed_tokens": progress.processed_tokens,
                "estimated_completion": progress.estimated_completion
            }
        else:
            return {"error": "Stream not found"}

    @app.route('/stream/<stream_id>/pause', method='POST')
    def pause_stream(stream_id):
        import asyncio
        success = asyncio.run(streaming_service.pause_stream(stream_id))

        response.content_type = 'application/json'
        return {"success": success}

    @app.route('/stream/<stream_id>/resume', method='POST')
    def resume_stream(stream_id):
        import asyncio
        success = asyncio.run(streaming_service.resume_stream(stream_id))

        response.content_type = 'application/json'
        return {"success": success}

    @app.route('/stream/<stream_id>/interrupt', method='POST')
    def interrupt_stream(stream_id):
        import asyncio
        success = asyncio.run(streaming_service.interrupt_stream(stream_id))

        response.content_type = 'application/json'
        return {"success": success}

    # DASHBOARD & MONITORING ENDPOINTS

    @app.route('/dashboard/data')
    def get_dashboard_data():
        time_range = int(request.query.get('time_range', 3600))

        import asyncio
        data = asyncio.run(dashboard_service.get_dashboard_data(time_range))

        response.content_type = 'application/json'
        return data

    @app.route('/dashboard/metrics')
    def get_prometheus_metrics():
        import asyncio
        metrics = asyncio.run(dashboard_service.get_prometheus_metrics())

        response.content_type = 'text/plain'
        return metrics

    @app.route('/dashboard/grafana')
    def get_grafana_dashboard():
        import asyncio
        dashboard = asyncio.run(dashboard_service.create_grafana_dashboard())

        response.content_type = 'application/json'
        return dashboard

    # ANALYTICS ENDPOINTS

    @app.route('/analytics/summary')
    def get_analytics_summary():
        import asyncio
        summary = asyncio.run(analytics_engine.get_analytics_summary())

        response.content_type = 'application/json'
        return summary

    @app.route('/analytics/predictions')
    def get_predictions():
        metric_name = request.query.get('metric_name', 'cpu_usage')
        time_horizon = int(request.query.get('time_horizon', 3600))

        import asyncio
        predictions = asyncio.run(analytics_engine.get_predictions(metric_name, time_horizon))

        response.content_type = 'application/json'
        return {"predictions": [{"metric_name": p.metric_name, "predicted_value": p.predicted_value, "confidence": p.confidence, "timestamp": p.timestamp} for p in predictions]}

    @app.route('/analytics/anomalies')
    def get_anomalies():
        time_range = int(request.query.get('time_range', 3600))

        import asyncio
        anomalies = asyncio.run(analytics_engine.get_anomalies(time_range))

        response.content_type = 'application/json'
        return {"anomalies": [{"metric_name": a.metric_name, "timestamp": a.timestamp, "severity": a.severity, "description": a.description} for a in anomalies]}

    @app.route('/analytics/insights')
    def get_insights():
        category = request.query.get('category')
        time_range = int(request.query.get('time_range', 86400))

        import asyncio
        insights = asyncio.run(analytics_engine.get_insights(category, time_range))

        response.content_type = 'application/json'
        return {"insights": [{"id": i.id, "category": i.category, "title": i.title, "description": i.description, "impact": i.impact, "recommendations": i.recommendations} for i in insights]}

    # Industry Pack Endpoints - E-commerce Churn Prevention
    @app.route('/industry/ecommerce/churn/analyze', method='POST')
    async def analyze_churn():
        data = request.json
        if not data or 'tenant_id' not in data or 'customer_data' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Tenant ID and customer data are required"}

        # Check if tenant has the e-commerce churn prevention pack (placeholder logic)
        async with dashboard_service.db_pool.acquire() as conn:
            has_pack = await conn.fetchval(
                "SELECT COUNT(*) > 0 FROM tenant_management.tenants WHERE id = $1 AND industry_pack = 'ecommerce_churn_prevention'",
                data['tenant_id']
            )
            if not has_pack:
                response.status = 403
                response.content_type = 'application/json'
                return {"detail": "Industry pack not enabled for this tenant"}

        result = await churn_prevention.analyze_churn_risk(data['tenant_id'], data['customer_data'])

        response.content_type = 'application/json'
        return result

    @app.route('/industry/ecommerce/churn/intervention', method='POST')
    async def execute_intervention_endpoint():
        data = request.json
        if not data or 'tenant_id' not in data or 'customer_id' not in data or 'intervention' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Tenant ID, Customer ID, and intervention details are required"}

        # Check if tenant has the e-commerce churn prevention pack (placeholder logic)
        async with dashboard_service.db_pool.acquire() as conn:
            has_pack = await conn.fetchval(
                "SELECT COUNT(*) > 0 FROM tenant_management.tenants WHERE id = $1 AND industry_pack = 'ecommerce_churn_prevention'",
                data['tenant_id']
            )
            if not has_pack:
                response.status = 403
                response.content_type = 'application/json'
                return {"detail": "Industry pack not enabled for this tenant"}

        result = await churn_prevention.execute_intervention(
            data['tenant_id'], data['customer_id'], data['intervention']
        )

        response.content_type = 'application/json'
        return result

    # SCALING ENDPOINTS

    @app.route('/scaling/status')
    def get_scaling_status():
        import asyncio
        status = asyncio.run(auto_scaler.get_scaling_status())

        response.content_type = 'application/json'
        return status

    @app.route('/scaling/manual', method='POST')
    def manual_scale():
        data = request.json
        if not data or 'service_name' not in data or 'target_instances' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Service name and target instances required"}

        import asyncio
        success = asyncio.run(auto_scaler.manual_scale(
            data['service_name'],
            data['target_instances']
        ))

        response.content_type = 'application/json'
        return {"success": success}

    # Customer Support Endpoints
    @app.route('/support/chat', method='POST')
    def handle_chat():
        data = request.json
        if not data or 'tenant_id' not in data or 'user_id' not in data or 'message' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Tenant ID, User ID, and message are required"}

        import asyncio
        result = asyncio.run(customer_support.handle_chat_request(
            data['tenant_id'],
            data['user_id'],
            data['message']
        ))

        response.content_type = 'application/json'
        return result

    @app.route('/support/ticket', method='POST')
    def create_ticket():
        data = request.json
        if not data or 'tenant_id' not in data or 'user_id' not in data or 'issue' not in data:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Tenant ID, User ID, and issue description are required"}

        import asyncio
        result = asyncio.run(customer_support.create_support_ticket(
            data['tenant_id'],
            data['user_id'],
            data['issue'],
            data.get('priority', 'medium')
        ))

        response.content_type = 'application/json'
        return result

    @app.route('/support/tutorial', method='GET')
    def get_tutorial():
        tenant_id = request.query.get('tenant_id')
        user_id = request.query.get('user_id')
        topic = request.query.get('topic', 'getting_started')

        if not tenant_id or not user_id:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Tenant ID and User ID are required"}

        import asyncio
        result = asyncio.run(customer_support.get_onboarding_tutorial(tenant_id, user_id, topic))

        response.content_type = 'application/json'
        return result

    # Overage Billing Endpoints
    @app.route('/billing/overage/summary')
    def get_overage_summary():
        tenant_id = request.query.get('tenant_id')
        period = request.query.get('period', 'current_month')

        if not tenant_id:
            response.status = 400
            response.content_type = 'application/json'
            return {"detail": "Tenant ID required"}

        import asyncio
        summary = asyncio.run(overage_billing.get_overage_summary(tenant_id, period))

        response.content_type = 'application/json'
        return summary

    # Start server
    print(f"\n=== {agent.config.name} SUPERCHARGED Server Mode ===")
    print(f"🚀 Multi-Modal AI: Vision, Audio, Code Generation")
    print(f"🤖 Agent Swarms: Coordinated specialist agents")
    print(f"⚡ Real-time Streaming: Live AI responses")
    print(f"📊 Predictive Analytics: AI-powered insights")
    print(f"🔄 Auto-scaling: Dynamic load balancing")
    print(f"💳 Overage Billing: Credits-per-task model for Pro tier")
    print(f"📚 Automated Documentation: User guides and API docs")
    print(f"🛠️ Customer Support: 24/7 AI chatbots and ticketing")
    print(f"Starting server at http://{host}:{port}")
    run(app, host=host, port=port)


def run_task_mode(agent: Agent, task: str):
    """
    Run in task mode

    Args:
        agent: Agent instance
        task: Task to run
    """
    print(f"\n=== {agent.config.name} Task Mode ===")
    print(f"Running task: {task}")

    # Create a task
    task_id = agent.state_manager.create_task(
        name="Command-line Task",
        description=task,
        priority=5
    )

    # Start the task
    agent.state_manager.start_task(task_id)

    # Run the agent
    start_time = time.time()
    result = agent.run(task=task_id)
    end_time = time.time()

    print(f"\nTask completed in {end_time - start_time:.2f} seconds.")
    print(f"Result: {result}")


def main():
    """Main entry point"""
    # Parse command-line arguments
    args = parse_args()

    # Set up logging
    setup_logging(
        level=args.log_level,
        log_to_file=args.log_file is not None,
        log_dir=os.path.dirname(args.log_file) if args.log_file else None,
        app_name="agent"
    )

    # Load configuration
    config = load_config(args.config)

    # Set up agent
    agent = setup_agent(config, args)

    # Start dashboard metrics collection, analytics, and auto-scaling
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.ensure_future(dashboard_service.metrics_collector.start_metrics_collection()))
        loop.run_until_complete(asyncio.ensure_future(analytics_engine.start_analytics()))
        loop.run_until_complete(asyncio.ensure_future(auto_scaler.start_scaling()))
        loop.run_until_complete(asyncio.ensure_future(advanced_load_balancer.initialize_advanced_balancing()))
        # Initialize and start overage billing system
        overage_billing = OverageBilling(dashboard_service.db_pool, dashboard_service.redis_client)
        loop.run_until_complete(asyncio.ensure_future(overage_billing.start()))
        # Initialize and start security automation engine with continuous scanning
        from AgentSystem.security.security_automation_engine import SecurityAutomationEngine
        security_engine = SecurityAutomationEngine({}, dashboard_service.db_pool, dashboard_service.redis_client)
        loop.run_until_complete(asyncio.ensure_future(security_engine.start_continuous_scanning()))
        # Initialize and start documentation generator
        from AgentSystem.documentation.doc_generator import DocGenerator
        doc_generator = DocGenerator(dashboard_service.db_pool, dashboard_service.redis_client)
        loop.run_until_complete(asyncio.ensure_future(doc_generator.start()))
        loop.run_until_complete(asyncio.ensure_future(doc_generator.generate_initial_docs()))
        # Initialize and start customer support system
        from AgentSystem.support.customer_support import CustomerSupport
        customer_support = CustomerSupport(dashboard_service.db_pool, dashboard_service.redis_client)
        loop.run_until_complete(asyncio.ensure_future(customer_support.start()))
    except Exception as e:
        logger.error(f"Error starting dashboard metrics collection, analytics, or auto-scaling: {e}")

    # Run in appropriate mode
    if args.mode == "interactive":
        run_interactive_mode(agent)
    elif args.mode == "server":
        run_server_mode(agent, args.host, args.port)
    elif args.mode == "task":
        if not args.task:
            logger.error("Task required for task mode")
            return
        run_task_mode(agent, args.task)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
