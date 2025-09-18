"""
Documentation and Knowledge Base Generator for AgentSystem
Automates the creation of user guides, API documentation, and self-help resources
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import markdown
import json
import re
from pathlib import Path
import shutil
import aioredis
import asyncpg
from AgentSystem.services.ai import ai_service

logger = logging.getLogger(__name__)

class DocGenerator:
    """Generates and manages documentation and knowledge base content for AgentSystem"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis, output_dir: str = "docs"):
        """Initialize the documentation generator"""
        self.db_pool = db_pool
        self.redis = redis_client
        self.output_dir = Path(output_dir)
        self.templates_dir = Path(__file__).parent / "templates"
        self._running = False
        self._update_task = None
        self.update_interval = 86400  # Daily updates by default

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Documentation Generator initialized")

    async def start(self):
        """Start the documentation generator with background updates"""
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Documentation Generator started")

    async def stop(self):
        """Stop the documentation generator and complete any pending updates"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("Documentation Generator stopped")

    async def generate_initial_docs(self):
        """Generate initial set of documentation and knowledge base content"""
        logger.info("Generating initial documentation set")

        # Generate core documentation
        await self._generate_user_guide()
        await self._generate_api_docs()
        await self._generate_knowledge_base()

        # Generate index page
        await self._generate_index_page()

        logger.info("Initial documentation generation completed")

    async def _update_loop(self):
        """Background task to periodically update documentation"""
        while self._running:
            try:
                await asyncio.sleep(self.update_interval)
                logger.info("Performing periodic documentation update")
                await self.generate_initial_docs()  # Regenerate all docs to ensure they're up-to-date
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in documentation update loop: {e}")

    async def _generate_user_guide(self):
        """Generate comprehensive user guide for AgentSystem"""
        logger.info("Generating user guide")

        # Structure of the user guide
        user_guide_structure = {
            "Introduction": {
                "Overview": "An overview of AgentSystem and its capabilities.",
                "Getting Started": "How to quickly set up and start using AgentSystem."
            },
            "Core Features": {
                "AI Agents": "Details on specialized AI agents for marketing, sales, etc.",
                "Integrations": "How to connect AgentSystem with other platforms like Salesforce and HubSpot.",
                "Workflow Automation": "Creating and managing automated workflows."
            },
            "Advanced Usage": {
                "Customization": "Customizing agents and workflows for specific needs.",
                "Scaling": "Managing multi-region deployment and auto-scaling."
            },
            "Troubleshooting": {
                "Common Issues": "Solutions to frequently encountered problems.",
                "Support": "How to get help and support from the AgentSystem team."
            }
        }

        # Generate content for each section using AI if available
        content = {}
        for section, subsections in user_guide_structure.items():
            content[section] = {}
            for subsection, description in subsections.items():
                markdown_content = await self._generate_content(f"Generate detailed documentation content for '{subsection}' in the '{section}' section of the AgentSystem user guide. {description}")
                content[section][subsection] = markdown_content

        # Render the user guide
        user_guide_html = await self._render_user_guide(content)

        # Save the user guide
        output_path = self.output_dir / "user_guide.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(user_guide_html)

        logger.info(f"User guide generated at {output_path}")

    async def _generate_api_docs(self):
        """Generate API documentation for AgentSystem"""
        logger.info("Generating API documentation")

        # Define API endpoints structure (this could be dynamically extracted from code in a real system)
        api_endpoints = {
            "Authentication": [
                {"endpoint": "/api/auth/login", "method": "POST", "description": "Authenticate user and receive access token."},
                {"endpoint": "/api/auth/refresh", "method": "POST", "description": "Refresh access token using refresh token."}
            ],
            "Tasks": [
                {"endpoint": "/api/tasks", "method": "GET", "description": "Retrieve list of tasks."},
                {"endpoint": "/api/tasks/{id}", "method": "GET", "description": "Retrieve specific task details."},
                {"endpoint": "/api/tasks", "method": "POST", "description": "Create a new task for an agent."}
            ],
            "Agents": [
                {"endpoint": "/api/agents", "method": "GET", "description": "List available AI agents."},
                {"endpoint": "/api/agents/{id}/status", "method": "GET", "description": "Check status of specific agent."}
            ],
            "Billing": [
                {"endpoint": "/api/billing/usage", "method": "GET", "description": "Retrieve current usage and billing information."},
                {"endpoint": "/billing/overage/summary", "method": "GET", "description": "Get summary of overage charges for Pro tier."}
            ]
        }

        # Generate detailed content for each endpoint
        detailed_endpoints = {}
        for category, endpoints in api_endpoints.items():
            detailed_endpoints[category] = []
            for endpoint in endpoints:
                detailed_content = await self._generate_content(
                    f"Generate detailed API documentation for the {endpoint['method']} {endpoint['endpoint']} endpoint. "
                    f"Include parameters, response format, and example usage. {endpoint['description']}"
                )
                detailed_endpoints[category].append({
                    "method": endpoint['method'],
                    "endpoint": endpoint['endpoint'],
                    "description": endpoint['description'],
                    "details": detailed_content
                })

        # Render API documentation
        api_docs_html = await self._render_api_docs(detailed_endpoints)

        # Save API documentation
        output_path = self.output_dir / "api_docs.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(api_docs_html)

        logger.info(f"API documentation generated at {output_path}")

    async def _generate_knowledge_base(self):
        """Generate knowledge base articles for common issues and questions"""
        logger.info("Generating knowledge base")

        # Define common topics for knowledge base
        kb_topics = {
            "Setup and Installation": [
                {"title": "Installing AgentSystem on Your Infrastructure", "description": "Step-by-step guide to install AgentSystem."},
                {"title": "Configuring Initial Settings", "description": "How to configure settings for optimal performance."}
            ],
            "Usage Tips": [
                {"title": "Optimizing AI Agent Performance", "description": "Tips to get the best results from AI agents."},
                {"title": "Creating Effective Workflows", "description": "Best practices for workflow automation."}
            ],
            "Troubleshooting": [
                {"title": "Resolving Connection Issues with Integrations", "description": "Fix problems with third-party integrations."},
                {"title": "Handling API Rate Limit Errors", "description": "What to do when you hit API rate limits."}
            ],
            "Billing and Licensing": [
                {"title": "Understanding Your Bill and Overage Charges", "description": "Explanation of billing details and overage costs for Pro tier."},
                {"title": "Upgrading or Downgrading Your Plan", "description": "How to change your subscription tier."}
            ]
        }

        # Generate content for each topic
        kb_content = {}
        for category, articles in kb_topics.items():
            kb_content[category] = []
            for article in articles:
                content = await self._generate_content(
                    f"Generate a detailed knowledge base article titled '{article['title']}' "
                    f"under the category '{category}'. {article['description']}"
                )
                kb_content[category].append({
                    "title": article['title'],
                    "content": content
                })

        # Render knowledge base
        kb_html = await self._render_knowledge_base(kb_content)

        # Save knowledge base
        output_path = self.output_dir / "knowledge_base.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(kb_html)

        logger.info(f"Knowledge base generated at {output_path}")

    async def _generate_index_page(self):
        """Generate an index page linking to all documentation resources"""
        logger.info("Generating documentation index page")

        index_content = {
            "title": "AgentSystem Documentation",
            "sections": [
                {"title": "User Guide", "link": "user_guide.html", "description": "Comprehensive guide to using AgentSystem."},
                {"title": "API Documentation", "link": "api_docs.html", "description": "Detailed documentation for AgentSystem APIs."},
                {"title": "Knowledge Base", "link": "knowledge_base.html", "description": "Solutions to common issues and best practices."}
            ]
        }

        index_html = await self._render_index_page(index_content)

        output_path = self.output_dir / "index.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(index_html)

        logger.info(f"Documentation index page generated at {output_path}")

    async def _generate_content(self, prompt: str) -> str:
        """Generate content using AI service if available, otherwise return placeholder"""
        try:
            if ai_service:
                result = await ai_service.generate_text(prompt, max_tokens=2000)
                return result.get("text", "# Content Generation Failed\n\nUnable to generate content.")
            else:
                return f"# Placeholder Content\n\nContent for '{prompt[:50]}...' will be generated here."
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return f"# Error Generating Content\n\nFailed to generate content due to: {str(e)}"

    async def _render_user_guide(self, content: Dict[str, Dict[str, str]]) -> str:
        """Render the user guide content into HTML"""
        template_path = self.templates_dir / "user_guide_template.html"
        if not template_path.exists():
            logger.warning("User guide template not found, using basic rendering")
            html_content = "<html><head><title>AgentSystem User Guide</title></head><body>"
            html_content += "<h1>AgentSystem User Guide</h1>"
            for section, subsections in content.items():
                html_content += f"<h2>{section}</h2>"
                for subsection, text in subsections.items():
                    html_content += f"<h3>{subsection}</h3>"
                    html_content += markdown.markdown(text)
            html_content += "</body></html>"
            return html_content

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        # Replace placeholders in template with content
        rendered_content = ""
        for section, subsections in content.items():
            rendered_content += f"<h2>{section}</h2>"
            for subsection, text in subsections.items():
                rendered_content += f"<h3>{subsection}</h3>"
                rendered_content += markdown.markdown(text)

        return template.replace("{{CONTENT}}", rendered_content)

    async def _render_api_docs(self, endpoints: Dict[str, List[Dict[str, Any]]]) -> str:
        """Render API documentation into HTML"""
        template_path = self.templates_dir / "api_docs_template.html"
        if not template_path.exists():
            logger.warning("API docs template not found, using basic rendering")
            html_content = "<html><head><title>AgentSystem API Documentation</title></head><body>"
            html_content += "<h1>AgentSystem API Documentation</h1>"
            for category, endpoint_list in endpoints.items():
                html_content += f"<h2>{category}</h2>"
                for endpoint in endpoint_list:
                    html_content += f"<h3>{endpoint['method']} {endpoint['endpoint']}</h3>"
                    html_content += f"<p>{endpoint['description']}</p>"
                    html_content += markdown.markdown(endpoint['details'])
            html_content += "</body></html>"
            return html_content

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        # Replace placeholders in template with content
        rendered_content = ""
        for category, endpoint_list in endpoints.items():
            rendered_content += f"<h2>{category}</h2>"
            for endpoint in endpoint_list:
                rendered_content += f"<h3>{endpoint['method']} {endpoint['endpoint']}</h3>"
                rendered_content += f"<p>{endpoint['description']}</p>"
                rendered_content += markdown.markdown(endpoint['details'])

        return template.replace("{{CONTENT}}", rendered_content)

    async def _render_knowledge_base(self, content: Dict[str, List[Dict[str, str]]]) -> str:
        """Render knowledge base articles into HTML"""
        template_path = self.templates_dir / "knowledge_base_template.html"
        if not template_path.exists():
            logger.warning("Knowledge base template not found, using basic rendering")
            html_content = "<html><head><title>AgentSystem Knowledge Base</title></head><body>"
            html_content += "<h1>AgentSystem Knowledge Base</h1>"
            for category, articles in content.items():
                html_content += f"<h2>{category}</h2>"
                for article in articles:
                    html_content += f"<h3>{article['title']}</h3>"
                    html_content += markdown.markdown(article['content'])
            html_content += "</body></html>"
            return html_content

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        # Replace placeholders in template with content
        rendered_content = ""
        for category, articles in content.items():
            rendered_content += f"<h2>{category}</h2>"
            for article in articles:
                rendered_content += f"<h3>{article['title']}</h3>"
                rendered_content += markdown.markdown(article['content'])

        return template.replace("{{CONTENT}}", rendered_content)

    async def _render_index_page(self, content: Dict[str, Any]) -> str:
        """Render the index page for documentation"""
        template_path = self.templates_dir / "index_template.html"
        if not template_path.exists():
            logger.warning("Index template not found, using basic rendering")
            html_content = "<html><head><title>AgentSystem Documentation</title></head><body>"
            html_content += f"<h1>{content['title']}</h1>"
            for section in content['sections']:
                html_content += f"<h2><a href='{section['link']}'>{section['title']}</a></h2>"
                html_content += f"<p>{section['description']}</p>"
            html_content += "</body></html>"
            return html_content

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        # Replace placeholders in template with content
        sections_html = ""
        for section in content['sections']:
            sections_html += f"<div class='section'><h2><a href='{section['link']}'>{section['title']}</a></h2>"
            sections_html += f"<p>{section['description']}</p></div>"

        return template.replace("{{TITLE}}", content['title']).replace("{{SECTIONS}}", sections_html)
