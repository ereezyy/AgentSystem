
"""
Webhook Marketplace - AgentSystem Profit Machine
Comprehensive marketplace for 3rd party integrations and webhook management
"""

import asyncio
import json
import logging
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import asyncpg
import aiohttp
from pydantic import BaseModel, Field, validator
import uuid
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)

class WebhookStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    SUSPENDED = "suspended"

class IntegrationType(Enum):
    CRM = "crm"
    EMAIL_MARKETING = "email_marketing"
    SOCIAL_MEDIA = "social_media"
    ECOMMERCE = "ecommerce"
    PROJECT_MANAGEMENT = "project_management"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    PAYMENT = "payment"
    STORAGE = "storage"
    PRODUCTIVITY = "productivity"
    CUSTOM = "custom"

class EventType(Enum):
    # Document events
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_FAILED = "document.failed"

    # User events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"

    # Billing events
    SUBSCRIPTION_CREATED = "subscription.created"
    SUBSCRIPTION_UPDATED = "subscription.updated"
    PAYMENT_SUCCEEDED = "payment.succeeded"
    PAYMENT_FAILED = "payment.failed"

    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"

    # Agent events
    AGENT_RESPONSE = "agent.response"
    AGENT_ERROR = "agent.error"

    # Custom events
    CUSTOM_EVENT = "custom.event"

class AuthenticationType(Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    HMAC_SHA256 = "hmac_sha256"
    CUSTOM_HEADER = "custom_header"

@dataclass
class WebhookEndpoint:
    webhook_id: str
    tenant_id: str
    name: str
    url: str
    events: List[EventType]
    status: WebhookStatus
    authentication: AuthenticationType
    auth_config: Dict[str, Any]
    headers: Dict[str, str]
    retry_config: Dict[str, Any]
    filter_config: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    last_triggered: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = None

@dataclass
class IntegrationTemplate:
    template_id: str
    name: str
    description: str
    provider: str
    integration_type: IntegrationType
    logo_url: str
    documentation_url: str
    setup_instructions: List[str]
    required_credentials: List[str]
    supported_events: List[EventType]
    webhook_config: Dict[str, Any]
    example_payload: Dict[str, Any]
    is_verified: bool
    popularity_score: float
    tags: List[str]

@dataclass
class WebhookDelivery:
    delivery_id: str
    webhook_id: str
    event_type: EventType
    payload: Dict[str, Any]
    status_code: Optional[int]
    response_body: Optional[str]
    delivery_time: datetime
    response_time_ms: Optional[float]
    retry_count: int
    success: bool
    error_message: Optional[str] = None

class WebhookMarketplace:
    """
    Comprehensive webhook marketplace for 3rd party integrations
    """

    def __init__(self, db_pool: asyncpg.Pool, base_url: str):
        self.db_pool = db_pool
        self.base_url = base_url
        self.session = None
        self.integration_templates = {}
        self.webhook_handlers = {}
        self._initialize_templates()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _initialize_templates(self):
        """Initialize popular integration templates"""

        # Slack Integration
        self.integration_templates["slack"] = IntegrationTemplate(
            template_id="slack",
            name="Slack",
            description="Send notifications and updates to Slack channels",
            provider="Slack Technologies",
            integration_type=IntegrationType.COMMUNICATION,
            logo_url="https://cdn.brandfolder.io/5H442O3W/as/pl546j-7le8zk-5guop3/Slack_Monogram_Black.png",
            documentation_url="https://api.slack.com/messaging/webhooks",
            setup_instructions=[
                "Create a Slack app in your workspace",
                "Enable Incoming Webhooks",
                "Create a webhook URL for your channel",
                "Copy the webhook URL to AgentSystem"
            ],
            required_credentials=["webhook_url"],
            supported_events=[
                EventType.DOCUMENT_PROCESSED,
                EventType.WORKFLOW_COMPLETED,
                EventType.PAYMENT_SUCCEEDED,
                EventType.AGENT_RESPONSE
            ],
            webhook_config={
                "method": "POST",
                "content_type": "application/json",
                "authentication": AuthenticationType.NONE,
                "retry_attempts": 3,
                "timeout_seconds": 30
            },
            example_payload={
                "text": "Document processing completed",
                "attachments": [
                    {
                        "color": "good",
                        "fields": [
                            {"title": "Document", "value": "contract.pdf", "short": True},
                            {"title": "Status", "value": "Processed", "short": True}
                        ]
                    }
                ]
            },
            is_verified=True,
            popularity_score=9.5,
            tags=["communication", "notifications", "team", "popular"]
        )

        # Microsoft Teams Integration
        self.integration_templates["teams"] = IntegrationTemplate(
            template_id="teams",
            name="Microsoft Teams",
            description="Send notifications to Microsoft Teams channels",
            provider="Microsoft",
            integration_type=IntegrationType.COMMUNICATION,
            logo_url="https://upload.wikimedia.org/wikipedia/commons/c/c9/Microsoft_Office_Teams_%282018%E2%80%93present%29.svg",
            documentation_url="https://docs.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/",
            setup_instructions=[
                "Go to your Teams channel",
                "Click on the three dots (...) and select 'Connectors'",
                "Find 'Incoming Webhook' and click 'Configure'",
                "Provide a name and upload an image for your webhook",
                "Copy the webhook URL to AgentSystem"
            ],
            required_credentials=["webhook_url"],
            supported_events=[
                EventType.DOCUMENT_PROCESSED,
                EventType.WORKFLOW_COMPLETED,
                EventType.PAYMENT_FAILED,
                EventType.AGENT_ERROR
            ],
            webhook_config={
                "method": "POST",
                "content_type": "application/json",
                "authentication": AuthenticationType.NONE,
                "retry_attempts": 3,
                "timeout_seconds": 30
            },
            example_payload={
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "0076D7",
                "summary": "AgentSystem Notification",
                "sections": [{
                    "activityTitle": "Document Processing Complete",
                    "activitySubtitle": "contract.pdf has been processed successfully",
                    "facts": [
                        {"name": "Status", "value": "Completed"},
                        {"name": "Processing Time", "value": "2.3 seconds"}
                    ]
                }]
            },
            is_verified=True,
            popularity_score=8.7,
            tags=["communication", "microsoft", "enterprise", "notifications"]
        )

        # Discord Integration
        self.integration_templates["discord"] = IntegrationTemplate(
            template_id="discord",
            name="Discord",
            description="Send notifications to Discord channels",
            provider="Discord Inc.",
            integration_type=IntegrationType.COMMUNICATION,
            logo_url="https://assets-global.website-files.com/6257adef93867e50d84d30e2/636e0a6918e57475a843dcf5_icon_clyde_black_RGB.png",
            documentation_url="https://discord.com/developers/docs/resources/webhook",
            setup_instructions=[
                "Go to your Discord server settings",
                "Navigate to Integrations > Webhooks",
                "Click 'New Webhook'",
                "Configure the webhook and copy the URL",
                "Paste the webhook URL in AgentSystem"
            ],
            required_credentials=["webhook_url"],
            supported_events=[
                EventType.DOCUMENT_PROCESSED,
                EventType.WORKFLOW_COMPLETED,
                EventType.AGENT_RESPONSE
            ],
            webhook_config={
                "method": "POST",
                "content_type": "application/json",
                "authentication": AuthenticationType.NONE,
                "retry_attempts": 3,
                "timeout_seconds": 30
            },
            example_payload={
                "content": "Document processing notification",
                "embeds": [{
                    "title": "Document Processed",
                    "description": "contract.pdf has been successfully processed",
                    "color": 3066993,
                    "fields": [
                        {"name": "Status", "value": "Completed", "inline": True},
                        {"name": "Time", "value": "2.3s", "inline": True}
                    ]
                }]
            },
            is_verified=True,
            popularity_score=7.8,
            tags=["communication", "gaming", "community", "notifications"]
        )

        # Zapier Integration
        self.integration_templates["zapier"] = IntegrationTemplate(
            template_id="zapier",
            name="Zapier",
            description="Connect to 5000+ apps through Zapier webhooks",
            provider="Zapier",
            integration_type=IntegrationType.PRODUCTIVITY,
            logo_url="https://cdn.zapier.com/storage/services/6cf3f5a0d8b2d4c2c6ac7c5c4c8e6c6c/logo_256.png",
            documentation_url="https://zapier.com/apps/webhook/help",
            setup_instructions=[
                "Create a new Zap in Zapier",
                "Choose 'Webhooks by Zapier' as the trigger",
                "Select 'Catch Hook' as the trigger event",
                "Copy the webhook URL provided by Zapier",
                "Configure your action app in Zapier",
                "Test the integration"
            ],
            required_credentials=["webhook_url"],
            supported_events=[
                EventType.DOCUMENT_UPLOADED,
                EventType.DOCUMENT_PROCESSED,
                EventType.WORKFLOW_COMPLETED,
                EventType.USER_CREATED,
                EventType.PAYMENT_SUCCEEDED,
                EventType.CUSTOM_EVENT
            ],
            webhook_config={
                "method": "POST",
                "content_type": "application/json",
                "authentication": AuthenticationType.NONE,
                "retry_attempts": 5,
                "timeout_seconds": 30
            },
            example_payload={
                "event_type": "document.processed",
                "tenant_id": "tenant_123",
                "document_id": "doc_456",
                "document_name": "contract.pdf",
                "processing_result": "success",
                "timestamp": "2024-01-15T10:30:00Z"
            },
            is_verified=True,
            popularity_score=9.2,
            tags=["automation", "integration", "productivity", "popular"]
        )

        # Webhook.site (for testing)
        self.integration_templates["webhook_site"] = IntegrationTemplate(
            template_id="webhook_site",
            name="Webhook.site",
            description="Test and debug webhooks with temporary URLs",
            provider="Webhook.site",
            integration_type=IntegrationType.CUSTOM,
            logo_url="https://webhook.site/favicon.ico",
            documentation_url="https://webhook.site/",
            setup_instructions=[
                "Go to webhook.site",
                "Copy the unique URL provided",
                "Use this URL for testing webhooks",
                "View real-time requests in the browser"
            ],
            required_credentials=["webhook_url"],
            supported_events=list(EventType),
            webhook_config={
                "method": "POST",
                "content_type": "application/json",
                "authentication": AuthenticationType.NONE,
                "retry_attempts": 1,
                "timeout_seconds": 10
            },
            example_payload={
                "event_type": "test.event",
                "data": {"message": "This is a test webhook"}
            },
            is_verified=True,
            popularity_score=6.5,
            tags=["testing", "debugging", "development"]
        )

    async def create_webhook(self, tenant_id: str, webhook_config: Dict[str, Any]) -> WebhookEndpoint:
        """Create a new webhook endpoint"""

        webhook_id = str(uuid.uuid4())

        # Validate webhook URL
        if not self._validate_webhook_url(webhook_config['url']):
            raise ValueError("Invalid webhook URL")

        # Parse events
        events = [EventType(event) for event in webhook_config.get('events', [])]

        webhook = WebhookEndpoint(
            webhook_id=webhook_id,
            tenant_id=tenant_id,
            name=webhook_config['name'],
            url=webhook_config['url'],
            events=events,
            status=WebhookStatus.ACTIVE,
            authentication=AuthenticationType(webhook_config.get('authentication', 'none')),
            auth_config=webhook_config.get('auth_config', {}),
            headers=webhook_config.get('headers', {}),
            retry_config=webhook_config.get('retry_config', {
                'max_attempts': 3,
                'backoff_seconds': 5,
                'timeout_seconds': 30
            }),
            filter_config=webhook_config.get('filter_config'),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=webhook_config.get('metadata', {})
        )

        # Store in database
        await self._store_webhook(webhook)

        return webhook

    async def update_webhook(self, webhook_id: str, tenant_id: str,
                           updates: Dict[str, Any]) -> WebhookEndpoint:
        """Update an existing webhook"""

        webhook = await self.get_webhook(webhook_id, tenant_id)
        if not webhook:
            raise ValueError(f"Webhook {webhook_id} not found")

        # Update fields
        for field, value in updates.items():
            if hasattr(webhook, field):
                if field == 'events':
                    value = [EventType(event) for event in value]
                elif field == 'status':
                    value = WebhookStatus(value)
                elif field == 'authentication':
                    value = AuthenticationType(value)

                setattr(webhook, field, value)

        webhook.updated_at = datetime.now()

        # Update in database
        await self._update_webhook(webhook)

        return webhook

    async def delete_webhook(self, webhook_id: str, tenant_id: str) -> bool:
        """Delete a webhook endpoint"""

        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM integrations.webhooks
                WHERE webhook_id = $1 AND tenant_id = $2
            """, webhook_id, tenant_id)

            return result == "DELETE 1"

    async def get_webhook(self, webhook_id: str, tenant_id: str) -> Optional[WebhookEndpoint]:
        """Get a specific webhook"""

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM integrations.webhooks
                WHERE webhook_id = $1 AND tenant_id = $2
            """, webhook_id, tenant_id)

            if row:
                return self._row_to_webhook(row)

        return None

    async def list_webhooks(self, tenant_id: str,
                           status: Optional[WebhookStatus] = None) -> List[WebhookEndpoint]:
        """List webhooks for a tenant"""

        query = "SELECT * FROM integrations.webhooks WHERE tenant_id = $1"
        params = [tenant_id]

        if status:
            query += " AND status = $2"
            params.append(status.value)

        query += " ORDER BY created_at DESC"

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_webhook(row) for row in rows]

    async def trigger_webhook(self, event_type: EventType, tenant_id: str,
                            payload: Dict[str, Any]) -> List[WebhookDelivery]:
        """Trigger webhooks for a specific event"""

        # Get all active webhooks for this tenant and event type
        webhooks = await self._get_webhooks_for_event(tenant_id, event_type)

        deliveries = []

        for webhook in webhooks:
            # Apply filters if configured
            if webhook.filter_config and not self._passes_filter(payload, webhook.filter_config):
                continue

            # Prepare payload
            webhook_payload = self._prepare_webhook_payload(event_type, payload, webhook)

            # Deliver webhook
            delivery = await self._deliver_webhook(webhook, webhook_payload)
            deliveries.append(delivery)

            # Update webhook statistics
            await self._update_webhook_stats(webhook.webhook_id, delivery.success)

        return deliveries

    async def _deliver_webhook(self, webhook: WebhookEndpoint,
                             payload: Dict[str, Any]) -> WebhookDelivery:
        """Deliver a single webhook"""

        delivery_id = str(uuid.uuid4())
        start_time = datetime.now()

        delivery = WebhookDelivery(
            delivery_id=delivery_id,
            webhook_id=webhook.webhook_id,
            event_type=EventType(payload.get('event_type', 'custom.event')),
            payload=payload,
            status_code=None,
            response_body=None,
            delivery_time=start_time,
            response_time_ms=None,
            retry_count=0,
            success=False
        )

        max_attempts = webhook.retry_config.get('max_attempts', 3)
        backoff_seconds = webhook.retry_config.get('backoff_seconds', 5)
        timeout_seconds = webhook.retry_config.get('timeout_seconds', 30)

        for attempt in range(max_attempts):
            try:
                # Prepare headers
                headers = webhook.headers.copy()
                headers['Content-Type'] = 'application/json'
                headers['User-Agent'] = 'AgentSystem-Webhook/1.0'
                headers['X-AgentSystem-Event'] = payload.get('event_type', 'custom.event')
                headers['X-AgentSystem-Delivery'] = delivery_id

                # Add authentication
                self._add_authentication(headers, webhook, payload)

                # Make request
                async with self.session.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as response:
                    delivery.status_code = response.status
                    delivery.response_body = await response.text()
                    delivery.response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                    delivery.success = 200 <= response.status < 300

                    if delivery.success:
                        break
                    else:
                        delivery.error_message = f"HTTP {response.status}: {delivery.response_body}"

            except Exception as e:
                delivery.error_message = str(e)
                logger.error(f"Webhook delivery failed: {e}")

            delivery.retry_count = attempt + 1

            # Wait before retry (except on last attempt)
            if attempt < max_attempts - 1:
                await asyncio.sleep(backoff_seconds * (2 ** attempt))  # Exponential backoff

        # Store delivery record
        await self._store_delivery(delivery)

        return delivery

    def _add_authentication(self, headers: Dict[str, str],
                          webhook: WebhookEndpoint, payload: Dict[str, Any]):
        """Add authentication to webhook request"""

        if webhook.authentication == AuthenticationType.API_KEY:
            api_key = webhook.auth_config.get('api_key')
            key_header = webhook.auth_config.get('header', 'X-API-Key')
            if api_key:
                headers[key_header] = api_key

        elif webhook.authentication == AuthenticationType.BEARER_TOKEN:
            token = webhook.auth_config.get('token')
            if token:
                headers['Authorization'] = f'Bearer {token}'

        elif webhook.authentication == AuthenticationType.BASIC_AUTH:
            username = webhook.auth_config.get('username')
            password = webhook.auth_config.get('password')
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers['Authorization'] = f'Basic {credentials}'

        elif webhook.authentication == AuthenticationType.HMAC_SHA256:
            secret = webhook.auth_config.get('secret')
            if secret:
                payload_str = json.dumps(payload, sort_keys=True)
                signature = hmac.new(
                    secret.encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers['X-Signature-SHA256'] = f'sha256={signature}'

        elif webhook.authentication == AuthenticationType.CUSTOM_HEADER:
            custom_headers = webhook.auth_config.get('headers', {})
            headers.update(custom_headers)

    def _prepare_webhook_payload(self, event_type: EventType,
                               payload: Dict[str, Any],
                               webhook: WebhookEndpoint) -> Dict[str, Any]:
        """Prepare the webhook payload"""

        webhook_payload = {
            'event_type': event_type.value,
            'webhook_id': webhook.webhook_id,
            'tenant_id': webhook.tenant_id,
            'timestamp': datetime.now().isoformat(),
            'data': payload
        }

        return webhook_payload

    def _passes_filter(self, payload: Dict[str, Any],
                      filter_config: Dict[str, Any]) -> bool:
        """Check if payload passes the configured filters"""

        for field, condition in filter_config.items():
            if field not in payload:
                return False

            value = payload[field]

            if isinstance(condition, dict):
                operator = condition.get('operator', 'equals')
                expected = condition.get('value')

                if operator == 'equals' and value != expected:
                    return False
                elif operator == 'contains' and expected not in str(value):
                    return False
                elif operator == 'greater_than' and value <= expected:
                    return False
                elif operator == 'less_than' and value >= expected:
                    return False
            else:
                if value != condition:
                    return False

        return True

    def _validate_webhook_url(self, url: str) -> bool:
        """Validate webhook URL"""

        try:
            parsed = urlparse(url)
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except:
            return False

    async def get_integration_templates(self,
                                      integration_type: Optional[IntegrationType] = None,
                                      search_query: Optional[str] = None) -> List[IntegrationTemplate]:
        """Get available integration templates"""

        templates = list(self.integration_templates.values())

        # Filter by integration type
        if integration_type:
            templates = [t for t in templates if t.integration_type == integration_type]

        # Filter by search query
        if search_query:
            query_lower = search_query.lower()
            templates = [
                t for t in templates
                if query_lower in t.name.lower() or
                   query_lower in t.description.lower() or
                   any(query_lower in tag.lower() for tag in t.tags)
            ]

        # Sort by popularity
        templates.sort(key=lambda t: t.popularity_score, reverse=True)

        return templates

    async def get_webhook_analytics(self, tenant_id: str,
                                  days: int = 30) -> Dict[str, Any]:
        """Get webhook analytics for a tenant"""

        start_date = datetime.now() - timedelta(days=days)

        async with self.db_pool.acquire() as conn:
            # Get delivery statistics
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_deliveries,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_deliveries,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(retry_count) as avg_retries
                FROM integrations.webhook_deliveries wd
                JOIN integrations.webhooks w ON wd.webhook_id = w.webhook_id
                WHERE w.tenant_id = $1 AND wd.delivery_time >= $2
            """, tenant_id, start_date)

            # Get event type breakdown
            event_breakdown = await conn.fetch("""
                SELECT
                    event_type,
                    COUNT(*) as count,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful
                FROM integrations.webhook_deliveries wd
                JOIN integrations.webhooks w ON wd.webhook_id = w.webhook_id
                WHERE w.tenant_id = $1 AND wd.delivery_time >= $2
                GROUP BY event_type
                ORDER BY count DESC
            """, tenant_id, start_date)

            # Get webhook performance
            webhook_performance = await conn.fetch("""
                SELECT
                    w.webhook_id,
                    w.name,
                    w.url,
                    COUNT(wd.*) as total_deliveries,
                    SUM(CASE WHEN wd.success THEN 1 ELSE 0 END) as successful_deliveries,
                    AVG(wd.response_time_ms) as avg_response_time
                FROM integrations.webhooks w
                LEFT JOIN integrations.webhook_deliveries wd ON w.webhook_id = wd.webhook_id
                    AND wd.delivery_time >= $2
                WHERE w.tenant_id = $1
                GROUP BY w.webhook_id, w.name, w.url
                ORDER BY total_deliveries DESC
            """, tenant_id, start_date)

        return {
            'period_days': days,
            'total_deliveries': stats['total_deliveries'] or 0,
            'successful_deliveries': stats['successful_deliveries'] or 0,
            'success_rate': (stats['successful_deliveries'] / stats['total_deliveries']) * 100 if stats['total_deliveries'] else 0,
            'avg_response_time_ms': float(stats['avg_response_time'] or 0),
            'avg_retries': float(stats['avg_retries'] or 0),
            'event_breakdown': [dict(row) for row in event_breakdown],
            'webhook_performance': [dict(row) for row in webhook_performance]
        }

    # Database helper methods
    async def _store_webhook(self, webhook: WebhookEndpoint):
        """Store webhook in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO integrations.webhooks (
                    webhook_id, tenant_id, name, url, events, status,
                    authentication, auth_config, headers, retry_config,
                    filter_config, metadata, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
                webhook.webhook_id, webhook.tenant_id, webhook.name, webhook.url,
                json.dumps([e.value for e in webhook.events]), webhook.status.value,
                webhook.authentication.value, json.dumps(webhook.auth_config),
                json.dumps(webhook.headers), json.dumps(webhook.retry_config),
                json.dumps(webhook.filter_config) if webhook.filter_config else None,
                json.dumps(webhook.metadata), webhook.created_at, webhook.updated_at
            )

    async def _update_webhook(self, webhook: WebhookEndpoint):
        """Update webhook in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE integrations.webhooks SET
                    name = $3, url = $4, events = $5, status = $6,
                    authentication = $7, auth_config = $8, headers = $9,
                    retry_config = $10, filter_config = $11, metadata = $12,
                    updated_at = $13
                WHERE webhook_id = $1 AND tenant_id = $2
            """,
                webhook.webhook_id, webhook.tenant_id, webhook.name, webhook.url,
                json.dumps([e.value for e in webhook.events]), webhook.status.value,
                webhook.authentication.value, json.dumps(webhook.auth_config),
                json.dumps(webhook.headers), json.dumps(webhook.retry_config),
                json.dumps(webhook.filter_config) if webhook.filter_config else None,
                json.dumps(webhook.metadata), webhook.updated_at
            )

    async def _get_webhooks_for_event(self, tenant_id: str,
                                    event_type: EventType) -> List[WebhookEndpoint]:
        """Get active webhooks for a specific event type"""

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM integrations.webhooks
                WHERE tenant_id = $1 AND status = 'active'
                AND events::jsonb ? $2
            """, tenant_id, event_type.value)

            return [self._row_to_webhook(row) for row in rows]

    async def _store_delivery(self, delivery: WebhookDelivery):
        """Store webhook delivery record"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO integrations.webhook_deliveries (
                    delivery_id, webhook_id, event_type, payload, status_code,
                    response_body, delivery_time, response_time_ms, retry_count,
                    success, error_message
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                delivery.delivery_id, delivery.webhook_id, delivery.event_type.value,
                json.dumps(delivery.payload), delivery.status_code, delivery.response_body,
                delivery.delivery_time, delivery.response_time_ms, delivery.retry_count,
                delivery.success, delivery.error_message
            )

    async def _update_webhook_stats(self, webhook_id: str, success: bool):
        """Update webhook success/failure statistics"""

        async with self.db_pool.acquire() as conn:
            if success:
                await conn.execute("""
                    UPDATE integrations.webhooks
                    SET success_count = success_count + 1, last_triggered = NOW()
                    WHERE webhook_id = $1
                """, webhook_id)
            else:
                await conn.execute("""
                    UPDATE integrations.webhooks
                    SET failure_count = failure_count + 1, last_triggered = NOW()
                    WHERE webhook_id = $1
                """, webhook_id)

    def _row_to_webhook(self, row) -> WebhookEndpoint:
        """Convert database row to WebhookEndpoint object"""

        return WebhookEndpoint(
            webhook_id=row['webhook_id'],
            tenant_id=row['tenant_id'],
            name=row['name'],
            url=row['url'],
            events=[EventType(e) for e in json.loads(row['events'])],
            status=WebhookStatus(row['status']),
            authentication=AuthenticationType(row['authentication']),
            auth_config=json.loads(row['auth_config']) if row['auth_config'] else {},
            headers=json.loads(row['headers']) if row['headers'] else {},
            retry_config=json.loads(row['retry_config']) if row['retry_config'] else {},
            filter_config=json.loads(row['filter_config']) if row['filter_config'] else None,
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            last_triggered=row.get('last_triggered'),
            success_count=row.get('success_count', 0),
            failure_count=row.get('failure_count', 0),
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )

# Database schema for webhook marketplace
WEBHOOK_MARKETPLACE_SCHEMA_SQL = """
-- Webhook marketplace schema
CREATE SCHEMA IF NOT EXISTS integrations;

-- Webhooks table
CREATE TABLE IF NOT EXISTS integrations.webhooks (
    webhook_id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    url TEXT NOT NULL,
    events JSONB NOT NULL DEFAULT '[]',
    status VARCHAR(50) DEFAULT 'active',
    authentication VARCHAR(50) DEFAULT 'none',
    auth_config JSONB DEFAULT '{}',
    headers JSONB DEFAULT '{}',
    retry_config JSONB DEFAULT '{}',
    filter_config JSONB,
    metadata JSONB DEFAULT '{}',
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_triggered TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Webhook deliveries table
CREATE TABLE IF NOT EXISTS integrations.webhook_deliveries (
    delivery_id UUID PRIMARY KEY,
    webhook_id UUID REFERENCES integrations.webhooks(webhook_id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    status_code INTEGER,
    response_body TEXT,
    delivery_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time_ms FLOAT,
    retry_count INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT
);

-- Integration templates table
CREATE TABLE IF NOT EXISTS integrations.integration_templates (
    template_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    provider VARCHAR(500),
    integration_type VARCHAR(100),
    logo_url TEXT,
    documentation_url TEXT,
    setup_instructions JSONB DEFAULT '[]',
    required_credentials JSONB DEFAULT '[]',
    supported_events JSONB DEFAULT '[]',
    webhook_config JSONB DEFAULT '{}',
    example_payload JSONB DEFAULT '{}',
    is_verified BOOLEAN DEFAULT FALSE,
    popularity_score FLOAT DEFAULT 0,
    tags JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tenant integrations table (tracks which integrations tenants have enabled)
CREATE TABLE IF NOT EXISTS integrations.tenant_integrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    template_id VARCHAR(100) REFERENCES integrations.integration_templates(template_id),
    webhook_id UUID REFERENCES integrations.webhooks(webhook_id) ON DELETE CASCADE,
    configuration JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(tenant_id, template_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_webhooks_tenant_status
ON integrations.webhooks(tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_webhooks_events
ON integrations.webhooks USING gin(events);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_webhook_time
ON integrations.webhook_deliveries(webhook_id, delivery_time DESC);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_success
ON integrations.webhook_deliveries(success, delivery_time DESC);

CREATE INDEX IF NOT EXISTS idx_integration_templates_type
ON integrations.integration_templates(integration_type, popularity_score DESC);

CREATE INDEX IF NOT EXISTS idx_integration_templates_verified
ON integrations.integration_templates(is_verified, popularity_score DESC);

CREATE INDEX IF NOT EXISTS idx_tenant_integrations_tenant
ON integrations.tenant_integrations(tenant_id, is_active);

-- Views for analytics
CREATE OR REPLACE VIEW integrations.webhook_analytics AS
SELECT
    w.webhook_id,
    w.tenant_id,
    w.name,
    w.url,
    w.status,
    w.success_count,
    w.failure_count,
    CASE
        WHEN (w.success_count + w.failure_count) > 0
        THEN (w.success_count::float / (w.success_count + w.failure_count)) * 100
        ELSE 0
    END as success_rate,
    COUNT(wd.delivery_id) as recent_deliveries,
    AVG(wd.response_time_ms) as avg_response_time,
    w.last_triggered,
    w.created_at
FROM integrations.webhooks w
LEFT JOIN integrations.webhook_deliveries wd ON w.webhook_id = wd.webhook_id
    AND wd.delivery_time >= NOW() - INTERVAL '30 days'
GROUP BY w.webhook_id, w.tenant_id, w.name, w.url, w.status,
         w.success_count, w.failure_count, w.last_triggered, w.created_at;

-- Function to clean up old delivery records
CREATE OR REPLACE FUNCTION integrations.cleanup_old_deliveries()
RETURNS void AS $$
BEGIN
    DELETE FROM integrations.webhook_deliveries
    WHERE delivery_time < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- Trigger to update webhook updated_at timestamp
CREATE OR REPLACE FUNCTION integrations.update_webhook_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_webhook_updated_at
    BEFORE UPDATE ON integrations.webhooks
    FOR EACH ROW
    EXECUTE FUNCTION integrations.update_webhook_timestamp();

-- Comments for documentation
COMMENT ON TABLE integrations.webhooks IS 'Webhook endpoints configured by tenants for receiving event notifications';
COMMENT ON TABLE integrations.webhook_deliveries IS 'Log of webhook delivery attempts and their results';
COMMENT ON TABLE integrations.integration_templates IS 'Pre-built integration templates for popular services';
COMMENT ON TABLE integrations.tenant_integrations IS 'Tracks which integrations each tenant has enabled';
"""

# Pydantic models for API
class WebhookCreateRequest(BaseModel):
    name: str = Field(..., description="Webhook name")
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="List of event types to subscribe to")
    authentication: str = Field("none", description="Authentication type")
    auth_config: Optional[Dict[str, Any]] = Field(None, description="Authentication configuration")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers")
    retry_config: Optional[Dict[str, Any]] = Field(None, description="Retry configuration")
    filter_config: Optional[Dict[str, Any]] = Field(None, description="Event filtering configuration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class WebhookUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Webhook name")
    url: Optional[str] = Field(None, description="Webhook URL")
    events: Optional[List[str]] = Field(None, description="List of event types to subscribe to")
    status: Optional[str] = Field(None, description="Webhook status")
    authentication: Optional[str] = Field(None, description="Authentication type")
    auth_config: Optional[Dict[str, Any]] = Field(None, description="Authentication configuration")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers")
    retry_config: Optional[Dict[str, Any]] = Field(None, description="Retry configuration")
    filter_config: Optional[Dict[str, Any]] = Field(None, description="Event filtering configuration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class WebhookTestRequest(BaseModel):
    webhook_id: str = Field(..., description="Webhook ID to test")
    event_type: str = Field("test.event", description="Event type for test")
    test_payload: Optional[Dict[str, Any]] = Field(None, description="Custom test payload")

# Export main classes
__all__ = [
    'WebhookMarketplace', 'WebhookEndpoint', 'IntegrationTemplate', 'WebhookDelivery',
    'WebhookStatus', 'IntegrationType', 'EventType', 'AuthenticationType',
    'WebhookCreateRequest', 'WebhookUpdateRequest', 'WebhookTestRequest',
    'WEBHOOK_MARKETPLACE_SCHEMA_SQL'
]
