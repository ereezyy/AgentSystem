"""
Microsoft 365 Integration - AgentSystem Profit Machine
Comprehensive Microsoft 365 connector for Teams, Outlook, SharePoint, and OneDrive
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
import asyncpg
import aiohttp
from pydantic import BaseModel, Field, validator
import uuid
import base64
from urllib.parse import urlencode, parse_qs
import msal

logger = logging.getLogger(__name__)

class Microsoft365Service(Enum):
    TEAMS = "teams"
    OUTLOOK = "outlook"
    SHAREPOINT = "sharepoint"
    ONEDRIVE = "onedrive"
    CALENDAR = "calendar"
    POWER_AUTOMATE = "power_automate"

class TeamsEventType(Enum):
    MESSAGE = "message"
    MENTION = "mention"
    MEETING_START = "meeting_start"
    MEETING_END = "meeting_end"
    CHANNEL_CREATED = "channel_created"
    MEMBER_ADDED = "member_added"
    FILE_SHARED = "file_shared"

class OutlookEventType(Enum):
    EMAIL_RECEIVED = "email_received"
    CALENDAR_EVENT = "calendar_event"
    MEETING_REQUEST = "meeting_request"
    TASK_CREATED = "task_created"

class SharePointEventType(Enum):
    FILE_UPLOADED = "file_uploaded"
    FILE_MODIFIED = "file_modified"
    LIST_ITEM_CREATED = "list_item_created"
    WORKFLOW_COMPLETED = "workflow_completed"

@dataclass
class Microsoft365Tenant:
    tenant_id: str
    agentsystem_tenant_id: str
    tenant_name: str
    directory_id: str
    access_token: str
    refresh_token: str
    token_expires_at: datetime
    scope: str
    admin_consent: bool
    enabled_services: List[Microsoft365Service]
    webhook_url: Optional[str]
    is_active: bool
    installed_at: datetime
    updated_at: datetime

@dataclass
class TeamsChannel:
    channel_id: str
    tenant_id: str
    team_id: str
    channel_name: str
    channel_type: str
    ai_assistance_enabled: bool
    notification_types: List[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class OutlookEmail:
    email_id: str
    tenant_id: str
    mailbox_id: str
    subject: str
    sender: str
    recipients: List[str]
    body: str
    received_at: datetime
    ai_processed: bool
    ai_summary: Optional[str]
    ai_priority: Optional[str]
    created_at: datetime

@dataclass
class SharePointDocument:
    document_id: str
    tenant_id: str
    site_id: str
    document_name: str
    document_path: str
    content_type: str
    size_bytes: int
    ai_processed: bool
    ai_summary: Optional[str]
    ai_tags: List[str]
    created_at: datetime
    modified_at: datetime

class Microsoft365Connector:
    """
    Comprehensive Microsoft 365 integration connector
    """

    def __init__(self, db_pool: asyncpg.Pool, client_id: str, client_secret: str,
                 tenant_id: str = "common", openai_client=None):
        self.db_pool = db_pool
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.openai_client = openai_client
        self.session = None

        # Microsoft Graph API endpoints
        self.graph_url = "https://graph.microsoft.com/v1.0"
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"

        # Required scopes for full functionality
        self.scopes = [
            "https://graph.microsoft.com/User.Read",
            "https://graph.microsoft.com/Mail.ReadWrite",
            "https://graph.microsoft.com/Calendars.ReadWrite",
            "https://graph.microsoft.com/Files.ReadWrite.All",
            "https://graph.microsoft.com/Sites.ReadWrite.All",
            "https://graph.microsoft.com/ChannelMessage.Send",
            "https://graph.microsoft.com/Team.ReadBasic.All",
            "https://graph.microsoft.com/TeamMember.Read.All",
            "https://graph.microsoft.com/Directory.Read.All"
        ]

        # Initialize MSAL app
        self.msal_app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=self.authority
        )

        self.tenants_cache = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def get_oauth_url(self, agentsystem_tenant_id: str, redirect_uri: str,
                     admin_consent: bool = False) -> str:
        """Generate OAuth URL for Microsoft 365 app installation"""

        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': redirect_uri,
            'scope': ' '.join(self.scopes),
            'state': agentsystem_tenant_id,
            'response_mode': 'query'
        }

        if admin_consent:
            params['prompt'] = 'admin_consent'

        return f"{self.authority}/oauth2/v2.0/authorize?{urlencode(params)}"

    async def exchange_code_for_tokens(self, agentsystem_tenant_id: str, code: str,
                                     redirect_uri: str) -> Microsoft365Tenant:
        """Exchange authorization code for access tokens"""

        try:
            result = self.msal_app.acquire_token_by_authorization_code(
                code=code,
                scopes=self.scopes,
                redirect_uri=redirect_uri
            )

            if 'error' in result:
                raise ValueError(f"Token exchange failed: {result['error_description']}")

            # Get tenant information
            tenant_info = await self._get_tenant_info(result['access_token'])

            tenant_id = str(uuid.uuid4())

            tenant = Microsoft365Tenant(
                tenant_id=tenant_id,
                agentsystem_tenant_id=agentsystem_tenant_id,
                tenant_name=tenant_info.get('displayName', 'Unknown'),
                directory_id=tenant_info.get('id'),
                access_token=result['access_token'],
                refresh_token=result.get('refresh_token'),
                token_expires_at=datetime.now() + timedelta(seconds=result['expires_in']),
                scope=result.get('scope', ' '.join(self.scopes)),
                admin_consent=False,  # Will be updated based on permissions
                enabled_services=[Microsoft365Service.TEAMS, Microsoft365Service.OUTLOOK],
                webhook_url=None,
                is_active=True,
                installed_at=datetime.now(),
                updated_at=datetime.now()
            )

            await self._store_tenant(tenant)
            self.tenants_cache[tenant_id] = tenant

            return tenant

        except Exception as e:
            logger.error(f"Microsoft 365 token exchange failed: {e}")
            raise

    async def refresh_access_token(self, tenant: Microsoft365Tenant) -> Microsoft365Tenant:
        """Refresh expired access token"""

        try:
            result = self.msal_app.acquire_token_by_refresh_token(
                refresh_token=tenant.refresh_token,
                scopes=self.scopes
            )

            if 'error' in result:
                raise ValueError(f"Token refresh failed: {result['error_description']}")

            tenant.access_token = result['access_token']
            tenant.token_expires_at = datetime.now() + timedelta(seconds=result['expires_in'])
            tenant.updated_at = datetime.now()

            if 'refresh_token' in result:
                tenant.refresh_token = result['refresh_token']

            await self._update_tenant(tenant)
            self.tenants_cache[tenant.tenant_id] = tenant

            return tenant

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise

    async def send_teams_message(self, tenant_id: str, team_id: str, channel_id: str,
                               message: str, message_type: str = "text") -> bool:
        """Send message to Teams channel"""

        tenant = await self._get_tenant(tenant_id)
        if not tenant:
            return False

        # Ensure token is valid
        if datetime.now() >= tenant.token_expires_at:
            tenant = await self.refresh_access_token(tenant)

        url = f"{self.graph_url}/teams/{team_id}/channels/{channel_id}/messages"
        headers = {
            'Authorization': f'Bearer {tenant.access_token}',
            'Content-Type': 'application/json'
        }

        payload = {
            'body': {
                'contentType': 'html' if message_type == 'html' else 'text',
                'content': message
            }
        }

        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                return response.status == 201
        except Exception as e:
            logger.error(f"Failed to send Teams message: {e}")
            return False

    async def send_outlook_email(self, tenant_id: str, to_recipients: List[str],
                               subject: str, body: str, body_type: str = "html") -> bool:
        """Send email via Outlook"""

        tenant = await self._get_tenant(tenant_id)
        if not tenant:
            return False

        # Ensure token is valid
        if datetime.now() >= tenant.token_expires_at:
            tenant = await self.refresh_access_token(tenant)

        url = f"{self.graph_url}/me/sendMail"
        headers = {
            'Authorization': f'Bearer {tenant.access_token}',
            'Content-Type': 'application/json'
        }

        payload = {
            'message': {
                'subject': subject,
                'body': {
                    'contentType': body_type,
                    'content': body
                },
                'toRecipients': [
                    {
                        'emailAddress': {
                            'address': email
                        }
                    } for email in to_recipients
                ]
            }
        }

        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                return response.status == 202
        except Exception as e:
            logger.error(f"Failed to send Outlook email: {e}")
            return False

    async def create_calendar_event(self, tenant_id: str, subject: str, start_time: datetime,
                                  end_time: datetime, attendees: List[str] = None,
                                  body: str = None) -> Optional[str]:
        """Create calendar event in Outlook"""

        tenant = await self._get_tenant(tenant_id)
        if not tenant:
            return None

        # Ensure token is valid
        if datetime.now() >= tenant.token_expires_at:
            tenant = await self.refresh_access_token(tenant)

        url = f"{self.graph_url}/me/events"
        headers = {
            'Authorization': f'Bearer {tenant.access_token}',
            'Content-Type': 'application/json'
        }

        payload = {
            'subject': subject,
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': 'UTC'
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': 'UTC'
            }
        }

        if body:
            payload['body'] = {
                'contentType': 'html',
                'content': body
            }

        if attendees:
            payload['attendees'] = [
                {
                    'emailAddress': {
                        'address': email
                    }
                } for email in attendees
            ]

        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 201:
                    result = await response.json()
                    return result.get('id')
                return None
        except Exception as e:
            logger.error(f"Failed to create calendar event: {e}")
            return None

    async def upload_sharepoint_file(self, tenant_id: str, site_id: str,
                                   file_name: str, file_content: bytes,
                                   folder_path: str = "/") -> Optional[str]:
        """Upload file to SharePoint"""

        tenant = await self._get_tenant(tenant_id)
        if not tenant:
            return None

        # Ensure token is valid
        if datetime.now() >= tenant.token_expires_at:
            tenant = await self.refresh_access_token(tenant)

        # Construct upload URL
        file_path = f"{folder_path.rstrip('/')}/{file_name}"
        url = f"{self.graph_url}/sites/{site_id}/drive/root:{file_path}:/content"

        headers = {
            'Authorization': f'Bearer {tenant.access_token}',
            'Content-Type': 'application/octet-stream'
        }

        try:
            async with self.session.put(url, data=file_content, headers=headers) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    return result.get('id')
                return None
        except Exception as e:
            logger.error(f"Failed to upload SharePoint file: {e}")
            return None

    async def process_teams_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Teams webhook event"""

        try:
            event_type = webhook_data.get('changeType')
            resource_data = webhook_data.get('resourceData', {})

            if event_type == 'created' and 'messages' in webhook_data.get('resource', ''):
                # New message in Teams
                return await self._handle_teams_message(resource_data)

            elif event_type == 'created' and 'channels' in webhook_data.get('resource', ''):
                # New channel created
                return await self._handle_teams_channel_created(resource_data)

            return {'status': 'processed'}

        except Exception as e:
            logger.error(f"Teams webhook processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

    async def process_outlook_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Outlook webhook event"""

        try:
            event_type = webhook_data.get('changeType')
            resource_data = webhook_data.get('resourceData', {})

            if event_type == 'created' and 'messages' in webhook_data.get('resource', ''):
                # New email received
                return await self._handle_outlook_email(resource_data)

            elif event_type == 'created' and 'events' in webhook_data.get('resource', ''):
                # New calendar event
                return await self._handle_outlook_calendar_event(resource_data)

            return {'status': 'processed'}

        except Exception as e:
            logger.error(f"Outlook webhook processing failed: {e}")
            return {'status': 'error', 'message': str(e)}

    async def analyze_email_with_ai(self, tenant_id: str, email_content: str) -> Dict[str, Any]:
        """Analyze email content using AI"""

        if not self.openai_client:
            return {'summary': 'AI analysis not available', 'priority': 'medium'}

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze this email and provide: 1) A brief summary (max 100 words), 2) Priority level (high/medium/low), 3) Suggested action items. Format as JSON."
                    },
                    {
                        "role": "user",
                        "content": email_content
                    }
                ],
                max_tokens=300
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            logger.error(f"Email AI analysis failed: {e}")
            return {'summary': 'AI analysis failed', 'priority': 'medium'}

    async def analyze_document_with_ai(self, tenant_id: str, document_content: str,
                                     document_type: str) -> Dict[str, Any]:
        """Analyze document content using AI"""

        if not self.openai_client:
            return {'summary': 'AI analysis not available', 'tags': []}

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"Analyze this {document_type} document and provide: 1) A summary, 2) Key topics/tags, 3) Document classification. Format as JSON."
                    },
                    {
                        "role": "user",
                        "content": document_content[:4000]  # Limit content size
                    }
                ],
                max_tokens=400
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            logger.error(f"Document AI analysis failed: {e}")
            return {'summary': 'AI analysis failed', 'tags': []}

    # Private helper methods
    async def _get_tenant_info(self, access_token: str) -> Dict[str, Any]:
        """Get tenant information from Microsoft Graph"""

        url = f"{self.graph_url}/organization"
        headers = {'Authorization': f'Bearer {access_token}'}

        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                return result.get('value', [{}])[0] if result.get('value') else {}
            return {}

    async def _handle_teams_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Teams message event"""

        # Extract message details and process with AI if needed
        # Store in database and generate response if appropriate
        return {'status': 'message_processed'}

    async def _handle_teams_channel_created(self, channel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Teams channel created event"""

        # Store channel information in database
        return {'status': 'channel_processed'}

    async def _handle_outlook_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Outlook email event"""

        # Process email with AI analysis and store results
        return {'status': 'email_processed'}

    async def _handle_outlook_calendar_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Outlook calendar event"""

        # Process calendar event
        return {'status': 'calendar_processed'}

    # Database operations
    async def _store_tenant(self, tenant: Microsoft365Tenant):
        """Store Microsoft 365 tenant in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO integrations.microsoft365_tenants (
                    tenant_id, agentsystem_tenant_id, tenant_name, directory_id,
                    access_token, refresh_token, token_expires_at, scope,
                    admin_consent, enabled_services, webhook_url, is_active,
                    installed_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
                tenant.tenant_id, tenant.agentsystem_tenant_id, tenant.tenant_name,
                tenant.directory_id, tenant.access_token, tenant.refresh_token,
                tenant.token_expires_at, tenant.scope, tenant.admin_consent,
                [service.value for service in tenant.enabled_services],
                tenant.webhook_url, tenant.is_active, tenant.installed_at,
                tenant.updated_at
            )

    async def _update_tenant(self, tenant: Microsoft365Tenant):
        """Update Microsoft 365 tenant in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE integrations.microsoft365_tenants
                SET access_token = $2, refresh_token = $3, token_expires_at = $4,
                    updated_at = $5
                WHERE tenant_id = $1
            """, tenant.tenant_id, tenant.access_token, tenant.refresh_token,
                tenant.token_expires_at, tenant.updated_at)

    async def _get_tenant(self, tenant_id: str) -> Optional[Microsoft365Tenant]:
        """Get tenant from database"""

        if tenant_id in self.tenants_cache:
            return self.tenants_cache[tenant_id]

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM integrations.microsoft365_tenants
                WHERE tenant_id = $1 AND is_active = true
            """, tenant_id)

            if row:
                tenant = Microsoft365Tenant(
                    tenant_id=row['tenant_id'],
                    agentsystem_tenant_id=row['agentsystem_tenant_id'],
                    tenant_name=row['tenant_name'],
                    directory_id=row['directory_id'],
                    access_token=row['access_token'],
                    refresh_token=row['refresh_token'],
                    token_expires_at=row['token_expires_at'],
                    scope=row['scope'],
                    admin_consent=row['admin_consent'],
                    enabled_services=[Microsoft365Service(service) for service in row['enabled_services']],
                    webhook_url=row['webhook_url'],
                    is_active=row['is_active'],
                    installed_at=row['installed_at'],
                    updated_at=row['updated_at']
                )

                self.tenants_cache[tenant_id] = tenant
                return tenant

        return None

# Export main classes
__all__ = [
    'Microsoft365Connector', 'Microsoft365Tenant', 'TeamsChannel',
    'OutlookEmail', 'SharePointDocument', 'Microsoft365Service',
    'TeamsEventType', 'OutlookEventType', 'SharePointEventType'
]
