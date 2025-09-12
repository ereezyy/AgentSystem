"""
Slack Bot Integration - AgentSystem Profit Machine
Comprehensive Slack bot for team collaboration, notifications, and AI assistance
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
import hmac
import hashlib
import time
from urllib.parse import urlencode, parse_qs

logger = logging.getLogger(__name__)

class SlackEventType(Enum):
    MESSAGE = "message"
    APP_MENTION = "app_mention"
    REACTION_ADDED = "reaction_added"
    CHANNEL_CREATED = "channel_created"
    MEMBER_JOINED_CHANNEL = "member_joined_channel"
    FILE_SHARED = "file_shared"
    WORKFLOW_STEP_EXECUTE = "workflow_step_execute"

class SlackCommandType(Enum):
    ANALYZE = "/analyze"
    SUMMARIZE = "/summarize"
    GENERATE = "/generate"
    TRANSLATE = "/translate"
    SCHEDULE = "/schedule"
    REPORT = "/report"
    HELP = "/help"

class NotificationType(Enum):
    DOCUMENT_PROCESSED = "document_processed"
    WORKFLOW_COMPLETED = "workflow_completed"
    LEAD_QUALIFIED = "lead_qualified"
    DEAL_WON = "deal_won"
    CHURN_RISK = "churn_risk"
    SYSTEM_ALERT = "system_alert"
    USAGE_THRESHOLD = "usage_threshold"

@dataclass
class SlackWorkspace:
    workspace_id: str
    tenant_id: str
    team_id: str
    team_name: str
    bot_token: str
    bot_user_id: str
    access_token: str
    scope: str
    webhook_url: Optional[str]
    is_active: bool
    installed_at: datetime
    updated_at: datetime

@dataclass
class SlackChannel:
    channel_id: str
    workspace_id: str
    channel_name: str
    is_private: bool
    is_member: bool
    notification_types: List[NotificationType]
    ai_assistance_enabled: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class SlackMessage:
    message_id: str
    workspace_id: str
    channel_id: str
    user_id: str
    text: str
    timestamp: str
    thread_ts: Optional[str]
    message_type: str
    ai_response: Optional[str]
    processed_at: Optional[datetime]
    created_at: datetime

class SlackBot:
    """
    Comprehensive Slack bot for team collaboration and AI assistance
    """

    def __init__(self, db_pool: asyncpg.Pool, client_id: str, client_secret: str,
                 signing_secret: str, openai_client=None):
        self.db_pool = db_pool
        self.client_id = client_id
        self.client_secret = client_secret
        self.signing_secret = signing_secret
        self.openai_client = openai_client
        self.session = None
        self.workspaces_cache = {}

        # Slack API endpoints
        self.base_url = "https://slack.com/api"

        # Command handlers
        self.command_handlers = {
            SlackCommandType.ANALYZE: self._handle_analyze_command,
            SlackCommandType.SUMMARIZE: self._handle_summarize_command,
            SlackCommandType.GENERATE: self._handle_generate_command,
            SlackCommandType.TRANSLATE: self._handle_translate_command,
            SlackCommandType.SCHEDULE: self._handle_schedule_command,
            SlackCommandType.REPORT: self._handle_report_command,
            SlackCommandType.HELP: self._handle_help_command
        }

        # Event handlers
        self.event_handlers = {
            SlackEventType.MESSAGE: self._handle_message_event,
            SlackEventType.APP_MENTION: self._handle_mention_event,
            SlackEventType.REACTION_ADDED: self._handle_reaction_event,
            SlackEventType.FILE_SHARED: self._handle_file_shared_event
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def get_oauth_url(self, tenant_id: str, redirect_uri: str) -> str:
        """Generate OAuth URL for Slack app installation"""

        scopes = [
            'app_mentions:read', 'channels:history', 'channels:read', 'chat:write',
            'commands', 'files:read', 'groups:history', 'groups:read', 'im:history',
            'im:read', 'mpim:history', 'mpim:read', 'reactions:read', 'team:read',
            'users:read', 'workflow.steps:execute', 'channels:join', 'chat:write.public'
        ]

        params = {
            'client_id': self.client_id,
            'scope': ','.join(scopes),
            'redirect_uri': redirect_uri,
            'state': tenant_id
        }

        return f"https://slack.com/oauth/v2/authorize?{urlencode(params)}"

    async def exchange_code_for_tokens(self, tenant_id: str, code: str,
                                     redirect_uri: str) -> SlackWorkspace:
        """Exchange authorization code for access tokens"""

        token_url = f"{self.base_url}/oauth.v2.access"

        payload = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'redirect_uri': redirect_uri
        }

        async with self.session.post(token_url, data=payload) as response:
            if response.status == 200:
                token_data = await response.json()

                if token_data.get('ok'):
                    workspace_id = str(uuid.uuid4())

                    workspace = SlackWorkspace(
                        workspace_id=workspace_id,
                        tenant_id=tenant_id,
                        team_id=token_data['team']['id'],
                        team_name=token_data['team']['name'],
                        bot_token=token_data['access_token'],
                        bot_user_id=token_data['bot_user_id'],
                        access_token=token_data['access_token'],
                        scope=token_data['scope'],
                        webhook_url=token_data.get('incoming_webhook', {}).get('url'),
                        is_active=True,
                        installed_at=datetime.now(),
                        updated_at=datetime.now()
                    )

                    await self._store_workspace(workspace)
                    self.workspaces_cache[workspace_id] = workspace

                    return workspace
                else:
                    raise ValueError(f"OAuth exchange failed: {token_data.get('error')}")
            else:
                raise ValueError(f"HTTP error {response.status}")

    async def verify_request(self, headers: Dict[str, str], body: str) -> bool:
        """Verify Slack request signature"""

        timestamp = headers.get('X-Slack-Request-Timestamp', '')
        signature = headers.get('X-Slack-Signature', '')

        if not timestamp or not signature:
            return False

        # Check timestamp (prevent replay attacks)
        if abs(time.time() - int(timestamp)) > 60 * 5:  # 5 minutes
            return False

        # Verify signature
        sig_basestring = f"v0:{timestamp}:{body}"
        expected_signature = 'v0=' + hmac.new(
            self.signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected_signature, signature)

    async def handle_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Slack events"""

        event_type = event_data.get('type')

        if event_type == 'url_verification':
            return {'challenge': event_data.get('challenge')}

        if event_type == 'event_callback':
            event = event_data.get('event', {})
            event_type = SlackEventType(event.get('type'))

            handler = self.event_handlers.get(event_type)
            if handler:
                return await handler(event, event_data)

        return {'status': 'ok'}

    async def handle_slash_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Slack slash commands"""

        command = command_data.get('command')

        try:
            command_type = SlackCommandType(command)
            handler = self.command_handlers.get(command_type)

            if handler:
                return await handler(command_data)
            else:
                return {
                    'response_type': 'ephemeral',
                    'text': f"Unknown command: {command}"
                }

        except ValueError:
            return {
                'response_type': 'ephemeral',
                'text': f"Unsupported command: {command}"
            }

    async def _handle_message_event(self, event: Dict[str, Any],
                                  event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle message events"""

        # Skip bot messages and message changes
        if event.get('subtype') in ['bot_message', 'message_changed', 'message_deleted']:
            return {'status': 'ok'}

        workspace = await self._get_workspace_by_team_id(event_data.get('team_id'))
        if not workspace:
            return {'status': 'workspace_not_found'}

        # Store message
        message = SlackMessage(
            message_id=str(uuid.uuid4()),
            workspace_id=workspace.workspace_id,
            channel_id=event.get('channel'),
            user_id=event.get('user'),
            text=event.get('text', ''),
            timestamp=event.get('ts'),
            thread_ts=event.get('thread_ts'),
            message_type='message',
            ai_response=None,
            processed_at=None,
            created_at=datetime.now()
        )

        await self._store_message(message)

        # Check if message needs AI response
        if self._should_respond_to_message(message.text):
            ai_response = await self._generate_ai_response(message.text, workspace)

            if ai_response:
                await self._send_message(
                    workspace,
                    event.get('channel'),
                    ai_response,
                    thread_ts=event.get('ts')
                )

                message.ai_response = ai_response
                message.processed_at = datetime.now()
                await self._update_message(message)

        return {'status': 'ok'}

    async def _handle_mention_event(self, event: Dict[str, Any],
                                  event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle app mention events"""

        workspace = await self._get_workspace_by_team_id(event_data.get('team_id'))
        if not workspace:
            return {'status': 'workspace_not_found'}

        # Extract text without the mention
        text = event.get('text', '')
        bot_mention = f"<@{workspace.bot_user_id}>"
        if bot_mention in text:
            text = text.replace(bot_mention, '').strip()

        # Generate AI response
        ai_response = await self._generate_ai_response(text, workspace)

        if ai_response:
            await self._send_message(
                workspace,
                event.get('channel'),
                ai_response,
                thread_ts=event.get('ts')
            )

        return {'status': 'ok'}

    async def _handle_analyze_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /analyze command"""

        text = command_data.get('text', '').strip()

        if not text:
            return {
                'response_type': 'ephemeral',
                'text': 'Please provide text to analyze. Usage: `/analyze [your text]`'
            }

        # Generate analysis using AI
        analysis = await self._analyze_text(text)

        return {
            'response_type': 'in_channel',
            'blocks': [
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f"*Analysis Results:*\n{analysis}"
                    }
                }
            ]
        }

    async def _handle_help_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /help command"""

        help_text = """
*AgentSystem Bot Commands:*

• `/analyze [text]` - Analyze text for insights, sentiment, and key points
• `/summarize [text]` - Create a concise summary of long text
• `/generate [prompt]` - Generate content based on your prompt
• `/translate [text]` - Translate text to different languages
• `/schedule [task]` - Schedule tasks and reminders
• `/report` - Get usage and performance reports

*Features:*
• AI assistance in channels (mention @AgentSystem)
• Automatic document processing when files are shared
• Real-time notifications for workflows and alerts
• Integration with CRM and business tools

Need help? Contact your admin or visit our documentation.
        """

        return {
            'response_type': 'ephemeral',
            'text': help_text
        }

    async def send_notification(self, workspace_id: str, channel_id: str,
                              notification_type: NotificationType,
                              data: Dict[str, Any]) -> bool:
        """Send notification to Slack channel"""

        workspace = await self._get_workspace(workspace_id)
        if not workspace:
            return False

        # Generate notification message
        message = self._format_notification(notification_type, data)

        return await self._send_message(workspace, channel_id, message)

    def _format_notification(self, notification_type: NotificationType,
                           data: Dict[str, Any]) -> Dict[str, Any]:
        """Format notification message based on type"""

        if notification_type == NotificationType.DOCUMENT_PROCESSED:
            return {
                'blocks': [
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': f"📄 *Document Processed*\n{data.get('filename')} has been successfully processed."
                        }
                    }
                ]
            }

        elif notification_type == NotificationType.LEAD_QUALIFIED:
            return {
                'blocks': [
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': f"🎯 *New Qualified Lead*\n{data.get('lead_name')} from {data.get('company')}"
                        }
                    }
                ]
            }

        # Default notification format
        return {
            'text': f"{notification_type.value}: {json.dumps(data, indent=2)}"
        }

    async def _send_message(self, workspace: SlackWorkspace, channel: str,
                          message: Union[str, Dict], thread_ts: str = None) -> bool:
        """Send message to Slack channel"""

        url = f"{self.base_url}/chat.postMessage"
        headers = {
            'Authorization': f'Bearer {workspace.bot_token}',
            'Content-Type': 'application/json'
        }

        payload = {
            'channel': channel,
            'as_user': True
        }

        if isinstance(message, str):
            payload['text'] = message
        else:
            payload.update(message)

        if thread_ts:
            payload['thread_ts'] = thread_ts

        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                result = await response.json()
                return result.get('ok', False)
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False

    async def _generate_ai_response(self, text: str, workspace: SlackWorkspace) -> Optional[str]:
        """Generate AI response for user message"""

        if not self.openai_client:
            return None

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are AgentSystem AI assistant. Provide helpful, concise responses to team questions. Keep responses under 500 characters for Slack."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return None

    def _should_respond_to_message(self, text: str) -> bool:
        """Determine if message needs AI response"""

        trigger_keywords = [
            '?', 'help', 'how', 'what', 'why', 'when', 'where', 'who',
            'explain', 'analyze', 'summarize', 'generate', 'create'
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in trigger_keywords)

    # Database operations
    async def _store_workspace(self, workspace: SlackWorkspace):
        """Store Slack workspace in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO integrations.slack_workspaces (
                    workspace_id, tenant_id, team_id, team_name, bot_token,
                    bot_user_id, access_token, scope, webhook_url, is_active,
                    installed_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                workspace.workspace_id, workspace.tenant_id, workspace.team_id,
                workspace.team_name, workspace.bot_token, workspace.bot_user_id,
                workspace.access_token, workspace.scope, workspace.webhook_url,
                workspace.is_active, workspace.installed_at, workspace.updated_at
            )

    async def _get_workspace(self, workspace_id: str) -> Optional[SlackWorkspace]:
        """Get workspace from database"""

        if workspace_id in self.workspaces_cache:
            return self.workspaces_cache[workspace_id]

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM integrations.slack_workspaces
                WHERE workspace_id = $1
            """, workspace_id)

            if row:
                workspace = SlackWorkspace(
                    workspace_id=row['workspace_id'],
                    tenant_id=row['tenant_id'],
                    team_id=row['team_id'],
                    team_name=row['team_name'],
                    bot_token=row['bot_token'],
                    bot_user_id=row['bot_user_id'],
                    access_token=row['access_token'],
                    scope=row['scope'],
                    webhook_url=row['webhook_url'],
                    is_active=row['is_active'],
                    installed_at=row['installed_at'],
                    updated_at=row['updated_at']
                )

                self.workspaces_cache[workspace_id] = workspace
                return workspace

        return None

    async def _get_workspace_by_team_id(self, team_id: str) -> Optional[SlackWorkspace]:
        """Get workspace by Slack team ID"""

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM integrations.slack_workspaces
                WHERE team_id = $1 AND is_active = true
            """, team_id)

            if row:
                return SlackWorkspace(
                    workspace_id=row['workspace_id'],
                    tenant_id=row['tenant_id'],
                    team_id=row['team_id'],
                    team_name=row['team_name'],
                    bot_token=row['bot_token'],
                    bot_user_id=row['bot_user_id'],
                    access_token=row['access_token'],
                    scope=row['scope'],
                    webhook_url=row['webhook_url'],
                    is_active=row['is_active'],
                    installed_at=row['installed_at'],
                    updated_at=row['updated_at']
                )

        return None

    async def _store_message(self, message: SlackMessage):
        """Store message in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO integrations.slack_messages (
                    message_id, workspace_id, channel_id, user_id, text,
                    timestamp, thread_ts, message_type, ai_response,
                    processed_at, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                message.message_id, message.workspace_id, message.channel_id,
                message.user_id, message.text, message.timestamp, message.thread_ts,
                message.message_type, message.ai_response, message.processed_at,
                message.created_at
            )

    async def _update_message(self, message: SlackMessage):
        """Update message with AI response"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE integrations.slack_messages
                SET ai_response = $2, processed_at = $3
                WHERE message_id = $1
            """, message.message_id, message.ai_response, message.processed_at)

    async def _analyze_text(self, text: str) -> str:
        """Analyze text using AI"""

        if not self.openai_client:
            return "AI analysis not available"

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze the provided text for sentiment, key topics, and insights. Provide a concise analysis."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=200
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Analysis failed: {str(e)}"

    # Placeholder methods for additional functionality
    async def _handle_summarize_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /summarize command"""
        return {'response_type': 'ephemeral', 'text': 'Summarization feature coming soon!'}

    async def _handle_generate_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /generate command"""
        return {'response_type': 'ephemeral', 'text': 'Content generation feature coming soon!'}

    async def _handle_translate_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /translate command"""
        return {'response_type': 'ephemeral', 'text': 'Translation feature coming soon!'}

    async def _handle_schedule_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /schedule command"""
        return {'response_type': 'ephemeral', 'text': 'Scheduling feature coming soon!'}

    async def _handle_report_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /report command"""
        return {'response_type': 'ephemeral', 'text': 'Reporting feature coming soon!'}

    async def _handle_reaction_event(self, event: Dict[str, Any], event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reaction added events"""
        return {'status': 'ok'}

    async def _handle_file_shared_event(self, event: Dict[str, Any], event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file shared events"""
        return {'status': 'ok'}

# Database schema for Slack integration
SLACK_SCHEMA_SQL = """
-- Slack workspaces table
CREATE TABLE IF NOT EXISTS integrations.slack_workspaces (
    workspace_id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    team_id VARCHAR(50) NOT NULL UNIQUE,
    team_name VARCHAR(500) NOT NULL,
    bot_token TEXT NOT NULL,
    bot_user_id VARCHAR(50) NOT NULL,
    access_token TEXT NOT NULL,
    scope TEXT,
    webhook_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    installed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Slack messages log
CREATE TABLE IF NOT EXISTS integrations.slack_messages (
    message_id UUID PRIMARY KEY,
    workspace_id UUID REFERENCES integrations.slack_workspaces(workspace_id) ON DELETE CASCADE,
    channel_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    text TEXT NOT NULL,
    timestamp VARCHAR(50) NOT NULL,
    thread_ts VARCHAR(50),
    message_type VARCHAR(50) DEFAULT 'message',
    ai_response TEXT,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_slack_workspaces_tenant
ON integrations.slack_workspaces(tenant_id, is_active);

CREATE INDEX IF NOT EXISTS idx_slack_messages_workspace_channel
ON integrations.slack_messages(workspace_id, channel_id, created_at DESC);

-- Comments for documentation
COMMENT ON TABLE integrations.slack_workspaces IS 'Slack workspace connections with bot tokens';
COMMENT ON TABLE integrations.slack_messages IS 'Log of messages and AI responses for analytics';
"""

# Pydantic models for API
class SlackOAuthRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant ID")
    redirect_uri: str = Field(..., description="OAuth redirect URI")

class SlackTokenExchangeRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant ID")
    code: str = Field(..., description="Authorization code")
    redirect_uri: str = Field(..., description="OAuth redirect URI")

class SlackNotificationRequest(BaseModel):
    workspace_id: str = Field(..., description="Slack workspace ID")
    channel_id: str = Field(..., description="Slack channel ID")
    notification_type: str = Field(..., description="Type of notification")
    data: Dict[str, Any] = Field(..., description="Notification data")

# Export main classes
__all__ = [
    'SlackBot', 'SlackWorkspace', 'SlackChannel', 'SlackMessage',
    'SlackEventType', 'SlackCommandType', 'NotificationType',
    'SlackOAuthRequest', 'SlackTokenExchangeRequest', 'SlackNotificationRequest',
    'SLACK_SCHEMA_SQL'
]
