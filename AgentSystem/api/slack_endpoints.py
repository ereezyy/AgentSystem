"""
Slack Integration API Endpoints - AgentSystem Profit Machine
FastAPI endpoints for Slack bot OAuth, events, and commands
"""

from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import asyncio
import json
import logging
from datetime import datetime
import asyncpg
from urllib.parse import urlencode

from ..integrations.slack_bot import (
    SlackBot, SlackOAuthRequest, SlackTokenExchangeRequest,
    SlackNotificationRequest, NotificationType
)
from ..auth.tenant_auth import get_current_tenant
from ..database.connection import get_db_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/slack", tags=["Slack Integration"])
security = HTTPBearer()

# Pydantic models for API requests/responses
class SlackInstallRequest(BaseModel):
    redirect_uri: str = Field(..., description="OAuth redirect URI")

class SlackInstallResponse(BaseModel):
    oauth_url: str = Field(..., description="Slack OAuth authorization URL")
    state: str = Field(..., description="OAuth state parameter")

class SlackCallbackRequest(BaseModel):
    code: str = Field(..., description="OAuth authorization code")
    state: str = Field(..., description="OAuth state (tenant_id)")
    redirect_uri: str = Field(..., description="OAuth redirect URI")

class SlackWorkspaceResponse(BaseModel):
    workspace_id: str
    team_name: str
    team_id: str
    is_active: bool
    installed_at: datetime
    channels_count: int
    last_activity: Optional[datetime]

class SlackEventRequest(BaseModel):
    token: Optional[str] = None
    team_id: Optional[str] = None
    api_app_id: Optional[str] = None
    event: Optional[Dict[str, Any]] = None
    type: str
    challenge: Optional[str] = None

class SlackCommandRequest(BaseModel):
    token: str
    team_id: str
    team_domain: str
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    command: str
    text: str
    response_url: str
    trigger_id: str

class SlackAnalyticsResponse(BaseModel):
    workspace_id: str
    team_name: str
    total_messages: int
    ai_responses: int
    commands_executed: int
    active_users: int
    active_channels: int
    avg_response_time: float
    last_30_days: Dict[str, int]

# Global SlackBot instance (will be initialized with config)
slack_bot: Optional[SlackBot] = None

async def get_slack_bot() -> SlackBot:
    """Get initialized SlackBot instance"""
    global slack_bot
    if not slack_bot:
        # Initialize with environment variables or config
        db_pool = await get_db_pool()
        slack_bot = SlackBot(
            db_pool=db_pool,
            client_id="your_slack_client_id",  # From environment
            client_secret="your_slack_client_secret",  # From environment
            signing_secret="your_slack_signing_secret"  # From environment
        )
    return slack_bot

@router.post("/install", response_model=SlackInstallResponse)
async def initiate_slack_install(
    request: SlackInstallRequest,
    tenant = Depends(get_current_tenant),
    bot: SlackBot = Depends(get_slack_bot)
):
    """
    Initiate Slack app installation for a tenant
    """
    try:
        oauth_url = bot.get_oauth_url(
            tenant_id=tenant.id,
            redirect_uri=request.redirect_uri
        )

        return SlackInstallResponse(
            oauth_url=oauth_url,
            state=tenant.id
        )

    except Exception as e:
        logger.error(f"Slack install initiation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate Slack installation")

@router.post("/oauth/callback")
async def slack_oauth_callback(
    request: SlackCallbackRequest,
    bot: SlackBot = Depends(get_slack_bot)
):
    """
    Handle Slack OAuth callback and exchange code for tokens
    """
    try:
        async with bot:
            workspace = await bot.exchange_code_for_tokens(
                tenant_id=request.state,
                code=request.code,
                redirect_uri=request.redirect_uri
            )

        return {
            "success": True,
            "workspace_id": workspace.workspace_id,
            "team_name": workspace.team_name,
            "message": "Slack integration installed successfully"
        }

    except Exception as e:
        logger.error(f"Slack OAuth callback failed: {e}")
        raise HTTPException(status_code=400, detail=f"OAuth callback failed: {str(e)}")

@router.post("/events")
async def slack_events(
    request: Request,
    background_tasks: BackgroundTasks,
    bot: SlackBot = Depends(get_slack_bot)
):
    """
    Handle Slack events (messages, mentions, reactions, etc.)
    """
    try:
        # Get request body and headers
        body = await request.body()
        headers = dict(request.headers)

        # Verify request signature
        async with bot:
            if not await bot.verify_request(headers, body.decode()):
                raise HTTPException(status_code=401, detail="Invalid request signature")

        # Parse event data
        event_data = json.loads(body.decode())

        # Handle URL verification challenge
        if event_data.get("type") == "url_verification":
            return {"challenge": event_data.get("challenge")}

        # Process event in background
        background_tasks.add_task(process_slack_event, event_data, bot)

        return {"status": "ok"}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logger.error(f"Slack event handling failed: {e}")
        raise HTTPException(status_code=500, detail="Event processing failed")

@router.post("/commands")
async def slack_commands(
    request: Request,
    background_tasks: BackgroundTasks,
    bot: SlackBot = Depends(get_slack_bot)
):
    """
    Handle Slack slash commands
    """
    try:
        # Get form data from slash command
        form_data = await request.form()
        command_data = dict(form_data)

        # Verify request signature
        body = await request.body()
        headers = dict(request.headers)

        async with bot:
            if not await bot.verify_request(headers, body.decode()):
                raise HTTPException(status_code=401, detail="Invalid request signature")

        # Process command
        async with bot:
            response = await bot.handle_slash_command(command_data)

        return response

    except Exception as e:
        logger.error(f"Slack command handling failed: {e}")
        return {
            "response_type": "ephemeral",
            "text": f"Command processing failed: {str(e)}"
        }

@router.get("/workspaces", response_model=List[SlackWorkspaceResponse])
async def get_slack_workspaces(
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Get all Slack workspaces for a tenant
    """
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    w.*,
                    COUNT(c.channel_id) as channels_count,
                    MAX(m.created_at) as last_activity
                FROM integrations.slack_workspaces w
                LEFT JOIN integrations.slack_channels c ON w.workspace_id = c.workspace_id
                LEFT JOIN integrations.slack_messages m ON w.workspace_id = m.workspace_id
                WHERE w.tenant_id = $1
                GROUP BY w.workspace_id
                ORDER BY w.installed_at DESC
            """, tenant.id)

            workspaces = []
            for row in rows:
                workspaces.append(SlackWorkspaceResponse(
                    workspace_id=row['workspace_id'],
                    team_name=row['team_name'],
                    team_id=row['team_id'],
                    is_active=row['is_active'],
                    installed_at=row['installed_at'],
                    channels_count=row['channels_count'] or 0,
                    last_activity=row['last_activity']
                ))

            return workspaces

    except Exception as e:
        logger.error(f"Failed to fetch Slack workspaces: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch workspaces")

@router.delete("/workspaces/{workspace_id}")
async def remove_slack_workspace(
    workspace_id: str,
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Remove/deactivate a Slack workspace
    """
    try:
        async with db_pool.acquire() as conn:
            # Verify workspace belongs to tenant
            workspace = await conn.fetchrow("""
                SELECT workspace_id FROM integrations.slack_workspaces
                WHERE workspace_id = $1 AND tenant_id = $2
            """, workspace_id, tenant.id)

            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")

            # Deactivate workspace
            await conn.execute("""
                UPDATE integrations.slack_workspaces
                SET is_active = false, updated_at = NOW()
                WHERE workspace_id = $1
            """, workspace_id)

            return {"success": True, "message": "Workspace deactivated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove Slack workspace: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove workspace")

@router.post("/notifications")
async def send_slack_notification(
    request: SlackNotificationRequest,
    tenant = Depends(get_current_tenant),
    bot: SlackBot = Depends(get_slack_bot)
):
    """
    Send notification to Slack channel
    """
    try:
        # Verify workspace belongs to tenant
        db_pool = await get_db_pool()
        async with db_pool.acquire() as conn:
            workspace = await conn.fetchrow("""
                SELECT workspace_id FROM integrations.slack_workspaces
                WHERE workspace_id = $1 AND tenant_id = $2 AND is_active = true
            """, request.workspace_id, tenant.id)

            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")

        # Send notification
        notification_type = NotificationType(request.notification_type)
        async with bot:
            success = await bot.send_notification(
                workspace_id=request.workspace_id,
                channel_id=request.channel_id,
                notification_type=notification_type,
                data=request.data
            )

        if success:
            return {"success": True, "message": "Notification sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send notification")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid notification type: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to send notification")

@router.get("/analytics/{workspace_id}", response_model=SlackAnalyticsResponse)
async def get_slack_analytics(
    workspace_id: str,
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Get analytics for a Slack workspace
    """
    try:
        async with db_pool.acquire() as conn:
            # Verify workspace belongs to tenant
            workspace = await conn.fetchrow("""
                SELECT team_name FROM integrations.slack_workspaces
                WHERE workspace_id = $1 AND tenant_id = $2
            """, workspace_id, tenant.id)

            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")

            # Get overall statistics
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_messages,
                    COUNT(ai_response) as ai_responses,
                    COUNT(DISTINCT user_id) as active_users,
                    COUNT(DISTINCT channel_id) as active_channels,
                    AVG(processing_time_ms) as avg_response_time
                FROM integrations.slack_messages
                WHERE workspace_id = $1
                AND created_at > NOW() - INTERVAL '30 days'
            """, workspace_id)

            # Get commands count
            commands = await conn.fetchval("""
                SELECT COUNT(*) FROM integrations.slack_commands
                WHERE workspace_id = $1
                AND created_at > NOW() - INTERVAL '30 days'
            """, workspace_id)

            # Get daily activity for last 30 days
            daily_activity = await conn.fetch("""
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as messages
                FROM integrations.slack_messages
                WHERE workspace_id = $1
                AND created_at > NOW() - INTERVAL '30 days'
                GROUP BY DATE(created_at)
                ORDER BY date
            """, workspace_id)

            # Format daily activity
            last_30_days = {}
            for row in daily_activity:
                last_30_days[row['date'].isoformat()] = row['messages']

            return SlackAnalyticsResponse(
                workspace_id=workspace_id,
                team_name=workspace['team_name'],
                total_messages=stats['total_messages'] or 0,
                ai_responses=stats['ai_responses'] or 0,
                commands_executed=commands or 0,
                active_users=stats['active_users'] or 0,
                active_channels=stats['active_channels'] or 0,
                avg_response_time=float(stats['avg_response_time'] or 0),
                last_30_days=last_30_days
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get Slack analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@router.post("/channels/{workspace_id}/configure")
async def configure_slack_channel(
    workspace_id: str,
    channel_id: str,
    ai_assistance_enabled: bool = True,
    notification_types: List[str] = [],
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Configure Slack channel settings
    """
    try:
        async with db_pool.acquire() as conn:
            # Verify workspace belongs to tenant
            workspace = await conn.fetchrow("""
                SELECT workspace_id FROM integrations.slack_workspaces
                WHERE workspace_id = $1 AND tenant_id = $2
            """, workspace_id, tenant.id)

            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")

            # Upsert channel configuration
            await conn.execute("""
                INSERT INTO integrations.slack_channels (
                    channel_id, workspace_id, channel_name, ai_assistance_enabled,
                    notification_types, updated_at
                ) VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (channel_id)
                DO UPDATE SET
                    ai_assistance_enabled = EXCLUDED.ai_assistance_enabled,
                    notification_types = EXCLUDED.notification_types,
                    updated_at = NOW()
            """, channel_id, workspace_id, f"channel-{channel_id}",
                ai_assistance_enabled, notification_types)

            return {
                "success": True,
                "message": "Channel configuration updated successfully"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure Slack channel: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure channel")

# Background task functions
async def process_slack_event(event_data: Dict[str, Any], bot: SlackBot):
    """Process Slack event in background"""
    try:
        async with bot:
            await bot.handle_event(event_data)
    except Exception as e:
        logger.error(f"Background event processing failed: {e}")

# Health check endpoint
@router.get("/health")
async def slack_health_check():
    """Health check for Slack integration"""
    return {
        "status": "healthy",
        "service": "slack_integration",
        "timestamp": datetime.now().isoformat()
    }

# Export router
__all__ = ["router"]
