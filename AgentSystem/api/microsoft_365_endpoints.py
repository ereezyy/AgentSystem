"""
Microsoft 365 Integration API Endpoints - AgentSystem Profit Machine
FastAPI endpoints for Teams, Outlook, SharePoint OAuth, events, and automation
"""

from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import asyncio
import json
import logging
from datetime import datetime, timedelta
import asyncpg

from ..integrations.microsoft_365_connector import (
    Microsoft365Connector, Microsoft365Tenant, Microsoft365Service,
    TeamsEventType, OutlookEventType, SharePointEventType
)
from ..auth.tenant_auth import get_current_tenant
from ..database.connection import get_db_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/microsoft365", tags=["Microsoft 365 Integration"])
security = HTTPBearer()

# Pydantic models for API requests/responses
class Microsoft365InstallRequest(BaseModel):
    redirect_uri: str = Field(..., description="OAuth redirect URI")
    admin_consent: bool = Field(default=False, description="Request admin consent")
    services: List[str] = Field(default=["teams", "outlook"], description="Services to enable")

class Microsoft365InstallResponse(BaseModel):
    oauth_url: str = Field(..., description="Microsoft 365 OAuth authorization URL")
    state: str = Field(..., description="OAuth state parameter")

class Microsoft365CallbackRequest(BaseModel):
    code: str = Field(..., description="OAuth authorization code")
    state: str = Field(..., description="OAuth state (agentsystem_tenant_id)")
    redirect_uri: str = Field(..., description="OAuth redirect URI")
    admin_consent: Optional[bool] = Field(default=False)

class Microsoft365TenantResponse(BaseModel):
    tenant_id: str
    tenant_name: str
    directory_id: str
    enabled_services: List[str]
    is_active: bool
    installed_at: datetime
    teams_channels: int
    emails_processed: int
    documents_processed: int

class TeamsMessageRequest(BaseModel):
    team_id: str = Field(..., description="Teams team ID")
    channel_id: str = Field(..., description="Teams channel ID")
    message: str = Field(..., description="Message content")
    message_type: str = Field(default="text", description="Message type (text/html)")

class OutlookEmailRequest(BaseModel):
    to_recipients: List[str] = Field(..., description="Email recipients")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    body_type: str = Field(default="html", description="Body type (text/html)")

class CalendarEventRequest(BaseModel):
    subject: str = Field(..., description="Event subject")
    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")
    attendees: Optional[List[str]] = Field(default=None, description="Event attendees")
    body: Optional[str] = Field(default=None, description="Event description")

class SharePointUploadRequest(BaseModel):
    site_id: str = Field(..., description="SharePoint site ID")
    file_name: str = Field(..., description="File name")
    folder_path: str = Field(default="/", description="Folder path")
    file_content_base64: str = Field(..., description="Base64 encoded file content")

class Microsoft365WebhookRequest(BaseModel):
    subscription_id: str = Field(..., description="Webhook subscription ID")
    change_type: str = Field(..., description="Change type")
    resource: str = Field(..., description="Resource path")
    resource_data: Dict[str, Any] = Field(..., description="Resource data")

class Microsoft365AnalyticsResponse(BaseModel):
    tenant_id: str
    tenant_name: str
    teams_messages: int
    emails_processed: int
    calendar_events: int
    documents_processed: int
    ai_responses: int
    active_users: int
    total_tokens_used: int
    last_30_days: Dict[str, Dict[str, int]]

# Global Microsoft365Connector instance
m365_connector: Optional[Microsoft365Connector] = None

async def get_m365_connector() -> Microsoft365Connector:
    """Get initialized Microsoft365Connector instance"""
    global m365_connector
    if not m365_connector:
        # Initialize with environment variables or config
        db_pool = await get_db_pool()
        m365_connector = Microsoft365Connector(
            db_pool=db_pool,
            client_id="your_m365_client_id",  # From environment
            client_secret="your_m365_client_secret",  # From environment
            tenant_id="common"  # Multi-tenant
        )
    return m365_connector

@router.post("/install", response_model=Microsoft365InstallResponse)
async def initiate_microsoft365_install(
    request: Microsoft365InstallRequest,
    tenant = Depends(get_current_tenant),
    connector: Microsoft365Connector = Depends(get_m365_connector)
):
    """
    Initiate Microsoft 365 app installation for a tenant
    """
    try:
        oauth_url = connector.get_oauth_url(
            agentsystem_tenant_id=tenant.id,
            redirect_uri=request.redirect_uri,
            admin_consent=request.admin_consent
        )

        return Microsoft365InstallResponse(
            oauth_url=oauth_url,
            state=tenant.id
        )

    except Exception as e:
        logger.error(f"Microsoft 365 install initiation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate Microsoft 365 installation")

@router.post("/oauth/callback")
async def microsoft365_oauth_callback(
    request: Microsoft365CallbackRequest,
    connector: Microsoft365Connector = Depends(get_m365_connector)
):
    """
    Handle Microsoft 365 OAuth callback and exchange code for tokens
    """
    try:
        async with connector:
            tenant = await connector.exchange_code_for_tokens(
                agentsystem_tenant_id=request.state,
                code=request.code,
                redirect_uri=request.redirect_uri
            )

        return {
            "success": True,
            "tenant_id": tenant.tenant_id,
            "tenant_name": tenant.tenant_name,
            "directory_id": tenant.directory_id,
            "message": "Microsoft 365 integration installed successfully"
        }

    except Exception as e:
        logger.error(f"Microsoft 365 OAuth callback failed: {e}")
        raise HTTPException(status_code=400, detail=f"OAuth callback failed: {str(e)}")

@router.get("/tenants", response_model=List[Microsoft365TenantResponse])
async def get_microsoft365_tenants(
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Get all Microsoft 365 tenants for an AgentSystem tenant
    """
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    t.*,
                    COUNT(DISTINCT tc.channel_id) as teams_channels,
                    COUNT(DISTINCT oe.email_id) as emails_processed,
                    COUNT(DISTINCT sd.document_id) as documents_processed
                FROM integrations.microsoft365_tenants t
                LEFT JOIN integrations.microsoft365_teams_channels tc ON t.tenant_id = tc.tenant_id
                LEFT JOIN integrations.microsoft365_outlook_emails oe ON t.tenant_id = oe.tenant_id
                LEFT JOIN integrations.microsoft365_sharepoint_documents sd ON t.tenant_id = sd.tenant_id
                WHERE t.agentsystem_tenant_id = $1
                GROUP BY t.tenant_id
                ORDER BY t.installed_at DESC
            """, tenant.id)

            tenants = []
            for row in rows:
                tenants.append(Microsoft365TenantResponse(
                    tenant_id=row['tenant_id'],
                    tenant_name=row['tenant_name'],
                    directory_id=row['directory_id'],
                    enabled_services=row['enabled_services'],
                    is_active=row['is_active'],
                    installed_at=row['installed_at'],
                    teams_channels=row['teams_channels'] or 0,
                    emails_processed=row['emails_processed'] or 0,
                    documents_processed=row['documents_processed'] or 0
                ))

            return tenants

    except Exception as e:
        logger.error(f"Failed to fetch Microsoft 365 tenants: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tenants")

@router.delete("/tenants/{tenant_id}")
async def remove_microsoft365_tenant(
    tenant_id: str,
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Remove/deactivate a Microsoft 365 tenant
    """
    try:
        async with db_pool.acquire() as conn:
            # Verify tenant belongs to AgentSystem tenant
            m365_tenant = await conn.fetchrow("""
                SELECT tenant_id FROM integrations.microsoft365_tenants
                WHERE tenant_id = $1 AND agentsystem_tenant_id = $2
            """, tenant_id, tenant.id)

            if not m365_tenant:
                raise HTTPException(status_code=404, detail="Microsoft 365 tenant not found")

            # Deactivate tenant
            await conn.execute("""
                UPDATE integrations.microsoft365_tenants
                SET is_active = false, updated_at = NOW()
                WHERE tenant_id = $1
            """, tenant_id)

            return {"success": True, "message": "Microsoft 365 tenant deactivated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove Microsoft 365 tenant: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove tenant")

@router.post("/teams/send-message")
async def send_teams_message(
    request: TeamsMessageRequest,
    tenant_id: str,
    tenant = Depends(get_current_tenant),
    connector: Microsoft365Connector = Depends(get_m365_connector)
):
    """
    Send message to Teams channel
    """
    try:
        # Verify tenant access
        db_pool = await get_db_pool()
        async with db_pool.acquire() as conn:
            m365_tenant = await conn.fetchrow("""
                SELECT tenant_id FROM integrations.microsoft365_tenants
                WHERE tenant_id = $1 AND agentsystem_tenant_id = $2 AND is_active = true
            """, tenant_id, tenant.id)

            if not m365_tenant:
                raise HTTPException(status_code=404, detail="Microsoft 365 tenant not found")

        # Send message
        async with connector:
            success = await connector.send_teams_message(
                tenant_id=tenant_id,
                team_id=request.team_id,
                channel_id=request.channel_id,
                message=request.message,
                message_type=request.message_type
            )

        if success:
            return {"success": True, "message": "Teams message sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send Teams message")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send Teams message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")

@router.post("/outlook/send-email")
async def send_outlook_email(
    request: OutlookEmailRequest,
    tenant_id: str,
    tenant = Depends(get_current_tenant),
    connector: Microsoft365Connector = Depends(get_m365_connector)
):
    """
    Send email via Outlook
    """
    try:
        # Verify tenant access
        db_pool = await get_db_pool()
        async with db_pool.acquire() as conn:
            m365_tenant = await conn.fetchrow("""
                SELECT tenant_id FROM integrations.microsoft365_tenants
                WHERE tenant_id = $1 AND agentsystem_tenant_id = $2 AND is_active = true
            """, tenant_id, tenant.id)

            if not m365_tenant:
                raise HTTPException(status_code=404, detail="Microsoft 365 tenant not found")

        # Send email
        async with connector:
            success = await connector.send_outlook_email(
                tenant_id=tenant_id,
                to_recipients=request.to_recipients,
                subject=request.subject,
                body=request.body,
                body_type=request.body_type
            )

        if success:
            return {"success": True, "message": "Email sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send email")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send Outlook email: {e}")
        raise HTTPException(status_code=500, detail="Failed to send email")

@router.post("/calendar/create-event")
async def create_calendar_event(
    request: CalendarEventRequest,
    tenant_id: str,
    tenant = Depends(get_current_tenant),
    connector: Microsoft365Connector = Depends(get_m365_connector)
):
    """
    Create calendar event in Outlook
    """
    try:
        # Verify tenant access
        db_pool = await get_db_pool()
        async with db_pool.acquire() as conn:
            m365_tenant = await conn.fetchrow("""
                SELECT tenant_id FROM integrations.microsoft365_tenants
                WHERE tenant_id = $1 AND agentsystem_tenant_id = $2 AND is_active = true
            """, tenant_id, tenant.id)

            if not m365_tenant:
                raise HTTPException(status_code=404, detail="Microsoft 365 tenant not found")

        # Create calendar event
        async with connector:
            event_id = await connector.create_calendar_event(
                tenant_id=tenant_id,
                subject=request.subject,
                start_time=request.start_time,
                end_time=request.end_time,
                attendees=request.attendees,
                body=request.body
            )

        if event_id:
            return {"success": True, "event_id": event_id, "message": "Calendar event created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create calendar event")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create calendar event: {e}")
        raise HTTPException(status_code=500, detail="Failed to create calendar event")

@router.post("/sharepoint/upload-file")
async def upload_sharepoint_file(
    request: SharePointUploadRequest,
    tenant_id: str,
    tenant = Depends(get_current_tenant),
    connector: Microsoft365Connector = Depends(get_m365_connector)
):
    """
    Upload file to SharePoint
    """
    try:
        import base64

        # Verify tenant access
        db_pool = await get_db_pool()
        async with db_pool.acquire() as conn:
            m365_tenant = await conn.fetchrow("""
                SELECT tenant_id FROM integrations.microsoft365_tenants
                WHERE tenant_id = $1 AND agentsystem_tenant_id = $2 AND is_active = true
            """, tenant_id, tenant.id)

            if not m365_tenant:
                raise HTTPException(status_code=404, detail="Microsoft 365 tenant not found")

        # Decode file content
        file_content = base64.b64decode(request.file_content_base64)

        # Upload file
        async with connector:
            file_id = await connector.upload_sharepoint_file(
                tenant_id=tenant_id,
                site_id=request.site_id,
                file_name=request.file_name,
                file_content=file_content,
                folder_path=request.folder_path
            )

        if file_id:
            return {"success": True, "file_id": file_id, "message": "File uploaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to upload file")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload SharePoint file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")

@router.post("/webhooks/teams")
async def teams_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    connector: Microsoft365Connector = Depends(get_m365_connector)
):
    """
    Handle Teams webhook events
    """
    try:
        body = await request.body()
        webhook_data = json.loads(body.decode())

        # Process webhook in background
        background_tasks.add_task(process_teams_webhook, webhook_data, connector)

        return {"status": "ok"}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logger.error(f"Teams webhook handling failed: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

@router.post("/webhooks/outlook")
async def outlook_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    connector: Microsoft365Connector = Depends(get_m365_connector)
):
    """
    Handle Outlook webhook events
    """
    try:
        body = await request.body()
        webhook_data = json.loads(body.decode())

        # Process webhook in background
        background_tasks.add_task(process_outlook_webhook, webhook_data, connector)

        return {"status": "ok"}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logger.error(f"Outlook webhook handling failed: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

@router.get("/analytics/{tenant_id}", response_model=Microsoft365AnalyticsResponse)
async def get_microsoft365_analytics(
    tenant_id: str,
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Get analytics for a Microsoft 365 tenant
    """
    try:
        async with db_pool.acquire() as conn:
            # Verify tenant belongs to AgentSystem tenant
            m365_tenant = await conn.fetchrow("""
                SELECT tenant_name FROM integrations.microsoft365_tenants
                WHERE tenant_id = $1 AND agentsystem_tenant_id = $2
            """, tenant_id, tenant.id)

            if not m365_tenant:
                raise HTTPException(status_code=404, detail="Microsoft 365 tenant not found")

            # Get overall statistics for last 30 days
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(DISTINCT tm.message_id) as teams_messages,
                    COUNT(DISTINCT oe.email_id) as emails_processed,
                    COUNT(DISTINCT ce.event_id) as calendar_events,
                    COUNT(DISTINCT sd.document_id) as documents_processed,
                    COUNT(DISTINCT tm.message_id) FILTER (WHERE tm.ai_response IS NOT NULL) as ai_responses,
                    COUNT(DISTINCT COALESCE(tm.user_id, oe.sender_email)) as active_users,
                    COALESCE(SUM(tm.ai_response_tokens), 0) as total_tokens_used
                FROM integrations.microsoft365_tenants t
                LEFT JOIN integrations.microsoft365_teams_messages tm ON t.tenant_id = tm.tenant_id
                    AND tm.created_at > NOW() - INTERVAL '30 days'
                LEFT JOIN integrations.microsoft365_outlook_emails oe ON t.tenant_id = oe.tenant_id
                    AND oe.received_at > NOW() - INTERVAL '30 days'
                LEFT JOIN integrations.microsoft365_calendar_events ce ON t.tenant_id = ce.tenant_id
                    AND ce.created_at > NOW() - INTERVAL '30 days'
                LEFT JOIN integrations.microsoft365_sharepoint_documents sd ON t.tenant_id = sd.tenant_id
                    AND sd.modified_at > NOW() - INTERVAL '30 days'
                WHERE t.tenant_id = $1
            """, tenant_id)

            # Get daily activity for last 30 days
            daily_activity = await conn.fetch("""
                SELECT
                    date,
                    service_type,
                    total_events
                FROM integrations.microsoft365_analytics_daily
                WHERE tenant_id = $1
                AND date > CURRENT_DATE - INTERVAL '30 days'
                ORDER BY date, service_type
            """, tenant_id)

            # Format daily activity
            last_30_days = {}
            for row in daily_activity:
                date_str = row['date'].isoformat()
                if date_str not in last_30_days:
                    last_30_days[date_str] = {}
                last_30_days[date_str][row['service_type']] = row['total_events']

            return Microsoft365AnalyticsResponse(
                tenant_id=tenant_id,
                tenant_name=m365_tenant['tenant_name'],
                teams_messages=stats['teams_messages'] or 0,
                emails_processed=stats['emails_processed'] or 0,
                calendar_events=stats['calendar_events'] or 0,
                documents_processed=stats['documents_processed'] or 0,
                ai_responses=stats['ai_responses'] or 0,
                active_users=stats['active_users'] or 0,
                total_tokens_used=stats['total_tokens_used'] or 0,
                last_30_days=last_30_days
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get Microsoft 365 analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

# Background task functions
async def process_teams_webhook(webhook_data: Dict[str, Any], connector: Microsoft365Connector):
    """Process Teams webhook in background"""
    try:
        async with connector:
            await connector.process_teams_webhook(webhook_data)
    except Exception as e:
        logger.error(f"Background Teams webhook processing failed: {e}")

async def process_outlook_webhook(webhook_data: Dict[str, Any], connector: Microsoft365Connector):
    """Process Outlook webhook in background"""
    try:
        async with connector:
            await connector.process_outlook_webhook(webhook_data)
    except Exception as e:
        logger.error(f"Background Outlook webhook processing failed: {e}")

# Health check endpoint
@router.get("/health")
async def microsoft365_health_check():
    """Health check for Microsoft 365 integration"""
    return {
        "status": "healthy",
        "service": "microsoft365_integration",
        "timestamp": datetime.now().isoformat()
    }

# Export router
__all__ = ["router"]
