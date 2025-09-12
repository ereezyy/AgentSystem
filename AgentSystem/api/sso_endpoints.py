"""
Enterprise SSO API Endpoints - AgentSystem Profit Machine
FastAPI endpoints for Single Sign-On integration with Active Directory, Okta, etc.
"""

from fastapi import APIRouter, HTTPException, Request, Depends, Form, Query
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import asyncio
import json
import logging
from datetime import datetime, timedelta
import asyncpg
import base64

from ..auth.sso_integration import (
    SSOManager, SSOConfig, SSOUser, SSOSession,
    SSOProvider, SSOProtocol, UserRole
)
from ..auth.tenant_auth import get_current_tenant
from ..database.connection import get_db_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth/sso", tags=["Enterprise SSO"])
security = HTTPBearer()

# Pydantic models for API requests/responses
class CreateSSOConfigRequest(BaseModel):
    provider: str = Field(..., description="SSO provider (okta, active_directory, etc.)")
    provider_name: str = Field(..., description="Display name for provider")
    config_data: Dict[str, Any] = Field(..., description="Provider-specific configuration")
    is_default: bool = Field(default=False, description="Set as default SSO provider")

class SSOConfigResponse(BaseModel):
    config_id: str
    provider: str
    provider_name: str
    protocol: str
    is_active: bool
    is_default: bool
    created_at: datetime
    user_count: int
    last_login: Optional[datetime]

class SSOUserResponse(BaseModel):
    user_id: str
    email: str
    first_name: str
    last_name: str
    display_name: str
    roles: List[str]
    groups: List[str]
    provider: str
    is_active: bool
    last_login_at: Optional[datetime]
    login_count: int

class SSOSessionResponse(BaseModel):
    session_id: str
    user_id: str
    provider: str
    expires_at: datetime
    last_accessed_at: datetime
    ip_address: Optional[str]

class SSOAnalyticsResponse(BaseModel):
    total_users: int
    active_sessions: int
    logins_today: int
    logins_this_week: int
    failed_attempts_today: int
    top_providers: List[Dict[str, Any]]
    recent_activity: List[Dict[str, Any]]

class InitiateSSORequest(BaseModel):
    config_id: Optional[str] = Field(None, description="Specific SSO config to use")
    return_url: Optional[str] = Field(None, description="URL to redirect after login")

class LDAPLoginRequest(BaseModel):
    username: str = Field(..., description="LDAP username")
    password: str = Field(..., description="LDAP password")
    config_id: Optional[str] = Field(None, description="Specific LDAP config to use")

# Global SSOManager instance
sso_manager: Optional[SSOManager] = None

async def get_sso_manager() -> SSOManager:
    """Get initialized SSOManager instance"""
    global sso_manager
    if not sso_manager:
        db_pool = await get_db_pool()
        base_url = "https://api.agentsystem.com"  # From environment
        sso_manager = SSOManager(db_pool=db_pool, base_url=base_url)
    return sso_manager

@router.post("/config", response_model=SSOConfigResponse)
async def create_sso_config(
    request: CreateSSOConfigRequest,
    tenant = Depends(get_current_tenant),
    manager: SSOManager = Depends(get_sso_manager)
):
    """
    Create SSO configuration for a tenant
    """
    try:
        provider = SSOProvider(request.provider)

        config = await manager.create_sso_config(
            tenant_id=tenant.id,
            provider=provider,
            config_data={
                **request.config_data,
                'provider_name': request.provider_name,
                'is_default': request.is_default
            }
        )

        return SSOConfigResponse(
            config_id=config.config_id,
            provider=config.provider.value,
            provider_name=config.provider_name,
            protocol=config.protocol.value,
            is_active=config.is_active,
            is_default=config.is_default,
            created_at=config.created_at,
            user_count=0,
            last_login=None
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create SSO config: {e}")
        raise HTTPException(status_code=500, detail="Failed to create SSO configuration")

@router.get("/configs", response_model=List[SSOConfigResponse])
async def list_sso_configs(
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    List SSO configurations for a tenant
    """
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    c.*,
                    COUNT(u.user_id) as user_count,
                    MAX(u.last_login_at) as last_login
                FROM auth.sso_configs c
                LEFT JOIN auth.sso_users u ON c.tenant_id = u.tenant_id AND c.provider = u.provider
                WHERE c.tenant_id = $1
                GROUP BY c.config_id
                ORDER BY c.is_default DESC, c.created_at DESC
            """, tenant.id)

            configs = []
            for row in rows:
                configs.append(SSOConfigResponse(
                    config_id=row['config_id'],
                    provider=row['provider'],
                    provider_name=row['provider_name'],
                    protocol=row['protocol'],
                    is_active=row['is_active'],
                    is_default=row['is_default'],
                    created_at=row['created_at'],
                    user_count=row['user_count'] or 0,
                    last_login=row['last_login']
                ))

            return configs

    except Exception as e:
        logger.error(f"Failed to list SSO configs: {e}")
        raise HTTPException(status_code=500, detail="Failed to list SSO configurations")

@router.delete("/config/{config_id}")
async def delete_sso_config(
    config_id: str,
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Delete SSO configuration
    """
    try:
        async with db_pool.acquire() as conn:
            # Verify ownership
            config = await conn.fetchrow("""
                SELECT config_id FROM auth.sso_configs
                WHERE config_id = $1 AND tenant_id = $2
            """, config_id, tenant.id)

            if not config:
                raise HTTPException(status_code=404, detail="SSO configuration not found")

            # Delete configuration
            await conn.execute("""
                UPDATE auth.sso_configs
                SET is_active = FALSE, updated_at = NOW()
                WHERE config_id = $1
            """, config_id)

            return {"success": True, "message": "SSO configuration deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete SSO config: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete SSO configuration")

@router.post("/initiate")
async def initiate_sso_login(
    request: InitiateSSORequest,
    tenant = Depends(get_current_tenant),
    manager: SSOManager = Depends(get_sso_manager)
):
    """
    Initiate SSO login flow
    """
    try:
        async with manager:
            login_url = await manager.get_sso_login_url(
                tenant_id=tenant.id,
                config_id=request.config_id,
                return_url=request.return_url
            )

        return {"login_url": login_url}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to initiate SSO login: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate SSO login")

@router.post("/login/ldap")
async def ldap_login(
    request: LDAPLoginRequest,
    tenant = Depends(get_current_tenant),
    manager: SSOManager = Depends(get_sso_manager)
):
    """
    Direct LDAP authentication
    """
    try:
        async with manager:
            user = await manager.authenticate_user(
                tenant_id=tenant.id,
                username=request.username,
                password=request.password,
                config_id=request.config_id
            )

        if not user:
            # Log failed attempt
            await _log_login_attempt(
                tenant.id, request.username, "ldap", False, "Invalid credentials"
            )
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Create session
        session = await manager.create_sso_session(user)

        # Log successful login
        await _log_login_attempt(tenant.id, user.email, "ldap", True)

        return {
            "success": True,
            "session_id": session.session_id,
            "user": {
                "user_id": user.user_id,
                "email": user.email,
                "display_name": user.display_name,
                "roles": [role.value for role in user.roles]
            },
            "expires_at": session.expires_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LDAP login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.post("/saml/acs")
async def saml_acs(
    request: Request,
    SAMLResponse: str = Form(...),
    RelayState: Optional[str] = Form(None),
    manager: SSOManager = Depends(get_sso_manager)
):
    """
    SAML Assertion Consumer Service (ACS) endpoint
    """
    try:
        # Extract tenant ID from RelayState or URL
        tenant_id = RelayState or "default"  # This should be properly implemented

        request_data = {
            'SAMLResponse': SAMLResponse,
            'RelayState': RelayState
        }

        async with manager:
            user = await manager.handle_sso_callback(
                tenant_id=tenant_id,
                protocol=SSOProtocol.SAML2,
                request_data=request_data
            )

        # Create session
        session = await manager.create_sso_session(user)

        # Log successful login
        await _log_login_attempt(tenant_id, user.email, "saml", True)

        # Redirect to application with session
        redirect_url = RelayState or "/dashboard"
        response = RedirectResponse(url=f"{redirect_url}?session={session.session_id}")

        # Set secure cookie
        response.set_cookie(
            key="sso_session",
            value=session.session_id,
            max_age=28800,  # 8 hours
            httponly=True,
            secure=True,
            samesite="lax"
        )

        return response

    except Exception as e:
        logger.error(f"SAML ACS failed: {e}")
        raise HTTPException(status_code=400, detail="SAML authentication failed")

@router.get("/oidc/callback")
async def oidc_callback(
    request: Request,
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    manager: SSOManager = Depends(get_sso_manager)
):
    """
    OIDC callback endpoint
    """
    try:
        if error:
            raise HTTPException(status_code=400, detail=f"OIDC error: {error}")

        if not code or not state:
            raise HTTPException(status_code=400, detail="Missing code or state parameter")

        # Decode state to get tenant ID
        state_data = json.loads(base64.urlsafe_b64decode(state.encode()).decode())
        tenant_id = state_data.get('tenant_id')
        return_url = state_data.get('return_url', '/dashboard')

        request_data = {
            'code': code,
            'state': state
        }

        async with manager:
            user = await manager.handle_sso_callback(
                tenant_id=tenant_id,
                protocol=SSOProtocol.OIDC,
                request_data=request_data
            )

        # Create session
        session = await manager.create_sso_session(user)

        # Log successful login
        await _log_login_attempt(tenant_id, user.email, "oidc", True)

        # Redirect to application
        response = RedirectResponse(url=f"{return_url}?session={session.session_id}")

        # Set secure cookie
        response.set_cookie(
            key="sso_session",
            value=session.session_id,
            max_age=28800,  # 8 hours
            httponly=True,
            secure=True,
            samesite="lax"
        )

        return response

    except Exception as e:
        logger.error(f"OIDC callback failed: {e}")
        raise HTTPException(status_code=400, detail="OIDC authentication failed")

@router.post("/validate")
async def validate_session(
    session_id: str,
    manager: SSOManager = Depends(get_sso_manager)
):
    """
    Validate SSO session
    """
    try:
        async with manager:
            user = await manager.validate_sso_session(session_id)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired session")

        return {
            "valid": True,
            "user": {
                "user_id": user.user_id,
                "email": user.email,
                "display_name": user.display_name,
                "roles": [role.value for role in user.roles],
                "groups": user.groups
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session validation failed: {e}")
        raise HTTPException(status_code=500, detail="Session validation failed")

@router.post("/logout")
async def logout_session(
    session_id: str,
    manager: SSOManager = Depends(get_sso_manager)
):
    """
    Logout SSO session
    """
    try:
        async with manager:
            success = await manager.logout_sso_session(session_id)

        if success:
            return {"success": True, "message": "Logged out successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")

@router.get("/users", response_model=List[SSOUserResponse])
async def list_sso_users(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    active_only: bool = Query(True, description="Show only active users"),
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    List SSO users for a tenant
    """
    try:
        async with db_pool.acquire() as conn:
            query = """
                SELECT u.*, COALESCE(u.login_count, 0) as login_count
                FROM auth.sso_users u
                WHERE u.tenant_id = $1
            """
            params = [tenant.id]

            if provider:
                query += " AND u.provider = $2"
                params.append(provider)

            if active_only:
                if provider:
                    query += " AND u.is_active = $3"
                else:
                    query += " AND u.is_active = $2"
                params.append(True)

            query += " ORDER BY u.last_login_at DESC NULLS LAST"

            rows = await conn.fetch(query, *params)

            users = []
            for row in rows:
                users.append(SSOUserResponse(
                    user_id=row['user_id'],
                    email=row['email'],
                    first_name=row['first_name'] or '',
                    last_name=row['last_name'] or '',
                    display_name=row['display_name'] or '',
                    roles=row['roles'] or [],
                    groups=row['groups'] or [],
                    provider=row['provider'],
                    is_active=row['is_active'],
                    last_login_at=row['last_login_at'],
                    login_count=row['login_count']
                ))

            return users

    except Exception as e:
        logger.error(f"Failed to list SSO users: {e}")
        raise HTTPException(status_code=500, detail="Failed to list users")

@router.get("/sessions", response_model=List[SSOSessionResponse])
async def list_active_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    List active SSO sessions
    """
    try:
        async with db_pool.acquire() as conn:
            query = """
                SELECT s.* FROM auth.sso_sessions s
                JOIN auth.sso_users u ON s.user_id = u.user_id
                WHERE u.tenant_id = $1 AND s.is_active = TRUE AND s.expires_at > NOW()
            """
            params = [tenant.id]

            if user_id:
                query += " AND s.user_id = $2"
                params.append(user_id)

            query += " ORDER BY s.last_accessed_at DESC"

            rows = await conn.fetch(query, *params)

            sessions = []
            for row in rows:
                sessions.append(SSOSessionResponse(
                    session_id=row['session_id'],
                    user_id=row['user_id'],
                    provider=row['provider'],
                    expires_at=row['expires_at'],
                    last_accessed_at=row['last_accessed_at'],
                    ip_address=str(row['ip_address']) if row['ip_address'] else None
                ))

            return sessions

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")

@router.get("/analytics", response_model=SSOAnalyticsResponse)
async def get_sso_analytics(
    days: int = Query(default=30, le=90),
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Get SSO analytics for a tenant
    """
    try:
        async with db_pool.acquire() as conn:
            # Get overall stats
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(DISTINCT u.user_id) as total_users,
                    COUNT(DISTINCT s.session_id) as active_sessions,
                    COUNT(DISTINCT CASE WHEN s.created_at::date = CURRENT_DATE THEN s.session_id END) as logins_today,
                    COUNT(DISTINCT CASE WHEN s.created_at > CURRENT_DATE - INTERVAL '7 days' THEN s.session_id END) as logins_this_week
                FROM auth.sso_users u
                LEFT JOIN auth.sso_sessions s ON u.user_id = s.user_id AND s.is_active = TRUE
                WHERE u.tenant_id = $1
            """, tenant.id)

            # Get failed attempts today
            failed_attempts = await conn.fetchval("""
                SELECT COUNT(*) FROM auth.sso_login_attempts
                WHERE tenant_id = $1 AND success = FALSE
                AND attempted_at::date = CURRENT_DATE
            """, tenant.id)

            # Get top providers
            top_providers = await conn.fetch("""
                SELECT
                    provider,
                    COUNT(DISTINCT user_id) as user_count,
                    COUNT(DISTINCT CASE WHEN last_login_at > NOW() - INTERVAL '7 days' THEN user_id END) as active_users
                FROM auth.sso_users
                WHERE tenant_id = $1 AND is_active = TRUE
                GROUP BY provider
                ORDER BY user_count DESC
            """, tenant.id)

            # Get recent activity
            recent_activity = await conn.fetch("""
                SELECT
                    u.email,
                    u.display_name,
                    s.provider,
                    s.created_at,
                    'login' as activity_type
                FROM auth.sso_sessions s
                JOIN auth.sso_users u ON s.user_id = u.user_id
                WHERE u.tenant_id = $1 AND s.created_at > NOW() - INTERVAL '24 hours'
                ORDER BY s.created_at DESC
                LIMIT 10
            """, tenant.id)

            return SSOAnalyticsResponse(
                total_users=stats['total_users'] or 0,
                active_sessions=stats['active_sessions'] or 0,
                logins_today=stats['logins_today'] or 0,
                logins_this_week=stats['logins_this_week'] or 0,
                failed_attempts_today=failed_attempts or 0,
                top_providers=[
                    {
                        "provider": row['provider'],
                        "user_count": row['user_count'],
                        "active_users": row['active_users']
                    }
                    for row in top_providers
                ],
                recent_activity=[
                    {
                        "email": row['email'],
                        "display_name": row['display_name'],
                        "provider": row['provider'],
                        "activity_type": row['activity_type'],
                        "timestamp": row['created_at'].isoformat()
                    }
                    for row in recent_activity
                ]
            )

    except Exception as e:
        logger.error(f"Failed to get SSO analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@router.get("/metadata/{config_id}")
async def get_sso_metadata(
    config_id: str,
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Get SSO metadata for configuration (SAML metadata, OIDC discovery, etc.)
    """
    try:
        async with db_pool.acquire() as conn:
            # Verify config ownership
            config = await conn.fetchrow("""
                SELECT * FROM auth.sso_configs
                WHERE config_id = $1 AND tenant_id = $2 AND is_active = TRUE
            """, config_id, tenant.id)

            if not config:
                raise HTTPException(status_code=404, detail="SSO configuration not found")

            if config['protocol'] == 'saml2':
                # Generate SAML metadata
                metadata = _generate_saml_metadata(config)
                return HTMLResponse(content=metadata, media_type="application/xml")

            elif config['protocol'] == 'oidc':
                # Return OIDC configuration
                return {
                    "issuer": f"https://api.agentsystem.com/auth/sso/oidc/{config_id}",
                    "authorization_endpoint": f"https://api.agentsystem.com/auth/sso/oidc/{config_id}/auth",
                    "token_endpoint": f"https://api.agentsystem.com/auth/sso/oidc/{config_id}/token",
                    "userinfo_endpoint": f"https://api.agentsystem.com/auth/sso/oidc/{config_id}/userinfo",
                    "jwks_uri": f"https://api.agentsystem.com/auth/sso/oidc/{config_id}/jwks"
                }

            else:
                raise HTTPException(status_code=400, detail="Metadata not available for this protocol")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get SSO metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metadata")

# Helper functions
async def _log_login_attempt(tenant_id: str, email: str, provider: str,
                           success: bool, failure_reason: str = None):
    """Log login attempt for audit and security"""

    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO auth.sso_login_attempts (
                tenant_id, email, provider, ip_address, success, failure_reason
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """, tenant_id, email, provider, None, success, failure_reason)

def _generate_saml_metadata(config: Dict[str, Any]) -> str:
    """Generate SAML metadata XML"""

    entity_id = config['saml_entity_id']
    acs_url = f"https://api.agentsystem.com/api/v1/auth/sso/saml/acs"

    metadata = f"""<?xml version="1.0" encoding="UTF-8"?>
<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
                     entityID="{entity_id}">
    <md:SPSSODescriptor protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
        <md:AssertionConsumerService
            Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            Location="{acs_url}"
            index="0" />
    </md:SPSSODescriptor>
</md:EntityDescriptor>"""

    return metadata

# Health check endpoint
@router.get("/health")
async def sso_health_check():
    """Health check for SSO integration"""
    return {
        "status": "healthy",
        "service": "sso_integration",
        "timestamp": datetime.now().isoformat()
    }

# Export router
__all__ = ["router"]
