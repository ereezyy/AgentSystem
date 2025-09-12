
"""
Enterprise SSO Integration - AgentSystem Profit Machine
Single Sign-On integration with Active Directory, Okta, and other enterprise identity providers
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
import jwt
import base64
from urllib.parse import urlencode, parse_qs
import xml.etree.ElementTree as ET
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import uuid

logger = logging.getLogger(__name__)

class SSOProvider(Enum):
    ACTIVE_DIRECTORY = "active_directory"
    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE_WORKSPACE = "google_workspace"
    ONELOGIN = "onelogin"
    PING_IDENTITY = "ping_identity"
    SAML_GENERIC = "saml_generic"
    OIDC_GENERIC = "oidc_generic"

class SSOProtocol(Enum):
    SAML2 = "saml2"
    OIDC = "oidc"
    OAUTH2 = "oauth2"
    LDAP = "ldap"

class UserRole(Enum):
    USER = "user"
    ADMIN = "admin"
    OWNER = "owner"
    VIEWER = "viewer"

@dataclass
class SSOConfig:
    config_id: str
    tenant_id: str
    provider: SSOProvider
    protocol: SSOProtocol
    provider_name: str
    is_active: bool
    is_default: bool

    # SAML Configuration
    saml_entity_id: Optional[str] = None
    saml_sso_url: Optional[str] = None
    saml_slo_url: Optional[str] = None
    saml_certificate: Optional[str] = None
    saml_private_key: Optional[str] = None
    saml_metadata_url: Optional[str] = None

    # OIDC/OAuth2 Configuration
    oidc_client_id: Optional[str] = None
    oidc_client_secret: Optional[str] = None
    oidc_discovery_url: Optional[str] = None
    oidc_authorization_endpoint: Optional[str] = None
    oidc_token_endpoint: Optional[str] = None
    oidc_userinfo_endpoint: Optional[str] = None
    oidc_jwks_url: Optional[str] = None
    oidc_scopes: List[str] = None

    # LDAP Configuration
    ldap_server: Optional[str] = None
    ldap_port: int = 389
    ldap_base_dn: Optional[str] = None
    ldap_bind_dn: Optional[str] = None
    ldap_bind_password: Optional[str] = None
    ldap_user_filter: Optional[str] = None
    ldap_group_filter: Optional[str] = None

    # Attribute Mapping
    attribute_mapping: Dict[str, str] = None
    role_mapping: Dict[str, str] = None
    group_mapping: Dict[str, str] = None

    # Auto-provisioning
    auto_provision_users: bool = True
    auto_assign_roles: bool = True
    default_role: UserRole = UserRole.USER

    # Advanced Settings
    force_authn: bool = False
    sign_requests: bool = True
    encrypt_assertions: bool = False
    session_timeout_minutes: int = 480  # 8 hours

    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class SSOUser:
    user_id: str
    tenant_id: str
    external_id: str
    provider: SSOProvider
    email: str
    first_name: str
    last_name: str
    display_name: str
    roles: List[UserRole]
    groups: List[str]
    attributes: Dict[str, Any]
    is_active: bool
    last_login_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

@dataclass
class SSOSession:
    session_id: str
    user_id: str
    tenant_id: str
    provider: SSOProvider
    external_session_id: Optional[str]
    access_token: Optional[str]
    refresh_token: Optional[str]
    id_token: Optional[str]
    expires_at: datetime
    created_at: datetime
    last_accessed_at: datetime

class SSOManager:
    """
    Enterprise SSO integration manager
    """

    def __init__(self, db_pool: asyncpg.Pool, base_url: str):
        self.db_pool = db_pool
        self.base_url = base_url
        self.session = None
        self.configs_cache = {}

        # Provider-specific handlers
        self.saml_handler = SAMLHandler(self)
        self.oidc_handler = OIDCHandler(self)
        self.ldap_handler = LDAPHandler(self)

        # Default attribute mappings
        self.default_mappings = {
            SSOProvider.ACTIVE_DIRECTORY: {
                'email': 'mail',
                'first_name': 'givenName',
                'last_name': 'sn',
                'display_name': 'displayName',
                'groups': 'memberOf'
            },
            SSOProvider.OKTA: {
                'email': 'email',
                'first_name': 'given_name',
                'last_name': 'family_name',
                'display_name': 'name',
                'groups': 'groups'
            },
            SSOProvider.AZURE_AD: {
                'email': 'email',
                'first_name': 'given_name',
                'last_name': 'family_name',
                'display_name': 'name',
                'groups': 'groups'
            }
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def create_sso_config(self, tenant_id: str, provider: SSOProvider,
                               config_data: Dict[str, Any]) -> SSOConfig:
        """Create SSO configuration for a tenant"""

        config_id = str(uuid.uuid4())

        # Determine protocol based on provider
        protocol = self._get_default_protocol(provider)

        config = SSOConfig(
            config_id=config_id,
            tenant_id=tenant_id,
            provider=provider,
            protocol=protocol,
            provider_name=config_data.get('provider_name', provider.value),
            is_active=config_data.get('is_active', True),
            is_default=config_data.get('is_default', False),
            attribute_mapping=config_data.get('attribute_mapping',
                                            self.default_mappings.get(provider, {})),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Set protocol-specific configuration
        if protocol == SSOProtocol.SAML2:
            config.saml_entity_id = config_data.get('entity_id')
            config.saml_sso_url = config_data.get('sso_url')
            config.saml_slo_url = config_data.get('slo_url')
            config.saml_certificate = config_data.get('certificate')
            config.saml_metadata_url = config_data.get('metadata_url')

        elif protocol in [SSOProtocol.OIDC, SSOProtocol.OAUTH2]:
            config.oidc_client_id = config_data.get('client_id')
            config.oidc_client_secret = config_data.get('client_secret')
            config.oidc_discovery_url = config_data.get('discovery_url')
            config.oidc_scopes = config_data.get('scopes', ['openid', 'profile', 'email'])

        elif protocol == SSOProtocol.LDAP:
            config.ldap_server = config_data.get('server')
            config.ldap_port = config_data.get('port', 389)
            config.ldap_base_dn = config_data.get('base_dn')
            config.ldap_bind_dn = config_data.get('bind_dn')
            config.ldap_bind_password = config_data.get('bind_password')

        await self._store_sso_config(config)
        self.configs_cache[config_id] = config

        return config

    async def get_sso_login_url(self, tenant_id: str, config_id: str = None,
                               return_url: str = None) -> str:
        """Generate SSO login URL"""

        config = await self._get_sso_config(tenant_id, config_id)
        if not config:
            raise ValueError("SSO configuration not found")

        if config.protocol == SSOProtocol.SAML2:
            return await self.saml_handler.get_login_url(config, return_url)
        elif config.protocol in [SSOProtocol.OIDC, SSOProtocol.OAUTH2]:
            return await self.oidc_handler.get_login_url(config, return_url)
        else:
            raise ValueError(f"Protocol {config.protocol} does not support login URLs")

    async def handle_sso_callback(self, tenant_id: str, protocol: SSOProtocol,
                                 request_data: Dict[str, Any]) -> SSOUser:
        """Handle SSO authentication callback"""

        if protocol == SSOProtocol.SAML2:
            return await self.saml_handler.handle_callback(tenant_id, request_data)
        elif protocol in [SSOProtocol.OIDC, SSOProtocol.OAUTH2]:
            return await self.oidc_handler.handle_callback(tenant_id, request_data)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

    async def authenticate_user(self, tenant_id: str, username: str,
                              password: str, config_id: str = None) -> Optional[SSOUser]:
        """Authenticate user against SSO provider (for LDAP)"""

        config = await self._get_sso_config(tenant_id, config_id)
        if not config:
            return None

        if config.protocol == SSOProtocol.LDAP:
            return await self.ldap_handler.authenticate(config, username, password)
        else:
            raise ValueError(f"Direct authentication not supported for {config.protocol}")

    async def get_sso_user(self, user_id: str) -> Optional[SSOUser]:
        """Get SSO user by ID"""

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM auth.sso_users WHERE user_id = $1
            """, user_id)

            if row:
                return SSOUser(
                    user_id=row['user_id'],
                    tenant_id=row['tenant_id'],
                    external_id=row['external_id'],
                    provider=SSOProvider(row['provider']),
                    email=row['email'],
                    first_name=row['first_name'],
                    last_name=row['last_name'],
                    display_name=row['display_name'],
                    roles=[UserRole(role) for role in row['roles']],
                    groups=row['groups'] or [],
                    attributes=json.loads(row['attributes']) if row['attributes'] else {},
                    is_active=row['is_active'],
                    last_login_at=row['last_login_at'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

        return None

    async def create_sso_session(self, user: SSOUser, provider_data: Dict[str, Any] = None) -> SSOSession:
        """Create SSO session for authenticated user"""

        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=8)  # Default 8 hours

        session = SSOSession(
            session_id=session_id,
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            provider=user.provider,
            external_session_id=provider_data.get('session_id') if provider_data else None,
            access_token=provider_data.get('access_token') if provider_data else None,
            refresh_token=provider_data.get('refresh_token') if provider_data else None,
            id_token=provider_data.get('id_token') if provider_data else None,
            expires_at=expires_at,
            created_at=datetime.now(),
            last_accessed_at=datetime.now()
        )

        await self._store_sso_session(session)
        return session

    async def validate_sso_session(self, session_id: str) -> Optional[SSOUser]:
        """Validate SSO session and return user"""

        async with self.db_pool.acquire() as conn:
            # Get session
            session_row = await conn.fetchrow("""
                SELECT * FROM auth.sso_sessions
                WHERE session_id = $1 AND expires_at > NOW()
            """, session_id)

            if not session_row:
                return None

            # Update last accessed
            await conn.execute("""
                UPDATE auth.sso_sessions
                SET last_accessed_at = NOW()
                WHERE session_id = $1
            """, session_id)

            # Get user
            user = await self.get_sso_user(session_row['user_id'])
            if user and user.is_active:
                return user

        return None

    async def logout_sso_session(self, session_id: str) -> bool:
        """Logout SSO session"""

        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM auth.sso_sessions WHERE session_id = $1
            """, session_id)

            return result == "DELETE 1"

    def _get_default_protocol(self, provider: SSOProvider) -> SSOProtocol:
        """Get default protocol for provider"""

        protocol_mapping = {
            SSOProvider.ACTIVE_DIRECTORY: SSOProtocol.LDAP,
            SSOProvider.OKTA: SSOProtocol.SAML2,
            SSOProvider.AZURE_AD: SSOProtocol.OIDC,
            SSOProvider.GOOGLE_WORKSPACE: SSOProtocol.OIDC,
            SSOProvider.ONELOGIN: SSOProtocol.SAML2,
            SSOProvider.PING_IDENTITY: SSOProtocol.SAML2,
            SSOProvider.SAML_GENERIC: SSOProtocol.SAML2,
            SSOProvider.OIDC_GENERIC: SSOProtocol.OIDC
        }

        return protocol_mapping.get(provider, SSOProtocol.SAML2)

    async def _get_sso_config(self, tenant_id: str, config_id: str = None) -> Optional[SSOConfig]:
        """Get SSO configuration"""

        # Use specific config or default
        if config_id and config_id in self.configs_cache:
            return self.configs_cache[config_id]

        async with self.db_pool.acquire() as conn:
            if config_id:
                row = await conn.fetchrow("""
                    SELECT * FROM auth.sso_configs
                    WHERE config_id = $1 AND tenant_id = $2 AND is_active = true
                """, config_id, tenant_id)
            else:
                row = await conn.fetchrow("""
                    SELECT * FROM auth.sso_configs
                    WHERE tenant_id = $1 AND is_active = true AND is_default = true
                """, tenant_id)

            if row:
                config = SSOConfig(
                    config_id=row['config_id'],
                    tenant_id=row['tenant_id'],
                    provider=SSOProvider(row['provider']),
                    protocol=SSOProtocol(row['protocol']),
                    provider_name=row['provider_name'],
                    is_active=row['is_active'],
                    is_default=row['is_default'],
                    saml_entity_id=row['saml_entity_id'],
                    saml_sso_url=row['saml_sso_url'],
                    saml_slo_url=row['saml_slo_url'],
                    saml_certificate=row['saml_certificate'],
                    oidc_client_id=row['oidc_client_id'],
                    oidc_client_secret=row['oidc_client_secret'],
                    oidc_discovery_url=row['oidc_discovery_url'],
                    ldap_server=row['ldap_server'],
                    ldap_port=row['ldap_port'],
                    ldap_base_dn=row['ldap_base_dn'],
                    attribute_mapping=json.loads(row['attribute_mapping']) if row['attribute_mapping'] else {},
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

                self.configs_cache[config.config_id] = config
                return config

        return None

    # Database operations
    async def _store_sso_config(self, config: SSOConfig):
        """Store SSO configuration in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO auth.sso_configs (
                    config_id, tenant_id, provider, protocol, provider_name,
                    is_active, is_default, saml_entity_id, saml_sso_url, saml_slo_url,
                    saml_certificate, oidc_client_id, oidc_client_secret, oidc_discovery_url,
                    ldap_server, ldap_port, ldap_base_dn, ldap_bind_dn, ldap_bind_password,
                    attribute_mapping, role_mapping, group_mapping, auto_provision_users,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
            """,
                config.config_id, config.tenant_id, config.provider.value, config.protocol.value,
                config.provider_name, config.is_active, config.is_default, config.saml_entity_id,
                config.saml_sso_url, config.saml_slo_url, config.saml_certificate,
                config.oidc_client_id, config.oidc_client_secret, config.oidc_discovery_url,
                config.ldap_server, config.ldap_port, config.ldap_base_dn, config.ldap_bind_dn,
                config.ldap_bind_password, json.dumps(config.attribute_mapping),
                json.dumps(config.role_mapping) if config.role_mapping else None,
                json.dumps(config.group_mapping) if config.group_mapping else None,
                config.auto_provision_users, config.created_at, config.updated_at
            )

    async def _store_sso_user(self, user: SSOUser):
        """Store or update SSO user"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO auth.sso_users (
                    user_id, tenant_id, external_id, provider, email, first_name,
                    last_name, display_name, roles, groups, attributes, is_active,
                    last_login_at, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (external_id, tenant_id, provider)
                DO UPDATE SET
                    email = EXCLUDED.email,
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    display_name = EXCLUDED.display_name,
                    roles = EXCLUDED.roles,
                    groups = EXCLUDED.groups,
                    attributes = EXCLUDED.attributes,
                    is_active = EXCLUDED.is_active,
                    last_login_at = EXCLUDED.last_login_at,
                    updated_at = EXCLUDED.updated_at
            """,
                user.user_id, user.tenant_id, user.external_id, user.provider.value,
                user.email, user.first_name, user.last_name, user.display_name,
                [role.value for role in user.roles], user.groups, json.dumps(user.attributes),
                user.is_active, user.last_login_at, user.created_at, user.updated_at
            )

    async def _store_sso_session(self, session: SSOSession):
        """Store SSO session"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO auth.sso_sessions (
                    session_id, user_id, tenant_id, provider, external_session_id,
                    access_token, refresh_token, id_token, expires_at, created_at, last_accessed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                session.session_id, session.user_id, session.tenant_id, session.provider.value,
                session.external_session_id, session.access_token, session.refresh_token,
                session.id_token, session.expires_at, session.created_at, session.last_accessed_at
            )

class SAMLHandler:
    """SAML 2.0 authentication handler"""

    def __init__(self, sso_manager: SSOManager):
        self.sso_manager = sso_manager

    async def get_login_url(self, config: SSOConfig, return_url: str = None) -> str:
        """Generate SAML login URL"""

        # Create SAML AuthnRequest
        authn_request = self._create_authn_request(config, return_url)

        # Encode and redirect to IdP
        encoded_request = base64.b64encode(authn_request.encode()).decode()

        params = {
            'SAMLRequest': encoded_request,
            'RelayState': return_url or ''
        }

        return f"{config.saml_sso_url}?{urlencode(params)}"

    async def handle_callback(self, tenant_id: str, request_data: Dict[str, Any]) -> SSOUser:
        """Handle SAML response callback"""

        saml_response = request_data.get('SAMLResponse')
        if not saml_response:
            raise ValueError("Missing SAMLResponse")

        # Decode and parse SAML response
        decoded_response = base64.b64decode(saml_response).decode()

        # Parse XML and extract user attributes
        root = ET.fromstring(decoded_response)

        # Extract user information from SAML assertions
        user_data = self._extract_user_data(root, tenant_id)

        # Create or update user
        user = await self._provision_user(user_data)

        return user

    def _create_authn_request(self, config: SSOConfig, return_url: str = None) -> str:
        """Create SAML AuthnRequest"""

        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat() + 'Z'

        acs_url = f"{self.sso_manager.base_url}/api/v1/auth/sso/saml/acs"

        authn_request = f"""
        <samlp:AuthnRequest
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{request_id}"
            Version="2.0"
            IssueInstant="{timestamp}"
            Destination="{config.saml_sso_url}"
            AssertionConsumerServiceURL="{acs_url}"
            ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">
            <saml:Issuer>{config.saml_entity_id}</saml:Issuer>
        </samlp:AuthnRequest>
        """

        return authn_request

    def _extract_user_data(self, saml_root: ET.Element, tenant_id: str) -> Dict[str, Any]:
        """Extract user data from SAML response"""

        # This is a simplified implementation
        # In production, you'd need proper SAML parsing and validation

        # Extract attributes from SAML assertion
        attributes = {}

        # Find attribute statements
        for attr_stmt in saml_root.findall('.//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeStatement'):
            for attr in attr_stmt.findall('.//{urn:oasis:names:tc:SAML:2.0:assertion}Attribute'):
                attr_name = attr.get('Name')
                attr_values = [val.text for val in attr.findall('.//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeValue')]
                attributes[attr_name] = attr_values[0] if len(attr_values) == 1 else attr_values

        return {
            'tenant_id': tenant_id,
            'attributes': attributes,
            'provider': SSOProvider.SAML_GENERIC
        }

    async def _provision_user(self, user_data: Dict[str, Any]) -> SSOUser:
        """Provision user from SAML data"""

        # This would map SAML attributes to user fields
        # Implementation depends on specific attribute mapping

        user_id = str(uuid.uuid4())

        user = SSOUser(
            user_id=user_id,
            tenant_id=user_data['tenant_id'],
            external_id=user_data['attributes'].get('NameID', user_id),
            provider=user_data['provider'],
            email=user_data['attributes'].get('email', ''),
            first_name=user_data['attributes'].get('first_name', ''),
            last_name=user_data['attributes'].get('last_name', ''),
            display_name=user_data['attributes'].get('display_name', ''),
            roles=[UserRole.USER],
            groups=user_data['attributes'].get('groups', []),
            attributes=user_data['attributes'],
            is_active=True,
            last_login_at=datetime.now(),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        await self.sso_manager._store_sso_user(user)
        return user

class OIDCHandler:
    """OpenID Connect authentication handler"""

    def __init__(self, sso_manager: SSOManager):
        self.sso_manager = sso_manager

    async def get_login_url(self, config: SSOConfig, return_url: str = None) -> str:
        """Generate OIDC login URL"""

        # Discover endpoints if needed
        if not config.oidc_authorization_endpoint:
            await self._discover_endpoints(config)

        state = base64.urlsafe_b64encode(json.dumps({
            'tenant_id': config.tenant_id,
            'config_id': config.config_id,
            'return_url': return_url
        }).encode()).decode()

        params = {
            'client_id': config.oidc_client_id,
            'response_type': 'code',
            'scope': ' '.join(config.oidc_scopes or ['openid', 'profile', 'email']),
            'redirect_uri': f"{self.sso_manager.base_url}/api/v1/auth/sso/oidc/callback",
            'state': state
        }

        return f"{config.oidc_authorization_endpoint}?{urlencode(params)}"

    async def handle_callback(self, tenant_id: str, request_data: Dict[str, Any]) -> SSOUser:
        """Handle OIDC callback"""

        code = request_data.get('code')
        state = request_data.get('state')

        if not code:
            raise ValueError("Missing authorization code")

        # Decode state to get config
        state_data = json.loads(base64.urlsafe_b64decode(state.encode()).decode())
        config = await self.sso_manager._get_sso_config(state_data['tenant_id'], state_data['config_id'])

        if not config:
            raise ValueError("SSO configuration not found")

        # Exchange code for tokens
        tokens = await self._exchange_code_for_tokens(config, code)

        # Get user info
        user_info = await self._get_user_info(config, tokens['access_token'])

        # Provision user
        user = await self._provision_user(user_info, config, tokens)

        return user

    async def _discover_endpoints(self, config: SSOConfig):
        """Discover OIDC endpoints from discovery URL"""

        if not config.oidc_discovery_url:
            return

        async with self.sso_manager.session.get(config.oidc_discovery_url) as response:
            if response.status == 200:
                discovery_doc = await response.json()

                config.oidc_authorization_endpoint = discovery_doc.get('authorization_endpoint')
                config.oidc_token_endpoint = discovery_doc.get('token_endpoint')
                config.oidc_userinfo_endpoint = discovery_doc.get('userinfo_endpoint')
                config.oidc_jwks_url = discovery_doc.get('jwks_uri')

    async def _exchange_code_for_tokens(self, config: SSOConfig, code: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""

        token_data = {
            'grant_type': 'authorization_code',
            'client_id': config.oidc_client_id,
            'client_secret': config.oidc_client_secret,
            'code': code,
            'redirect_uri': f"{self.sso_manager.base_url}/api/v1/auth/sso/oidc/callback"
        }

        async with self.sso_manager.session.post(
            config.oidc_token_endpoint,
            data=token_data
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise ValueError(f"Token exchange failed: {response.status}")

    async def _get_user_info(self, config: SSOConfig, access_token: str) -> Dict[str, Any]:
        """Get user information from userinfo endpoint"""

        headers = {'Authorization': f'Bearer {access_token}'}

        async with self.sso_manager.session.get(
            config.oidc_userinfo_endpoint,
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise ValueError(f"Failed to get user info: {response.status}")

    async def _provision_user(self, user_info: Dict[str, Any], config: SSOConfig,
                            tokens: Dict[str, Any]) -> SSOUser:
        """Provision user from OIDC user info"""

        # Map attributes using config
        mapping = config.attribute_mapping or {}

        user_id = str(uuid.uuid4())
        external_id = user_info.get('sub') or user_info.get('id')

        user = SSOUser(
            user_id=user_id,
            tenant_id=config.tenant_id,
            external_id=external_id,
            provider=config.provider,
            email=user_info.get(mapping.get('email', 'email'), ''),
            first_name=user_info.get(mapping.get('first_name', 'given_name'), ''),
            last_name=user_info.get(mapping.get('last_name', 'family_name'), ''),
            display_name=user_info.get(mapping.get('display_name', 'name'), ''),
            roles=[UserRole.USER],  # Default role, can be mapped from groups
            groups=user_info.get(mapping.get('groups', 'groups'), []),
            attributes=user_info,
            is_active=True,
            last_login_at=datetime.now(),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        await self.sso_manager._store_sso_user(user)
        return user

class LDAPHandler:
    """LDAP authentication handler"""

    def __init__(self, sso_manager: SSOManager):
        self.sso_manager = sso_manager

    async def authenticate(self, config: SSOConfig, username: str, password: str) -> Optional[SSOUser]:
        """Authenticate user against LDAP"""

        try:
            import ldap3

            # Connect to LDAP server
            server = ldap3.Server(f"{config.ldap_server}:{config.ldap_port}")

            # Try to bind with user credentials
            user_dn = f"uid={username},{config.ldap_base_dn}"
            conn = ldap3.Connection(server, user=user_dn, password=password)

            if not conn.bind():
                return None

            # Search for user attributes
            search_filter = config.ldap_user_filter or f"(uid={username})"
            conn.search(config.ldap_base_dn, search_filter, attributes=['*'])

            if not conn.entries:
                return None

            user_entry = conn.entries[0]

            # Extract user attributes
            user_data = self._extract_ldap_attributes(user_entry, config)

            # Provision user
            user = await self._provision_user(user_data, config)

            conn.unbind()
            return user

        except ImportError:
            logger.error("ldap3 library not installed")
            return None
        except Exception as e:
            logger.error(f"LDAP authentication failed: {e}")
            return None

    def _extract_ldap_attributes(self, entry, config: SSOConfig) -> Dict[str, Any]:
        """Extract attributes from LDAP entry"""

        mapping = config.attribute_mapping or {}

        attributes = {}
        for key, ldap_attr in mapping.items():
            if hasattr(entry, ldap_attr):
                attr_value = getattr(entry, ldap_attr)
                attributes[key] = attr_value.value if hasattr(attr_value, 'value') else str(attr_value)

        return {
            'tenant_id': config.tenant_id,
            'provider': config.provider,
            'attributes': attributes,
            'external_id': entry.entry_dn
        }

    async def _provision_user(self, user_data: Dict[str, Any], config: SSOConfig) -> SSOUser:
        """Provision user from LDAP data"""

        user_id = str(uuid.uuid4())
        attrs = user_data['attributes']

        user = SSOUser(
            user_id=user_id,
            tenant_id=user_data['tenant_id'],
            external_id=user_data['external_id'],
            provider=user_data['provider'],
            email=attrs.get('email', ''),
            first_name=attrs.get('first_name', ''),
            last_name=attrs.get('last_name', ''),
            display_name=attrs.get('display_name', ''),
            roles=[UserRole.USER],
            groups=attrs.get('groups', []) if isinstance(attrs.get('groups'), list) else [attrs.get('groups', '')],
            attributes=attrs,
            is_active=True,
            last_login_at=datetime.now(),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        await self.sso_manager._store_sso_user(user)
        return user

# Export main classes
__all__ = [
    'SSOManager', 'SSOConfig', 'SSOUser', 'SSOSession',
    'SSOProvider', 'SSOProtocol', 'UserRole',
    'SAMLHandler', 'OIDCHandler', 'LDAPHandler'
]
