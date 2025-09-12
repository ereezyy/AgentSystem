-- Enterprise SSO Integration Database Schema
-- AgentSystem Profit Machine - Single Sign-On with Active Directory, Okta, etc.

-- Create auth schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS auth;

-- SSO configurations table
CREATE TABLE IF NOT EXISTS auth.sso_configs (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL, -- active_directory, okta, azure_ad, etc.
    protocol VARCHAR(20) NOT NULL, -- saml2, oidc, oauth2, ldap
    provider_name VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,

    -- SAML Configuration
    saml_entity_id VARCHAR(500),
    saml_sso_url TEXT,
    saml_slo_url TEXT,
    saml_certificate TEXT,
    saml_private_key TEXT,
    saml_metadata_url TEXT,

    -- OIDC/OAuth2 Configuration
    oidc_client_id VARCHAR(255),
    oidc_client_secret TEXT,
    oidc_discovery_url TEXT,
    oidc_authorization_endpoint TEXT,
    oidc_token_endpoint TEXT,
    oidc_userinfo_endpoint TEXT,
    oidc_jwks_url TEXT,
    oidc_scopes TEXT[], -- Array of scopes

    -- LDAP Configuration
    ldap_server VARCHAR(255),
    ldap_port INTEGER DEFAULT 389,
    ldap_base_dn VARCHAR(500),
    ldap_bind_dn VARCHAR(500),
    ldap_bind_password TEXT,
    ldap_user_filter VARCHAR(500),
    ldap_group_filter VARCHAR(500),
    ldap_use_ssl BOOLEAN DEFAULT FALSE,
    ldap_use_tls BOOLEAN DEFAULT FALSE,

    -- Attribute Mapping (JSON)
    attribute_mapping JSONB DEFAULT '{}',
    role_mapping JSONB DEFAULT '{}',
    group_mapping JSONB DEFAULT '{}',

    -- Auto-provisioning Settings
    auto_provision_users BOOLEAN DEFAULT TRUE,
    auto_assign_roles BOOLEAN DEFAULT TRUE,
    default_role VARCHAR(50) DEFAULT 'user',

    -- Advanced Settings
    force_authn BOOLEAN DEFAULT FALSE,
    sign_requests BOOLEAN DEFAULT TRUE,
    encrypt_assertions BOOLEAN DEFAULT FALSE,
    session_timeout_minutes INTEGER DEFAULT 480, -- 8 hours

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100),

    -- Constraints
    CONSTRAINT unique_default_per_tenant UNIQUE(tenant_id, is_default) DEFERRABLE INITIALLY DEFERRED,

    -- Indexes
    INDEX idx_sso_configs_tenant (tenant_id, is_active),
    INDEX idx_sso_configs_provider (provider, is_active)
);

-- SSO users table
CREATE TABLE IF NOT EXISTS auth.sso_users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    external_id VARCHAR(500) NOT NULL, -- User ID from external provider
    provider VARCHAR(50) NOT NULL,
    email VARCHAR(500) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    display_name VARCHAR(500),
    roles TEXT[] DEFAULT ARRAY['user']::TEXT[],
    groups TEXT[] DEFAULT ARRAY[]::TEXT[],
    attributes JSONB DEFAULT '{}', -- Additional user attributes
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    login_count INTEGER DEFAULT 0,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    password_changed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_external_user UNIQUE(external_id, tenant_id, provider),
    CONSTRAINT unique_email_per_tenant UNIQUE(email, tenant_id),

    -- Indexes
    INDEX idx_sso_users_tenant (tenant_id, is_active),
    INDEX idx_sso_users_email (email),
    INDEX idx_sso_users_external (external_id, provider),
    INDEX idx_sso_users_last_login (last_login_at DESC)
);

-- SSO sessions table
CREATE TABLE IF NOT EXISTS auth.sso_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.sso_users(user_id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    external_session_id VARCHAR(500), -- Session ID from external provider
    access_token TEXT,
    refresh_token TEXT,
    id_token TEXT,
    token_type VARCHAR(50) DEFAULT 'Bearer',
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    refresh_expires_at TIMESTAMP WITH TIME ZONE,
    scope TEXT,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes
    INDEX idx_sso_sessions_user (user_id, is_active),
    INDEX idx_sso_sessions_expires (expires_at) WHERE is_active = TRUE,
    INDEX idx_sso_sessions_external (external_session_id, provider)
);

-- SSO audit log
CREATE TABLE IF NOT EXISTS auth.sso_audit_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.sso_users(user_id) ON DELETE SET NULL,
    session_id UUID REFERENCES auth.sso_sessions(session_id) ON DELETE SET NULL,
    event_type VARCHAR(50) NOT NULL, -- login, logout, login_failed, session_expired, etc.
    provider VARCHAR(50) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    details JSONB DEFAULT '{}',
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    risk_score INTEGER DEFAULT 0, -- 0-100 risk assessment
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes
    INDEX idx_sso_audit_tenant_time (tenant_id, created_at DESC),
    INDEX idx_sso_audit_user_time (user_id, created_at DESC),
    INDEX idx_sso_audit_event (event_type, created_at DESC),
    INDEX idx_sso_audit_risk (risk_score DESC, created_at DESC) WHERE risk_score > 50
);

-- SSO group memberships
CREATE TABLE IF NOT EXISTS auth.sso_group_memberships (
    membership_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.sso_users(user_id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    group_name VARCHAR(255) NOT NULL,
    group_dn VARCHAR(500), -- Distinguished name for LDAP groups
    provider VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,

    -- Constraints
    CONSTRAINT unique_user_group UNIQUE(user_id, group_name, provider),

    -- Indexes
    INDEX idx_group_memberships_user (user_id, is_active),
    INDEX idx_group_memberships_group (group_name, provider, is_active)
);

-- SSO role assignments
CREATE TABLE IF NOT EXISTS auth.sso_role_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.sso_users(user_id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    role_name VARCHAR(100) NOT NULL, -- user, admin, owner, viewer
    granted_by VARCHAR(100),
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,

    -- Constraints
    CONSTRAINT unique_user_role UNIQUE(user_id, role_name),

    -- Indexes
    INDEX idx_role_assignments_user (user_id, is_active),
    INDEX idx_role_assignments_role (role_name, is_active)
);

-- SSO provider metadata cache
CREATE TABLE IF NOT EXISTS auth.sso_provider_metadata (
    metadata_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_id UUID REFERENCES auth.sso_configs(config_id) ON DELETE CASCADE,
    metadata_type VARCHAR(50) NOT NULL, -- saml_metadata, oidc_discovery, jwks
    metadata_content JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_config_metadata_type UNIQUE(config_id, metadata_type),

    -- Index
    INDEX idx_provider_metadata_expires (expires_at) WHERE expires_at IS NOT NULL
);

-- SSO login attempts tracking
CREATE TABLE IF NOT EXISTS auth.sso_login_attempts (
    attempt_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    email VARCHAR(500),
    provider VARCHAR(50) NOT NULL,
    ip_address INET NOT NULL,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    failure_reason VARCHAR(255),
    risk_factors JSONB DEFAULT '{}',
    attempted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes
    INDEX idx_login_attempts_email_time (email, attempted_at DESC),
    INDEX idx_login_attempts_ip_time (ip_address, attempted_at DESC),
    INDEX idx_login_attempts_failed (success, attempted_at DESC) WHERE success = FALSE
);

-- Functions for SSO management

-- Function to update user last login
CREATE OR REPLACE FUNCTION auth.update_user_last_login()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE auth.sso_users
    SET last_login_at = NOW(),
        login_count = login_count + 1,
        failed_login_attempts = 0
    WHERE user_id = NEW.user_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for session creation
DROP TRIGGER IF EXISTS trigger_session_login ON auth.sso_sessions;
CREATE TRIGGER trigger_session_login
    AFTER INSERT ON auth.sso_sessions
    FOR EACH ROW
    EXECUTE FUNCTION auth.update_user_last_login();

-- Function to cleanup expired sessions
CREATE OR REPLACE FUNCTION auth.cleanup_expired_sso_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Deactivate expired sessions
    UPDATE auth.sso_sessions
    SET is_active = FALSE
    WHERE expires_at < NOW() AND is_active = TRUE;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Delete very old sessions (older than 30 days)
    DELETE FROM auth.sso_sessions
    WHERE expires_at < NOW() - INTERVAL '30 days';

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to detect suspicious login activity
CREATE OR REPLACE FUNCTION auth.calculate_login_risk_score(
    p_email VARCHAR(500),
    p_ip_address INET,
    p_user_agent TEXT
) RETURNS INTEGER AS $$
DECLARE
    risk_score INTEGER := 0;
    recent_failures INTEGER;
    ip_history_count INTEGER;
    unusual_time BOOLEAN := FALSE;
BEGIN
    -- Check recent failed attempts for this email
    SELECT COUNT(*) INTO recent_failures
    FROM auth.sso_login_attempts
    WHERE email = p_email
    AND success = FALSE
    AND attempted_at > NOW() - INTERVAL '1 hour';

    -- Add risk for failed attempts
    risk_score := risk_score + (recent_failures * 15);

    -- Check if IP has been used before
    SELECT COUNT(*) INTO ip_history_count
    FROM auth.sso_login_attempts
    WHERE ip_address = p_ip_address
    AND attempted_at > NOW() - INTERVAL '30 days';

    -- Add risk for new IP
    IF ip_history_count = 0 THEN
        risk_score := risk_score + 25;
    END IF;

    -- Check for unusual login time (outside 6 AM - 10 PM)
    IF EXTRACT(HOUR FROM NOW()) < 6 OR EXTRACT(HOUR FROM NOW()) > 22 THEN
        risk_score := risk_score + 10;
    END IF;

    -- Cap risk score at 100
    RETURN LEAST(risk_score, 100);
END;
$$ LANGUAGE plpgsql;

-- Function to enforce SSO config constraints
CREATE OR REPLACE FUNCTION auth.enforce_sso_config_constraints()
RETURNS TRIGGER AS $$
BEGIN
    -- Only one default config per tenant
    IF NEW.is_default = TRUE THEN
        UPDATE auth.sso_configs
        SET is_default = FALSE
        WHERE tenant_id = NEW.tenant_id
        AND config_id != NEW.config_id
        AND is_default = TRUE;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for SSO config constraints
DROP TRIGGER IF EXISTS trigger_sso_config_constraints ON auth.sso_configs;
CREATE TRIGGER trigger_sso_config_constraints
    BEFORE INSERT OR UPDATE ON auth.sso_configs
    FOR EACH ROW
    EXECUTE FUNCTION auth.enforce_sso_config_constraints();

-- Views for common queries

-- Active SSO users view
CREATE OR REPLACE VIEW auth.v_active_sso_users AS
SELECT
    u.*,
    t.name as tenant_name,
    t.plan_type,
    s.session_id,
    s.last_accessed_at as last_session_activity,
    CASE
        WHEN s.expires_at > NOW() THEN TRUE
        ELSE FALSE
    END as has_active_session
FROM auth.sso_users u
JOIN tenant_management.tenants t ON u.tenant_id = t.id
LEFT JOIN auth.sso_sessions s ON u.user_id = s.user_id AND s.is_active = TRUE
WHERE u.is_active = TRUE;

-- SSO configuration summary view
CREATE OR REPLACE VIEW auth.v_sso_config_summary AS
SELECT
    c.*,
    t.name as tenant_name,
    COUNT(u.user_id) as total_users,
    COUNT(s.session_id) as active_sessions,
    MAX(u.last_login_at) as last_user_login
FROM auth.sso_configs c
JOIN tenant_management.tenants t ON c.tenant_id = t.id
LEFT JOIN auth.sso_users u ON c.tenant_id = u.tenant_id AND u.provider = c.provider
LEFT JOIN auth.sso_sessions s ON u.user_id = s.user_id AND s.is_active = TRUE
WHERE c.is_active = TRUE
GROUP BY c.config_id, t.name;

-- Recent SSO activity view
CREATE OR REPLACE VIEW auth.v_recent_sso_activity AS
SELECT
    'login' as activity_type,
    u.email,
    u.display_name,
    t.name as tenant_name,
    s.provider,
    s.ip_address,
    s.created_at as activity_time
FROM auth.sso_sessions s
JOIN auth.sso_users u ON s.user_id = u.user_id
JOIN tenant_management.tenants t ON s.tenant_id = t.id
WHERE s.created_at > NOW() - INTERVAL '24 hours'

UNION ALL

SELECT
    'audit' as activity_type,
    u.email,
    u.display_name,
    t.name as tenant_name,
    a.provider,
    a.ip_address,
    a.created_at as activity_time
FROM auth.sso_audit_log a
LEFT JOIN auth.sso_users u ON a.user_id = u.user_id
LEFT JOIN tenant_management.tenants t ON a.tenant_id = t.id
WHERE a.created_at > NOW() - INTERVAL '24 hours'

ORDER BY activity_time DESC;

-- Grant permissions
GRANT USAGE ON SCHEMA auth TO agentsystem_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA auth TO agentsystem_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA auth TO agentsystem_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA auth TO agentsystem_app;

-- Comments for documentation
COMMENT ON SCHEMA auth IS 'Schema for authentication and SSO integration';
COMMENT ON TABLE auth.sso_configs IS 'SSO provider configurations for tenants';
COMMENT ON TABLE auth.sso_users IS 'Users authenticated via SSO providers';
COMMENT ON TABLE auth.sso_sessions IS 'Active SSO sessions with tokens';
COMMENT ON TABLE auth.sso_audit_log IS 'Audit trail for SSO authentication events';
COMMENT ON TABLE auth.sso_group_memberships IS 'User group memberships from SSO providers';
COMMENT ON TABLE auth.sso_role_assignments IS 'Role assignments for SSO users';
COMMENT ON TABLE auth.sso_provider_metadata IS 'Cached metadata from SSO providers';
COMMENT ON TABLE auth.sso_login_attempts IS 'Login attempt tracking for security';

-- Sample SSO configurations for testing
/*
INSERT INTO auth.sso_configs (
    tenant_id, provider, protocol, provider_name, is_default,
    saml_entity_id, saml_sso_url, attribute_mapping
) VALUES (
    (SELECT id FROM tenant_management.tenants LIMIT 1),
    'okta',
    'saml2',
    'Okta SAML',
    true,
    'https://agentsystem.okta.com',
    'https://agentsystem.okta.com/app/agentsystem/sso/saml',
    '{"email": "email", "first_name": "firstName", "last_name": "lastName", "groups": "groups"}'
);
*/
