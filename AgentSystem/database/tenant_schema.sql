-- 🏢 AgentSystem Multi-Tenant Database Schema
-- This schema supports complete tenant isolation and scalability

-- ============================================================================
-- TENANT MANAGEMENT SCHEMA
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS tenant_management;

-- Main tenants table
CREATE TABLE IF NOT EXISTS tenant_management.tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    subdomain VARCHAR(100) UNIQUE NOT NULL,
    plan_type VARCHAR(50) NOT NULL DEFAULT 'starter',
    status VARCHAR(50) NOT NULL DEFAULT 'active', -- active, suspended, cancelled
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Billing information
    stripe_customer_id VARCHAR(255) UNIQUE,
    billing_email VARCHAR(255) NOT NULL,
    billing_cycle_day INTEGER DEFAULT 1, -- Day of month for billing
    current_period_start TIMESTAMP WITH TIME ZONE,
    current_period_end TIMESTAMP WITH TIME ZONE,

    -- Configuration
    settings JSONB DEFAULT '{}',
    branding JSONB DEFAULT '{}', -- White-label settings

    -- Limits and quotas
    monthly_token_limit INTEGER,
    api_rate_limit INTEGER DEFAULT 1000, -- requests per minute

    CONSTRAINT valid_plan_type CHECK (plan_type IN ('starter', 'professional', 'enterprise', 'custom')),
    CONSTRAINT valid_status CHECK (status IN ('active', 'suspended', 'cancelled', 'trial'))
);

-- Tenant API keys
CREATE TABLE IF NOT EXISTS tenant_management.tenant_api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    api_key_hash VARCHAR(255) NOT NULL UNIQUE, -- bcrypt hashed
    permissions JSONB DEFAULT '{}', -- Granular permissions
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Tenant users (for multi-user accounts)
CREATE TABLE IF NOT EXISTS tenant_management.tenant_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'member', -- owner, admin, member, viewer
    is_active BOOLEAN DEFAULT true,
    invited_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    joined_at TIMESTAMP WITH TIME ZONE,
    last_login_at TIMESTAMP WITH TIME ZONE,

    UNIQUE(tenant_id, email),
    CONSTRAINT valid_role CHECK (role IN ('owner', 'admin', 'member', 'viewer'))
);

-- ============================================================================
-- USAGE TRACKING AND BILLING SCHEMA
-- ============================================================================

-- Real-time usage tracking
CREATE TABLE IF NOT EXISTS tenant_management.usage_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,

    -- Event details
    event_type VARCHAR(100) NOT NULL, -- ai_request, agent_execution, api_call
    service_type VARCHAR(100) NOT NULL, -- openai_gpt4, claude_3, whisper, etc.

    -- Usage metrics
    tokens_used INTEGER DEFAULT 0,
    requests_count INTEGER DEFAULT 1,
    processing_time_ms INTEGER,

    -- Cost tracking
    provider_cost_usd DECIMAL(10, 6) DEFAULT 0, -- What we paid to AI provider
    calculated_cost_usd DECIMAL(10, 6) DEFAULT 0, -- What we charge customer

    -- Metadata
    request_metadata JSONB DEFAULT '{}',
    response_metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    billing_period_start DATE, -- For aggregation

    -- Indexing for fast queries
    INDEX idx_usage_events_tenant_period (tenant_id, billing_period_start),
    INDEX idx_usage_events_created_at (created_at),
    INDEX idx_usage_events_service_type (service_type)
);

-- Daily usage aggregates (for faster billing calculations)
CREATE TABLE IF NOT EXISTS tenant_management.daily_usage_summary (
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    service_type VARCHAR(100) NOT NULL,

    -- Aggregated metrics
    total_tokens INTEGER DEFAULT 0,
    total_requests INTEGER DEFAULT 0,
    total_provider_cost_usd DECIMAL(10, 4) DEFAULT 0,
    total_calculated_cost_usd DECIMAL(10, 4) DEFAULT 0,

    -- Performance metrics
    avg_processing_time_ms INTEGER,
    success_rate DECIMAL(5, 4), -- 0.0 to 1.0

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (tenant_id, date, service_type)
);

-- Monthly billing summaries
CREATE TABLE IF NOT EXISTS tenant_management.monthly_bills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,

    -- Billing period
    billing_month INTEGER NOT NULL, -- 1-12
    billing_year INTEGER NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Subscription charges
    base_subscription_fee DECIMAL(10, 2) DEFAULT 0,

    -- Usage charges
    included_tokens INTEGER DEFAULT 0,
    total_tokens_used INTEGER DEFAULT 0,
    overage_tokens INTEGER DEFAULT 0,
    overage_charges DECIMAL(10, 2) DEFAULT 0,

    -- Total billing
    subtotal DECIMAL(10, 2) DEFAULT 0,
    tax_amount DECIMAL(10, 2) DEFAULT 0,
    total_amount DECIMAL(10, 2) DEFAULT 0,

    -- Payment tracking
    stripe_invoice_id VARCHAR(255),
    payment_status VARCHAR(50) DEFAULT 'pending', -- pending, paid, failed, refunded
    paid_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    usage_breakdown JSONB DEFAULT '{}', -- Detailed usage by service

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(tenant_id, billing_month, billing_year),
    CONSTRAINT valid_payment_status CHECK (payment_status IN ('pending', 'paid', 'failed', 'refunded'))
);

-- ============================================================================
-- AGENT AND WORKFLOW SCHEMA
-- ============================================================================

-- Custom agent definitions per tenant
CREATE TABLE IF NOT EXISTS tenant_management.custom_agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,

    -- Agent details
    name VARCHAR(255) NOT NULL,
    description TEXT,
    agent_type VARCHAR(100) NOT NULL, -- marketing, sales, customer_success, custom

    -- Configuration
    capabilities JSONB DEFAULT '[]', -- Array of capability strings
    system_prompt TEXT,
    model_preferences JSONB DEFAULT '{}', -- Preferred AI models
    parameters JSONB DEFAULT '{}', -- Custom parameters

    -- Usage and performance
    execution_count INTEGER DEFAULT 0,
    avg_execution_time_ms INTEGER,
    success_rate DECIMAL(5, 4) DEFAULT 1.0,

    -- Management
    is_active BOOLEAN DEFAULT true,
    created_by UUID REFERENCES tenant_management.tenant_users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Workflow definitions
CREATE TABLE IF NOT EXISTS tenant_management.workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,

    -- Workflow details
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workflow_type VARCHAR(100), -- automation, integration, custom

    -- Configuration
    definition JSONB NOT NULL, -- Complete workflow definition
    triggers JSONB DEFAULT '[]', -- Trigger conditions
    schedule JSONB DEFAULT '{}', -- Scheduling configuration

    -- Status and metrics
    is_active BOOLEAN DEFAULT true,
    execution_count INTEGER DEFAULT 0,
    last_execution_at TIMESTAMP WITH TIME ZONE,
    avg_execution_time_ms INTEGER,
    success_rate DECIMAL(5, 4) DEFAULT 1.0,

    -- Management
    created_by UUID REFERENCES tenant_management.tenant_users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Workflow execution logs
CREATE TABLE IF NOT EXISTS tenant_management.workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES tenant_management.workflows(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,

    -- Execution details
    status VARCHAR(50) NOT NULL, -- running, completed, failed, cancelled
    trigger_type VARCHAR(100), -- manual, scheduled, webhook, api

    -- Performance metrics
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER,

    -- Results
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    error_details JSONB DEFAULT '{}',

    -- Resource usage
    tokens_consumed INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 6) DEFAULT 0,

    CONSTRAINT valid_execution_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
);

-- ============================================================================
-- INTEGRATIONS AND WEBHOOKS SCHEMA
-- ============================================================================

-- Third-party integrations per tenant
CREATE TABLE IF NOT EXISTS tenant_management.integrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,

    -- Integration details
    integration_type VARCHAR(100) NOT NULL, -- salesforce, hubspot, slack, teams
    name VARCHAR(255) NOT NULL,

    -- Configuration
    config JSONB NOT NULL, -- Integration-specific configuration
    credentials JSONB NOT NULL, -- Encrypted credentials

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_sync_at TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(50) DEFAULT 'pending', -- pending, syncing, success, error
    error_details JSONB DEFAULT '{}',

    -- Management
    created_by UUID REFERENCES tenant_management.tenant_users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(tenant_id, integration_type, name)
);

-- Webhook endpoints
CREATE TABLE IF NOT EXISTS tenant_management.webhook_endpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,

    -- Webhook details
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    secret_key VARCHAR(255), -- For signature verification

    -- Events and filtering
    event_types JSONB DEFAULT '[]', -- Array of event types to send
    filters JSONB DEFAULT '{}', -- Additional filtering criteria

    -- Configuration
    headers JSONB DEFAULT '{}', -- Custom headers
    retry_count INTEGER DEFAULT 3,
    timeout_seconds INTEGER DEFAULT 30,

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_delivery_at TIMESTAMP WITH TIME ZONE,
    delivery_success_rate DECIMAL(5, 4) DEFAULT 1.0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- PERFORMANCE AND ANALYTICS SCHEMA
-- ============================================================================

-- System performance metrics per tenant
CREATE TABLE IF NOT EXISTS tenant_management.performance_metrics (
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL, -- response_time, success_rate, cost_efficiency
    time_bucket TIMESTAMP WITH TIME ZONE NOT NULL, -- 5-minute buckets

    -- Metric values
    value DECIMAL(12, 6) NOT NULL,
    count INTEGER DEFAULT 1,
    min_value DECIMAL(12, 6),
    max_value DECIMAL(12, 6),

    -- Metadata
    service_type VARCHAR(100),
    additional_dimensions JSONB DEFAULT '{}',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (tenant_id, metric_type, time_bucket, service_type)
);

-- Customer success metrics
CREATE TABLE IF NOT EXISTS tenant_management.customer_health_scores (
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    calculated_date DATE NOT NULL,

    -- Health score components
    usage_score INTEGER DEFAULT 0, -- 0-100
    engagement_score INTEGER DEFAULT 0, -- 0-100
    support_score INTEGER DEFAULT 0, -- 0-100
    payment_score INTEGER DEFAULT 0, -- 0-100

    -- Overall health
    overall_health_score INTEGER DEFAULT 0, -- 0-100
    health_trend VARCHAR(20), -- improving, stable, declining
    churn_risk_score DECIMAL(5, 4) DEFAULT 0, -- 0.0 to 1.0

    -- Predictions
    predicted_clv DECIMAL(10, 2),
    predicted_churn_date DATE,

    -- Metadata
    factors JSONB DEFAULT '{}', -- Contributing factors
    recommendations JSONB DEFAULT '[]', -- Action recommendations

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (tenant_id, calculated_date)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Usage tracking indexes
CREATE INDEX IF NOT EXISTS idx_usage_events_tenant_service_date
ON tenant_management.usage_events(tenant_id, service_type, created_at);

CREATE INDEX IF NOT EXISTS idx_daily_usage_tenant_date
ON tenant_management.daily_usage_summary(tenant_id, date);

-- Billing indexes
CREATE INDEX IF NOT EXISTS idx_monthly_bills_tenant_period
ON tenant_management.monthly_bills(tenant_id, billing_year, billing_month);

-- Agent and workflow indexes
CREATE INDEX IF NOT EXISTS idx_custom_agents_tenant_active
ON tenant_management.custom_agents(tenant_id, is_active);

CREATE INDEX IF NOT EXISTS idx_workflows_tenant_active
ON tenant_management.workflows(tenant_id, is_active);

CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_date
ON tenant_management.workflow_executions(workflow_id, started_at);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_tenant_type_time
ON tenant_management.performance_metrics(tenant_id, metric_type, time_bucket);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update tenant updated_at timestamp
CREATE OR REPLACE FUNCTION tenant_management.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at columns
CREATE TRIGGER update_tenants_updated_at
    BEFORE UPDATE ON tenant_management.tenants
    FOR EACH ROW EXECUTE FUNCTION tenant_management.update_updated_at_column();

CREATE TRIGGER update_custom_agents_updated_at
    BEFORE UPDATE ON tenant_management.custom_agents
    FOR EACH ROW EXECUTE FUNCTION tenant_management.update_updated_at_column();

CREATE TRIGGER update_workflows_updated_at
    BEFORE UPDATE ON tenant_management.workflows
    FOR EACH ROW EXECUTE FUNCTION tenant_management.update_updated_at_column();

-- Function to calculate monthly usage aggregates
CREATE OR REPLACE FUNCTION tenant_management.aggregate_daily_usage()
RETURNS VOID AS $$
BEGIN
    INSERT INTO tenant_management.daily_usage_summary (
        tenant_id, date, service_type, total_tokens, total_requests,
        total_provider_cost_usd, total_calculated_cost_usd,
        avg_processing_time_ms, success_rate
    )
    SELECT
        tenant_id,
        DATE(created_at) as date,
        service_type,
        SUM(tokens_used) as total_tokens,
        COUNT(*) as total_requests,
        SUM(provider_cost_usd) as total_provider_cost_usd,
        SUM(calculated_cost_usd) as total_calculated_cost_usd,
        AVG(processing_time_ms)::INTEGER as avg_processing_time_ms,
        AVG(CASE WHEN tokens_used > 0 THEN 1.0 ELSE 0.0 END) as success_rate
    FROM tenant_management.usage_events
    WHERE DATE(created_at) = CURRENT_DATE - INTERVAL '1 day'
    GROUP BY tenant_id, DATE(created_at), service_type
    ON CONFLICT (tenant_id, date, service_type)
    DO UPDATE SET
        total_tokens = EXCLUDED.total_tokens,
        total_requests = EXCLUDED.total_requests,
        total_provider_cost_usd = EXCLUDED.total_provider_cost_usd,
        total_calculated_cost_usd = EXCLUDED.total_calculated_cost_usd,
        avg_processing_time_ms = EXCLUDED.avg_processing_time_ms,
        success_rate = EXCLUDED.success_rate,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- Insert default pricing tiers configuration
INSERT INTO tenant_management.tenants (id, name, subdomain, plan_type, billing_email, settings) VALUES
(
    '00000000-0000-0000-0000-000000000001'::UUID,
    'AgentSystem Demo',
    'demo',
    'starter',
    'demo@agentsystem.ai',
    '{
        "pricing_tiers": {
            "starter": {
                "monthly_fee": 99,
                "included_tokens": 100000,
                "overage_rate": 0.002,
                "features": ["basic_ai", "email_support", "5_custom_agents"]
            },
            "professional": {
                "monthly_fee": 299,
                "included_tokens": 500000,
                "overage_rate": 0.0015,
                "features": ["all_ai", "agent_swarms", "priority_support", "unlimited_agents", "integrations"]
            },
            "enterprise": {
                "monthly_fee": 999,
                "included_tokens": 2000000,
                "overage_rate": 0.001,
                "features": ["unlimited", "custom_agents", "dedicated_support", "white_label", "sso", "compliance"]
            },
            "custom": {
                "monthly_fee": "negotiable",
                "included_tokens": "unlimited",
                "overage_rate": 0.0005,
                "features": ["everything", "custom_development", "dedicated_infrastructure"]
            }
        }
    }'
) ON CONFLICT (id) DO NOTHING;

-- Create indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_tenants_subdomain ON tenant_management.tenants(subdomain);
CREATE INDEX IF NOT EXISTS idx_tenants_stripe_customer ON tenant_management.tenants(stripe_customer_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON tenant_management.tenant_api_keys(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_tenant_users_email ON tenant_management.tenant_users(tenant_id, email);

COMMENT ON SCHEMA tenant_management IS 'Multi-tenant architecture for AgentSystem profit machine';
