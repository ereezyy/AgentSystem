-- Slack Integration Database Schema
-- AgentSystem Profit Machine - Slack Bot Integration

-- Create integrations schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS integrations;

-- Slack workspaces table
CREATE TABLE IF NOT EXISTS integrations.slack_workspaces (
    workspace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_tenant_team UNIQUE(tenant_id, team_id)
);

-- Slack channels configuration
CREATE TABLE IF NOT EXISTS integrations.slack_channels (
    channel_id VARCHAR(50) PRIMARY KEY,
    workspace_id UUID REFERENCES integrations.slack_workspaces(workspace_id) ON DELETE CASCADE,
    channel_name VARCHAR(500) NOT NULL,
    is_private BOOLEAN DEFAULT FALSE,
    is_member BOOLEAN DEFAULT FALSE,
    ai_assistance_enabled BOOLEAN DEFAULT TRUE,
    notification_types TEXT[] DEFAULT ARRAY[]::TEXT[],
    auto_respond BOOLEAN DEFAULT FALSE,
    response_delay_seconds INTEGER DEFAULT 2,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Slack messages log for analytics and AI training
CREATE TABLE IF NOT EXISTS integrations.slack_messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID REFERENCES integrations.slack_workspaces(workspace_id) ON DELETE CASCADE,
    channel_id VARCHAR(50) REFERENCES integrations.slack_channels(channel_id) ON DELETE CASCADE,
    user_id VARCHAR(50) NOT NULL,
    user_name VARCHAR(500),
    text TEXT NOT NULL,
    timestamp VARCHAR(50) NOT NULL,
    thread_ts VARCHAR(50),
    message_type VARCHAR(50) DEFAULT 'message',
    ai_response TEXT,
    ai_response_tokens INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    sentiment_score DECIMAL(3,2), -- -1.0 to 1.0
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes for performance
    INDEX idx_slack_messages_workspace_time (workspace_id, created_at DESC),
    INDEX idx_slack_messages_channel_time (channel_id, created_at DESC),
    INDEX idx_slack_messages_user (user_id, created_at DESC)
);

-- Slack commands usage tracking
CREATE TABLE IF NOT EXISTS integrations.slack_commands (
    command_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID REFERENCES integrations.slack_workspaces(workspace_id) ON DELETE CASCADE,
    channel_id VARCHAR(50),
    user_id VARCHAR(50) NOT NULL,
    command_name VARCHAR(100) NOT NULL,
    command_text TEXT,
    response_text TEXT,
    execution_time_ms INTEGER,
    tokens_used INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Index for analytics
    INDEX idx_slack_commands_workspace_time (workspace_id, created_at DESC),
    INDEX idx_slack_commands_type (command_name, created_at DESC)
);

-- Slack notifications queue
CREATE TABLE IF NOT EXISTS integrations.slack_notifications (
    notification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID REFERENCES integrations.slack_workspaces(workspace_id) ON DELETE CASCADE,
    channel_id VARCHAR(50),
    notification_type VARCHAR(100) NOT NULL,
    title VARCHAR(500),
    message TEXT NOT NULL,
    data JSONB,
    priority INTEGER DEFAULT 5, -- 1 (highest) to 10 (lowest)
    scheduled_for TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sent_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'pending', -- pending, sent, failed, cancelled
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Index for queue processing
    INDEX idx_slack_notifications_queue (status, scheduled_for, priority),
    INDEX idx_slack_notifications_workspace (workspace_id, created_at DESC)
);

-- Slack user preferences
CREATE TABLE IF NOT EXISTS integrations.slack_user_preferences (
    user_id VARCHAR(50) PRIMARY KEY,
    workspace_id UUID REFERENCES integrations.slack_workspaces(workspace_id) ON DELETE CASCADE,
    user_name VARCHAR(500),
    timezone VARCHAR(100) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    ai_assistance_enabled BOOLEAN DEFAULT TRUE,
    notification_preferences JSONB DEFAULT '{}',
    daily_summary BOOLEAN DEFAULT TRUE,
    weekly_report BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Unique constraint
    CONSTRAINT unique_user_workspace UNIQUE(user_id, workspace_id)
);

-- Slack workflow automations
CREATE TABLE IF NOT EXISTS integrations.slack_workflows (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID REFERENCES integrations.slack_workspaces(workspace_id) ON DELETE CASCADE,
    workflow_name VARCHAR(500) NOT NULL,
    trigger_type VARCHAR(100) NOT NULL, -- message, mention, reaction, file_upload, schedule
    trigger_config JSONB NOT NULL,
    action_type VARCHAR(100) NOT NULL, -- ai_response, notification, webhook, api_call
    action_config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    execution_count INTEGER DEFAULT 0,
    last_executed_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Slack analytics aggregations
CREATE TABLE IF NOT EXISTS integrations.slack_analytics_daily (
    date DATE NOT NULL,
    workspace_id UUID REFERENCES integrations.slack_workspaces(workspace_id) ON DELETE CASCADE,
    total_messages INTEGER DEFAULT 0,
    ai_responses INTEGER DEFAULT 0,
    commands_executed INTEGER DEFAULT 0,
    notifications_sent INTEGER DEFAULT 0,
    active_users INTEGER DEFAULT 0,
    active_channels INTEGER DEFAULT 0,
    total_tokens_used INTEGER DEFAULT 0,
    avg_response_time_ms DECIMAL(10,2),
    avg_sentiment_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Primary key and constraints
    PRIMARY KEY (date, workspace_id),
    INDEX idx_slack_analytics_workspace_date (workspace_id, date DESC)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_slack_workspaces_tenant
ON integrations.slack_workspaces(tenant_id, is_active);

CREATE INDEX IF NOT EXISTS idx_slack_workspaces_team
ON integrations.slack_workspaces(team_id) WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_slack_channels_workspace
ON integrations.slack_channels(workspace_id, is_member);

CREATE INDEX IF NOT EXISTS idx_slack_messages_timestamp
ON integrations.slack_messages(timestamp);

CREATE INDEX IF NOT EXISTS idx_slack_messages_ai_response
ON integrations.slack_messages(workspace_id) WHERE ai_response IS NOT NULL;

-- Functions for analytics and automation

-- Function to update workspace activity
CREATE OR REPLACE FUNCTION integrations.update_workspace_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE integrations.slack_workspaces
    SET updated_at = NOW()
    WHERE workspace_id = NEW.workspace_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for workspace activity tracking
DROP TRIGGER IF EXISTS trigger_workspace_activity ON integrations.slack_messages;
CREATE TRIGGER trigger_workspace_activity
    AFTER INSERT ON integrations.slack_messages
    FOR EACH ROW
    EXECUTE FUNCTION integrations.update_workspace_activity();

-- Function to calculate daily analytics
CREATE OR REPLACE FUNCTION integrations.calculate_daily_slack_analytics(target_date DATE DEFAULT CURRENT_DATE)
RETURNS VOID AS $$
BEGIN
    INSERT INTO integrations.slack_analytics_daily (
        date, workspace_id, total_messages, ai_responses, commands_executed,
        notifications_sent, active_users, active_channels, total_tokens_used,
        avg_response_time_ms, avg_sentiment_score
    )
    SELECT
        target_date,
        m.workspace_id,
        COUNT(*) as total_messages,
        COUNT(m.ai_response) as ai_responses,
        COALESCE(c.commands_executed, 0) as commands_executed,
        COALESCE(n.notifications_sent, 0) as notifications_sent,
        COUNT(DISTINCT m.user_id) as active_users,
        COUNT(DISTINCT m.channel_id) as active_channels,
        COALESCE(SUM(m.ai_response_tokens), 0) as total_tokens_used,
        AVG(m.processing_time_ms) as avg_response_time_ms,
        AVG(m.sentiment_score) as avg_sentiment_score
    FROM integrations.slack_messages m
    LEFT JOIN (
        SELECT workspace_id, COUNT(*) as commands_executed
        FROM integrations.slack_commands
        WHERE DATE(created_at) = target_date
        GROUP BY workspace_id
    ) c ON m.workspace_id = c.workspace_id
    LEFT JOIN (
        SELECT workspace_id, COUNT(*) as notifications_sent
        FROM integrations.slack_notifications
        WHERE DATE(sent_at) = target_date AND status = 'sent'
        GROUP BY workspace_id
    ) n ON m.workspace_id = n.workspace_id
    WHERE DATE(m.created_at) = target_date
    GROUP BY m.workspace_id, c.commands_executed, n.notifications_sent
    ON CONFLICT (date, workspace_id)
    DO UPDATE SET
        total_messages = EXCLUDED.total_messages,
        ai_responses = EXCLUDED.ai_responses,
        commands_executed = EXCLUDED.commands_executed,
        notifications_sent = EXCLUDED.notifications_sent,
        active_users = EXCLUDED.active_users,
        active_channels = EXCLUDED.active_channels,
        total_tokens_used = EXCLUDED.total_tokens_used,
        avg_response_time_ms = EXCLUDED.avg_response_time_ms,
        avg_sentiment_score = EXCLUDED.avg_sentiment_score;
END;
$$ LANGUAGE plpgsql;

-- Function to clean old data (data retention)
CREATE OR REPLACE FUNCTION integrations.cleanup_slack_data(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete old messages (keep AI responses longer)
    DELETE FROM integrations.slack_messages
    WHERE created_at < NOW() - INTERVAL '1 day' * retention_days
    AND ai_response IS NULL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Delete old notifications
    DELETE FROM integrations.slack_notifications
    WHERE created_at < NOW() - INTERVAL '1 day' * (retention_days / 2)
    AND status IN ('sent', 'failed', 'cancelled');

    -- Delete old commands (keep for shorter period)
    DELETE FROM integrations.slack_commands
    WHERE created_at < NOW() - INTERVAL '1 day' * (retention_days / 3);

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Views for common queries

-- Active workspaces view
CREATE OR REPLACE VIEW integrations.v_active_slack_workspaces AS
SELECT
    w.*,
    t.name as tenant_name,
    t.plan_type,
    COUNT(c.channel_id) as total_channels,
    COUNT(CASE WHEN c.ai_assistance_enabled THEN 1 END) as ai_enabled_channels
FROM integrations.slack_workspaces w
JOIN tenant_management.tenants t ON w.tenant_id = t.id
LEFT JOIN integrations.slack_channels c ON w.workspace_id = c.workspace_id
WHERE w.is_active = true
GROUP BY w.workspace_id, t.name, t.plan_type;

-- Recent activity view
CREATE OR REPLACE VIEW integrations.v_slack_recent_activity AS
SELECT
    w.team_name,
    c.channel_name,
    m.user_name,
    m.text,
    m.ai_response,
    m.created_at
FROM integrations.slack_messages m
JOIN integrations.slack_workspaces w ON m.workspace_id = w.workspace_id
JOIN integrations.slack_channels c ON m.channel_id = c.channel_id
WHERE m.created_at > NOW() - INTERVAL '24 hours'
ORDER BY m.created_at DESC;

-- Usage statistics view
CREATE OR REPLACE VIEW integrations.v_slack_usage_stats AS
SELECT
    w.workspace_id,
    w.team_name,
    COUNT(DISTINCT m.channel_id) as active_channels,
    COUNT(DISTINCT m.user_id) as active_users,
    COUNT(*) as total_messages,
    COUNT(m.ai_response) as ai_responses,
    SUM(m.ai_response_tokens) as total_tokens,
    AVG(m.processing_time_ms) as avg_processing_time,
    MAX(m.created_at) as last_activity
FROM integrations.slack_workspaces w
LEFT JOIN integrations.slack_messages m ON w.workspace_id = m.workspace_id
WHERE w.is_active = true
AND m.created_at > NOW() - INTERVAL '30 days'
GROUP BY w.workspace_id, w.team_name;

-- Grant permissions
GRANT USAGE ON SCHEMA integrations TO agentsystem_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA integrations TO agentsystem_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA integrations TO agentsystem_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA integrations TO agentsystem_app;

-- Comments for documentation
COMMENT ON SCHEMA integrations IS 'Schema for third-party integrations including Slack, Teams, etc.';
COMMENT ON TABLE integrations.slack_workspaces IS 'Slack workspace connections with OAuth tokens';
COMMENT ON TABLE integrations.slack_channels IS 'Channel-specific configuration and preferences';
COMMENT ON TABLE integrations.slack_messages IS 'Message log for analytics and AI training';
COMMENT ON TABLE integrations.slack_commands IS 'Slash command usage tracking';
COMMENT ON TABLE integrations.slack_notifications IS 'Notification queue and delivery tracking';
COMMENT ON TABLE integrations.slack_user_preferences IS 'Per-user settings and preferences';
COMMENT ON TABLE integrations.slack_workflows IS 'Automated workflow configurations';
COMMENT ON TABLE integrations.slack_analytics_daily IS 'Daily aggregated usage statistics';

-- Sample data for testing (optional)
/*
INSERT INTO integrations.slack_workspaces (
    workspace_id, tenant_id, team_id, team_name, bot_token, bot_user_id, access_token, scope
) VALUES (
    gen_random_uuid(),
    (SELECT id FROM tenant_management.tenants LIMIT 1),
    'T1234567890',
    'Test Workspace',
    'xoxb-test-token',
    'U1234567890',
    'xoxp-test-access-token',
    'app_mentions:read,channels:history,chat:write'
);
*/
