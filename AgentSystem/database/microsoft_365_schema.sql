-- Microsoft 365 Integration Database Schema
-- AgentSystem Profit Machine - Teams, Outlook, SharePoint Integration

-- Microsoft 365 tenants table
CREATE TABLE IF NOT EXISTS integrations.microsoft365_tenants (
    tenant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agentsystem_tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    tenant_name VARCHAR(500) NOT NULL,
    directory_id VARCHAR(100) NOT NULL UNIQUE,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    token_expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    scope TEXT,
    admin_consent BOOLEAN DEFAULT FALSE,
    enabled_services TEXT[] DEFAULT ARRAY[]::TEXT[],
    webhook_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    installed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_agentsystem_directory UNIQUE(agentsystem_tenant_id, directory_id)
);

-- Teams channels configuration
CREATE TABLE IF NOT EXISTS integrations.microsoft365_teams_channels (
    channel_id VARCHAR(100) PRIMARY KEY,
    tenant_id UUID REFERENCES integrations.microsoft365_tenants(tenant_id) ON DELETE CASCADE,
    team_id VARCHAR(100) NOT NULL,
    channel_name VARCHAR(500) NOT NULL,
    channel_type VARCHAR(50) DEFAULT 'standard',
    ai_assistance_enabled BOOLEAN DEFAULT TRUE,
    notification_types TEXT[] DEFAULT ARRAY[]::TEXT[],
    auto_respond BOOLEAN DEFAULT FALSE,
    response_delay_seconds INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Teams messages log
CREATE TABLE IF NOT EXISTS integrations.microsoft365_teams_messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES integrations.microsoft365_tenants(tenant_id) ON DELETE CASCADE,
    team_id VARCHAR(100) NOT NULL,
    channel_id VARCHAR(100) REFERENCES integrations.microsoft365_teams_channels(channel_id) ON DELETE CASCADE,
    user_id VARCHAR(100) NOT NULL,
    user_name VARCHAR(500),
    message_text TEXT NOT NULL,
    message_type VARCHAR(50) DEFAULT 'message',
    mention_bot BOOLEAN DEFAULT FALSE,
    ai_response TEXT,
    ai_response_tokens INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    sentiment_score DECIMAL(3,2), -- -1.0 to 1.0
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes for performance
    INDEX idx_teams_messages_tenant_time (tenant_id, created_at DESC),
    INDEX idx_teams_messages_channel_time (channel_id, created_at DESC),
    INDEX idx_teams_messages_user (user_id, created_at DESC)
);

-- Outlook emails processing
CREATE TABLE IF NOT EXISTS integrations.microsoft365_outlook_emails (
    email_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES integrations.microsoft365_tenants(tenant_id) ON DELETE CASCADE,
    outlook_message_id VARCHAR(200) NOT NULL UNIQUE,
    mailbox_id VARCHAR(100) NOT NULL,
    subject VARCHAR(1000),
    sender_email VARCHAR(500) NOT NULL,
    sender_name VARCHAR(500),
    recipients TEXT[] NOT NULL,
    cc_recipients TEXT[],
    bcc_recipients TEXT[],
    body_preview TEXT,
    body_content TEXT,
    importance VARCHAR(20) DEFAULT 'normal', -- low, normal, high
    is_read BOOLEAN DEFAULT FALSE,
    has_attachments BOOLEAN DEFAULT FALSE,
    received_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ai_processed BOOLEAN DEFAULT FALSE,
    ai_summary TEXT,
    ai_priority VARCHAR(20), -- low, medium, high, urgent
    ai_category VARCHAR(100),
    ai_action_items TEXT[],
    ai_sentiment DECIMAL(3,2),
    auto_replied BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes for performance
    INDEX idx_outlook_emails_tenant_time (tenant_id, received_at DESC),
    INDEX idx_outlook_emails_sender (sender_email, received_at DESC),
    INDEX idx_outlook_emails_unprocessed (tenant_id, ai_processed, received_at) WHERE ai_processed = FALSE
);

-- Outlook calendar events
CREATE TABLE IF NOT EXISTS integrations.microsoft365_calendar_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES integrations.microsoft365_tenants(tenant_id) ON DELETE CASCADE,
    outlook_event_id VARCHAR(200) NOT NULL UNIQUE,
    calendar_id VARCHAR(100) NOT NULL,
    subject VARCHAR(1000) NOT NULL,
    organizer_email VARCHAR(500) NOT NULL,
    organizer_name VARCHAR(500),
    attendees JSONB DEFAULT '[]',
    location_name VARCHAR(500),
    location_address TEXT,
    body_content TEXT,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    is_all_day BOOLEAN DEFAULT FALSE,
    is_recurring BOOLEAN DEFAULT FALSE,
    recurrence_pattern JSONB,
    importance VARCHAR(20) DEFAULT 'normal',
    sensitivity VARCHAR(20) DEFAULT 'normal', -- normal, personal, private, confidential
    show_as VARCHAR(20) DEFAULT 'busy', -- free, tentative, busy, oof, workingElsewhere
    response_status VARCHAR(20) DEFAULT 'none', -- none, organizer, tentativelyAccepted, accepted, declined
    reminder_minutes INTEGER,
    ai_processed BOOLEAN DEFAULT FALSE,
    ai_preparation_notes TEXT,
    ai_follow_up_actions TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes for performance
    INDEX idx_calendar_events_tenant_time (tenant_id, start_time),
    INDEX idx_calendar_events_organizer (organizer_email, start_time),
    INDEX idx_calendar_events_upcoming (tenant_id, start_time) WHERE start_time > NOW()
);

-- SharePoint documents
CREATE TABLE IF NOT EXISTS integrations.microsoft365_sharepoint_documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES integrations.microsoft365_tenants(tenant_id) ON DELETE CASCADE,
    sharepoint_item_id VARCHAR(200) NOT NULL,
    site_id VARCHAR(100) NOT NULL,
    site_name VARCHAR(500),
    document_name VARCHAR(1000) NOT NULL,
    document_path TEXT NOT NULL,
    file_extension VARCHAR(20),
    content_type VARCHAR(200),
    size_bytes BIGINT DEFAULT 0,
    author_email VARCHAR(500),
    author_name VARCHAR(500),
    last_modified_by_email VARCHAR(500),
    last_modified_by_name VARCHAR(500),
    version_number VARCHAR(50),
    download_url TEXT,
    web_url TEXT,
    ai_processed BOOLEAN DEFAULT FALSE,
    ai_summary TEXT,
    ai_tags TEXT[],
    ai_category VARCHAR(100),
    ai_content_type VARCHAR(100), -- document, spreadsheet, presentation, image, video, etc.
    ai_language VARCHAR(10),
    ai_key_topics TEXT[],
    text_content TEXT, -- Extracted text for AI processing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_sharepoint_site_item UNIQUE(site_id, sharepoint_item_id),

    -- Indexes for performance
    INDEX idx_sharepoint_docs_tenant_time (tenant_id, modified_at DESC),
    INDEX idx_sharepoint_docs_site (site_id, modified_at DESC),
    INDEX idx_sharepoint_docs_author (author_email, modified_at DESC),
    INDEX idx_sharepoint_docs_unprocessed (tenant_id, ai_processed, modified_at) WHERE ai_processed = FALSE
);

-- OneDrive files (similar to SharePoint but for personal storage)
CREATE TABLE IF NOT EXISTS integrations.microsoft365_onedrive_files (
    file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES integrations.microsoft365_tenants(tenant_id) ON DELETE CASCADE,
    onedrive_item_id VARCHAR(200) NOT NULL UNIQUE,
    drive_id VARCHAR(100) NOT NULL,
    owner_email VARCHAR(500) NOT NULL,
    owner_name VARCHAR(500),
    file_name VARCHAR(1000) NOT NULL,
    file_path TEXT NOT NULL,
    file_extension VARCHAR(20),
    mime_type VARCHAR(200),
    size_bytes BIGINT DEFAULT 0,
    parent_folder_id VARCHAR(200),
    parent_folder_path TEXT,
    download_url TEXT,
    web_url TEXT,
    shared BOOLEAN DEFAULT FALSE,
    sharing_permissions JSONB DEFAULT '[]',
    ai_processed BOOLEAN DEFAULT FALSE,
    ai_summary TEXT,
    ai_tags TEXT[],
    ai_category VARCHAR(100),
    text_content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes for performance
    INDEX idx_onedrive_files_tenant_time (tenant_id, modified_at DESC),
    INDEX idx_onedrive_files_owner (owner_email, modified_at DESC),
    INDEX idx_onedrive_files_unprocessed (tenant_id, ai_processed, modified_at) WHERE ai_processed = FALSE
);

-- Microsoft 365 webhooks subscriptions
CREATE TABLE IF NOT EXISTS integrations.microsoft365_webhooks (
    webhook_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES integrations.microsoft365_tenants(tenant_id) ON DELETE CASCADE,
    subscription_id VARCHAR(200) NOT NULL UNIQUE,
    resource_type VARCHAR(100) NOT NULL, -- messages, events, driveItems, etc.
    resource_path TEXT NOT NULL,
    change_types TEXT[] NOT NULL, -- created, updated, deleted
    notification_url TEXT NOT NULL,
    client_state VARCHAR(500),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_notification_at TIMESTAMP WITH TIME ZONE,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Index for monitoring expiring webhooks
    INDEX idx_webhooks_expiring (expires_at) WHERE is_active = TRUE
);

-- Microsoft 365 automation workflows
CREATE TABLE IF NOT EXISTS integrations.microsoft365_workflows (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES integrations.microsoft365_tenants(tenant_id) ON DELETE CASCADE,
    workflow_name VARCHAR(500) NOT NULL,
    service_type VARCHAR(50) NOT NULL, -- teams, outlook, sharepoint, onedrive
    trigger_type VARCHAR(100) NOT NULL, -- new_email, new_message, file_upload, meeting_start
    trigger_config JSONB NOT NULL,
    action_type VARCHAR(100) NOT NULL, -- ai_response, send_email, create_task, upload_file
    action_config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    execution_count INTEGER DEFAULT 0,
    last_executed_at TIMESTAMP WITH TIME ZONE,
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Microsoft 365 analytics aggregations
CREATE TABLE IF NOT EXISTS integrations.microsoft365_analytics_daily (
    date DATE NOT NULL,
    tenant_id UUID REFERENCES integrations.microsoft365_tenants(tenant_id) ON DELETE CASCADE,
    service_type VARCHAR(50) NOT NULL, -- teams, outlook, sharepoint, onedrive
    total_events INTEGER DEFAULT 0,
    ai_processed_events INTEGER DEFAULT 0,
    messages_sent INTEGER DEFAULT 0,
    emails_processed INTEGER DEFAULT 0,
    files_processed INTEGER DEFAULT 0,
    meetings_created INTEGER DEFAULT 0,
    active_users INTEGER DEFAULT 0,
    total_tokens_used INTEGER DEFAULT 0,
    avg_processing_time_ms DECIMAL(10,2),
    avg_sentiment_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Primary key and constraints
    PRIMARY KEY (date, tenant_id, service_type),
    INDEX idx_m365_analytics_tenant_date (tenant_id, date DESC),
    INDEX idx_m365_analytics_service_date (service_type, date DESC)
);

-- Microsoft 365 user preferences
CREATE TABLE IF NOT EXISTS integrations.microsoft365_user_preferences (
    user_id VARCHAR(100) NOT NULL,
    tenant_id UUID REFERENCES integrations.microsoft365_tenants(tenant_id) ON DELETE CASCADE,
    user_email VARCHAR(500) NOT NULL,
    user_name VARCHAR(500),
    timezone VARCHAR(100) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    ai_assistance_enabled BOOLEAN DEFAULT TRUE,
    auto_respond_enabled BOOLEAN DEFAULT FALSE,
    notification_preferences JSONB DEFAULT '{}',
    email_ai_processing BOOLEAN DEFAULT TRUE,
    calendar_ai_assistance BOOLEAN DEFAULT TRUE,
    document_ai_processing BOOLEAN DEFAULT TRUE,
    daily_summary_enabled BOOLEAN DEFAULT TRUE,
    weekly_report_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Primary key
    PRIMARY KEY (user_id, tenant_id),

    -- Unique constraint for email per tenant
    CONSTRAINT unique_email_tenant UNIQUE(user_email, tenant_id)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_m365_tenants_agentsystem
ON integrations.microsoft365_tenants(agentsystem_tenant_id, is_active);

CREATE INDEX IF NOT EXISTS idx_m365_tenants_directory
ON integrations.microsoft365_tenants(directory_id) WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_teams_channels_tenant
ON integrations.microsoft365_teams_channels(tenant_id, ai_assistance_enabled);

CREATE INDEX IF NOT EXISTS idx_outlook_emails_priority
ON integrations.microsoft365_outlook_emails(tenant_id, ai_priority, received_at DESC)
WHERE ai_processed = true AND ai_priority IN ('high', 'urgent');

CREATE INDEX IF NOT EXISTS idx_calendar_events_today
ON integrations.microsoft365_calendar_events(tenant_id, start_time)
WHERE DATE(start_time) = CURRENT_DATE;

CREATE INDEX IF NOT EXISTS idx_sharepoint_docs_recent
ON integrations.microsoft365_sharepoint_documents(tenant_id, modified_at DESC)
WHERE modified_at > NOW() - INTERVAL '7 days';

-- Functions for analytics and automation

-- Function to update tenant activity
CREATE OR REPLACE FUNCTION integrations.update_m365_tenant_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE integrations.microsoft365_tenants
    SET updated_at = NOW()
    WHERE tenant_id = NEW.tenant_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for tenant activity tracking
DROP TRIGGER IF EXISTS trigger_m365_teams_activity ON integrations.microsoft365_teams_messages;
CREATE TRIGGER trigger_m365_teams_activity
    AFTER INSERT ON integrations.microsoft365_teams_messages
    FOR EACH ROW
    EXECUTE FUNCTION integrations.update_m365_tenant_activity();

DROP TRIGGER IF EXISTS trigger_m365_outlook_activity ON integrations.microsoft365_outlook_emails;
CREATE TRIGGER trigger_m365_outlook_activity
    AFTER INSERT ON integrations.microsoft365_outlook_emails
    FOR EACH ROW
    EXECUTE FUNCTION integrations.update_m365_tenant_activity();

-- Function to calculate daily Microsoft 365 analytics
CREATE OR REPLACE FUNCTION integrations.calculate_daily_m365_analytics(target_date DATE DEFAULT CURRENT_DATE)
RETURNS VOID AS $$
BEGIN
    -- Teams analytics
    INSERT INTO integrations.microsoft365_analytics_daily (
        date, tenant_id, service_type, total_events, ai_processed_events,
        messages_sent, active_users, total_tokens_used, avg_processing_time_ms, avg_sentiment_score
    )
    SELECT
        target_date,
        tenant_id,
        'teams',
        COUNT(*) as total_events,
        COUNT(ai_response) as ai_processed_events,
        COUNT(*) as messages_sent,
        COUNT(DISTINCT user_id) as active_users,
        COALESCE(SUM(ai_response_tokens), 0) as total_tokens_used,
        AVG(processing_time_ms) as avg_processing_time_ms,
        AVG(sentiment_score) as avg_sentiment_score
    FROM integrations.microsoft365_teams_messages
    WHERE DATE(created_at) = target_date
    GROUP BY tenant_id
    ON CONFLICT (date, tenant_id, service_type)
    DO UPDATE SET
        total_events = EXCLUDED.total_events,
        ai_processed_events = EXCLUDED.ai_processed_events,
        messages_sent = EXCLUDED.messages_sent,
        active_users = EXCLUDED.active_users,
        total_tokens_used = EXCLUDED.total_tokens_used,
        avg_processing_time_ms = EXCLUDED.avg_processing_time_ms,
        avg_sentiment_score = EXCLUDED.avg_sentiment_score;

    -- Outlook analytics
    INSERT INTO integrations.microsoft365_analytics_daily (
        date, tenant_id, service_type, total_events, ai_processed_events, emails_processed
    )
    SELECT
        target_date,
        tenant_id,
        'outlook',
        COUNT(*) as total_events,
        COUNT(*) FILTER (WHERE ai_processed = true) as ai_processed_events,
        COUNT(*) as emails_processed
    FROM integrations.microsoft365_outlook_emails
    WHERE DATE(received_at) = target_date
    GROUP BY tenant_id
    ON CONFLICT (date, tenant_id, service_type)
    DO UPDATE SET
        total_events = EXCLUDED.total_events,
        ai_processed_events = EXCLUDED.ai_processed_events,
        emails_processed = EXCLUDED.emails_processed;

    -- SharePoint analytics
    INSERT INTO integrations.microsoft365_analytics_daily (
        date, tenant_id, service_type, total_events, ai_processed_events, files_processed
    )
    SELECT
        target_date,
        tenant_id,
        'sharepoint',
        COUNT(*) as total_events,
        COUNT(*) FILTER (WHERE ai_processed = true) as ai_processed_events,
        COUNT(*) as files_processed
    FROM integrations.microsoft365_sharepoint_documents
    WHERE DATE(modified_at) = target_date
    GROUP BY tenant_id
    ON CONFLICT (date, tenant_id, service_type)
    DO UPDATE SET
        total_events = EXCLUDED.total_events,
        ai_processed_events = EXCLUDED.ai_processed_events,
        files_processed = EXCLUDED.files_processed;
END;
$$ LANGUAGE plpgsql;

-- Function to clean old Microsoft 365 data
CREATE OR REPLACE FUNCTION integrations.cleanup_m365_data(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    temp_count INTEGER;
BEGIN
    -- Delete old Teams messages (keep AI responses longer)
    DELETE FROM integrations.microsoft365_teams_messages
    WHERE created_at < NOW() - INTERVAL '1 day' * retention_days
    AND ai_response IS NULL;

    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;

    -- Delete old emails (keep important ones longer)
    DELETE FROM integrations.microsoft365_outlook_emails
    WHERE created_at < NOW() - INTERVAL '1 day' * retention_days
    AND ai_priority NOT IN ('high', 'urgent');

    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;

    -- Delete old calendar events
    DELETE FROM integrations.microsoft365_calendar_events
    WHERE end_time < NOW() - INTERVAL '1 day' * (retention_days / 2);

    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Views for common queries

-- Active Microsoft 365 tenants view
CREATE OR REPLACE VIEW integrations.v_active_m365_tenants AS
SELECT
    t.*,
    ag.name as agentsystem_tenant_name,
    ag.plan_type,
    COUNT(tc.channel_id) as teams_channels_count,
    COUNT(DISTINCT tm.user_id) as active_teams_users,
    COUNT(oe.email_id) as total_emails_processed,
    COUNT(sd.document_id) as total_documents_processed
FROM integrations.microsoft365_tenants t
JOIN tenant_management.tenants ag ON t.agentsystem_tenant_id = ag.id
LEFT JOIN integrations.microsoft365_teams_channels tc ON t.tenant_id = tc.tenant_id
LEFT JOIN integrations.microsoft365_teams_messages tm ON t.tenant_id = tm.tenant_id
    AND tm.created_at > NOW() - INTERVAL '30 days'
LEFT JOIN integrations.microsoft365_outlook_emails oe ON t.tenant_id = oe.tenant_id
    AND oe.received_at > NOW() - INTERVAL '30 days'
LEFT JOIN integrations.microsoft365_sharepoint_documents sd ON t.tenant_id = sd.tenant_id
    AND sd.modified_at > NOW() - INTERVAL '30 days'
WHERE t.is_active = true
GROUP BY t.tenant_id, ag.name, ag.plan_type;

-- Recent Microsoft 365 activity view
CREATE OR REPLACE VIEW integrations.v_m365_recent_activity AS
SELECT
    'teams' as service_type,
    t.tenant_name,
    tc.channel_name as resource_name,
    tm.user_name as user_name,
    tm.message_text as content,
    tm.ai_response,
    tm.created_at
FROM integrations.microsoft365_teams_messages tm
JOIN integrations.microsoft365_tenants t ON tm.tenant_id = t.tenant_id
JOIN integrations.microsoft365_teams_channels tc ON tm.channel_id = tc.channel_id
WHERE tm.created_at > NOW() - INTERVAL '24 hours'

UNION ALL

SELECT
    'outlook' as service_type,
    t.tenant_name,
    oe.subject as resource_name,
    oe.sender_name as user_name,
    oe.body_preview as content,
    oe.ai_summary as ai_response,
    oe.received_at as created_at
FROM integrations.microsoft365_outlook_emails oe
JOIN integrations.microsoft365_tenants t ON oe.tenant_id = t.tenant_id
WHERE oe.received_at > NOW() - INTERVAL '24 hours'

ORDER BY created_at DESC;

-- Grant permissions
GRANT USAGE ON SCHEMA integrations TO agentsystem_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA integrations TO agentsystem_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA integrations TO agentsystem_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA integrations TO agentsystem_app;

-- Comments for documentation
COMMENT ON TABLE integrations.microsoft365_tenants IS 'Microsoft 365 tenant connections with OAuth tokens';
COMMENT ON TABLE integrations.microsoft365_teams_channels IS 'Teams channel configuration and preferences';
COMMENT ON TABLE integrations.microsoft365_teams_messages IS 'Teams message log for AI processing and analytics';
COMMENT ON TABLE integrations.microsoft365_outlook_emails IS 'Outlook email processing and AI analysis';
COMMENT ON TABLE integrations.microsoft365_calendar_events IS 'Calendar events with AI assistance';
COMMENT ON TABLE integrations.microsoft365_sharepoint_documents IS 'SharePoint document processing and analysis';
COMMENT ON TABLE integrations.microsoft365_onedrive_files IS 'OneDrive file processing and analysis';
COMMENT ON TABLE integrations.microsoft365_webhooks IS 'Microsoft 365 webhook subscriptions';
COMMENT ON TABLE integrations.microsoft365_workflows IS 'Automated workflow configurations';
COMMENT ON TABLE integrations.microsoft365_analytics_daily IS 'Daily aggregated usage statistics';
COMMENT ON TABLE integrations.microsoft365_user_preferences IS 'Per-user settings and preferences';
