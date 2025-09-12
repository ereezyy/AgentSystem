-- Workflow Builder Database Schema
-- AgentSystem Profit Machine - Zapier-style Visual Workflow Automation

-- Create workflows schema
CREATE SCHEMA IF NOT EXISTS workflows;

-- Workflow folders for organization
CREATE TABLE IF NOT EXISTS workflows.workflow_folders (
    folder_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    parent_folder_id UUID REFERENCES workflows.workflow_folders(folder_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    color VARCHAR(7) DEFAULT '#6366f1', -- Hex color
    icon VARCHAR(50) DEFAULT 'folder',
    is_system BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_folder_name_per_tenant UNIQUE(tenant_id, parent_folder_id, name)
);

-- Main workflows table
CREATE TABLE IF NOT EXISTS workflows.workflows (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    folder_id UUID REFERENCES workflows.workflow_folders(folder_id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'draft', -- draft, active, paused, disabled, error
    nodes JSONB NOT NULL DEFAULT '[]',
    connections JSONB NOT NULL DEFAULT '[]',
    variables JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    is_template BOOLEAN DEFAULT FALSE,
    template_category VARCHAR(100),
    execution_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    last_executed_at TIMESTAMP WITH TIME ZONE,
    last_error TEXT,
    avg_execution_time_ms INTEGER,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes
    INDEX idx_workflows_tenant_status (tenant_id, status),
    INDEX idx_workflows_folder (folder_id),
    INDEX idx_workflows_template (is_template, template_category),
    INDEX idx_workflows_tags USING GIN(tags),
    INDEX idx_workflows_updated (updated_at DESC)
);

-- Workflow executions table
CREATE TABLE IF NOT EXISTS workflows.workflow_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows.workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL, -- running, completed, failed, cancelled, timeout
    trigger_type VARCHAR(50),
    trigger_data JSONB NOT NULL DEFAULT '{}',
    execution_context JSONB DEFAULT '{}',
    current_node_id VARCHAR(100),
    nodes_executed TEXT[] DEFAULT ARRAY[]::TEXT[],
    error_message TEXT,
    error_node_id VARCHAR(100),
    execution_time_ms INTEGER,
    memory_usage_mb DECIMAL(10,2),
    tokens_used INTEGER DEFAULT 0,
    api_calls_made INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Indexes for performance
    INDEX idx_executions_workflow_time (workflow_id, started_at DESC),
    INDEX idx_executions_tenant_status (tenant_id, status, started_at DESC),
    INDEX idx_executions_status_time (status, started_at DESC)
);

-- Workflow execution steps (detailed logging)
CREATE TABLE IF NOT EXISTS workflows.workflow_execution_steps (
    step_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID REFERENCES workflows.workflow_executions(execution_id) ON DELETE CASCADE,
    node_id VARCHAR(100) NOT NULL,
    node_name VARCHAR(255),
    node_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL, -- started, completed, failed, skipped
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    execution_time_ms INTEGER,
    tokens_used INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Index for step tracking
    INDEX idx_execution_steps_execution (execution_id, started_at),
    INDEX idx_execution_steps_node (node_id, started_at DESC)
);

-- Workflow schedules table
CREATE TABLE IF NOT EXISTS workflows.workflow_schedules (
    schedule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows.workflows(workflow_id) ON DELETE CASCADE,
    trigger_node_id VARCHAR(100) NOT NULL,
    schedule_config JSONB NOT NULL,
    timezone VARCHAR(50) DEFAULT 'UTC',
    is_active BOOLEAN DEFAULT TRUE,
    next_run_at TIMESTAMP WITH TIME ZONE,
    last_run_at TIMESTAMP WITH TIME ZONE,
    run_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Unique constraint
    CONSTRAINT unique_workflow_trigger_schedule UNIQUE(workflow_id, trigger_node_id),

    -- Index for scheduler
    INDEX idx_schedules_next_run (is_active, next_run_at) WHERE is_active = TRUE
);

-- Workflow webhooks table
CREATE TABLE IF NOT EXISTS workflows.workflow_webhooks (
    webhook_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows.workflows(workflow_id) ON DELETE CASCADE,
    trigger_node_id VARCHAR(100) NOT NULL,
    webhook_url TEXT NOT NULL,
    webhook_token VARCHAR(255) NOT NULL,
    webhook_config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    request_count INTEGER DEFAULT 0,
    last_request_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Unique constraint
    CONSTRAINT unique_workflow_trigger_webhook UNIQUE(workflow_id, trigger_node_id),

    -- Index for webhook lookup
    INDEX idx_webhooks_token (webhook_token) WHERE is_active = TRUE
);

-- Workflow templates marketplace
CREATE TABLE IF NOT EXISTS workflows.workflow_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    difficulty_level VARCHAR(20) DEFAULT 'beginner', -- beginner, intermediate, advanced
    estimated_setup_time_minutes INTEGER,
    workflow_definition JSONB NOT NULL,
    preview_image_url TEXT,
    documentation_url TEXT,
    author_name VARCHAR(255),
    author_email VARCHAR(255),
    version VARCHAR(20) DEFAULT '1.0.0',
    downloads_count INTEGER DEFAULT 0,
    rating_average DECIMAL(3,2) DEFAULT 0.0,
    rating_count INTEGER DEFAULT 0,
    is_featured BOOLEAN DEFAULT FALSE,
    is_approved BOOLEAN DEFAULT FALSE,
    required_integrations TEXT[] DEFAULT ARRAY[]::TEXT[],
    pricing_model VARCHAR(20) DEFAULT 'free', -- free, paid, freemium
    price_usd DECIMAL(10,2) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes for marketplace
    INDEX idx_templates_category (category, is_approved),
    INDEX idx_templates_featured (is_featured, is_approved, rating_average DESC),
    INDEX idx_templates_popular (downloads_count DESC, rating_average DESC),
    INDEX idx_templates_tags USING GIN(tags),
    INDEX idx_templates_integrations USING GIN(required_integrations)
);

-- Workflow template ratings
CREATE TABLE IF NOT EXISTS workflows.workflow_template_ratings (
    rating_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id UUID REFERENCES workflows.workflow_templates(template_id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review TEXT,
    is_helpful BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Unique constraint
    CONSTRAINT unique_template_tenant_rating UNIQUE(template_id, tenant_id)
);

-- Workflow analytics aggregations
CREATE TABLE IF NOT EXISTS workflows.workflow_analytics_daily (
    date DATE NOT NULL,
    workflow_id UUID REFERENCES workflows.workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    executions_total INTEGER DEFAULT 0,
    executions_successful INTEGER DEFAULT 0,
    executions_failed INTEGER DEFAULT 0,
    avg_execution_time_ms INTEGER DEFAULT 0,
    total_tokens_used INTEGER DEFAULT 0,
    total_api_calls INTEGER DEFAULT 0,
    unique_triggers INTEGER DEFAULT 0,
    data_processed_mb DECIMAL(12,2) DEFAULT 0.0,
    cost_usd DECIMAL(10,4) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Primary key
    PRIMARY KEY (date, workflow_id),

    -- Indexes
    INDEX idx_workflow_analytics_tenant_date (tenant_id, date DESC),
    INDEX idx_workflow_analytics_workflow_date (workflow_id, date DESC)
);

-- Workflow sharing and collaboration
CREATE TABLE IF NOT EXISTS workflows.workflow_shares (
    share_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows.workflows(workflow_id) ON DELETE CASCADE,
    shared_by_tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    shared_with_tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    permission_level VARCHAR(20) DEFAULT 'view', -- view, execute, edit, admin
    share_token VARCHAR(255) UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT no_self_share CHECK (shared_by_tenant_id != shared_with_tenant_id),

    -- Indexes
    INDEX idx_workflow_shares_token (share_token) WHERE is_active = TRUE,
    INDEX idx_workflow_shares_shared_with (shared_with_tenant_id, is_active)
);

-- Workflow comments and collaboration
CREATE TABLE IF NOT EXISTS workflows.workflow_comments (
    comment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows.workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    parent_comment_id UUID REFERENCES workflows.workflow_comments(comment_id) ON DELETE CASCADE,
    node_id VARCHAR(100), -- Optional: comment on specific node
    content TEXT NOT NULL,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes
    INDEX idx_workflow_comments_workflow (workflow_id, created_at DESC),
    INDEX idx_workflow_comments_node (node_id, is_resolved)
);

-- Workflow versions and history
CREATE TABLE IF NOT EXISTS workflows.workflow_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows.workflows(workflow_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    version_name VARCHAR(255),
    changelog TEXT,
    workflow_definition JSONB NOT NULL,
    is_current BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_workflow_version UNIQUE(workflow_id, version_number),

    -- Index
    INDEX idx_workflow_versions_workflow (workflow_id, version_number DESC)
);

-- Functions for workflow analytics and automation

-- Function to update workflow statistics
CREATE OR REPLACE FUNCTION workflows.update_workflow_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Update execution count
        UPDATE workflows.workflows
        SET execution_count = execution_count + 1,
            last_executed_at = NEW.started_at
        WHERE workflow_id = NEW.workflow_id;

    ELSIF TG_OP = 'UPDATE' AND OLD.status != NEW.status THEN
        -- Update success/error counts
        IF NEW.status = 'completed' THEN
            UPDATE workflows.workflows
            SET success_count = success_count + 1,
                avg_execution_time_ms = (
                    COALESCE(avg_execution_time_ms * (execution_count - 1), 0) + COALESCE(NEW.execution_time_ms, 0)
                ) / execution_count
            WHERE workflow_id = NEW.workflow_id;

        ELSIF NEW.status = 'failed' THEN
            UPDATE workflows.workflows
            SET error_count = error_count + 1,
                last_error = NEW.error_message
            WHERE workflow_id = NEW.workflow_id;
        END IF;
    END IF;

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Triggers for workflow statistics
DROP TRIGGER IF EXISTS trigger_workflow_execution_stats ON workflows.workflow_executions;
CREATE TRIGGER trigger_workflow_execution_stats
    AFTER INSERT OR UPDATE ON workflows.workflow_executions
    FOR EACH ROW
    EXECUTE FUNCTION workflows.update_workflow_stats();

-- Function to calculate daily workflow analytics
CREATE OR REPLACE FUNCTION workflows.calculate_daily_workflow_analytics(target_date DATE DEFAULT CURRENT_DATE)
RETURNS VOID AS $$
BEGIN
    INSERT INTO workflows.workflow_analytics_daily (
        date, workflow_id, tenant_id, executions_total, executions_successful,
        executions_failed, avg_execution_time_ms, total_tokens_used, total_api_calls,
        unique_triggers, cost_usd
    )
    SELECT
        target_date,
        e.workflow_id,
        e.tenant_id,
        COUNT(*) as executions_total,
        COUNT(*) FILTER (WHERE e.status = 'completed') as executions_successful,
        COUNT(*) FILTER (WHERE e.status = 'failed') as executions_failed,
        AVG(e.execution_time_ms)::INTEGER as avg_execution_time_ms,
        COALESCE(SUM(e.tokens_used), 0) as total_tokens_used,
        COALESCE(SUM(e.api_calls_made), 0) as total_api_calls,
        COUNT(DISTINCT e.trigger_type) as unique_triggers,
        COALESCE(SUM(e.tokens_used * 0.002), 0.0) as cost_usd -- Estimated cost
    FROM workflows.workflow_executions e
    WHERE DATE(e.started_at) = target_date
    GROUP BY e.workflow_id, e.tenant_id
    ON CONFLICT (date, workflow_id)
    DO UPDATE SET
        executions_total = EXCLUDED.executions_total,
        executions_successful = EXCLUDED.executions_successful,
        executions_failed = EXCLUDED.executions_failed,
        avg_execution_time_ms = EXCLUDED.avg_execution_time_ms,
        total_tokens_used = EXCLUDED.total_tokens_used,
        total_api_calls = EXCLUDED.total_api_calls,
        unique_triggers = EXCLUDED.unique_triggers,
        cost_usd = EXCLUDED.cost_usd;
END;
$$ LANGUAGE plpgsql;

-- Function to clean old workflow data
CREATE OR REPLACE FUNCTION workflows.cleanup_workflow_data(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    temp_count INTEGER;
BEGIN
    -- Delete old executions (keep successful ones longer)
    DELETE FROM workflows.workflow_executions
    WHERE started_at < NOW() - INTERVAL '1 day' * retention_days
    AND status = 'failed';

    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;

    -- Delete very old successful executions
    DELETE FROM workflows.workflow_executions
    WHERE started_at < NOW() - INTERVAL '1 day' * (retention_days * 2)
    AND status = 'completed';

    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;

    -- Delete old execution steps
    DELETE FROM workflows.workflow_execution_steps
    WHERE started_at < NOW() - INTERVAL '1 day' * retention_days;

    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Views for common queries

-- Active workflows view
CREATE OR REPLACE VIEW workflows.v_active_workflows AS
SELECT
    w.*,
    t.name as tenant_name,
    t.plan_type,
    f.name as folder_name,
    CASE
        WHEN w.execution_count > 0 THEN (w.success_count::DECIMAL / w.execution_count * 100)
        ELSE 0
    END as success_rate_percent
FROM workflows.workflows w
JOIN tenant_management.tenants t ON w.tenant_id = t.id
LEFT JOIN workflows.workflow_folders f ON w.folder_id = f.folder_id
WHERE w.status = 'active';

-- Workflow execution summary view
CREATE OR REPLACE VIEW workflows.v_workflow_execution_summary AS
SELECT
    w.workflow_id,
    w.name as workflow_name,
    w.tenant_id,
    COUNT(e.execution_id) as total_executions,
    COUNT(e.execution_id) FILTER (WHERE e.status = 'completed') as successful_executions,
    COUNT(e.execution_id) FILTER (WHERE e.status = 'failed') as failed_executions,
    COUNT(e.execution_id) FILTER (WHERE e.started_at > NOW() - INTERVAL '24 hours') as executions_last_24h,
    AVG(e.execution_time_ms) as avg_execution_time_ms,
    MAX(e.started_at) as last_execution_at
FROM workflows.workflows w
LEFT JOIN workflows.workflow_executions e ON w.workflow_id = e.workflow_id
GROUP BY w.workflow_id, w.name, w.tenant_id;

-- Popular workflow templates view
CREATE OR REPLACE VIEW workflows.v_popular_templates AS
SELECT
    t.*,
    COALESCE(AVG(r.rating), 0) as avg_rating,
    COUNT(r.rating_id) as review_count
FROM workflows.workflow_templates t
LEFT JOIN workflows.workflow_template_ratings r ON t.template_id = r.template_id
WHERE t.is_approved = true
GROUP BY t.template_id
ORDER BY t.downloads_count DESC, avg_rating DESC;

-- Grant permissions
GRANT USAGE ON SCHEMA workflows TO agentsystem_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA workflows TO agentsystem_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA workflows TO agentsystem_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA workflows TO agentsystem_app;

-- Comments for documentation
COMMENT ON SCHEMA workflows IS 'Schema for visual workflow builder and automation';
COMMENT ON TABLE workflows.workflows IS 'Main workflows with visual node definitions';
COMMENT ON TABLE workflows.workflow_executions IS 'Workflow execution tracking and results';
COMMENT ON TABLE workflows.workflow_execution_steps IS 'Detailed step-by-step execution logging';
COMMENT ON TABLE workflows.workflow_schedules IS 'Scheduled workflow triggers';
COMMENT ON TABLE workflows.workflow_webhooks IS 'Webhook-based workflow triggers';
COMMENT ON TABLE workflows.workflow_templates IS 'Marketplace of reusable workflow templates';
COMMENT ON TABLE workflows.workflow_analytics_daily IS 'Daily aggregated workflow analytics';
COMMENT ON TABLE workflows.workflow_shares IS 'Workflow sharing and collaboration';
COMMENT ON TABLE workflows.workflow_versions IS 'Workflow version control and history';

-- Sample workflow templates
INSERT INTO workflows.workflow_templates (
    name, description, category, subcategory, tags, difficulty_level,
    workflow_definition, author_name, is_approved, required_integrations
) VALUES
(
    'Lead Qualification Assistant',
    'Automatically qualify incoming leads using AI analysis and route to appropriate sales team',
    'Sales & Marketing',
    'Lead Management',
    ARRAY['sales', 'lead-qualification', 'ai', 'automation'],
    'beginner',
    '{"nodes": [{"node_id": "trigger-1", "node_type": "trigger", "name": "New Lead Form", "config": {"trigger_type": "webhook"}}], "connections": []}',
    'AgentSystem Team',
    true,
    ARRAY['webhook', 'ai_service']
),
(
    'Customer Support Ticket Router',
    'Analyze support tickets and automatically route to the right department with priority scoring',
    'Customer Support',
    'Ticket Management',
    ARRAY['support', 'routing', 'ai', 'prioritization'],
    'intermediate',
    '{"nodes": [{"node_id": "trigger-1", "node_type": "trigger", "name": "New Ticket", "config": {"trigger_type": "email_received"}}], "connections": []}',
    'AgentSystem Team',
    true,
    ARRAY['email', 'ai_service', 'slack']
),
(
    'Document Processing Pipeline',
    'Extract data from uploaded documents, analyze content, and store structured information',
    'Document Management',
    'Data Extraction',
    ARRAY['documents', 'ocr', 'data-extraction', 'ai'],
    'advanced',
    '{"nodes": [{"node_id": "trigger-1", "node_type": "trigger", "name": "File Upload", "config": {"trigger_type": "file_upload"}}], "connections": []}',
    'AgentSystem Team',
    true,
    ARRAY['file_storage', 'ai_service', 'document_processing']
) ON CONFLICT DO NOTHING;
