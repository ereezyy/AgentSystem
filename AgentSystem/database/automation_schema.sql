-- Workflow Automation Platform Database Schema - AgentSystem Profit Machine
-- No-code workflow automation with visual builder and execution engine

-- Create automation schema
CREATE SCHEMA IF NOT EXISTS automation;

-- Workflow definitions table
CREATE TABLE automation.workflows (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(50) DEFAULT '1.0.0',
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    category VARCHAR(100) DEFAULT 'general',
    tags JSONB DEFAULT '[]',
    trigger_config JSONB NOT NULL,
    nodes_config JSONB NOT NULL,
    global_variables JSONB DEFAULT '{}',
    error_handling JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'paused', 'disabled', 'error')),
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    is_template BOOLEAN DEFAULT FALSE,
    template_source UUID REFERENCES automation.workflow_templates(template_id),
    execution_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2) DEFAULT 0.00,
    avg_duration_ms INTEGER DEFAULT 0,
    total_cost DECIMAL(12,4) DEFAULT 0.00
);

-- Workflow templates table
CREATE TABLE automation.workflow_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    tags JSONB DEFAULT '[]',
    use_cases JSONB DEFAULT '[]',
    complexity_level VARCHAR(50) DEFAULT 'beginner' CHECK (complexity_level IN ('beginner', 'intermediate', 'advanced', 'expert')),
    estimated_setup_time INTEGER DEFAULT 30, -- minutes
    template_config JSONB NOT NULL,
    preview_image_url VARCHAR(500),
    documentation_url VARCHAR(500),
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    is_public BOOLEAN DEFAULT TRUE,
    is_featured BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    avg_rating DECIMAL(3,2) DEFAULT 0.00,
    rating_count INTEGER DEFAULT 0,
    required_integrations JSONB DEFAULT '[]',
    supported_triggers JSONB DEFAULT '[]'
);

-- Workflow executions table
CREATE TABLE automation.workflow_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES automation.workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    trigger_data JSONB DEFAULT '{}',
    execution_context JSONB DEFAULT '{}',
    current_node VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout')),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    total_duration_ms INTEGER DEFAULT 0,
    nodes_executed INTEGER DEFAULT 0,
    nodes_successful INTEGER DEFAULT 0,
    nodes_failed INTEGER DEFAULT 0,
    total_cost DECIMAL(10,4) DEFAULT 0.00,
    error_details TEXT,
    output_data JSONB,
    triggered_by VARCHAR(100),
    execution_priority INTEGER DEFAULT 5,
    retry_count INTEGER DEFAULT 0,
    parent_execution_id UUID REFERENCES automation.workflow_executions(execution_id)
);

-- Node executions table
CREATE TABLE automation.node_executions (
    node_execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES automation.workflow_executions(execution_id) ON DELETE CASCADE,
    node_id VARCHAR(255) NOT NULL,
    node_type VARCHAR(100) NOT NULL,
    input_data JSONB DEFAULT '{}',
    output_data JSONB,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout')),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER DEFAULT 0,
    cost DECIMAL(8,4) DEFAULT 0.00,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    execution_order INTEGER,
    ai_provider VARCHAR(100),
    tokens_used INTEGER DEFAULT 0,
    cache_hit BOOLEAN DEFAULT FALSE
);

-- Workflow triggers table
CREATE TABLE automation.workflow_triggers (
    trigger_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES automation.workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    trigger_type VARCHAR(100) NOT NULL CHECK (trigger_type IN ('manual', 'schedule', 'webhook', 'event', 'email', 'api_call', 'file_upload', 'form_submission')),
    trigger_name VARCHAR(255) NOT NULL,
    configuration JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    webhook_url VARCHAR(500),
    webhook_secret VARCHAR(255),
    schedule_expression VARCHAR(255), -- cron expression
    event_filters JSONB DEFAULT '{}',
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_triggered TIMESTAMP WITH TIME ZONE,
    trigger_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0
);

-- Workflow variables table
CREATE TABLE automation.workflow_variables (
    variable_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES automation.workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    variable_name VARCHAR(255) NOT NULL,
    variable_type VARCHAR(50) DEFAULT 'string' CHECK (variable_type IN ('string', 'number', 'boolean', 'object', 'array', 'secret')),
    default_value JSONB,
    is_required BOOLEAN DEFAULT FALSE,
    is_secret BOOLEAN DEFAULT FALSE,
    description TEXT,
    validation_rules JSONB DEFAULT '{}',
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(workflow_id, variable_name)
);

-- Workflow connections table (for visual editor)
CREATE TABLE automation.workflow_connections (
    connection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES automation.workflows(workflow_id) ON DELETE CASCADE,
    source_node_id VARCHAR(255) NOT NULL,
    target_node_id VARCHAR(255) NOT NULL,
    connection_type VARCHAR(50) DEFAULT 'success' CHECK (connection_type IN ('success', 'error', 'condition_true', 'condition_false', 'always')),
    conditions JSONB DEFAULT '{}',
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Workflow schedules table
CREATE TABLE automation.workflow_schedules (
    schedule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES automation.workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    schedule_name VARCHAR(255) NOT NULL,
    cron_expression VARCHAR(255) NOT NULL,
    timezone VARCHAR(100) DEFAULT 'UTC',
    is_active BOOLEAN DEFAULT TRUE,
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    max_executions INTEGER,
    execution_count INTEGER DEFAULT 0,
    last_execution TIMESTAMP WITH TIME ZONE,
    next_execution TIMESTAMP WITH TIME ZONE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Workflow webhooks table
CREATE TABLE automation.workflow_webhooks (
    webhook_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES automation.workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    webhook_name VARCHAR(255) NOT NULL,
    webhook_url VARCHAR(500) UNIQUE NOT NULL,
    webhook_secret VARCHAR(255),
    allowed_methods JSONB DEFAULT '["POST"]',
    content_type VARCHAR(100) DEFAULT 'application/json',
    authentication JSONB DEFAULT '{}',
    rate_limit INTEGER DEFAULT 100, -- requests per minute
    is_active BOOLEAN DEFAULT TRUE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_triggered TIMESTAMP WITH TIME ZONE,
    trigger_count INTEGER DEFAULT 0
);

-- Workflow integrations table
CREATE TABLE automation.workflow_integrations (
    integration_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES automation.workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    integration_type VARCHAR(100) NOT NULL,
    integration_name VARCHAR(255) NOT NULL,
    configuration JSONB NOT NULL,
    credentials JSONB DEFAULT '{}', -- encrypted
    is_active BOOLEAN DEFAULT TRUE,
    last_sync TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Workflow analytics table
CREATE TABLE automation.workflow_analytics (
    analytics_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES automation.workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    date_period DATE NOT NULL,
    period_type VARCHAR(20) DEFAULT 'daily' CHECK (period_type IN ('hourly', 'daily', 'weekly', 'monthly')),
    executions_count INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    failed_executions INTEGER DEFAULT 0,
    avg_duration_ms INTEGER DEFAULT 0,
    total_cost DECIMAL(10,4) DEFAULT 0.00,
    total_tokens_used INTEGER DEFAULT 0,
    unique_triggers INTEGER DEFAULT 0,
    error_rate DECIMAL(5,2) DEFAULT 0.00,
    performance_score DECIMAL(5,2) DEFAULT 0.00,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(workflow_id, date_period, period_type)
);

-- Template ratings table
CREATE TABLE automation.template_ratings (
    rating_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id UUID NOT NULL REFERENCES automation.workflow_templates(template_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review TEXT,
    is_featured BOOLEAN DEFAULT FALSE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(template_id, tenant_id)
);

-- Workflow sharing table
CREATE TABLE automation.workflow_sharing (
    sharing_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES automation.workflows(workflow_id) ON DELETE CASCADE,
    owner_tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    shared_with_tenant_id UUID REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    sharing_type VARCHAR(50) DEFAULT 'private' CHECK (sharing_type IN ('private', 'public', 'organization', 'specific')),
    permissions JSONB DEFAULT '{"read": true, "execute": false, "modify": false}',
    expiry_date TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_workflows_tenant_id ON automation.workflows(tenant_id);
CREATE INDEX idx_workflows_status ON automation.workflows(status);
CREATE INDEX idx_workflows_category ON automation.workflows(category);
CREATE INDEX idx_workflows_created_date ON automation.workflows(created_date);

CREATE INDEX idx_workflow_executions_workflow_id ON automation.workflow_executions(workflow_id);
CREATE INDEX idx_workflow_executions_tenant_id ON automation.workflow_executions(tenant_id);
CREATE INDEX idx_workflow_executions_status ON automation.workflow_executions(status);
CREATE INDEX idx_workflow_executions_start_time ON automation.workflow_executions(start_time);

CREATE INDEX idx_node_executions_execution_id ON automation.node_executions(execution_id);
CREATE INDEX idx_node_executions_node_type ON automation.node_executions(node_type);
CREATE INDEX idx_node_executions_status ON automation.node_executions(status);

CREATE INDEX idx_workflow_triggers_workflow_id ON automation.workflow_triggers(workflow_id);
CREATE INDEX idx_workflow_triggers_tenant_id ON automation.workflow_triggers(tenant_id);
CREATE INDEX idx_workflow_triggers_trigger_type ON automation.workflow_triggers(trigger_type);
CREATE INDEX idx_workflow_triggers_is_active ON automation.workflow_triggers(is_active);

CREATE INDEX idx_workflow_templates_category ON automation.workflow_templates(category);
CREATE INDEX idx_workflow_templates_is_public ON automation.workflow_templates(is_public);
CREATE INDEX idx_workflow_templates_is_featured ON automation.workflow_templates(is_featured);
CREATE INDEX idx_workflow_templates_usage_count ON automation.workflow_templates(usage_count);

CREATE INDEX idx_workflow_analytics_workflow_id ON automation.workflow_analytics(workflow_id);
CREATE INDEX idx_workflow_analytics_date_period ON automation.workflow_analytics(date_period);
CREATE INDEX idx_workflow_analytics_period_type ON automation.workflow_analytics(period_type);

-- Create GIN indexes for JSONB columns
CREATE INDEX idx_workflows_tags_gin ON automation.workflows USING GIN(tags);
CREATE INDEX idx_workflows_trigger_config_gin ON automation.workflows USING GIN(trigger_config);
CREATE INDEX idx_workflows_nodes_config_gin ON automation.workflows USING GIN(nodes_config);

CREATE INDEX idx_workflow_templates_tags_gin ON automation.workflow_templates USING GIN(tags);
CREATE INDEX idx_workflow_templates_use_cases_gin ON automation.workflow_templates USING GIN(use_cases);
CREATE INDEX idx_workflow_templates_config_gin ON automation.workflow_templates USING GIN(template_config);

-- Create functions for automation
CREATE OR REPLACE FUNCTION automation.update_workflow_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update workflow execution stats
    UPDATE automation.workflows
    SET
        execution_count = (
            SELECT COUNT(*)
            FROM automation.workflow_executions
            WHERE workflow_id = NEW.workflow_id
        ),
        success_rate = (
            SELECT COALESCE(
                (COUNT(*) FILTER (WHERE status = 'completed')::DECIMAL / NULLIF(COUNT(*), 0)) * 100,
                0
            )
            FROM automation.workflow_executions
            WHERE workflow_id = NEW.workflow_id
        ),
        avg_duration_ms = (
            SELECT COALESCE(AVG(total_duration_ms), 0)
            FROM automation.workflow_executions
            WHERE workflow_id = NEW.workflow_id AND status = 'completed'
        ),
        total_cost = (
            SELECT COALESCE(SUM(total_cost), 0)
            FROM automation.workflow_executions
            WHERE workflow_id = NEW.workflow_id
        )
    WHERE workflow_id = NEW.workflow_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for workflow stats
CREATE TRIGGER trigger_update_workflow_stats
    AFTER INSERT OR UPDATE ON automation.workflow_executions
    FOR EACH ROW
    EXECUTE FUNCTION automation.update_workflow_stats();

-- Create function for template rating updates
CREATE OR REPLACE FUNCTION automation.update_template_rating()
RETURNS TRIGGER AS $$
BEGIN
    -- Update template rating stats
    UPDATE automation.workflow_templates
    SET
        avg_rating = (
            SELECT COALESCE(AVG(rating), 0)
            FROM automation.template_ratings
            WHERE template_id = COALESCE(NEW.template_id, OLD.template_id)
        ),
        rating_count = (
            SELECT COUNT(*)
            FROM automation.template_ratings
            WHERE template_id = COALESCE(NEW.template_id, OLD.template_id)
        )
    WHERE template_id = COALESCE(NEW.template_id, OLD.template_id);

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create trigger for template ratings
CREATE TRIGGER trigger_update_template_rating
    AFTER INSERT OR UPDATE OR DELETE ON automation.template_ratings
    FOR EACH ROW
    EXECUTE FUNCTION automation.update_template_rating();

-- Insert default workflow templates
INSERT INTO automation.workflow_templates (
    name, description, category, tags, use_cases, complexity_level,
    estimated_setup_time, template_config, is_public, is_featured
) VALUES
(
    'Lead Nurturing Campaign',
    'Automated email sequence for nurturing leads through the sales funnel',
    'sales',
    '["email", "automation", "sales", "nurturing"]',
    '["Lead qualification", "Email marketing", "Sales automation"]',
    'beginner',
    15,
    '{
        "trigger": {
            "type": "webhook",
            "name": "New Lead Trigger",
            "configuration": {
                "webhook_url": "/webhooks/new-lead",
                "method": "POST"
            }
        },
        "nodes": [
            {
                "id": "node_1",
                "type": "email_send",
                "name": "Welcome Email",
                "configuration": {
                    "email": {
                        "to": "{lead_email}",
                        "subject": "Welcome to {company_name}!",
                        "body": "Thank you for your interest in our services..."
                    }
                },
                "position": {"x": 100, "y": 100}
            },
            {
                "id": "node_2",
                "type": "delay",
                "name": "Wait 3 Days",
                "configuration": {
                    "delay": {"seconds": 259200}
                },
                "position": {"x": 100, "y": 200}
            },
            {
                "id": "node_3",
                "type": "email_send",
                "name": "Follow-up Email",
                "configuration": {
                    "email": {
                        "to": "{lead_email}",
                        "subject": "How can we help you succeed?",
                        "body": "We wanted to follow up on your interest..."
                    }
                },
                "position": {"x": 100, "y": 300}
            }
        ]
    }',
    true,
    true
),
(
    'Customer Onboarding',
    'Complete customer onboarding workflow with welcome sequence and setup tasks',
    'customer_success',
    '["onboarding", "customer_success", "automation", "welcome"]',
    '["New customer setup", "Account activation", "Product introduction"]',
    'intermediate',
    30,
    '{
        "trigger": {
            "type": "event",
            "name": "New Customer Signup",
            "configuration": {
                "event_type": "customer.created"
            }
        },
        "nodes": [
            {
                "id": "node_1",
                "type": "email_send",
                "name": "Welcome Email",
                "configuration": {
                    "email": {
                        "to": "{customer_email}",
                        "subject": "Welcome aboard, {customer_name}!",
                        "body": "We are excited to have you as a customer..."
                    }
                },
                "position": {"x": 100, "y": 100}
            },
            {
                "id": "node_2",
                "type": "api_request",
                "name": "Create Customer Account",
                "configuration": {
                    "api": {
                        "url": "/api/customers/{customer_id}/setup",
                        "method": "POST",
                        "body": {"setup_type": "standard"}
                    }
                },
                "position": {"x": 100, "y": 200}
            },
            {
                "id": "node_3",
                "type": "notification",
                "name": "Notify Success Team",
                "configuration": {
                    "notification": {
                        "type": "slack",
                        "channel": "#customer-success",
                        "message": "New customer {customer_name} has been onboarded"
                    }
                },
                "position": {"x": 100, "y": 300}
            }
        ]
    }',
    true,
    true
),
(
    'Support Ticket Routing',
    'Intelligent support ticket routing based on priority and category',
    'customer_service',
    '["support", "routing", "automation", "customer_service"]',
    '["Ticket management", "Support automation", "Priority routing"]',
    'advanced',
    45,
    '{
        "trigger": {
            "type": "webhook",
            "name": "New Support Ticket",
            "configuration": {
                "webhook_url": "/webhooks/support-ticket",
                "method": "POST"
            }
        },
        "nodes": [
            {
                "id": "node_1",
                "type": "condition",
                "name": "Check Priority",
                "configuration": {
                    "condition": {
                        "type": "simple",
                        "field": "priority",
                        "operator": "equals",
                        "value": "high"
                    }
                },
                "position": {"x": 100, "y": 100}
            },
            {
                "id": "node_2",
                "type": "notification",
                "name": "Alert Senior Support",
                "configuration": {
                    "notification": {
                        "type": "email",
                        "to": "senior-support@company.com",
                        "subject": "High Priority Ticket: {ticket_subject}"
                    }
                },
                "position": {"x": 200, "y": 150}
            },
            {
                "id": "node_3",
                "type": "api_request",
                "name": "Assign to Queue",
                "configuration": {
                    "api": {
                        "url": "/api/tickets/{ticket_id}/assign",
                        "method": "POST",
                        "body": {"queue": "general"}
                    }
                },
                "position": {"x": 200, "y": 250}
            }
        ]
    }',
    true,
    false
);

-- Grant permissions
GRANT USAGE ON SCHEMA automation TO agentsystem_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA automation TO agentsystem_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA automation TO agentsystem_app;
