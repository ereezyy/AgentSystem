-- AI Agent Marketplace Database Schema - AgentSystem Profit Machine
-- Revolutionary platform for browsing, creating, and monetizing AI agents

-- Create marketplace schema
CREATE SCHEMA IF NOT EXISTS marketplace;

-- Agent category enum
CREATE TYPE marketplace.agent_category AS ENUM (
    'customer_service', 'sales_automation', 'marketing', 'operations',
    'analytics', 'content_creation', 'data_processing', 'integration',
    'workflow_automation', 'specialized'
);

-- Agent status enum
CREATE TYPE marketplace.agent_status AS ENUM (
    'draft', 'published', 'deprecated', 'private'
);

-- Agent pricing model enum
CREATE TYPE marketplace.agent_pricing_model AS ENUM (
    'free', 'one_time', 'subscription', 'usage_based', 'revenue_share'
);

-- Agent capability enum
CREATE TYPE marketplace.agent_capability AS ENUM (
    'text_generation', 'data_analysis', 'api_integration', 'workflow_execution',
    'decision_making', 'learning_adaptation', 'multi_modal', 'real_time_processing'
);

-- Agents table (marketplace catalog)
CREATE TABLE marketplace.agents (
    agent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    category marketplace.agent_category NOT NULL,
    capabilities JSONB NOT NULL DEFAULT '[]', -- Array of agent_capability values
    version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    author_id UUID NOT NULL,
    author_name VARCHAR(200) NOT NULL,
    pricing_model marketplace.agent_pricing_model NOT NULL DEFAULT 'free',
    price DECIMAL(10,2) DEFAULT 0,
    currency VARCHAR(3) DEFAULT 'USD',
    status marketplace.agent_status NOT NULL DEFAULT 'draft',
    tags JSONB DEFAULT '[]',
    use_cases JSONB DEFAULT '[]',
    supported_integrations JSONB DEFAULT '[]',
    min_requirements JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    download_count INTEGER DEFAULT 0,
    rating_average DECIMAL(3,2) DEFAULT 0,
    rating_count INTEGER DEFAULT 0,
    revenue_generated DECIMAL(15,2) DEFAULT 0,
    is_featured BOOLEAN DEFAULT false,
    created_date TIMESTAMPTZ DEFAULT NOW(),
    updated_date TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (author_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Agent definitions table (implementation details)
CREATE TABLE marketplace.agent_definitions (
    agent_id UUID PRIMARY KEY,
    configuration JSONB NOT NULL DEFAULT '{}',
    prompt_templates JSONB NOT NULL DEFAULT '{}',
    workflow_logic JSONB NOT NULL DEFAULT '{}',
    api_endpoints JSONB DEFAULT '[]',
    event_handlers JSONB DEFAULT '{}',
    validation_rules JSONB DEFAULT '[]',
    dependencies JSONB DEFAULT '[]',
    custom_code TEXT,
    test_cases JSONB DEFAULT '[]',
    security_config JSONB DEFAULT '{}',
    resource_limits JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (agent_id) REFERENCES marketplace.agents(agent_id) ON DELETE CASCADE
);

-- Agent instances table (deployed agents)
CREATE TABLE marketplace.agent_instances (
    instance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    instance_name VARCHAR(200) NOT NULL,
    configuration JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active', -- active, paused, stopped, error
    deployment_date TIMESTAMPTZ DEFAULT NOW(),
    last_execution TIMESTAMPTZ,
    execution_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 1.0,
    avg_execution_time DECIMAL(10,2) DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    resource_usage JSONB DEFAULT '{}',
    monthly_cost DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (agent_id) REFERENCES marketplace.agents(agent_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, instance_name)
);

-- Agent executions table (execution history)
CREATE TABLE marketplace.agent_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instance_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    trigger_type VARCHAR(100) NOT NULL, -- manual, scheduled, webhook, event
    input_data JSONB,
    output_data JSONB,
    execution_time_ms INTEGER DEFAULT 0,
    status VARCHAR(50) NOT NULL, -- success, error, timeout, cancelled
    error_message TEXT,
    cost DECIMAL(8,4) DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    FOREIGN KEY (instance_id) REFERENCES marketplace.agent_instances(instance_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Agent purchases table
CREATE TABLE marketplace.agent_purchases (
    purchase_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    buyer_id UUID NOT NULL,
    seller_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    purchase_type marketplace.agent_pricing_model NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    stripe_payment_intent_id VARCHAR(200),
    license_type VARCHAR(100) DEFAULT 'standard', -- standard, enterprise, unlimited
    expires_at TIMESTAMPTZ,
    purchase_date TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (buyer_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (seller_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (agent_id) REFERENCES marketplace.agents(agent_id) ON DELETE CASCADE
);

-- Agent ratings and reviews table
CREATE TABLE marketplace.agent_ratings (
    rating_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review TEXT,
    helpful_votes INTEGER DEFAULT 0,
    rating_date TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (agent_id) REFERENCES marketplace.agents(agent_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, agent_id)
);

-- Agent monetization table
CREATE TABLE marketplace.agent_monetization (
    monetization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL,
    tenant_id UUID NOT NULL, -- Author/seller
    stripe_product_id VARCHAR(200),
    stripe_price_id VARCHAR(200),
    pricing_model marketplace.agent_pricing_model NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    revenue_share_percent DECIMAL(5,2) DEFAULT 70, -- Platform takes 30%
    total_revenue DECIMAL(15,2) DEFAULT 0,
    total_purchases INTEGER DEFAULT 0,
    payout_schedule VARCHAR(50) DEFAULT 'monthly', -- weekly, monthly, quarterly
    last_payout_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (agent_id) REFERENCES marketplace.agents(agent_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(agent_id)
);

-- Agent installations tracking table
CREATE TABLE marketplace.agent_installations (
    installation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    instance_id UUID NOT NULL,
    installation_date TIMESTAMPTZ DEFAULT NOW(),
    uninstallation_date TIMESTAMPTZ,
    installation_source VARCHAR(100), -- marketplace, custom, import

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (agent_id) REFERENCES marketplace.agents(agent_id) ON DELETE CASCADE,
    FOREIGN KEY (instance_id) REFERENCES marketplace.agent_instances(instance_id) ON DELETE CASCADE
);

-- Agent search index table
CREATE TABLE marketplace.agent_search_index (
    agent_id UUID PRIMARY KEY,
    search_data JSONB NOT NULL,
    search_vector tsvector,
    indexed_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (agent_id) REFERENCES marketplace.agents(agent_id) ON DELETE CASCADE
);

-- Agent builder templates table
CREATE TABLE marketplace.builder_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(200) NOT NULL,
    category marketplace.agent_category NOT NULL,
    description TEXT NOT NULL,
    template_config JSONB NOT NULL,
    default_prompts JSONB DEFAULT '{}',
    sample_workflows JSONB DEFAULT '[]',
    required_capabilities JSONB DEFAULT '[]',
    is_premium BOOLEAN DEFAULT false,
    usage_count INTEGER DEFAULT 0,
    created_by UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (created_by) REFERENCES billing.tenants(tenant_id) ON DELETE SET NULL
);

-- Agent performance metrics table
CREATE TABLE marketplace.agent_performance (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL,
    tenant_id UUID NOT NULL, -- Instance owner
    instance_id UUID NOT NULL,
    metric_date DATE NOT NULL,
    execution_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    avg_execution_time DECIMAL(10,2) DEFAULT 0,
    total_cost DECIMAL(10,2) DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cpu_usage_avg DECIMAL(5,2) DEFAULT 0,
    memory_usage_avg DECIMAL(5,2) DEFAULT 0,
    user_satisfaction_score DECIMAL(3,2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (agent_id) REFERENCES marketplace.agents(agent_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (instance_id) REFERENCES marketplace.agent_instances(instance_id) ON DELETE CASCADE,
    UNIQUE(instance_id, metric_date)
);

-- Agent marketplace categories table
CREATE TABLE marketplace.categories (
    category_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category_name marketplace.agent_category NOT NULL UNIQUE,
    display_name VARCHAR(200) NOT NULL,
    description TEXT,
    icon_url VARCHAR(500),
    agent_count INTEGER DEFAULT 0,
    featured_agents JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT true,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_marketplace_agents_author ON marketplace.agents(author_id);
CREATE INDEX idx_marketplace_agents_category ON marketplace.agents(category);
CREATE INDEX idx_marketplace_agents_status ON marketplace.agents(status);
CREATE INDEX idx_marketplace_agents_pricing ON marketplace.agents(pricing_model);
CREATE INDEX idx_marketplace_agents_rating ON marketplace.agents(rating_average DESC);
CREATE INDEX idx_marketplace_agents_downloads ON marketplace.agents(download_count DESC);
CREATE INDEX idx_marketplace_agents_featured ON marketplace.agents(is_featured);

CREATE INDEX idx_agent_instances_tenant ON marketplace.agent_instances(tenant_id);
CREATE INDEX idx_agent_instances_agent ON marketplace.agent_instances(agent_id);
CREATE INDEX idx_agent_instances_status ON marketplace.agent_instances(status);
CREATE INDEX idx_agent_instances_deployment ON marketplace.agent_instances(deployment_date DESC);

CREATE INDEX idx_agent_executions_instance ON marketplace.agent_executions(instance_id);
CREATE INDEX idx_agent_executions_tenant ON marketplace.agent_executions(tenant_id);
CREATE INDEX idx_agent_executions_started ON marketplace.agent_executions(started_at DESC);
CREATE INDEX idx_agent_executions_status ON marketplace.agent_executions(status);

CREATE INDEX idx_agent_purchases_buyer ON marketplace.agent_purchases(buyer_id);
CREATE INDEX idx_agent_purchases_seller ON marketplace.agent_purchases(seller_id);
CREATE INDEX idx_agent_purchases_agent ON marketplace.agent_purchases(agent_id);
CREATE INDEX idx_agent_purchases_date ON marketplace.agent_purchases(purchase_date DESC);

CREATE INDEX idx_agent_ratings_agent ON marketplace.agent_ratings(agent_id);
CREATE INDEX idx_agent_ratings_rating ON marketplace.agent_ratings(rating DESC);
CREATE INDEX idx_agent_ratings_date ON marketplace.agent_ratings(rating_date DESC);

CREATE INDEX idx_agent_monetization_tenant ON marketplace.agent_monetization(tenant_id);
CREATE INDEX idx_agent_monetization_agent ON marketplace.agent_monetization(agent_id);

CREATE INDEX idx_agent_installations_tenant ON marketplace.agent_installations(tenant_id);
CREATE INDEX idx_agent_installations_agent ON marketplace.agent_installations(agent_id);
CREATE INDEX idx_agent_installations_date ON marketplace.agent_installations(installation_date DESC);

CREATE INDEX idx_agent_search_index_vector ON marketplace.agent_search_index USING gin(search_vector);

CREATE INDEX idx_builder_templates_category ON marketplace.builder_templates(category);
CREATE INDEX idx_builder_templates_premium ON marketplace.builder_templates(is_premium);
CREATE INDEX idx_builder_templates_usage ON marketplace.builder_templates(usage_count DESC);

CREATE INDEX idx_agent_performance_agent ON marketplace.agent_performance(agent_id);
CREATE INDEX idx_agent_performance_tenant ON marketplace.agent_performance(tenant_id);
CREATE INDEX idx_agent_performance_instance ON marketplace.agent_performance(instance_id);
CREATE INDEX idx_agent_performance_date ON marketplace.agent_performance(metric_date DESC);

-- Create updated_at trigger function (reuse existing one)
CREATE TRIGGER update_marketplace_agents_updated_at
    BEFORE UPDATE ON marketplace.agents
    FOR EACH ROW EXECUTE FUNCTION analytics.update_updated_at_column();

CREATE TRIGGER update_agent_instances_updated_at
    BEFORE UPDATE ON marketplace.agent_instances
    FOR EACH ROW EXECUTE FUNCTION analytics.update_updated_at_column();

CREATE TRIGGER update_agent_monetization_updated_at
    BEFORE UPDATE ON marketplace.agent_monetization
    FOR EACH ROW EXECUTE FUNCTION analytics.update_updated_at_column();

CREATE TRIGGER update_builder_templates_updated_at
    BEFORE UPDATE ON marketplace.builder_templates
    FOR EACH ROW EXECUTE FUNCTION analytics.update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE marketplace.agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE marketplace.agent_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE marketplace.agent_instances ENABLE ROW LEVEL SECURITY;
ALTER TABLE marketplace.agent_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE marketplace.agent_purchases ENABLE ROW LEVEL SECURITY;
ALTER TABLE marketplace.agent_ratings ENABLE ROW LEVEL SECURITY;
ALTER TABLE marketplace.agent_monetization ENABLE ROW LEVEL SECURITY;
ALTER TABLE marketplace.agent_installations ENABLE ROW LEVEL SECURITY;
ALTER TABLE marketplace.agent_performance ENABLE ROW LEVEL SECURITY;

-- RLS policies - Note: Marketplace has different access patterns
-- Published agents are public, but management requires ownership
CREATE POLICY agents_public_read ON marketplace.agents
    FOR SELECT USING (status IN ('published'));

CREATE POLICY agents_author_manage ON marketplace.agents
    FOR ALL USING (author_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY agent_definitions_author_only ON marketplace.agent_definitions
    USING (agent_id IN (SELECT agent_id FROM marketplace.agents WHERE author_id = current_setting('app.current_tenant_id')::UUID));

CREATE POLICY agent_instances_tenant_isolation ON marketplace.agent_instances
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY agent_executions_tenant_isolation ON marketplace.agent_executions
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY agent_purchases_participant ON marketplace.agent_purchases
    USING (buyer_id = current_setting('app.current_tenant_id')::UUID OR seller_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY agent_ratings_tenant_isolation ON marketplace.agent_ratings
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY agent_monetization_author_only ON marketplace.agent_monetization
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY agent_installations_tenant_isolation ON marketplace.agent_installations
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY agent_performance_tenant_isolation ON marketplace.agent_performance
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant permissions
GRANT USAGE ON SCHEMA marketplace TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA marketplace TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA marketplace TO agentsystem_api;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA marketplace TO agentsystem_api;

-- Create views for analytics and reporting
CREATE VIEW marketplace.agent_popularity AS
SELECT
    a.agent_id,
    a.name,
    a.category,
    a.author_name,
    a.download_count,
    a.rating_average,
    a.rating_count,
    COUNT(ai.instance_id) as active_instances,
    SUM(ae.execution_count) as total_executions,
    AVG(ae.success_rate) as avg_success_rate
FROM marketplace.agents a
LEFT JOIN marketplace.agent_instances ai ON a.agent_id = ai.agent_id AND ai.status = 'active'
LEFT JOIN marketplace.agent_executions ae ON ai.instance_id = ae.instance_id
WHERE a.status = 'published'
GROUP BY a.agent_id, a.name, a.category, a.author_name, a.download_count, a.rating_average, a.rating_count
ORDER BY a.download_count DESC, a.rating_average DESC;

CREATE VIEW marketplace.author_performance AS
SELECT
    a.author_id,
    a.author_name,
    COUNT(a.agent_id) as total_agents,
    COUNT(CASE WHEN a.status = 'published' THEN 1 END) as published_agents,
    SUM(a.download_count) as total_downloads,
    AVG(a.rating_average) as avg_rating,
    SUM(am.total_revenue) as total_revenue,
    COUNT(ap.purchase_id) as total_sales
FROM marketplace.agents a
LEFT JOIN marketplace.agent_monetization am ON a.agent_id = am.agent_id
LEFT JOIN marketplace.agent_purchases ap ON a.agent_id = ap.agent_id
GROUP BY a.author_id, a.author_name
ORDER BY total_revenue DESC NULLS LAST, total_downloads DESC;

CREATE VIEW marketplace.category_performance AS
SELECT
    c.category_name,
    c.display_name,
    COUNT(a.agent_id) as total_agents,
    COUNT(CASE WHEN a.status = 'published' THEN 1 END) as published_agents,
    AVG(a.rating_average) as avg_rating,
    SUM(a.download_count) as total_downloads,
    COUNT(ai.instance_id) as active_instances
FROM marketplace.categories c
LEFT JOIN marketplace.agents a ON c.category_name = a.category
LEFT JOIN marketplace.agent_instances ai ON a.agent_id = ai.agent_id AND ai.status = 'active'
GROUP BY c.category_name, c.display_name, c.sort_order
ORDER BY c.sort_order, total_downloads DESC;

CREATE VIEW marketplace.execution_analytics AS
SELECT
    ae.tenant_id,
    ai.agent_id,
    a.name as agent_name,
    DATE(ae.started_at) as execution_date,
    COUNT(*) as total_executions,
    COUNT(CASE WHEN ae.status = 'success' THEN 1 END) as successful_executions,
    AVG(ae.execution_time_ms) as avg_execution_time,
    SUM(ae.cost) as total_cost,
    SUM(ae.tokens_used) as total_tokens
FROM marketplace.agent_executions ae
JOIN marketplace.agent_instances ai ON ae.instance_id = ai.instance_id
JOIN marketplace.agents a ON ai.agent_id = a.agent_id
WHERE ae.started_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY ae.tenant_id, ai.agent_id, a.name, DATE(ae.started_at)
ORDER BY execution_date DESC, total_executions DESC;

-- Grant permissions on views
GRANT SELECT ON marketplace.agent_popularity TO agentsystem_api;
GRANT SELECT ON marketplace.author_performance TO agentsystem_api;
GRANT SELECT ON marketplace.category_performance TO agentsystem_api;
GRANT SELECT ON marketplace.execution_analytics TO agentsystem_api;

-- Create materialized view for dashboard performance
CREATE MATERIALIZED VIEW marketplace.marketplace_dashboard_stats AS
SELECT
    COUNT(DISTINCT a.agent_id) as total_agents,
    COUNT(DISTINCT CASE WHEN a.status = 'published' THEN a.agent_id END) as published_agents,
    COUNT(DISTINCT a.author_id) as total_authors,
    COUNT(DISTINCT ai.tenant_id) as active_customers,
    SUM(a.download_count) as total_downloads,
    AVG(a.rating_average) as avg_marketplace_rating,
    COUNT(ai.instance_id) as total_instances,
    COUNT(CASE WHEN ai.status = 'active' THEN 1 END) as active_instances,
    SUM(am.total_revenue) as total_marketplace_revenue,
    COUNT(ap.purchase_id) as total_purchases,
    COUNT(CASE WHEN ae.started_at >= CURRENT_DATE - INTERVAL '24 hours' THEN 1 END) as executions_last_24h
FROM marketplace.agents a
LEFT JOIN marketplace.agent_instances ai ON a.agent_id = ai.agent_id
LEFT JOIN marketplace.agent_monetization am ON a.agent_id = am.agent_id
LEFT JOIN marketplace.agent_purchases ap ON a.agent_id = ap.agent_id
LEFT JOIN marketplace.agent_executions ae ON ai.instance_id = ae.instance_id;

-- Create index on materialized view
CREATE INDEX idx_marketplace_dashboard_stats ON marketplace.marketplace_dashboard_stats USING btree (total_agents);

-- Grant permissions on materialized view
GRANT SELECT ON marketplace.marketplace_dashboard_stats TO agentsystem_api;

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION marketplace.refresh_marketplace_dashboard_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY marketplace.marketplace_dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on function
GRANT EXECUTE ON FUNCTION marketplace.refresh_marketplace_dashboard_stats() TO agentsystem_api;

-- Function to update search vector
CREATE OR REPLACE FUNCTION marketplace.update_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english',
        COALESCE(NEW.search_data->>'name', '') || ' ' ||
        COALESCE(NEW.search_data->>'description', '') || ' ' ||
        COALESCE(NEW.search_data->>'tags', '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for search vector updates
CREATE TRIGGER update_agent_search_vector
    BEFORE INSERT OR UPDATE ON marketplace.agent_search_index
    FOR EACH ROW EXECUTE FUNCTION marketplace.update_search_vector();

-- Function to calculate agent revenue share
CREATE OR REPLACE FUNCTION marketplace.calculate_revenue_share(
    p_agent_id UUID,
    p_purchase_amount DECIMAL
) RETURNS DECIMAL AS $$
DECLARE
    v_revenue_share_percent DECIMAL;
    v_author_share DECIMAL;
BEGIN
    -- Get revenue share percentage
    SELECT revenue_share_percent INTO v_revenue_share_percent
    FROM marketplace.agent_monetization
    WHERE agent_id = p_agent_id;

    IF v_revenue_share_percent IS NULL THEN
        v_revenue_share_percent := 70; -- Default 70% to author
    END IF;

    v_author_share := p_purchase_amount * (v_revenue_share_percent / 100);

    RETURN ROUND(v_author_share, 2);
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on revenue share function
GRANT EXECUTE ON FUNCTION marketplace.calculate_revenue_share TO agentsystem_api;

-- Function to update agent metrics after execution
CREATE OR REPLACE FUNCTION marketplace.update_agent_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update instance metrics
    UPDATE marketplace.agent_instances
    SET execution_count = execution_count + 1,
        last_execution = NEW.completed_at,
        success_rate = CASE
            WHEN NEW.status = 'success' THEN
                (success_rate * execution_count + 1) / (execution_count + 1)
            ELSE
                (success_rate * execution_count) / (execution_count + 1)
        END,
        avg_execution_time = (avg_execution_time * execution_count + NEW.execution_time_ms) / (execution_count + 1),
        error_count = CASE WHEN NEW.status = 'error' THEN error_count + 1 ELSE error_count END,
        monthly_cost = monthly_cost + NEW.cost
    WHERE instance_id = NEW.instance_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic metrics updates
CREATE TRIGGER update_agent_metrics_trigger
    AFTER INSERT ON marketplace.agent_executions
    FOR EACH ROW EXECUTE FUNCTION marketplace.update_agent_metrics();

-- Insert default categories
INSERT INTO marketplace.categories (category_name, display_name, description, sort_order) VALUES
('customer_service', 'Customer Service', 'AI agents for customer support and service automation', 1),
('sales_automation', 'Sales Automation', 'AI agents for sales process automation and lead management', 2),
('marketing', 'Marketing', 'AI agents for marketing automation and content generation', 3),
('operations', 'Operations', 'AI agents for business operations and process automation', 4),
('analytics', 'Analytics', 'AI agents for data analysis and business intelligence', 5),
('content_creation', 'Content Creation', 'AI agents for content generation and creative tasks', 6),
('data_processing', 'Data Processing', 'AI agents for data transformation and processing', 7),
('integration', 'Integration', 'AI agents for system integration and API management', 8),
('workflow_automation', 'Workflow Automation', 'AI agents for workflow and process automation', 9),
('specialized', 'Specialized', 'Industry-specific and specialized AI agents', 10);

-- Insert default builder templates
INSERT INTO marketplace.builder_templates (template_name, category, description, template_config, default_prompts) VALUES
('Customer Support Bot', 'customer_service', 'Template for building customer support chatbots',
 '{"type": "chatbot", "capabilities": ["text_generation", "api_integration"]}',
 '{"greeting": "Hello! How can I help you today?", "escalation": "Let me connect you with a human agent."}'),

('Lead Qualification Agent', 'sales_automation', 'Template for building lead qualification agents',
 '{"type": "workflow", "capabilities": ["data_analysis", "decision_making"]}',
 '{"qualification": "Based on your requirements, let me assess your fit.", "follow_up": "I''ll schedule a follow-up call."}'),

('Content Generator', 'content_creation', 'Template for building content generation agents',
 '{"type": "generator", "capabilities": ["text_generation", "multi_modal"]}',
 '{"blog_post": "Generate a blog post about {topic}", "social_media": "Create social media content for {platform}"}'),

('Data Analyzer', 'analytics', 'Template for building data analysis agents',
 '{"type": "analyzer", "capabilities": ["data_analysis", "data_processing"]}',
 '{"analysis": "Analyze the provided data and generate insights", "report": "Create a summary report of findings"}'),

('Workflow Automator', 'workflow_automation', 'Template for building workflow automation agents',
 '{"type": "workflow", "capabilities": ["workflow_execution", "api_integration"]}',
 '{"start": "Starting workflow automation", "complete": "Workflow completed successfully"}'
);
