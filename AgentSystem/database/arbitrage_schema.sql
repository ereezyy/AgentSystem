
-- AI Arbitrage Engine Database Schema - AgentSystem Profit Machine
-- Intelligent AI provider routing and cost optimization

-- Create optimization schema
CREATE SCHEMA IF NOT EXISTS optimization;

-- AI provider enum
CREATE TYPE optimization.ai_provider AS ENUM (
    'openai', 'anthropic', 'google', 'azure_openai', 'aws_bedrock',
    'cohere', 'mistral', 'together', 'fireworks', 'groq', 'replicate', 'huggingface'
);

-- Model capability enum
CREATE TYPE optimization.model_capability AS ENUM (
    'text_generation', 'code_generation', 'image_generation', 'image_analysis',
    'embedding', 'function_calling', 'long_context', 'multimodal'
);

-- Routing strategy enum
CREATE TYPE optimization.routing_strategy AS ENUM (
    'cost_optimal', 'quality_optimal', 'latency_optimal', 'balanced', 'fallback_cascade'
);

-- AI models table
CREATE TABLE optimization.ai_models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider optimization.ai_provider NOT NULL,
    model_name VARCHAR(200) NOT NULL,
    capabilities optimization.model_capability[] NOT NULL,
    cost_per_input_token DECIMAL(12,8) NOT NULL,
    cost_per_output_token DECIMAL(12,8) NOT NULL,
    max_context_length INTEGER NOT NULL,
    max_output_tokens INTEGER NOT NULL,
    quality_score DECIMAL(5,2) NOT NULL CHECK (quality_score >= 0 AND quality_score <= 100),
    average_latency_ms DECIMAL(8,2) NOT NULL,
    availability_score DECIMAL(5,2) NOT NULL CHECK (availability_score >= 0 AND availability_score <= 100),
    rate_limit_rpm INTEGER NOT NULL,
    rate_limit_tpm INTEGER NOT NULL,
    supports_streaming BOOLEAN DEFAULT false,
    supports_function_calling BOOLEAN DEFAULT false,
    supports_vision BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(provider, model_name)
);

-- AI pricing table for real-time pricing updates
CREATE TABLE optimization.ai_pricing (
    pricing_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider optimization.ai_provider NOT NULL,
    model_name VARCHAR(200) NOT NULL,
    cost_per_input_token DECIMAL(12,8) NOT NULL,
    cost_per_output_token DECIMAL(12,8) NOT NULL,
    effective_date TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(provider, model_name)
);

-- Provider metrics table
CREATE TABLE optimization.provider_metrics (
    provider optimization.ai_provider PRIMARY KEY,
    success_rate DECIMAL(5,4) NOT NULL CHECK (success_rate >= 0 AND success_rate <= 1),
    average_latency_ms DECIMAL(8,2) NOT NULL,
    average_cost_per_request DECIMAL(10,6) NOT NULL,
    total_requests_24h INTEGER DEFAULT 0,
    total_cost_24h DECIMAL(12,2) DEFAULT 0,
    quality_score DECIMAL(5,2) NOT NULL CHECK (quality_score >= 0 AND quality_score <= 100),
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- Arbitrage decisions table
CREATE TABLE optimization.arbitrage_decisions (
    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL UNIQUE,
    tenant_id UUID NOT NULL,
    capability optimization.model_capability NOT NULL,
    input_tokens INTEGER NOT NULL,
    estimated_output_tokens INTEGER NOT NULL,
    strategy optimization.routing_strategy NOT NULL,
    selected_model VARCHAR(300) NOT NULL, -- provider:model_name format
    estimated_cost DECIMAL(10,6) NOT NULL,
    estimated_latency_ms DECIMAL(8,2) NOT NULL,
    cost_savings_percent DECIMAL(5,2) DEFAULT 0,
    routing_reason TEXT,
    confidence_score DECIMAL(5,2) CHECK (confidence_score >= 0 AND confidence_score <= 100),
    fallback_models JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Arbitrage outcomes table for tracking actual results
CREATE TABLE optimization.arbitrage_outcomes (
    outcome_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL,
    actual_cost DECIMAL(10,6) NOT NULL,
    actual_latency_ms DECIMAL(8,2) NOT NULL,
    quality_rating DECIMAL(5,2) CHECK (quality_rating >= 0 AND quality_rating <= 100),
    success BOOLEAN NOT NULL,
    error_details TEXT,
    response_tokens INTEGER,
    completion_reason VARCHAR(100), -- stop, length, function_call, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (request_id) REFERENCES optimization.arbitrage_decisions(request_id) ON DELETE CASCADE
);

-- Model performance history table
CREATE TABLE optimization.model_performance_history (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_identifier VARCHAR(300) NOT NULL, -- provider:model_name format
    request_id UUID NOT NULL,
    cost_accuracy DECIMAL(5,4) CHECK (cost_accuracy >= 0 AND cost_accuracy <= 1),
    latency_accuracy DECIMAL(5,4) CHECK (latency_accuracy >= 0 AND latency_accuracy <= 1),
    quality_rating DECIMAL(5,2) CHECK (quality_rating >= 0 AND quality_rating <= 100),
    success BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (request_id) REFERENCES optimization.arbitrage_decisions(request_id) ON DELETE CASCADE
);

-- Model failures table for retraining triggers
CREATE TABLE optimization.model_failures (
    failure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL,
    error_details TEXT NOT NULL,
    error_category VARCHAR(100), -- timeout, rate_limit, api_error, quality_issue
    provider_response_code INTEGER,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (request_id) REFERENCES optimization.arbitrage_decisions(request_id) ON DELETE CASCADE
);

-- Cost optimization rules table
CREATE TABLE optimization.cost_optimization_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(300) NOT NULL,
    description TEXT,
    conditions JSONB NOT NULL, -- JSON conditions for rule matching
    actions JSONB NOT NULL,    -- Actions to take when rule matches
    priority INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    cost_savings_target DECIMAL(5,2) DEFAULT 0, -- Target cost savings percentage
    times_triggered INTEGER DEFAULT 0,
    total_savings_achieved DECIMAL(12,2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Provider usage tracking table
CREATE TABLE optimization.provider_usage (
    usage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    provider optimization.ai_provider NOT NULL,
    model_name VARCHAR(200) NOT NULL,
    requests_count INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0,
    usage_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, provider, model_name, usage_date)
);

-- Routing analytics table
CREATE TABLE optimization.routing_analytics (
    analytics_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    date DATE NOT NULL,
    total_requests INTEGER DEFAULT 0,
    total_cost DECIMAL(12,6) DEFAULT 0,
    total_savings DECIMAL(12,6) DEFAULT 0,
    average_latency_ms DECIMAL(8,2) DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 0,
    top_provider VARCHAR(100),
    top_strategy optimization.routing_strategy,
    cost_optimal_requests INTEGER DEFAULT 0,
    quality_optimal_requests INTEGER DEFAULT 0,
    latency_optimal_requests INTEGER DEFAULT 0,
    balanced_requests INTEGER DEFAULT 0,
    fallback_requests INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, date)
);

-- Provider health checks table
CREATE TABLE optimization.provider_health_checks (
    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider optimization.ai_provider NOT NULL,
    endpoint_url VARCHAR(500),
    response_time_ms DECIMAL(8,2),
    status_code INTEGER,
    is_healthy BOOLEAN NOT NULL,
    error_message TEXT,
    check_timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Model benchmarks table
CREATE TABLE optimization.model_benchmarks (
    benchmark_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_identifier VARCHAR(300) NOT NULL, -- provider:model_name format
    benchmark_type VARCHAR(100) NOT NULL, -- quality, speed, cost_efficiency
    test_dataset VARCHAR(200),
    score DECIMAL(8,4) NOT NULL,
    percentile_rank DECIMAL(5,2), -- 0-100 percentile among all models
    test_date TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Request batching table for bulk optimization
CREATE TABLE optimization.request_batches (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    batch_size INTEGER NOT NULL,
    total_input_tokens INTEGER NOT NULL,
    total_output_tokens INTEGER NOT NULL,
    estimated_cost DECIMAL(10,6) NOT NULL,
    actual_cost DECIMAL(10,6),
    processing_time_ms INTEGER,
    selected_provider optimization.ai_provider NOT NULL,
    selected_model VARCHAR(200) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Batch request items table
CREATE TABLE optimization.batch_request_items (
    item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    request_id UUID NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER,
    individual_cost DECIMAL(10,6),
    processing_order INTEGER,
    status VARCHAR(50) DEFAULT 'pending',
    error_details TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (batch_id) REFERENCES optimization.request_batches(batch_id) ON DELETE CASCADE,
    FOREIGN KEY (request_id) REFERENCES optimization.arbitrage_decisions(request_id) ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX idx_ai_models_provider ON optimization.ai_models(provider);
CREATE INDEX idx_ai_models_capabilities ON optimization.ai_models USING GIN(capabilities);
CREATE INDEX idx_ai_models_active ON optimization.ai_models(is_active);
CREATE INDEX idx_ai_models_quality_score ON optimization.ai_models(quality_score DESC);
CREATE INDEX idx_ai_models_cost ON optimization.ai_models(cost_per_input_token, cost_per_output_token);

CREATE INDEX idx_ai_pricing_provider_model ON optimization.ai_pricing(provider, model_name);
CREATE INDEX idx_ai_pricing_updated_at ON optimization.ai_pricing(updated_at DESC);

CREATE INDEX idx_arbitrage_decisions_tenant ON optimization.arbitrage_decisions(tenant_id);
CREATE INDEX idx_arbitrage_decisions_created_at ON optimization.arbitrage_decisions(created_at DESC);
CREATE INDEX idx_arbitrage_decisions_strategy ON optimization.arbitrage_decisions(strategy);
CREATE INDEX idx_arbitrage_decisions_capability ON optimization.arbitrage_decisions(capability);
CREATE INDEX idx_arbitrage_decisions_selected_model ON optimization.arbitrage_decisions(selected_model);

CREATE INDEX idx_arbitrage_outcomes_request_id ON optimization.arbitrage_outcomes(request_id);
CREATE INDEX idx_arbitrage_outcomes_success ON optimization.arbitrage_outcomes(success);
CREATE INDEX idx_arbitrage_outcomes_created_at ON optimization.arbitrage_outcomes(created_at DESC);

CREATE INDEX idx_model_performance_model ON optimization.model_performance_history(model_identifier);
CREATE INDEX idx_model_performance_created_at ON optimization.model_performance_history(created_at DESC);
CREATE INDEX idx_model_performance_success ON optimization.model_performance_history(success);

CREATE INDEX idx_model_failures_created_at ON optimization.model_failures(created_at DESC);
CREATE INDEX idx_model_failures_error_category ON optimization.model_failures(error_category);

CREATE INDEX idx_cost_optimization_rules_tenant ON optimization.cost_optimization_rules(tenant_id);
CREATE INDEX idx_cost_optimization_rules_active ON optimization.cost_optimization_rules(is_active);
CREATE INDEX idx_cost_optimization_rules_priority ON optimization.cost_optimization_rules(priority DESC);

CREATE INDEX idx_provider_usage_tenant_date ON optimization.provider_usage(tenant_id, usage_date DESC);
CREATE INDEX idx_provider_usage_provider ON optimization.provider_usage(provider);

CREATE INDEX idx_routing_analytics_tenant_date ON optimization.routing_analytics(tenant_id, date DESC);

CREATE INDEX idx_provider_health_checks_provider ON optimization.provider_health_checks(provider);
CREATE INDEX idx_provider_health_checks_timestamp ON optimization.provider_health_checks(check_timestamp DESC);
CREATE INDEX idx_provider_health_checks_healthy ON optimization.provider_health_checks(is_healthy);

CREATE INDEX idx_model_benchmarks_model ON optimization.model_benchmarks(model_identifier);
CREATE INDEX idx_model_benchmarks_type ON optimization.model_benchmarks(benchmark_type);
CREATE INDEX idx_model_benchmarks_score ON optimization.model_benchmarks(score DESC);

CREATE INDEX idx_request_batches_tenant ON optimization.request_batches(tenant_id);
CREATE INDEX idx_request_batches_status ON optimization.request_batches(status);
CREATE INDEX idx_request_batches_created_at ON optimization.request_batches(created_at DESC);

CREATE INDEX idx_batch_request_items_batch ON optimization.batch_request_items(batch_id);
CREATE INDEX idx_batch_request_items_status ON optimization.batch_request_items(status);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION optimization.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_ai_models_updated_at BEFORE UPDATE ON optimization.ai_models FOR EACH ROW EXECUTE FUNCTION optimization.update_updated_at_column();
CREATE TRIGGER update_ai_pricing_updated_at BEFORE UPDATE ON optimization.ai_pricing FOR EACH ROW EXECUTE FUNCTION optimization.update_updated_at_column();
CREATE TRIGGER update_cost_optimization_rules_updated_at BEFORE UPDATE ON optimization.cost_optimization_rules FOR EACH ROW EXECUTE FUNCTION optimization.update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE optimization.arbitrage_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE optimization.arbitrage_outcomes ENABLE ROW LEVEL SECURITY;
ALTER TABLE optimization.cost_optimization_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE optimization.provider_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE optimization.routing_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE optimization.request_batches ENABLE ROW LEVEL SECURITY;

-- RLS policies for tenant isolation
CREATE POLICY arbitrage_decisions_tenant_isolation ON optimization.arbitrage_decisions
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY arbitrage_outcomes_tenant_isolation ON optimization.arbitrage_outcomes
    USING (EXISTS (
        SELECT 1 FROM optimization.arbitrage_decisions ad
        WHERE ad.request_id = arbitrage_outcomes.request_id
        AND ad.tenant_id = current_setting('app.current_tenant_id')::UUID
    ));

CREATE POLICY cost_optimization_rules_tenant_isolation ON optimization.cost_optimization_rules
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY provider_usage_tenant_isolation ON optimization.provider_usage
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY routing_analytics_tenant_isolation ON optimization.routing_analytics
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY request_batches_tenant_isolation ON optimization.request_batches
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant permissions
GRANT USAGE ON SCHEMA optimization TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA optimization TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA optimization TO agentsystem_api;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA optimization TO agentsystem_api;

-- Create views for analytics and reporting
CREATE VIEW optimization.cost_savings_summary AS
SELECT
    ad.tenant_id,
    DATE(ad.created_at) as date,
    COUNT(*) as total_requests,
    AVG(ad.cost_savings_percent) as avg_savings_percent,
    SUM(ad.estimated_cost) as total_estimated_cost,
    SUM(ao.actual_cost) as total_actual_cost,
    SUM(CASE WHEN ao.actual_cost IS NOT NULL
        THEN (ad.estimated_cost / (1 - ad.cost_savings_percent/100)) - ao.actual_cost
        ELSE 0 END) as actual_savings,
    AVG(ao.actual_latency_ms) as avg_latency_ms,
    COUNT(CASE WHEN ao.success THEN 1 END) * 100.0 / COUNT(*) as success_rate
FROM optimization.arbitrage_decisions ad
LEFT JOIN optimization.arbitrage_outcomes ao ON ad.request_id = ao.request_id
GROUP BY ad.tenant_id, DATE(ad.created_at);

CREATE VIEW optimization.provider_performance_summary AS
SELECT
    SPLIT_PART(ad.selected_model, ':', 1) as provider,
    SPLIT_PART(ad.selected_model, ':', 2) as model_name,
    COUNT(*) as total_requests,
    AVG(ad.cost_savings_percent) as avg_cost_savings,
    AVG(ad.confidence_score) as avg_confidence,
    AVG(ao.actual_latency_ms) as avg_latency_ms,
    COUNT(CASE WHEN ao.success THEN 1 END) * 100.0 / COUNT(*) as success_rate,
    AVG(ao.quality_rating) as avg_quality_rating,
    SUM(ao.actual_cost) as total_cost,
    MAX(ad.created_at) as last_used
FROM optimization.arbitrage_decisions ad
LEFT JOIN optimization.arbitrage_outcomes ao ON ad.request_id = ao.request_id
WHERE ad.created_at > NOW() - INTERVAL '30 days'
GROUP BY SPLIT_PART(ad.selected_model, ':', 1), SPLIT_PART(ad.selected_model, ':', 2);

CREATE VIEW optimization.tenant_arbitrage_dashboard AS
SELECT
    ad.tenant_id,
    COUNT(*) as total_requests,
    AVG(ad.cost_savings_percent) as avg_cost_savings,
    SUM(ad.estimated_cost) as total_estimated_cost,
    SUM(ao.actual_cost) as total_actual_cost,
    AVG(ao.actual_latency_ms) as avg_latency_ms,
    COUNT(CASE WHEN ao.success THEN 1 END) * 100.0 / COUNT(*) as success_rate,
    COUNT(DISTINCT SPLIT_PART(ad.selected_model, ':', 1)) as providers_used,
    COUNT(DISTINCT ad.selected_model) as models_used,
    MAX(ad.created_at) as last_request_at
FROM optimization.arbitrage_decisions ad
LEFT JOIN optimization.arbitrage_outcomes ao ON ad.request_id = ao.request_id
WHERE ad.created_at > NOW() - INTERVAL '7 days'
GROUP BY ad.tenant_id;

-- Grant permissions on views
GRANT SELECT ON optimization.cost_savings_summary TO agentsystem_api;
GRANT SELECT ON optimization.provider_performance_summary TO agentsystem_api;
GRANT SELECT ON optimization.tenant_arbitrage_dashboard TO agentsystem_api;

-- Insert sample AI models for testing
INSERT INTO optimization.ai_models (
    provider, model_name, capabilities, cost_per_input_token, cost_per_output_token,
    max_context_length, max_output_tokens, quality_score, average_latency_ms,
    availability_score, rate_limit_rpm, rate_limit_tpm, supports_streaming,
    supports_function_calling, supports_vision
) VALUES
-- OpenAI Models
('openai', 'gpt-4', ARRAY['text_generation', 'function_calling']::optimization.model_capability[],
 0.00003, 0.00006, 8192, 4096, 95.0, 800, 99.5, 500, 150000, true, true, false),

('openai', 'gpt-4-turbo', ARRAY['text_generation', 'function_calling', 'multimodal']::optimization.model_capability[],
 0.00001, 0.00003, 128000, 4096, 93.0, 600, 99.2, 800, 300000, true, true, true),

('openai', 'gpt-3.5-turbo', ARRAY['text_generation', 'function_calling']::optimization.model_capability[],
 0.000001, 0.000002, 16384, 4096, 85.0, 400, 99.8, 3500, 1000000, true, true, false),

-- Anthropic Models
('anthropic', 'claude-3-opus', ARRAY['text_generation', 'multimodal']::optimization.model_capability[],
 0.000015, 0.000075, 200000, 4096, 97.0, 1200, 99.0, 400, 200000, true, false, true),

('anthropic', 'claude-3-sonnet', ARRAY['text_generation', 'multimodal']::optimization.model_capability[],
 0.000003, 0.000015, 200000, 4096, 92.0, 800, 99.3, 600, 400000, true, false, true),

('anthropic', 'claude-3-haiku', ARRAY['text_generation']::optimization.model_capability[],
 0.00000025, 0.00000125, 200000, 4096, 88.0, 300, 99.7, 2000, 1000000, true, false, false),

-- Google Models
('google', 'gemini-pro', ARRAY['text_generation', 'multimodal']::optimization.model_capability[],
 0.0000005, 0.0000015, 32768, 8192, 90.0, 700, 98.5, 1000, 500000, true, false, true),

('google', 'gemini-pro-vision', ARRAY['text_generation', 'image_analysis', 'multimodal']::optimization.model_capability[],
 0.0000005, 0.0000015, 16384, 2048, 89.0, 900, 98.0, 800, 400000, true, false, true),

-- Cohere Models
('cohere', 'command-r', ARRAY['text_generation']::optimization.model_capability[],
 0.0000005, 0.0000015, 128000, 4096, 87.0, 600, 99.0, 1000, 600000, true, false, false),

('cohere', 'command-r-plus', ARRAY['text_generation']::optimization.model_capability[],
 0.000003, 0.000015, 128000, 4096, 91.0, 800, 98.8, 800, 400000, true, false, false),

-- Mistral Models
('mistral', 'mistral-large', ARRAY['text_generation', 'function_calling']::optimization.model_capability[],
 0.000008, 0.000024, 32768, 8192, 89.0, 700, 98.5, 600, 300000, true, true, false),

('mistral', 'mistral-medium', ARRAY['text_generation']::optimization.model_capability[],
 0.0000027, 0.0000081, 32768, 8192, 86.0, 500, 99.0, 1000, 500000, true, false, false);

-- Insert sample pricing data
INSERT INTO optimization.ai_pricing (provider, model_name, cost_per_input_token, cost_per_output_token)
SELECT provider, model_name, cost_per_input_token, cost_per_output_token
FROM optimization.ai_models;

-- Insert sample provider metrics
INSERT INTO optimization.provider_metrics (
    provider, success_rate, average_latency_ms, average_cost_per_request,
    total_requests_24h, total_cost_24h, quality_score
) VALUES
('openai', 0.995, 650, 0.0025, 15000, 37.50, 92.0),
('anthropic', 0.992, 950, 0.0035, 8000, 28.00, 94.0),
('google', 0.985, 750, 0.0018, 12000, 21.60, 89.0),
('cohere', 0.990, 550, 0.0015, 6000, 9.00, 87.0),
('mistral', 0.988, 600, 0.0020, 4000, 8.00, 88.0);

-- Create materialized view for fast analytics
CREATE MATERIALIZED VIEW optimization.daily_arbitrage_stats AS
SELECT
    DATE(ad.created_at) as date,
    ad.tenant_id,
    COUNT(*) as total_requests,
    AVG(ad.cost_savings_percent) as avg_cost_savings,
    SUM(ad.estimated_cost) as total_estimated_cost,
    SUM(ao.actual_cost) as total_actual_cost,
    AVG(ao.actual_latency_ms) as avg_latency_ms,
    COUNT(CASE WHEN ao.success THEN 1 END) * 100.0 / COUNT(*) as success_rate,
    COUNT(DISTINCT SPLIT_PART(ad.selected_model, ':', 1)) as unique_providers,
    MODE() WITHIN GROUP (ORDER BY ad.strategy) as most_used_strategy,
    MODE() WITHIN GROUP (ORDER BY SPLIT_PART(ad.selected_model, ':', 1)) as most_used_provider
FROM optimization.arbitrage_decisions ad
LEFT JOIN optimization.arbitrage_outcomes ao ON ad.request_id = ao.request_id
GROUP BY DATE(ad.created_at), ad.tenant_id;

-- Create index on materialized view
CREATE INDEX idx_daily_arbitrage_stats_date_tenant ON optimization.daily_arbitrage_stats(date DESC, tenant_id);

-- Grant permissions on materialized view
GRANT SELECT ON optimization.daily_arbitrage_stats TO agentsystem_api;

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION optimization.refresh_daily_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY optimization.daily_arbitrage_stats;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on function
GRANT EXECUTE ON FUNCTION optimization.refresh_daily_stats() TO agentsystem_api;
