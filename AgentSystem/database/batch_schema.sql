-- Batch Processing Database Schema - AgentSystem Profit Machine
-- Advanced batch optimization for bulk AI operations and cost reduction

-- Create batch schema
CREATE SCHEMA IF NOT EXISTS batch;

-- Batch strategy enum
CREATE TYPE batch.batch_strategy AS ENUM (
    'time_based', 'size_based', 'cost_based', 'latency_based', 'adaptive'
);

-- Batch priority enum
CREATE TYPE batch.batch_priority AS ENUM (
    'low', 'normal', 'high', 'critical'
);

-- Batch status enum
CREATE TYPE batch.batch_status AS ENUM (
    'queued', 'processing', 'completed', 'failed', 'cancelled', 'retrying'
);

-- Request status enum
CREATE TYPE batch.request_status AS ENUM (
    'pending', 'queued', 'processing', 'completed', 'failed', 'cancelled', 'retrying'
);

-- Batch requests table
CREATE TABLE batch.requests (
    request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    batch_id UUID,
    model VARCHAR(200) NOT NULL,
    capability VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    priority batch.batch_priority DEFAULT 'normal',
    status batch.request_status DEFAULT 'pending',
    max_latency_ms INTEGER,
    cost_budget DECIMAL(10,6),
    callback_url VARCHAR(500),
    metadata JSONB DEFAULT '{}',
    estimated_tokens INTEGER DEFAULT 0,
    estimated_cost DECIMAL(10,6) DEFAULT 0,
    actual_tokens INTEGER,
    actual_cost DECIMAL(10,6),
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    response_content TEXT,
    error_details TEXT,
    processing_time_ms INTEGER,
    deadline TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Batches table
CREATE TABLE batch.batches (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    model VARCHAR(200) NOT NULL,
    capability VARCHAR(100) NOT NULL,
    strategy batch.batch_strategy NOT NULL,
    status batch.batch_status DEFAULT 'queued',
    priority batch.batch_priority DEFAULT 'normal',
    batch_size INTEGER DEFAULT 0,
    estimated_cost DECIMAL(10,6) DEFAULT 0,
    estimated_tokens INTEGER DEFAULT 0,
    estimated_processing_time DECIMAL(8,2) DEFAULT 0,
    actual_cost DECIMAL(10,6),
    actual_tokens INTEGER,
    actual_processing_time DECIMAL(8,2),
    cost_savings DECIMAL(10,6) DEFAULT 0,
    cost_savings_percent DECIMAL(5,2) DEFAULT 0,
    throughput_improvement DECIMAL(5,2) DEFAULT 0,
    scheduled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_details TEXT,

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Batch configurations table
CREATE TABLE batch.configs (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    strategy batch.batch_strategy DEFAULT 'adaptive',
    max_batch_size INTEGER DEFAULT 50,
    min_batch_size INTEGER DEFAULT 2,
    max_wait_time_seconds INTEGER DEFAULT 30,
    cost_threshold DECIMAL(10,6) DEFAULT 10.0,
    enable_smart_grouping BOOLEAN DEFAULT true,
    enable_priority_queue BOOLEAN DEFAULT true,
    retry_failed_requests BOOLEAN DEFAULT true,
    parallel_batches INTEGER DEFAULT 3,
    cost_savings_target DECIMAL(5,2) DEFAULT 20.0,
    auto_optimization BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id)
);

-- Batch performance metrics table
CREATE TABLE batch.performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    batch_id UUID,
    metric_date DATE NOT NULL,
    total_requests INTEGER DEFAULT 0,
    total_batches INTEGER DEFAULT 0,
    avg_batch_size DECIMAL(5,2) DEFAULT 0,
    avg_processing_time DECIMAL(8,2) DEFAULT 0,
    total_cost_savings DECIMAL(12,6) DEFAULT 0,
    cost_savings_percent DECIMAL(5,2) DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 0,
    throughput_per_hour DECIMAL(8,2) DEFAULT 0,
    avg_latency_reduction DECIMAL(5,2) DEFAULT 0,
    efficiency_score DECIMAL(5,2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (batch_id) REFERENCES batch.batches(batch_id) ON DELETE SET NULL,
    UNIQUE(tenant_id, metric_date)
);

-- Batch optimization history table
CREATE TABLE batch.optimization_history (
    optimization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    optimization_type VARCHAR(100) NOT NULL, -- config_update, strategy_change, size_optimization
    old_config JSONB,
    new_config JSONB,
    performance_before JSONB,
    performance_after JSONB,
    improvement_percent DECIMAL(5,2),
    optimization_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Batch queue monitoring table
CREATE TABLE batch.queue_metrics (
    queue_metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    priority batch.batch_priority NOT NULL,
    queue_size INTEGER DEFAULT 0,
    avg_wait_time_seconds DECIMAL(8,2) DEFAULT 0,
    max_wait_time_seconds DECIMAL(8,2) DEFAULT 0,
    throughput_per_minute DECIMAL(8,2) DEFAULT 0,
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Batch processing logs table
CREATE TABLE batch.processing_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    log_level VARCHAR(20) NOT NULL, -- INFO, WARN, ERROR, DEBUG
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    processing_stage VARCHAR(50), -- queuing, grouping, processing, completion
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (batch_id) REFERENCES batch.batches(batch_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Batch cost analysis table
CREATE TABLE batch.cost_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    batch_id UUID NOT NULL,
    individual_cost DECIMAL(12,6) NOT NULL, -- Cost if processed individually
    batch_cost DECIMAL(12,6) NOT NULL,      -- Actual batch cost
    cost_savings DECIMAL(12,6) NOT NULL,    -- Absolute savings
    savings_percent DECIMAL(5,2) NOT NULL,  -- Percentage savings
    provider_discounts DECIMAL(12,6) DEFAULT 0,
    efficiency_gains DECIMAL(12,6) DEFAULT 0,
    overhead_costs DECIMAL(12,6) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (batch_id) REFERENCES batch.batches(batch_id) ON DELETE CASCADE
);

-- Batch scheduling table
CREATE TABLE batch.schedules (
    schedule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    schedule_name VARCHAR(200) NOT NULL,
    cron_expression VARCHAR(100) NOT NULL,
    batch_config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    last_run TIMESTAMPTZ,
    next_run TIMESTAMPTZ,
    total_runs INTEGER DEFAULT 0,
    successful_runs INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX idx_batch_requests_tenant ON batch.requests(tenant_id);
CREATE INDEX idx_batch_requests_status ON batch.requests(status);
CREATE INDEX idx_batch_requests_priority ON batch.requests(priority DESC);
CREATE INDEX idx_batch_requests_created_at ON batch.requests(created_at DESC);
CREATE INDEX idx_batch_requests_deadline ON batch.requests(deadline);
CREATE INDEX idx_batch_requests_batch_id ON batch.requests(batch_id);
CREATE INDEX idx_batch_requests_model ON batch.requests(tenant_id, model);

CREATE INDEX idx_batches_tenant ON batch.batches(tenant_id);
CREATE INDEX idx_batches_status ON batch.batches(status);
CREATE INDEX idx_batches_priority ON batch.batches(priority DESC);
CREATE INDEX idx_batches_created_at ON batch.batches(created_at DESC);
CREATE INDEX idx_batches_scheduled_at ON batch.batches(scheduled_at);
CREATE INDEX idx_batches_model ON batch.batches(tenant_id, model);

CREATE INDEX idx_batch_configs_tenant ON batch.configs(tenant_id);

CREATE INDEX idx_performance_metrics_tenant_date ON batch.performance_metrics(tenant_id, metric_date DESC);
CREATE INDEX idx_performance_metrics_batch ON batch.performance_metrics(batch_id);

CREATE INDEX idx_optimization_history_tenant ON batch.optimization_history(tenant_id);
CREATE INDEX idx_optimization_history_created_at ON batch.optimization_history(created_at DESC);

CREATE INDEX idx_queue_metrics_tenant_priority ON batch.queue_metrics(tenant_id, priority);
CREATE INDEX idx_queue_metrics_timestamp ON batch.queue_metrics(timestamp DESC);

CREATE INDEX idx_processing_logs_batch ON batch.processing_logs(batch_id);
CREATE INDEX idx_processing_logs_tenant ON batch.processing_logs(tenant_id);
CREATE INDEX idx_processing_logs_timestamp ON batch.processing_logs(timestamp DESC);
CREATE INDEX idx_processing_logs_level ON batch.processing_logs(log_level);

CREATE INDEX idx_cost_analysis_tenant ON batch.cost_analysis(tenant_id);
CREATE INDEX idx_cost_analysis_batch ON batch.cost_analysis(batch_id);
CREATE INDEX idx_cost_analysis_savings ON batch.cost_analysis(savings_percent DESC);

CREATE INDEX idx_batch_schedules_tenant ON batch.schedules(tenant_id);
CREATE INDEX idx_batch_schedules_active ON batch.schedules(is_active);
CREATE INDEX idx_batch_schedules_next_run ON batch.schedules(next_run);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION batch.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_batch_requests_updated_at BEFORE UPDATE ON batch.requests FOR EACH ROW EXECUTE FUNCTION batch.update_updated_at_column();
CREATE TRIGGER update_batch_configs_updated_at BEFORE UPDATE ON batch.configs FOR EACH ROW EXECUTE FUNCTION batch.update_updated_at_column();
CREATE TRIGGER update_batch_schedules_updated_at BEFORE UPDATE ON batch.schedules FOR EACH ROW EXECUTE FUNCTION batch.update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE batch.requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE batch.batches ENABLE ROW LEVEL SECURITY;
ALTER TABLE batch.configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE batch.performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE batch.optimization_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE batch.queue_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE batch.processing_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE batch.cost_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE batch.schedules ENABLE ROW LEVEL SECURITY;

-- RLS policies for tenant isolation
CREATE POLICY batch_requests_tenant_isolation ON batch.requests
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY batch_batches_tenant_isolation ON batch.batches
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY batch_configs_tenant_isolation ON batch.configs
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY batch_performance_metrics_tenant_isolation ON batch.performance_metrics
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY batch_optimization_history_tenant_isolation ON batch.optimization_history
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY batch_queue_metrics_tenant_isolation ON batch.queue_metrics
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY batch_processing_logs_tenant_isolation ON batch.processing_logs
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY batch_cost_analysis_tenant_isolation ON batch.cost_analysis
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY batch_schedules_tenant_isolation ON batch.schedules
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant permissions
GRANT USAGE ON SCHEMA batch TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA batch TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA batch TO agentsystem_api;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA batch TO agentsystem_api;

-- Create views for analytics and reporting
CREATE VIEW batch.batch_efficiency_summary AS
SELECT
    b.tenant_id,
    DATE(b.created_at) as date,
    COUNT(*) as total_batches,
    AVG(b.batch_size) as avg_batch_size,
    AVG(b.actual_processing_time) as avg_processing_time,
    SUM(b.cost_savings) as total_cost_savings,
    AVG(b.cost_savings_percent) as avg_cost_savings_percent,
    COUNT(CASE WHEN b.status = 'completed' THEN 1 END) * 100.0 / COUNT(*) as success_rate,
    SUM(b.batch_size) / EXTRACT(EPOCH FROM MAX(b.completed_at) - MIN(b.started_at)) * 3600 as throughput_per_hour
FROM batch.batches b
WHERE b.status IN ('completed', 'failed')
GROUP BY b.tenant_id, DATE(b.created_at);

CREATE VIEW batch.request_processing_summary AS
SELECT
    r.tenant_id,
    r.model,
    r.capability,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN r.status = 'completed' THEN 1 END) as completed_requests,
    COUNT(CASE WHEN r.status = 'failed' THEN 1 END) as failed_requests,
    AVG(r.processing_time_ms) as avg_processing_time_ms,
    AVG(r.actual_cost) as avg_cost,
    SUM(r.actual_cost) as total_cost,
    AVG(EXTRACT(EPOCH FROM (r.completed_at - r.created_at))) as avg_total_latency_seconds
FROM batch.requests r
WHERE r.status IN ('completed', 'failed')
GROUP BY r.tenant_id, r.model, r.capability;

CREATE VIEW batch.cost_savings_analysis AS
SELECT
    ca.tenant_id,
    DATE(ca.created_at) as date,
    COUNT(*) as total_batches,
    SUM(ca.individual_cost) as total_individual_cost,
    SUM(ca.batch_cost) as total_batch_cost,
    SUM(ca.cost_savings) as total_cost_savings,
    AVG(ca.savings_percent) as avg_savings_percent,
    SUM(ca.provider_discounts) as total_provider_discounts,
    SUM(ca.efficiency_gains) as total_efficiency_gains
FROM batch.cost_analysis ca
GROUP BY ca.tenant_id, DATE(ca.created_at);

CREATE VIEW batch.queue_performance AS
SELECT
    qm.tenant_id,
    qm.priority,
    AVG(qm.queue_size) as avg_queue_size,
    MAX(qm.queue_size) as max_queue_size,
    AVG(qm.avg_wait_time_seconds) as avg_wait_time,
    MAX(qm.max_wait_time_seconds) as max_wait_time,
    AVG(qm.throughput_per_minute) as avg_throughput_per_minute,
    COUNT(*) as measurement_count
FROM batch.queue_metrics qm
WHERE qm.timestamp > NOW() - INTERVAL '24 hours'
GROUP BY qm.tenant_id, qm.priority;

-- Grant permissions on views
GRANT SELECT ON batch.batch_efficiency_summary TO agentsystem_api;
GRANT SELECT ON batch.request_processing_summary TO agentsystem_api;
GRANT SELECT ON batch.cost_savings_analysis TO agentsystem_api;
GRANT SELECT ON batch.queue_performance TO agentsystem_api;

-- Create materialized view for fast dashboard queries
CREATE MATERIALIZED VIEW batch.tenant_dashboard_stats AS
SELECT
    r.tenant_id,
    COUNT(DISTINCT r.request_id) as total_requests,
    COUNT(DISTINCT b.batch_id) as total_batches,
    AVG(b.batch_size) as avg_batch_size,
    SUM(CASE WHEN r.status = 'completed' THEN 1 ELSE 0 END) as completed_requests,
    SUM(CASE WHEN r.status = 'failed' THEN 1 ELSE 0 END) as failed_requests,
    SUM(CASE WHEN r.status = 'pending' THEN 1 ELSE 0 END) as pending_requests,
    SUM(CASE WHEN r.status = 'processing' THEN 1 ELSE 0 END) as processing_requests,
    AVG(r.processing_time_ms) as avg_processing_time_ms,
    SUM(b.cost_savings) as total_cost_savings,
    AVG(b.cost_savings_percent) as avg_cost_savings_percent,
    COUNT(DISTINCT r.model) as unique_models,
    MAX(r.created_at) as last_request_at,
    COUNT(CASE WHEN r.created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as requests_last_24h
FROM batch.requests r
LEFT JOIN batch.batches b ON r.batch_id = b.batch_id
GROUP BY r.tenant_id;

-- Create index on materialized view
CREATE INDEX idx_tenant_dashboard_stats_tenant ON batch.tenant_dashboard_stats(tenant_id);

-- Grant permissions on materialized view
GRANT SELECT ON batch.tenant_dashboard_stats TO agentsystem_api;

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION batch.refresh_dashboard_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY batch.tenant_dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on function
GRANT EXECUTE ON FUNCTION batch.refresh_dashboard_stats() TO agentsystem_api;

-- Function to calculate batch efficiency score
CREATE OR REPLACE FUNCTION batch.calculate_batch_efficiency(p_tenant_id UUID, p_days INTEGER DEFAULT 7)
RETURNS TABLE (
    efficiency_score DECIMAL(5,2),
    cost_savings_score DECIMAL(5,2),
    throughput_score DECIMAL(5,2),
    latency_score DECIMAL(5,2),
    recommendation TEXT
) AS $$
DECLARE
    v_cost_savings DECIMAL(5,2);
    v_throughput DECIMAL(8,2);
    v_avg_latency DECIMAL(8,2);
    v_efficiency_score DECIMAL(5,2);
    v_recommendation TEXT;
BEGIN
    -- Get batch performance metrics
    SELECT
        COALESCE(AVG(cost_savings_percent), 0),
        COALESCE(AVG(throughput_per_hour), 0),
        COALESCE(AVG(avg_processing_time), 0)
    INTO v_cost_savings, v_throughput, v_avg_latency
    FROM batch.batch_efficiency_summary
    WHERE tenant_id = p_tenant_id
    AND date > CURRENT_DATE - INTERVAL '%s days' % p_days;

    -- Calculate component scores (0-100)
    v_cost_savings := LEAST(v_cost_savings * 2, 100); -- Scale cost savings

    -- Calculate efficiency score
    v_efficiency_score := (v_cost_savings * 0.4 +
                          LEAST(v_throughput / 10, 100) * 0.3 +
                          GREATEST(100 - v_avg_latency, 0) * 0.3);

    -- Generate recommendation
    IF v_cost_savings < 10 THEN
        v_recommendation := 'Low cost savings detected. Consider increasing batch sizes or enabling smart grouping.';
    ELSIF v_throughput < 50 THEN
        v_recommendation := 'Low throughput detected. Consider optimizing batch processing strategy.';
    ELSIF v_efficiency_score > 80 THEN
        v_recommendation := 'Excellent batch performance. Consider expanding to more use cases.';
    ELSE
        v_recommendation := 'Good batch performance. Monitor and optimize based on usage patterns.';
    END IF;

    RETURN QUERY SELECT
        v_efficiency_score,
        v_cost_savings,
        LEAST(v_throughput / 10, 100),
        GREATEST(100 - v_avg_latency, 0),
        v_recommendation;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on efficiency function
GRANT EXECUTE ON FUNCTION batch.calculate_batch_efficiency TO agentsystem_api;

-- Insert default batch configurations for existing tenants
INSERT INTO batch.configs (tenant_id, strategy)
SELECT tenant_id, 'adaptive'
FROM billing.tenants
WHERE tenant_id NOT IN (SELECT tenant_id FROM batch.configs);

-- Create function to automatically update batch size based on performance
CREATE OR REPLACE FUNCTION batch.auto_optimize_batch_size()
RETURNS void AS $$
DECLARE
    tenant_record RECORD;
    performance_data RECORD;
    new_batch_size INTEGER;
BEGIN
    -- Loop through all tenants with batch configs
    FOR tenant_record IN
        SELECT tenant_id FROM batch.configs WHERE auto_optimization = true
    LOOP
        -- Get recent performance data
        SELECT
            AVG(batch_size) as avg_batch_size,
            AVG(cost_savings_percent) as avg_savings,
            AVG(actual_processing_time) as avg_time
        INTO performance_data
        FROM batch.batches
        WHERE tenant_id = tenant_record.tenant_id
        AND created_at > NOW() - INTERVAL '7 days'
        AND status = 'completed';

        -- Optimize batch size based on performance
        IF performance_data.avg_savings < 15 AND performance_data.avg_batch_size < 40 THEN
            new_batch_size := LEAST(performance_data.avg_batch_size * 1.2, 50);
        ELSIF performance_data.avg_time > 30 AND performance_data.avg_batch_size > 10 THEN
            new_batch_size := GREATEST(performance_data.avg_batch_size * 0.8, 5);
        ELSE
            CONTINUE; -- No optimization needed
        END IF;

        -- Update configuration
        UPDATE batch.configs
        SET max_batch_size = new_batch_size,
            updated_at = NOW()
        WHERE tenant_id = tenant_record.tenant_id;

        -- Log optimization
        INSERT INTO batch.optimization_history (
            tenant_id, optimization_type, old_config, new_config,
            optimization_reason
        ) VALUES (
            tenant_record.tenant_id, 'size_optimization',
            jsonb_build_object('max_batch_size', performance_data.avg_batch_size),
            jsonb_build_object('max_batch_size', new_batch_size),
            'Automatic optimization based on performance metrics'
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on auto optimization function
GRANT EXECUTE ON FUNCTION batch.auto_optimize_batch_size() TO agentsystem_api;
