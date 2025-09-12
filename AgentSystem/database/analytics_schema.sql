-- AgentSystem Advanced Analytics Database Schema
-- Comprehensive analytics and business intelligence tables

-- Analytics metrics storage
CREATE TABLE analytics_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    change_percent DECIMAL(8,4),
    trend VARCHAR(20) CHECK (trend IN ('up', 'down', 'stable')),
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    category VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_analytics_tenant_date (tenant_id, created_at),
    INDEX idx_analytics_category (category),
    INDEX idx_analytics_metric_name (metric_name)
);

-- Business insights storage
CREATE TABLE business_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    impact_level VARCHAR(20) CHECK (impact_level IN ('high', 'medium', 'low')),
    category VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(3,2) CHECK (confidence_score BETWEEN 0 AND 1),
    data_points JSONB,
    recommendations JSONB,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'resolved', 'dismissed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_insights_tenant_status (tenant_id, status),
    INDEX idx_insights_impact (impact_level),
    INDEX idx_insights_category (category)
);

-- Dashboard configurations
CREATE TABLE dashboard_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    dashboard_name VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(tenant_id, dashboard_name),
    INDEX idx_dashboard_tenant (tenant_id)
);

-- Real-time monitoring sessions
CREATE TABLE realtime_monitoring (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_id VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'stopped')),
    update_interval INTEGER DEFAULT 60,
    last_update TIMESTAMP,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(tenant_id, session_id),
    INDEX idx_monitoring_status (status)
);

-- Analytics alerts
CREATE TABLE analytics_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    alert_type VARCHAR(20) CHECK (alert_type IN ('critical', 'warning', 'info')),
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    action_required BOOLEAN DEFAULT FALSE,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolved_by UUID REFERENCES users(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_alerts_tenant_status (tenant_id, resolved),
    INDEX idx_alerts_type (alert_type),
    INDEX idx_alerts_action_required (action_required)
);

-- Predictive forecasts
CREATE TABLE predictive_forecasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    forecast_type VARCHAR(50) NOT NULL,
    forecast_period_days INTEGER NOT NULL,
    forecast_data JSONB NOT NULL,
    confidence_level DECIMAL(3,2),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMP,

    INDEX idx_forecasts_tenant_type (tenant_id, forecast_type),
    INDEX idx_forecasts_valid (valid_until)
);

-- System health metrics
CREATE TABLE system_health_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    uptime_percentage DECIMAL(5,2),
    response_time_avg DECIMAL(8,2),
    error_rate DECIMAL(5,4),
    throughput_requests_per_second DECIMAL(10,2),
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_percent DECIMAL(5,2),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_health_tenant_time (tenant_id, timestamp),
    INDEX idx_health_timestamp (timestamp)
);

-- API performance metrics
CREATE TABLE api_performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    response_time_ms DECIMAL(8,2) NOT NULL,
    status_code INTEGER NOT NULL,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_api_perf_tenant_endpoint (tenant_id, endpoint),
    INDEX idx_api_perf_timestamp (timestamp),
    INDEX idx_api_perf_status (status_code)
);

-- Support tickets for satisfaction tracking
CREATE TABLE support_tickets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    subject VARCHAR(200) NOT NULL,
    description TEXT,
    priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
    satisfaction_score INTEGER CHECK (satisfaction_score BETWEEN 1 AND 5),
    satisfaction_feedback TEXT,
    assigned_to UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,

    INDEX idx_tickets_tenant_status (tenant_id, status),
    INDEX idx_tickets_satisfaction (satisfaction_score),
    INDEX idx_tickets_priority (priority)
);

-- Revenue analytics materialized view for performance
CREATE MATERIALIZED VIEW revenue_analytics_daily AS
SELECT
    tenant_id,
    DATE(created_at) as date,
    SUM(amount) as daily_revenue,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_transaction_amount
FROM billing_transactions
WHERE status = 'completed'
GROUP BY tenant_id, DATE(created_at);

-- Usage analytics materialized view
CREATE MATERIALIZED VIEW usage_analytics_daily AS
SELECT
    tenant_id,
    DATE(timestamp) as date,
    SUM(api_calls) as daily_api_calls,
    SUM(ai_tokens) as daily_ai_tokens,
    SUM(cost) as daily_cost,
    COUNT(DISTINCT user_id) as active_users
FROM usage_tracking
GROUP BY tenant_id, DATE(timestamp);

-- Create indexes on materialized views
CREATE INDEX idx_revenue_analytics_tenant_date ON revenue_analytics_daily(tenant_id, date);
CREATE INDEX idx_usage_analytics_tenant_date ON usage_analytics_daily(tenant_id, date);

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY revenue_analytics_daily;
    REFRESH MATERIALIZED VIEW CONCURRENTLY usage_analytics_daily;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update analytics metrics when new data arrives
CREATE OR REPLACE FUNCTION update_analytics_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update real-time metrics when billing transactions are added
    IF TG_TABLE_NAME = 'billing_transactions' AND NEW.status = 'completed' THEN
        INSERT INTO analytics_metrics (tenant_id, metric_name, metric_value, category, period_start, period_end)
        VALUES (NEW.tenant_id, 'daily_revenue', NEW.amount, 'revenue',
                DATE_TRUNC('day', NEW.created_at), DATE_TRUNC('day', NEW.created_at) + INTERVAL '1 day');
    END IF;

    -- Update usage metrics
    IF TG_TABLE_NAME = 'usage_tracking' THEN
        INSERT INTO analytics_metrics (tenant_id, metric_name, metric_value, category, period_start, period_end)
        VALUES (NEW.tenant_id, 'daily_api_calls', NEW.api_calls, 'usage',
                DATE_TRUNC('day', NEW.timestamp), DATE_TRUNC('day', NEW.timestamp) + INTERVAL '1 day');
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER billing_analytics_trigger
    AFTER INSERT ON billing_transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_analytics_metrics();

CREATE TRIGGER usage_analytics_trigger
    AFTER INSERT ON usage_tracking
    FOR EACH ROW
    EXECUTE FUNCTION update_analytics_metrics();

-- Function to calculate customer lifetime value
CREATE OR REPLACE FUNCTION calculate_clv(p_tenant_id UUID, p_customer_id UUID)
RETURNS DECIMAL(15,2) AS $$
DECLARE
    avg_monthly_revenue DECIMAL(15,2);
    customer_lifespan_months INTEGER;
    clv DECIMAL(15,2);
BEGIN
    -- Calculate average monthly revenue for customer
    SELECT COALESCE(AVG(monthly_revenue), 0)
    INTO avg_monthly_revenue
    FROM (
        SELECT
            DATE_TRUNC('month', created_at) as month,
            SUM(amount) as monthly_revenue
        FROM billing_transactions
        WHERE tenant_id = p_tenant_id
        AND customer_id = p_customer_id
        AND status = 'completed'
        GROUP BY DATE_TRUNC('month', created_at)
    ) monthly_data;

    -- Estimate customer lifespan (simplified)
    SELECT COALESCE(
        EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / (30 * 24 * 3600), 1
    )
    INTO customer_lifespan_months
    FROM billing_transactions
    WHERE tenant_id = p_tenant_id
    AND customer_id = p_customer_id;

    -- Calculate CLV
    clv := avg_monthly_revenue * customer_lifespan_months;

    RETURN clv;
END;
$$ LANGUAGE plpgsql;

-- Function to detect anomalies in metrics
CREATE OR REPLACE FUNCTION detect_metric_anomalies(p_tenant_id UUID, p_metric_name VARCHAR(100))
RETURNS TABLE(anomaly_date DATE, anomaly_value DECIMAL(15,4), expected_range_min DECIMAL(15,4), expected_range_max DECIMAL(15,4)) AS $$
BEGIN
    RETURN QUERY
    WITH metric_stats AS (
        SELECT
            AVG(metric_value) as avg_value,
            STDDEV(metric_value) as stddev_value
        FROM analytics_metrics
        WHERE tenant_id = p_tenant_id
        AND metric_name = p_metric_name
        AND created_at >= CURRENT_DATE - INTERVAL '30 days'
    ),
    daily_metrics AS (
        SELECT
            DATE(created_at) as metric_date,
            AVG(metric_value) as daily_value
        FROM analytics_metrics
        WHERE tenant_id = p_tenant_id
        AND metric_name = p_metric_name
        AND created_at >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY DATE(created_at)
    )
    SELECT
        dm.metric_date,
        dm.daily_value,
        ms.avg_value - (2 * ms.stddev_value) as min_range,
        ms.avg_value + (2 * ms.stddev_value) as max_range
    FROM daily_metrics dm
    CROSS JOIN metric_stats ms
    WHERE dm.daily_value < (ms.avg_value - (2 * ms.stddev_value))
       OR dm.daily_value > (ms.avg_value + (2 * ms.stddev_value));
END;
$$ LANGUAGE plpgsql;

-- Create scheduled job to refresh analytics views (requires pg_cron extension)
-- SELECT cron.schedule('refresh-analytics', '0 1 * * *', 'SELECT refresh_analytics_views();');

COMMENT ON TABLE analytics_metrics IS 'Stores calculated analytics metrics for business intelligence';
COMMENT ON TABLE business_insights IS 'AI-generated business insights and recommendations';
COMMENT ON TABLE dashboard_configs IS 'Custom dashboard configurations for tenants';
COMMENT ON TABLE realtime_monitoring IS 'Real-time monitoring session management';
COMMENT ON TABLE analytics_alerts IS 'System-generated alerts for critical business metrics';
COMMENT ON TABLE predictive_forecasts IS 'Predictive forecasting data for business planning';
COMMENT ON TABLE system_health_metrics IS 'System performance and health monitoring';
COMMENT ON TABLE api_performance_metrics IS 'API endpoint performance tracking';
COMMENT ON TABLE support_tickets IS 'Customer support tickets for satisfaction tracking';
