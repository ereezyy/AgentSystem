-- Competitive Intelligence Database Schema - AgentSystem Profit Machine
-- Advanced competitive analysis and market intelligence system

-- Create intelligence schema
CREATE SCHEMA IF NOT EXISTS intelligence;

-- Competitor tier enum
CREATE TYPE intelligence.competitor_tier AS ENUM (
    'direct', 'indirect', 'substitute', 'adjacent'
);

-- Intelligence type enum
CREATE TYPE intelligence.intelligence_type AS ENUM (
    'pricing', 'features', 'marketing', 'funding', 'hiring',
    'partnerships', 'product_updates', 'customer_reviews'
);

-- Monitoring frequency enum
CREATE TYPE intelligence.monitoring_frequency AS ENUM (
    'daily', 'weekly', 'monthly', 'quarterly'
);

-- Threat level enum
CREATE TYPE intelligence.threat_level AS ENUM (
    'low', 'medium', 'high', 'critical'
);

-- Competitors table
CREATE TABLE intelligence.competitors (
    competitor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(200) NOT NULL,
    website VARCHAR(500),
    tier intelligence.competitor_tier NOT NULL DEFAULT 'direct',
    market_cap DECIMAL(15,2),
    funding_raised DECIMAL(15,2),
    employee_count INTEGER,
    founded_year INTEGER,
    headquarters VARCHAR(200),
    key_products JSONB DEFAULT '[]',
    target_markets JSONB DEFAULT '[]',
    key_executives JSONB DEFAULT '[]',
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, name)
);

-- Competitive intelligence table
CREATE TABLE intelligence.competitive_intelligence (
    intelligence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competitor_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    intelligence_type intelligence.intelligence_type NOT NULL,
    title VARCHAR(300) NOT NULL,
    summary TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    source_url VARCHAR(1000),
    source_type VARCHAR(100) NOT NULL, -- website, news, social, api, manual
    confidence_score DECIMAL(3,2) DEFAULT 0.5 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    threat_level intelligence.threat_level DEFAULT 'medium',
    impact_assessment TEXT,
    recommended_actions JSONB DEFAULT '[]',
    detected_date TIMESTAMPTZ DEFAULT NOW(),
    expiry_date TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'active', -- active, archived, dismissed
    reviewed_by VARCHAR(200),
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (competitor_id) REFERENCES intelligence.competitors(competitor_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Market trends table
CREATE TABLE intelligence.market_trends (
    trend_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    trend_category VARCHAR(100) NOT NULL, -- pricing, features, market_size, customer_behavior
    trend_title VARCHAR(300) NOT NULL,
    trend_description TEXT NOT NULL,
    trend_direction VARCHAR(20) NOT NULL, -- up, down, stable
    confidence_level DECIMAL(3,2) DEFAULT 0.5 CHECK (confidence_level >= 0 AND confidence_level <= 1),
    supporting_data JSONB DEFAULT '{}',
    impact_on_business TEXT,
    strategic_implications JSONB DEFAULT '[]',
    identified_date TIMESTAMPTZ DEFAULT NOW(),
    expiry_date TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Monitoring configurations table
CREATE TABLE intelligence.monitoring_configs (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    competitor_id UUID NOT NULL,
    monitoring_frequency intelligence.monitoring_frequency DEFAULT 'daily',
    intelligence_types JSONB DEFAULT '[]', -- Types of intelligence to monitor
    is_active BOOLEAN DEFAULT true,
    last_run TIMESTAMPTZ,
    next_run TIMESTAMPTZ,
    run_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    monitoring_rules JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (competitor_id) REFERENCES intelligence.competitors(competitor_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, competitor_id)
);

-- Competitor alerts table
CREATE TABLE intelligence.competitor_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    competitor_id UUID NOT NULL,
    alert_type VARCHAR(100) NOT NULL, -- price_change, feature_release, funding, partnership
    title VARCHAR(300) NOT NULL,
    message TEXT NOT NULL,
    severity intelligence.threat_level DEFAULT 'medium',
    alert_triggers JSONB NOT NULL, -- Conditions that triggered the alert
    triggered_by_intelligence UUID, -- Reference to intelligence item
    status VARCHAR(50) DEFAULT 'active', -- active, acknowledged, resolved, dismissed
    acknowledged_by VARCHAR(200),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (competitor_id) REFERENCES intelligence.competitors(competitor_id) ON DELETE CASCADE,
    FOREIGN KEY (triggered_by_intelligence) REFERENCES intelligence.competitive_intelligence(intelligence_id) ON DELETE SET NULL
);

-- Competitive reports table
CREATE TABLE intelligence.competitive_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    report_name VARCHAR(200) NOT NULL,
    report_type VARCHAR(100) DEFAULT 'comprehensive', -- comprehensive, pricing, features, threats
    analysis_period_days INTEGER NOT NULL,
    competitors_analyzed INTEGER DEFAULT 0,
    intelligence_items_analyzed INTEGER DEFAULT 0,
    executive_summary TEXT,
    competitive_score DECIMAL(5,2) DEFAULT 50,
    market_position VARCHAR(100),
    key_findings JSONB DEFAULT '[]',
    strategic_recommendations JSONB DEFAULT '[]',
    threat_assessment JSONB DEFAULT '{}',
    opportunities JSONB DEFAULT '[]',
    market_trends JSONB DEFAULT '[]',
    report_data JSONB, -- Full report data
    generated_date TIMESTAMPTZ DEFAULT NOW(),
    generated_by VARCHAR(200),
    is_public BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Competitor pricing history table
CREATE TABLE intelligence.competitor_pricing_history (
    pricing_history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    competitor_id UUID NOT NULL,
    product_tier VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    pricing_model VARCHAR(100), -- monthly, annual, usage_based, hybrid
    features_included JSONB DEFAULT '[]',
    promotional_offer VARCHAR(200),
    effective_date TIMESTAMPTZ DEFAULT NOW(),
    source_url VARCHAR(1000),
    confidence_score DECIMAL(3,2) DEFAULT 0.8,
    data_collection_method VARCHAR(100), -- scraping, manual, api
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (competitor_id) REFERENCES intelligence.competitors(competitor_id) ON DELETE CASCADE
);

-- Market analysis jobs table
CREATE TABLE intelligence.analysis_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    job_type VARCHAR(100) NOT NULL, -- landscape_analysis, competitor_monitoring, trend_analysis
    job_config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'queued', -- queued, running, completed, failed
    progress DECIMAL(5,2) DEFAULT 0, -- 0-100%
    results JSONB DEFAULT '{}',
    error_details TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Competitive insights table
CREATE TABLE intelligence.competitive_insights (
    insight_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    insight_category VARCHAR(100) NOT NULL, -- opportunity, threat, trend, gap
    title VARCHAR(300) NOT NULL,
    description TEXT NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium', -- low, medium, high, critical
    estimated_impact VARCHAR(100), -- revenue, market_share, competitive_position
    action_required BOOLEAN DEFAULT false,
    recommended_actions JSONB DEFAULT '[]',
    supporting_intelligence JSONB DEFAULT '[]', -- References to intelligence items
    insight_date TIMESTAMPTZ DEFAULT NOW(),
    expiry_date TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Competitor feature matrix table
CREATE TABLE intelligence.competitor_features (
    feature_matrix_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    competitor_id UUID NOT NULL,
    feature_category VARCHAR(100) NOT NULL,
    feature_name VARCHAR(200) NOT NULL,
    has_feature BOOLEAN DEFAULT false,
    feature_quality_score DECIMAL(3,2) DEFAULT 0, -- 0-1 quality assessment
    feature_description TEXT,
    pricing_tier_required VARCHAR(100),
    launch_date TIMESTAMPTZ,
    source_verification VARCHAR(200),
    last_verified TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (competitor_id) REFERENCES intelligence.competitors(competitor_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, competitor_id, feature_name)
);

-- Create indexes for performance
CREATE INDEX idx_competitors_tenant ON intelligence.competitors(tenant_id);
CREATE INDEX idx_competitors_tier ON intelligence.competitors(tier);
CREATE INDEX idx_competitors_active ON intelligence.competitors(is_active);

CREATE INDEX idx_competitive_intelligence_tenant ON intelligence.competitive_intelligence(tenant_id);
CREATE INDEX idx_competitive_intelligence_competitor ON intelligence.competitive_intelligence(competitor_id);
CREATE INDEX idx_competitive_intelligence_type ON intelligence.competitive_intelligence(intelligence_type);
CREATE INDEX idx_competitive_intelligence_threat ON intelligence.competitive_intelligence(threat_level);
CREATE INDEX idx_competitive_intelligence_date ON intelligence.competitive_intelligence(detected_date DESC);

CREATE INDEX idx_market_trends_tenant ON intelligence.market_trends(tenant_id);
CREATE INDEX idx_market_trends_category ON intelligence.market_trends(trend_category);
CREATE INDEX idx_market_trends_date ON intelligence.market_trends(identified_date DESC);

CREATE INDEX idx_monitoring_configs_tenant ON intelligence.monitoring_configs(tenant_id);
CREATE INDEX idx_monitoring_configs_competitor ON intelligence.monitoring_configs(competitor_id);
CREATE INDEX idx_monitoring_configs_active ON intelligence.monitoring_configs(is_active);
CREATE INDEX idx_monitoring_configs_next_run ON intelligence.monitoring_configs(next_run);

CREATE INDEX idx_competitor_alerts_tenant ON intelligence.competitor_alerts(tenant_id);
CREATE INDEX idx_competitor_alerts_competitor ON intelligence.competitor_alerts(competitor_id);
CREATE INDEX idx_competitor_alerts_status ON intelligence.competitor_alerts(status);
CREATE INDEX idx_competitor_alerts_severity ON intelligence.competitor_alerts(severity);

CREATE INDEX idx_competitive_reports_tenant ON intelligence.competitive_reports(tenant_id);
CREATE INDEX idx_competitive_reports_date ON intelligence.competitive_reports(generated_date DESC);
CREATE INDEX idx_competitive_reports_type ON intelligence.competitive_reports(report_type);

CREATE INDEX idx_competitor_pricing_history_tenant ON intelligence.competitor_pricing_history(tenant_id);
CREATE INDEX idx_competitor_pricing_history_competitor ON intelligence.competitor_pricing_history(competitor_id);
CREATE INDEX idx_competitor_pricing_history_date ON intelligence.competitor_pricing_history(effective_date DESC);

CREATE INDEX idx_analysis_jobs_tenant ON intelligence.analysis_jobs(tenant_id);
CREATE INDEX idx_analysis_jobs_status ON intelligence.analysis_jobs(status);
CREATE INDEX idx_analysis_jobs_type ON intelligence.analysis_jobs(job_type);

CREATE INDEX idx_competitive_insights_tenant ON intelligence.competitive_insights(tenant_id);
CREATE INDEX idx_competitive_insights_category ON intelligence.competitive_insights(insight_category);
CREATE INDEX idx_competitive_insights_priority ON intelligence.competitive_insights(priority);
CREATE INDEX idx_competitive_insights_date ON intelligence.competitive_insights(insight_date DESC);

CREATE INDEX idx_competitor_features_tenant ON intelligence.competitor_features(tenant_id);
CREATE INDEX idx_competitor_features_competitor ON intelligence.competitor_features(competitor_id);
CREATE INDEX idx_competitor_features_category ON intelligence.competitor_features(feature_category);

-- Create updated_at trigger function (reuse existing one)
CREATE TRIGGER update_monitoring_configs_updated_at
    BEFORE UPDATE ON intelligence.monitoring_configs
    FOR EACH ROW EXECUTE FUNCTION analytics.update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE intelligence.competitors ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.competitive_intelligence ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.market_trends ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.monitoring_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.competitor_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.competitive_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.competitor_pricing_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.analysis_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.competitive_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence.competitor_features ENABLE ROW LEVEL SECURITY;

-- RLS policies for tenant isolation
CREATE POLICY competitors_tenant_isolation ON intelligence.competitors
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY competitive_intelligence_tenant_isolation ON intelligence.competitive_intelligence
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY market_trends_tenant_isolation ON intelligence.market_trends
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY monitoring_configs_tenant_isolation ON intelligence.monitoring_configs
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY competitor_alerts_tenant_isolation ON intelligence.competitor_alerts
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY competitive_reports_tenant_isolation ON intelligence.competitive_reports
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY competitor_pricing_history_tenant_isolation ON intelligence.competitor_pricing_history
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY analysis_jobs_tenant_isolation ON intelligence.analysis_jobs
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY competitive_insights_tenant_isolation ON intelligence.competitive_insights
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY competitor_features_tenant_isolation ON intelligence.competitor_features
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant permissions
GRANT USAGE ON SCHEMA intelligence TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA intelligence TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA intelligence TO agentsystem_api;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA intelligence TO agentsystem_api;

-- Create views for analytics and reporting
CREATE VIEW intelligence.threat_summary AS
SELECT
    ci.tenant_id,
    ci.threat_level,
    COUNT(*) as threat_count,
    COUNT(DISTINCT ci.competitor_id) as competitors_with_threats,
    AVG(ci.confidence_score) as avg_confidence,
    COUNT(CASE WHEN ci.detected_date >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END) as recent_threats
FROM intelligence.competitive_intelligence ci
WHERE ci.status = 'active'
AND ci.expiry_date > NOW()
GROUP BY ci.tenant_id, ci.threat_level
ORDER BY ci.tenant_id,
    CASE ci.threat_level
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        ELSE 4
    END;

CREATE VIEW intelligence.competitor_activity_summary AS
SELECT
    c.tenant_id,
    c.competitor_id,
    c.name as competitor_name,
    c.tier,
    COUNT(ci.intelligence_id) as total_intelligence_items,
    COUNT(CASE WHEN ci.detected_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as recent_activity,
    MAX(ci.detected_date) as last_activity_date,
    AVG(CASE WHEN ci.threat_level = 'critical' THEN 4
             WHEN ci.threat_level = 'high' THEN 3
             WHEN ci.threat_level = 'medium' THEN 2
             ELSE 1 END) as avg_threat_score,
    COUNT(CASE WHEN ci.threat_level IN ('high', 'critical') THEN 1 END) as high_threat_items
FROM intelligence.competitors c
LEFT JOIN intelligence.competitive_intelligence ci ON c.competitor_id = ci.competitor_id
WHERE c.is_active = true
GROUP BY c.tenant_id, c.competitor_id, c.name, c.tier
ORDER BY avg_threat_score DESC NULLS LAST, recent_activity DESC;

CREATE VIEW intelligence.market_intelligence_dashboard AS
SELECT
    ci.tenant_id,
    ci.intelligence_type,
    COUNT(*) as total_items,
    COUNT(CASE WHEN ci.detected_date >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END) as items_last_7d,
    COUNT(CASE WHEN ci.detected_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as items_last_30d,
    COUNT(CASE WHEN ci.threat_level IN ('high', 'critical') THEN 1 END) as high_threat_items,
    AVG(ci.confidence_score) as avg_confidence,
    MAX(ci.detected_date) as latest_intelligence
FROM intelligence.competitive_intelligence ci
WHERE ci.status = 'active'
GROUP BY ci.tenant_id, ci.intelligence_type
ORDER BY ci.tenant_id, total_items DESC;

CREATE VIEW intelligence.competitive_positioning AS
SELECT
    c.tenant_id,
    c.tier,
    COUNT(*) as competitor_count,
    AVG(c.market_cap) as avg_market_cap,
    AVG(c.funding_raised) as avg_funding,
    AVG(c.employee_count) as avg_employees,
    COUNT(CASE WHEN ci.threat_level IN ('high', 'critical') THEN 1 END) as active_threats,
    MAX(ci.detected_date) as last_threat_date
FROM intelligence.competitors c
LEFT JOIN intelligence.competitive_intelligence ci ON c.competitor_id = ci.competitor_id
    AND ci.detected_date >= CURRENT_DATE - INTERVAL '30 days'
WHERE c.is_active = true
GROUP BY c.tenant_id, c.tier
ORDER BY c.tenant_id, competitor_count DESC;

-- Grant permissions on views
GRANT SELECT ON intelligence.threat_summary TO agentsystem_api;
GRANT SELECT ON intelligence.competitor_activity_summary TO agentsystem_api;
GRANT SELECT ON intelligence.market_intelligence_dashboard TO agentsystem_api;
GRANT SELECT ON intelligence.competitive_positioning TO agentsystem_api;

-- Create materialized view for dashboard performance
CREATE MATERIALIZED VIEW intelligence.intelligence_dashboard_stats AS
SELECT
    ci.tenant_id,
    COUNT(DISTINCT c.competitor_id) as total_competitors_monitored,
    COUNT(ci.intelligence_id) as total_intelligence_items,
    COUNT(CASE WHEN ci.detected_date >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END) as intelligence_last_7d,
    COUNT(CASE WHEN ci.threat_level = 'critical' THEN 1 END) as critical_threats,
    COUNT(CASE WHEN ci.threat_level = 'high' THEN 1 END) as high_threats,
    COUNT(CASE WHEN ci.status = 'active' THEN 1 END) as active_intelligence,
    AVG(ci.confidence_score) as avg_confidence_score,
    COUNT(DISTINCT ca.alert_id) as active_alerts,
    MAX(ci.detected_date) as latest_intelligence_date
FROM intelligence.competitors c
LEFT JOIN intelligence.competitive_intelligence ci ON c.competitor_id = ci.competitor_id
LEFT JOIN intelligence.competitor_alerts ca ON c.competitor_id = ca.competitor_id
    AND ca.status = 'active'
WHERE c.is_active = true
GROUP BY ci.tenant_id;

-- Create index on materialized view
CREATE INDEX idx_intelligence_dashboard_stats_tenant ON intelligence.intelligence_dashboard_stats(tenant_id);

-- Grant permissions on materialized view
GRANT SELECT ON intelligence.intelligence_dashboard_stats TO agentsystem_api;

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION intelligence.refresh_intelligence_dashboard_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY intelligence.intelligence_dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on function
GRANT EXECUTE ON FUNCTION intelligence.refresh_intelligence_dashboard_stats() TO agentsystem_api;

-- Function to create competitive alert
CREATE OR REPLACE FUNCTION intelligence.create_competitive_alert(
    p_tenant_id UUID,
    p_competitor_id UUID,
    p_alert_type VARCHAR,
    p_title VARCHAR,
    p_message TEXT,
    p_severity intelligence.threat_level DEFAULT 'medium',
    p_intelligence_id UUID DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_alert_id UUID;
BEGIN
    INSERT INTO intelligence.competitor_alerts (
        tenant_id, competitor_id, alert_type, title, message,
        severity, triggered_by_intelligence, alert_triggers
    ) VALUES (
        p_tenant_id, p_competitor_id, p_alert_type, p_title, p_message,
        p_severity, p_intelligence_id, '{}'::JSONB
    ) RETURNING alert_id INTO v_alert_id;

    RETURN v_alert_id;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on alert function
GRANT EXECUTE ON FUNCTION intelligence.create_competitive_alert TO agentsystem_api;

-- Function to calculate competitive threat score
CREATE OR REPLACE FUNCTION intelligence.calculate_threat_score(
    p_tenant_id UUID,
    p_competitor_id UUID,
    p_days_back INTEGER DEFAULT 30
) RETURNS DECIMAL AS $$
DECLARE
    v_threat_score DECIMAL := 0;
    v_intel_record RECORD;
BEGIN
    -- Calculate threat score based on recent intelligence
    FOR v_intel_record IN
        SELECT threat_level, confidence_score
        FROM intelligence.competitive_intelligence
        WHERE tenant_id = p_tenant_id
        AND competitor_id = p_competitor_id
        AND detected_date >= NOW() - INTERVAL '%s days' % p_days_back
        AND status = 'active'
    LOOP
        CASE v_intel_record.threat_level
            WHEN 'critical' THEN v_threat_score := v_threat_score + (10 * v_intel_record.confidence_score);
            WHEN 'high' THEN v_threat_score := v_threat_score + (5 * v_intel_record.confidence_score);
            WHEN 'medium' THEN v_threat_score := v_threat_score + (2 * v_intel_record.confidence_score);
            WHEN 'low' THEN v_threat_score := v_threat_score + (1 * v_intel_record.confidence_score);
        END CASE;
    END LOOP;

    RETURN ROUND(v_threat_score, 2);
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on threat score function
GRANT EXECUTE ON FUNCTION intelligence.calculate_threat_score TO agentsystem_api;

-- Function to auto-escalate high-threat alerts
CREATE OR REPLACE FUNCTION intelligence.auto_escalate_competitive_alerts()
RETURNS void AS $$
DECLARE
    alert_record RECORD;
BEGIN
    -- Escalate unacknowledged critical threats after 1 hour
    FOR alert_record IN
        SELECT alert_id, tenant_id, competitor_id, title
        FROM intelligence.competitor_alerts
        WHERE status = 'active'
        AND severity = 'critical'
        AND acknowledged_at IS NULL
        AND created_at < NOW() - INTERVAL '1 hour'
    LOOP
        -- Update alert status
        UPDATE intelligence.competitor_alerts
        SET status = 'escalated'
        WHERE alert_id = alert_record.alert_id;

        -- TODO: Send escalation notification
        -- This would integrate with notification system
    END LOOP;

    -- Escalate high threats after 4 hours
    FOR alert_record IN
        SELECT alert_id, tenant_id, competitor_id, title
        FROM intelligence.competitor_alerts
        WHERE status = 'active'
        AND severity = 'high'
        AND acknowledged_at IS NULL
        AND created_at < NOW() - INTERVAL '4 hours'
    LOOP
        -- Update alert status
        UPDATE intelligence.competitor_alerts
        SET status = 'escalated'
        WHERE alert_id = alert_record.alert_id;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on escalation function
GRANT EXECUTE ON FUNCTION intelligence.auto_escalate_competitive_alerts() TO agentsystem_api;

-- Insert sample competitors for demo (optional)
-- This would be populated by the tenant during onboarding
