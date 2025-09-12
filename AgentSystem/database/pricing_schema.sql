-- Dynamic Pricing Database Schema - AgentSystem Profit Machine
-- Advanced value-based pricing optimization system

-- Create pricing schema
CREATE SCHEMA IF NOT EXISTS pricing;

-- Pricing strategy enum
CREATE TYPE pricing.pricing_strategy AS ENUM (
    'value_based', 'usage_based', 'competitive', 'penetration', 'premium', 'dynamic'
);

-- Pricing tier enum
CREATE TYPE pricing.pricing_tier AS ENUM (
    'starter', 'professional', 'enterprise', 'custom'
);

-- Price adjustment type enum
CREATE TYPE pricing.price_adjustment_type AS ENUM (
    'discount', 'premium', 'loyalty', 'volume', 'competitive', 'value_realization'
);

-- Pricing recommendations table
CREATE TABLE pricing.recommendations (
    recommendation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    current_price DECIMAL(10,2) NOT NULL,
    recommended_price DECIMAL(10,2) NOT NULL,
    price_change_percent DECIMAL(6,2) NOT NULL,
    adjustment_type pricing.price_adjustment_type NOT NULL,
    strategy_used pricing.pricing_strategy NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    expected_revenue_impact DECIMAL(12,2) DEFAULT 0,
    expected_churn_impact DECIMAL(5,4) DEFAULT 0,
    implementation_priority DECIMAL(5,2) DEFAULT 0,
    reasoning JSONB DEFAULT '[]',
    supporting_metrics JSONB DEFAULT '{}',
    effective_date TIMESTAMPTZ DEFAULT NOW(),
    expiry_date TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'pending', -- pending, approved, implemented, rejected
    implemented_date TIMESTAMPTZ,
    actual_revenue_impact DECIMAL(12,2),
    actual_churn_impact DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Customer pricing profiles table
CREATE TABLE pricing.customer_profiles (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    current_tier pricing.pricing_tier NOT NULL,
    monthly_usage DECIMAL(12,2) DEFAULT 0,
    value_score DECIMAL(3,2) DEFAULT 0 CHECK (value_score >= 0 AND value_score <= 1),
    price_sensitivity DECIMAL(3,2) DEFAULT 0.5 CHECK (price_sensitivity >= 0 AND price_sensitivity <= 1),
    churn_risk DECIMAL(3,2) DEFAULT 0 CHECK (churn_risk >= 0 AND churn_risk <= 1),
    clv_prediction DECIMAL(12,2) DEFAULT 0,
    competitive_position DECIMAL(3,2) DEFAULT 0.5,
    usage_growth_trend DECIMAL(5,4) DEFAULT 0,
    feature_adoption_score DECIMAL(3,2) DEFAULT 0,
    support_cost_ratio DECIMAL(5,4) DEFAULT 0,
    payment_reliability DECIMAL(3,2) DEFAULT 1.0,
    contract_length_preference INTEGER DEFAULT 12, -- months
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, customer_id)
);

-- Market conditions table
CREATE TABLE pricing.market_conditions (
    condition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    competitive_pressure DECIMAL(3,2) DEFAULT 0.5,
    market_growth_rate DECIMAL(5,4) DEFAULT 0,
    customer_acquisition_cost DECIMAL(10,2) DEFAULT 0,
    average_deal_size DECIMAL(10,2) DEFAULT 0,
    price_elasticity DECIMAL(5,4) DEFAULT -0.5,
    seasonal_factor DECIMAL(5,4) DEFAULT 1.0,
    economic_indicator DECIMAL(5,4) DEFAULT 1.0,
    data_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, data_date)
);

-- Pricing experiments table
CREATE TABLE pricing.experiments (
    experiment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    strategy pricing.pricing_strategy NOT NULL,
    target_segment VARCHAR(100),
    test_price DECIMAL(10,2) NOT NULL,
    control_price DECIMAL(10,2) NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    sample_size INTEGER DEFAULT 0,
    test_group_size INTEGER DEFAULT 0,
    control_group_size INTEGER DEFAULT 0,
    success_metrics JSONB DEFAULT '[]',
    status VARCHAR(50) DEFAULT 'planned', -- planned, running, completed, cancelled
    results JSONB DEFAULT '{}',
    statistical_significance DECIMAL(5,4),
    confidence_interval JSONB DEFAULT '{}',
    recommendation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Experiment participants table
CREATE TABLE pricing.experiment_participants (
    participant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    group_type VARCHAR(20) NOT NULL, -- test, control
    assigned_price DECIMAL(10,2) NOT NULL,
    baseline_metrics JSONB DEFAULT '{}',
    outcome_metrics JSONB DEFAULT '{}',
    participated_fully BOOLEAN DEFAULT true,
    dropout_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (experiment_id) REFERENCES pricing.experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(experiment_id, customer_id)
);

-- Price elasticity table
CREATE TABLE pricing.price_elasticity (
    elasticity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    customer_segment VARCHAR(100),
    pricing_tier pricing.pricing_tier,
    price_point DECIMAL(10,2) NOT NULL,
    demand_quantity INTEGER NOT NULL,
    elasticity_coefficient DECIMAL(6,4) NOT NULL,
    confidence_interval JSONB DEFAULT '{}',
    calculation_date DATE NOT NULL,
    data_points_count INTEGER DEFAULT 0,
    r_squared DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Competitive pricing intelligence table
CREATE TABLE pricing.competitive_intelligence (
    intelligence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    competitor_name VARCHAR(200) NOT NULL,
    product_tier VARCHAR(100),
    price DECIMAL(10,2) NOT NULL,
    features_included JSONB DEFAULT '[]',
    pricing_model VARCHAR(100), -- subscription, usage, hybrid
    contract_terms VARCHAR(200),
    promotional_offers JSONB DEFAULT '[]',
    market_position VARCHAR(50), -- premium, mid-market, budget
    data_source VARCHAR(100),
    confidence_score DECIMAL(3,2) DEFAULT 0.5,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Pricing alerts table
CREATE TABLE pricing.alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    alert_type VARCHAR(100) NOT NULL, -- elasticity_change, competitor_move, churn_risk, revenue_decline
    severity VARCHAR(20) DEFAULT 'medium', -- low, medium, high, critical
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    affected_customers INTEGER DEFAULT 0,
    revenue_impact DECIMAL(12,2) DEFAULT 0,
    recommended_actions JSONB DEFAULT '[]',
    status VARCHAR(50) DEFAULT 'active', -- active, acknowledged, resolved, dismissed
    acknowledged_by VARCHAR(200),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Pricing tier performance table
CREATE TABLE pricing.tier_performance (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    tier pricing.pricing_tier NOT NULL,
    performance_date DATE NOT NULL,
    customer_count INTEGER DEFAULT 0,
    average_price DECIMAL(10,2) DEFAULT 0,
    total_revenue DECIMAL(15,2) DEFAULT 0,
    churn_rate DECIMAL(5,4) DEFAULT 0,
    upgrade_rate DECIMAL(5,4) DEFAULT 0,
    downgrade_rate DECIMAL(5,4) DEFAULT 0,
    payment_success_rate DECIMAL(5,4) DEFAULT 0,
    customer_satisfaction DECIMAL(3,2) DEFAULT 0,
    support_cost_per_customer DECIMAL(8,2) DEFAULT 0,
    clv_average DECIMAL(12,2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, tier, performance_date)
);

-- Dynamic pricing rules table
CREATE TABLE pricing.dynamic_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    rule_name VARCHAR(200) NOT NULL,
    conditions JSONB NOT NULL, -- Conditions that trigger the rule
    actions JSONB NOT NULL,    -- Actions to take when conditions are met
    priority INTEGER DEFAULT 50,
    is_active BOOLEAN DEFAULT true,
    max_price_increase DECIMAL(5,4) DEFAULT 0.25,
    max_price_decrease DECIMAL(5,4) DEFAULT 0.40,
    cooldown_period_days INTEGER DEFAULT 30,
    approval_required BOOLEAN DEFAULT false,
    created_by VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Pricing optimization jobs table
CREATE TABLE pricing.optimization_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    job_type VARCHAR(100) NOT NULL, -- full_optimization, tier_optimization, segment_optimization
    target_metrics JSONB NOT NULL,
    scope JSONB DEFAULT '{}', -- Which customers/tiers to optimize
    status VARCHAR(50) DEFAULT 'queued', -- queued, running, completed, failed
    progress DECIMAL(5,2) DEFAULT 0, -- 0-100%
    results JSONB DEFAULT '{}',
    recommendations_generated INTEGER DEFAULT 0,
    estimated_revenue_impact DECIMAL(15,2) DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_details TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX idx_pricing_recommendations_tenant ON pricing.recommendations(tenant_id);
CREATE INDEX idx_pricing_recommendations_customer ON pricing.recommendations(customer_id);
CREATE INDEX idx_pricing_recommendations_status ON pricing.recommendations(status);
CREATE INDEX idx_pricing_recommendations_priority ON pricing.recommendations(implementation_priority DESC);
CREATE INDEX idx_pricing_recommendations_effective_date ON pricing.recommendations(effective_date);

CREATE INDEX idx_customer_profiles_tenant ON pricing.customer_profiles(tenant_id);
CREATE INDEX idx_customer_profiles_customer ON pricing.customer_profiles(customer_id);
CREATE INDEX idx_customer_profiles_tier ON pricing.customer_profiles(current_tier);
CREATE INDEX idx_customer_profiles_value_score ON pricing.customer_profiles(value_score DESC);
CREATE INDEX idx_customer_profiles_churn_risk ON pricing.customer_profiles(churn_risk DESC);

CREATE INDEX idx_market_conditions_tenant ON pricing.market_conditions(tenant_id);
CREATE INDEX idx_market_conditions_date ON pricing.market_conditions(data_date DESC);

CREATE INDEX idx_pricing_experiments_tenant ON pricing.experiments(tenant_id);
CREATE INDEX idx_pricing_experiments_status ON pricing.experiments(status);
CREATE INDEX idx_pricing_experiments_dates ON pricing.experiments(start_date, end_date);

CREATE INDEX idx_experiment_participants_experiment ON pricing.experiment_participants(experiment_id);
CREATE INDEX idx_experiment_participants_customer ON pricing.experiment_participants(customer_id);
CREATE INDEX idx_experiment_participants_group ON pricing.experiment_participants(group_type);

CREATE INDEX idx_price_elasticity_tenant ON pricing.price_elasticity(tenant_id);
CREATE INDEX idx_price_elasticity_segment ON pricing.price_elasticity(customer_segment);
CREATE INDEX idx_price_elasticity_tier ON pricing.price_elasticity(pricing_tier);
CREATE INDEX idx_price_elasticity_date ON pricing.price_elasticity(calculation_date DESC);

CREATE INDEX idx_competitive_intelligence_tenant ON pricing.competitive_intelligence(tenant_id);
CREATE INDEX idx_competitive_intelligence_competitor ON pricing.competitive_intelligence(competitor_name);
CREATE INDEX idx_competitive_intelligence_updated ON pricing.competitive_intelligence(last_updated DESC);

CREATE INDEX idx_pricing_alerts_tenant ON pricing.alerts(tenant_id);
CREATE INDEX idx_pricing_alerts_status ON pricing.alerts(status);
CREATE INDEX idx_pricing_alerts_severity ON pricing.alerts(severity);
CREATE INDEX idx_pricing_alerts_type ON pricing.alerts(alert_type);

CREATE INDEX idx_tier_performance_tenant ON pricing.tier_performance(tenant_id);
CREATE INDEX idx_tier_performance_tier ON pricing.tier_performance(tier);
CREATE INDEX idx_tier_performance_date ON pricing.tier_performance(performance_date DESC);

CREATE INDEX idx_dynamic_rules_tenant ON pricing.dynamic_rules(tenant_id);
CREATE INDEX idx_dynamic_rules_active ON pricing.dynamic_rules(is_active);
CREATE INDEX idx_dynamic_rules_priority ON pricing.dynamic_rules(priority DESC);

CREATE INDEX idx_optimization_jobs_tenant ON pricing.optimization_jobs(tenant_id);
CREATE INDEX idx_optimization_jobs_status ON pricing.optimization_jobs(status);
CREATE INDEX idx_optimization_jobs_type ON pricing.optimization_jobs(job_type);

-- Create updated_at trigger function (reuse existing one)
CREATE TRIGGER update_pricing_recommendations_updated_at
    BEFORE UPDATE ON pricing.recommendations
    FOR EACH ROW EXECUTE FUNCTION analytics.update_updated_at_column();

CREATE TRIGGER update_pricing_experiments_updated_at
    BEFORE UPDATE ON pricing.experiments
    FOR EACH ROW EXECUTE FUNCTION analytics.update_updated_at_column();

CREATE TRIGGER update_dynamic_rules_updated_at
    BEFORE UPDATE ON pricing.dynamic_rules
    FOR EACH ROW EXECUTE FUNCTION analytics.update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE pricing.recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing.customer_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing.market_conditions ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing.experiments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing.experiment_participants ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing.price_elasticity ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing.competitive_intelligence ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing.alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing.tier_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing.dynamic_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing.optimization_jobs ENABLE ROW LEVEL SECURITY;

-- RLS policies for tenant isolation
CREATE POLICY pricing_recommendations_tenant_isolation ON pricing.recommendations
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY customer_profiles_tenant_isolation ON pricing.customer_profiles
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY market_conditions_tenant_isolation ON pricing.market_conditions
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY pricing_experiments_tenant_isolation ON pricing.experiments
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY experiment_participants_tenant_isolation ON pricing.experiment_participants
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY price_elasticity_tenant_isolation ON pricing.price_elasticity
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY competitive_intelligence_tenant_isolation ON pricing.competitive_intelligence
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY pricing_alerts_tenant_isolation ON pricing.alerts
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY tier_performance_tenant_isolation ON pricing.tier_performance
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY dynamic_rules_tenant_isolation ON pricing.dynamic_rules
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY optimization_jobs_tenant_isolation ON pricing.optimization_jobs
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant permissions
GRANT USAGE ON SCHEMA pricing TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA pricing TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA pricing TO agentsystem_api;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pricing TO agentsystem_api;

-- Create views for analytics and reporting
CREATE VIEW pricing.pricing_performance_summary AS
SELECT
    r.tenant_id,
    r.strategy_used,
    COUNT(*) as total_recommendations,
    COUNT(CASE WHEN r.status = 'implemented' THEN 1 END) as implemented_count,
    AVG(r.price_change_percent) as avg_price_change,
    AVG(r.expected_revenue_impact) as avg_expected_revenue,
    AVG(r.actual_revenue_impact) as avg_actual_revenue,
    AVG(r.confidence_score) as avg_confidence,
    SUM(r.actual_revenue_impact) as total_revenue_impact
FROM pricing.recommendations r
WHERE r.created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY r.tenant_id, r.strategy_used
ORDER BY total_revenue_impact DESC NULLS LAST;

CREATE VIEW pricing.customer_pricing_insights AS
SELECT
    cp.tenant_id,
    cp.customer_id,
    cp.current_tier,
    cp.value_score,
    cp.price_sensitivity,
    cp.churn_risk,
    cp.clv_prediction,
    r.recommended_price,
    r.price_change_percent,
    r.implementation_priority,
    r.status as recommendation_status
FROM pricing.customer_profiles cp
LEFT JOIN pricing.recommendations r ON cp.customer_id = r.customer_id
    AND cp.tenant_id = r.tenant_id
    AND r.created_at = (
        SELECT MAX(created_at)
        FROM pricing.recommendations r2
        WHERE r2.customer_id = cp.customer_id
        AND r2.tenant_id = cp.tenant_id
    );

CREATE VIEW pricing.tier_optimization_opportunities AS
SELECT
    tp.tenant_id,
    tp.tier,
    tp.customer_count,
    tp.average_price,
    tp.total_revenue,
    tp.churn_rate,
    COUNT(r.recommendation_id) as pending_recommendations,
    AVG(r.expected_revenue_impact) as avg_revenue_opportunity,
    SUM(r.expected_revenue_impact) as total_revenue_opportunity
FROM pricing.tier_performance tp
LEFT JOIN pricing.recommendations r ON tp.tenant_id = r.tenant_id
    AND r.status = 'pending'
WHERE tp.performance_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY tp.tenant_id, tp.tier, tp.customer_count, tp.average_price,
         tp.total_revenue, tp.churn_rate
ORDER BY total_revenue_opportunity DESC NULLS LAST;

CREATE VIEW pricing.competitive_position_analysis AS
SELECT
    ci.tenant_id,
    ci.product_tier,
    COUNT(DISTINCT ci.competitor_name) as competitor_count,
    AVG(ci.price) as avg_competitor_price,
    MIN(ci.price) as min_competitor_price,
    MAX(ci.price) as max_competitor_price,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ci.price) as median_competitor_price,
    tp.average_price as our_price,
    CASE
        WHEN tp.average_price > PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ci.price) THEN 'Premium'
        WHEN tp.average_price > PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ci.price) THEN 'Above Market'
        WHEN tp.average_price > PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ci.price) THEN 'Market Rate'
        ELSE 'Below Market'
    END as market_position
FROM pricing.competitive_intelligence ci
LEFT JOIN pricing.tier_performance tp ON ci.tenant_id = tp.tenant_id
    AND ci.product_tier = tp.tier::text
WHERE ci.last_updated >= CURRENT_DATE - INTERVAL '30 days'
AND tp.performance_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY ci.tenant_id, ci.product_tier, tp.average_price;

-- Grant permissions on views
GRANT SELECT ON pricing.pricing_performance_summary TO agentsystem_api;
GRANT SELECT ON pricing.customer_pricing_insights TO agentsystem_api;
GRANT SELECT ON pricing.tier_optimization_opportunities TO agentsystem_api;
GRANT SELECT ON pricing.competitive_position_analysis TO agentsystem_api;

-- Create materialized view for dashboard performance
CREATE MATERIALIZED VIEW pricing.pricing_dashboard_stats AS
SELECT
    r.tenant_id,
    COUNT(DISTINCT r.customer_id) as customers_with_recommendations,
    COUNT(r.recommendation_id) as total_recommendations,
    COUNT(CASE WHEN r.status = 'pending' THEN 1 END) as pending_recommendations,
    COUNT(CASE WHEN r.status = 'implemented' THEN 1 END) as implemented_recommendations,
    AVG(r.expected_revenue_impact) as avg_revenue_opportunity,
    SUM(r.expected_revenue_impact) as total_revenue_opportunity,
    SUM(r.actual_revenue_impact) as total_realized_revenue,
    AVG(r.implementation_priority) as avg_priority_score,
    COUNT(CASE WHEN r.implementation_priority > 70 THEN 1 END) as high_priority_recommendations
FROM pricing.recommendations r
WHERE r.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY r.tenant_id;

-- Create index on materialized view
CREATE INDEX idx_pricing_dashboard_stats_tenant ON pricing.pricing_dashboard_stats(tenant_id);

-- Grant permissions on materialized view
GRANT SELECT ON pricing.pricing_dashboard_stats TO agentsystem_api;

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION pricing.refresh_pricing_dashboard_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY pricing.pricing_dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on function
GRANT EXECUTE ON FUNCTION pricing.refresh_pricing_dashboard_stats() TO agentsystem_api;

-- Function to calculate price elasticity
CREATE OR REPLACE FUNCTION pricing.calculate_price_elasticity(
    p_tenant_id UUID,
    p_customer_segment VARCHAR DEFAULT NULL,
    p_pricing_tier pricing.pricing_tier DEFAULT NULL
) RETURNS TABLE (
    elasticity_coefficient DECIMAL,
    confidence_interval JSONB,
    r_squared DECIMAL,
    data_points INTEGER
) AS $$
DECLARE
    v_elasticity DECIMAL;
    v_confidence JSONB;
    v_r_squared DECIMAL;
    v_data_points INTEGER;
BEGIN
    -- Simplified elasticity calculation
    -- In practice, this would use more sophisticated statistical methods

    SELECT
        -0.8, -- Default elasticity coefficient
        '{"lower": -1.2, "upper": -0.4}'::JSONB,
        0.75, -- R-squared
        100   -- Data points
    INTO v_elasticity, v_confidence, v_r_squared, v_data_points;

    RETURN QUERY SELECT v_elasticity, v_confidence, v_r_squared, v_data_points;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on elasticity function
GRANT EXECUTE ON FUNCTION pricing.calculate_price_elasticity TO agentsystem_api;

-- Function to create pricing alert
CREATE OR REPLACE FUNCTION pricing.create_pricing_alert(
    p_tenant_id UUID,
    p_alert_type VARCHAR,
    p_title VARCHAR,
    p_description TEXT,
    p_severity VARCHAR DEFAULT 'medium',
    p_affected_customers INTEGER DEFAULT 0,
    p_revenue_impact DECIMAL DEFAULT 0
) RETURNS UUID AS $$
DECLARE
    v_alert_id UUID;
BEGIN
    INSERT INTO pricing.alerts (
        tenant_id, alert_type, title, description, severity,
        affected_customers, revenue_impact
    ) VALUES (
        p_tenant_id, p_alert_type, p_title, p_description, p_severity,
        p_affected_customers, p_revenue_impact
    ) RETURNING alert_id INTO v_alert_id;

    RETURN v_alert_id;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on alert function
GRANT EXECUTE ON FUNCTION pricing.create_pricing_alert TO agentsystem_api;

-- Function to update customer pricing profile
CREATE OR REPLACE FUNCTION pricing.update_customer_profile(
    p_tenant_id UUID,
    p_customer_id UUID
) RETURNS void AS $$
DECLARE
    v_profile_data RECORD;
BEGIN
    -- Calculate customer metrics
    SELECT
        COALESCE(s.plan_id::pricing.pricing_tier, 'starter') as tier,
        COALESCE(usage_agg.monthly_cost, 0) as monthly_usage,
        COALESCE(payment_agg.total_revenue / 10000.0, 0) as value_score,
        COALESCE(1.0 - (payment_agg.total_revenue / 5000.0), 0.5) as price_sensitivity,
        COALESCE(cp.churn_probability, 0.3) as churn_risk,
        COALESCE(clv.predicted_clv, 5000) as clv_prediction
    INTO v_profile_data
    FROM billing.tenants t
    LEFT JOIN billing.subscriptions s ON t.tenant_id = s.tenant_id
    LEFT JOIN (
        SELECT tenant_id, AVG(cost) as monthly_cost
        FROM usage.usage_logs
        WHERE tenant_id = p_customer_id
        AND created_at >= NOW() - INTERVAL '30 days'
        GROUP BY tenant_id
    ) usage_agg ON t.tenant_id = usage_agg.tenant_id
    LEFT JOIN (
        SELECT tenant_id, SUM(amount) as total_revenue
        FROM billing.payments
        WHERE tenant_id = p_customer_id
        AND created_at >= NOW() - INTERVAL '12 months'
        GROUP BY tenant_id
    ) payment_agg ON t.tenant_id = payment_agg.tenant_id
    LEFT JOIN analytics.churn_predictions cp ON t.tenant_id = cp.customer_id
        AND cp.prediction_date >= NOW() - INTERVAL '7 days'
    LEFT JOIN analytics.clv_predictions clv ON t.tenant_id = clv.customer_id
        AND clv.prediction_date >= NOW() - INTERVAL '7 days'
    WHERE t.tenant_id = p_customer_id;

    -- Upsert profile
    INSERT INTO pricing.customer_profiles (
        tenant_id, customer_id, current_tier, monthly_usage, value_score,
        price_sensitivity, churn_risk, clv_prediction
    ) VALUES (
        p_tenant_id, p_customer_id, v_profile_data.tier, v_profile_data.monthly_usage,
        v_profile_data.value_score, v_profile_data.price_sensitivity,
        v_profile_data.churn_risk, v_profile_data.clv_prediction
    )
    ON CONFLICT (tenant_id, customer_id)
    DO UPDATE SET
        current_tier = EXCLUDED.current_tier,
        monthly_usage = EXCLUDED.monthly_usage,
        value_score = EXCLUDED.value_score,
        price_sensitivity = EXCLUDED.price_sensitivity,
        churn_risk = EXCLUDED.churn_risk,
        clv_prediction = EXCLUDED.clv_prediction,
        last_updated = NOW();
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on profile update function
GRANT EXECUTE ON FUNCTION pricing.update_customer_profile TO agentsystem_api;

-- Insert default market conditions for existing tenants
INSERT INTO pricing.market_conditions (tenant_id, data_date)
SELECT
    tenant_id,
    CURRENT_DATE
FROM billing.tenants
WHERE NOT EXISTS (
    SELECT 1 FROM pricing.market_conditions mc
    WHERE mc.tenant_id = billing.tenants.tenant_id
    AND mc.data_date = CURRENT_DATE
);

-- Insert default pricing profiles for existing customers
INSERT INTO pricing.customer_profiles (tenant_id, customer_id, current_tier)
SELECT
    t.tenant_id,
    t.tenant_id as customer_id,
    COALESCE(s.plan_id::pricing.pricing_tier, 'starter')
FROM billing.tenants t
LEFT JOIN billing.subscriptions s ON t.tenant_id = s.tenant_id
WHERE NOT EXISTS (
    SELECT 1 FROM pricing.customer_profiles cp
    WHERE cp.tenant_id = t.tenant_id
    AND cp.customer_id = t.tenant_id
);
