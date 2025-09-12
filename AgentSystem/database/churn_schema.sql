
-- Churn Prediction and Intervention Database Schema - AgentSystem Profit Machine
-- Advanced churn prediction and automated intervention system

-- Extend analytics schema for churn-specific tables
-- Note: This extends the existing analytics schema

-- Churn risk level enum
CREATE TYPE analytics.churn_risk_level AS ENUM (
    'very_low', 'low', 'medium', 'high', 'very_high', 'critical'
);

-- Intervention type enum
CREATE TYPE analytics.intervention_type AS ENUM (
    'email_outreach', 'phone_call', 'discount_offer', 'feature_training',
    'account_review', 'product_demo', 'customer_success_call', 'retention_campaign'
);

-- Intervention status enum
CREATE TYPE analytics.intervention_status AS ENUM (
    'pending', 'scheduled', 'in_progress', 'completed', 'failed', 'cancelled'
);

-- Churn model type enum
CREATE TYPE analytics.churn_model_type AS ENUM (
    'logistic_regression', 'random_forest', 'gradient_boosting', 'ensemble'
);

-- Churn predictions table
CREATE TABLE analytics.churn_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    churn_probability DECIMAL(5,4) NOT NULL CHECK (churn_probability >= 0 AND churn_probability <= 1),
    risk_level analytics.churn_risk_level NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    time_to_churn_days INTEGER,
    key_risk_factors JSONB DEFAULT '[]',
    protective_factors JSONB DEFAULT '[]',
    early_warning_signals JSONB DEFAULT '[]',
    feature_importance JSONB DEFAULT '{}',
    model_used analytics.churn_model_type NOT NULL,
    prediction_date TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Churn features table for ML training
CREATE TABLE analytics.churn_features (
    feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    snapshot_date DATE NOT NULL,

    -- Usage patterns
    usage_trend_30d DECIMAL(6,2) DEFAULT 0,
    usage_trend_7d DECIMAL(6,2) DEFAULT 0,
    days_since_last_login INTEGER DEFAULT 0,
    session_frequency_decline DECIMAL(5,2) DEFAULT 0,
    feature_usage_decline DECIMAL(5,2) DEFAULT 0,
    api_calls_decline DECIMAL(5,2) DEFAULT 0,

    -- Engagement metrics
    support_ticket_frequency DECIMAL(5,2) DEFAULT 0,
    support_satisfaction_score DECIMAL(3,2) DEFAULT 5.0,
    feature_adoption_rate DECIMAL(3,2) DEFAULT 0,
    onboarding_completion DECIMAL(3,2) DEFAULT 0,
    training_attendance INTEGER DEFAULT 0,

    -- Billing and subscription
    payment_failures INTEGER DEFAULT 0,
    billing_issues INTEGER DEFAULT 0,
    plan_downgrades INTEGER DEFAULT 0,
    contract_renewal_date TIMESTAMPTZ,
    days_to_renewal INTEGER,
    payment_delays INTEGER DEFAULT 0,

    -- Behavioral indicators
    complaint_frequency INTEGER DEFAULT 0,
    cancellation_attempts INTEGER DEFAULT 0,
    competitor_mentions INTEGER DEFAULT 0,
    negative_feedback_score DECIMAL(3,2) DEFAULT 0,
    response_rate_decline DECIMAL(5,2) DEFAULT 0,

    -- Account health
    account_age_days INTEGER DEFAULT 0,
    total_spent DECIMAL(12,2) DEFAULT 0,
    monthly_spend_trend DECIMAL(6,2) DEFAULT 0,
    user_count_change DECIMAL(5,2) DEFAULT 0,
    integration_usage INTEGER DEFAULT 0,

    -- Comparative metrics
    peer_usage_comparison DECIMAL(5,2) DEFAULT 0,
    industry_benchmark_gap DECIMAL(5,2) DEFAULT 0,
    value_realization_score DECIMAL(3,2) DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, customer_id, snapshot_date)
);

-- Intervention plans table
CREATE TABLE analytics.intervention_plans (
    plan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    churn_probability DECIMAL(5,4) NOT NULL,
    risk_level analytics.churn_risk_level NOT NULL,
    interventions JSONB NOT NULL DEFAULT '[]',
    priority_score DECIMAL(5,2) NOT NULL DEFAULT 0,
    estimated_success_rate DECIMAL(3,2) NOT NULL DEFAULT 0,
    estimated_cost DECIMAL(10,2) NOT NULL DEFAULT 0,
    estimated_clv_impact DECIMAL(12,2) NOT NULL DEFAULT 0,
    created_date TIMESTAMPTZ DEFAULT NOW(),
    target_completion_date TIMESTAMPTZ,
    assigned_agent VARCHAR(200),
    status VARCHAR(50) DEFAULT 'active',
    actual_success BOOLEAN,
    actual_cost DECIMAL(10,2),
    actual_clv_impact DECIMAL(12,2),
    completion_date TIMESTAMPTZ,

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Churn interventions execution table
CREATE TABLE analytics.churn_interventions (
    intervention_execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    plan_id UUID NOT NULL,
    intervention_id VARCHAR(100) NOT NULL,
    intervention_type analytics.intervention_type NOT NULL,
    status analytics.intervention_status NOT NULL DEFAULT 'pending',
    scheduled_date TIMESTAMPTZ,
    execution_date TIMESTAMPTZ,
    completion_date TIMESTAMPTZ,
    assigned_agent VARCHAR(200),
    outcome JSONB DEFAULT '{}',
    engagement_score DECIMAL(3,2),
    success_metrics JSONB DEFAULT '{}',
    cost DECIMAL(10,2) DEFAULT 0,
    duration_minutes INTEGER,
    follow_up_required BOOLEAN DEFAULT false,
    follow_up_date TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (plan_id) REFERENCES analytics.intervention_plans(plan_id) ON DELETE CASCADE
);

-- Intervention outcomes tracking table
CREATE TABLE analytics.intervention_outcomes (
    outcome_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    plan_id UUID NOT NULL,
    intervention_date TIMESTAMPTZ NOT NULL,
    pre_intervention_churn_prob DECIMAL(5,4) NOT NULL,
    post_intervention_churn_prob DECIMAL(5,4),
    prevented_churn BOOLEAN DEFAULT false,
    churn_date TIMESTAMPTZ,
    retention_period_days INTEGER,
    clv_impact DECIMAL(12,2) DEFAULT 0,
    intervention_cost DECIMAL(10,2) DEFAULT 0,
    roi DECIMAL(8,2), -- Return on Investment
    customer_satisfaction_change DECIMAL(3,2),
    usage_change_percent DECIMAL(6,2),
    engagement_improvement DECIMAL(3,2),
    evaluation_date TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (plan_id) REFERENCES analytics.intervention_plans(plan_id) ON DELETE CASCADE
);

-- Churn model performance table
CREATE TABLE analytics.churn_model_performance (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    model_type analytics.churn_model_type NOT NULL,
    evaluation_date DATE NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,
    accuracy DECIMAL(5,4) DEFAULT 0,
    precision_score DECIMAL(5,4) DEFAULT 0,
    recall DECIMAL(5,4) DEFAULT 0,
    f1_score DECIMAL(5,4) DEFAULT 0,
    auc_score DECIMAL(5,4) DEFAULT 0,
    avg_confidence DECIMAL(3,2) DEFAULT 0,
    calibration_error DECIMAL(5,4) DEFAULT 0,
    feature_drift_score DECIMAL(3,2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, model_type, evaluation_date)
);

-- Churn alerts table
CREATE TABLE analytics.churn_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    alert_type VARCHAR(100) NOT NULL, -- high_risk, critical_risk, early_warning
    risk_level analytics.churn_risk_level NOT NULL,
    churn_probability DECIMAL(5,4) NOT NULL,
    trigger_factors JSONB DEFAULT '[]',
    alert_message TEXT NOT NULL,
    priority_score DECIMAL(5,2) NOT NULL DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active', -- active, acknowledged, resolved, dismissed
    acknowledged_by VARCHAR(200),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,
    escalation_level INTEGER DEFAULT 1,
    escalated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Retention campaigns table
CREATE TABLE analytics.retention_campaigns (
    campaign_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    campaign_name VARCHAR(200) NOT NULL,
    campaign_type VARCHAR(100) NOT NULL, -- email_series, discount_offer, training_program
    target_risk_levels analytics.churn_risk_level[] NOT NULL,
    campaign_config JSONB NOT NULL DEFAULT '{}',
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    target_customer_count INTEGER DEFAULT 0,
    enrolled_customer_count INTEGER DEFAULT 0,
    completed_customer_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 0,
    total_cost DECIMAL(12,2) DEFAULT 0,
    total_clv_impact DECIMAL(15,2) DEFAULT 0,
    roi DECIMAL(8,2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Campaign enrollments table
CREATE TABLE analytics.campaign_enrollments (
    enrollment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    campaign_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    enrollment_date TIMESTAMPTZ DEFAULT NOW(),
    completion_date TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'enrolled', -- enrolled, in_progress, completed, dropped_out
    engagement_score DECIMAL(3,2) DEFAULT 0,
    outcome_success BOOLEAN,
    churn_prevented BOOLEAN DEFAULT false,
    cost DECIMAL(10,2) DEFAULT 0,
    clv_impact DECIMAL(12,2) DEFAULT 0,
    feedback_score DECIMAL(3,2),
    notes TEXT,

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (campaign_id) REFERENCES analytics.retention_campaigns(campaign_id) ON DELETE CASCADE,
    UNIQUE(campaign_id, customer_id)
);

-- Create indexes for performance
CREATE INDEX idx_churn_predictions_tenant ON analytics.churn_predictions(tenant_id);
CREATE INDEX idx_churn_predictions_customer ON analytics.churn_predictions(customer_id);
CREATE INDEX idx_churn_predictions_risk_level ON analytics.churn_predictions(risk_level);
CREATE INDEX idx_churn_predictions_date ON analytics.churn_predictions(prediction_date DESC);
CREATE INDEX idx_churn_predictions_probability ON analytics.churn_predictions(churn_probability DESC);

CREATE INDEX idx_churn_features_tenant ON analytics.churn_features(tenant_id);
CREATE INDEX idx_churn_features_customer ON analytics.churn_features(customer_id);
CREATE INDEX idx_churn_features_date ON analytics.churn_features(snapshot_date DESC);

CREATE INDEX idx_intervention_plans_tenant ON analytics.intervention_plans(tenant_id);
CREATE INDEX idx_intervention_plans_customer ON analytics.intervention_plans(customer_id);
CREATE INDEX idx_intervention_plans_risk_level ON analytics.intervention_plans(risk_level);
CREATE INDEX idx_intervention_plans_priority ON analytics.intervention_plans(priority_score DESC);
CREATE INDEX idx_intervention_plans_status ON analytics.intervention_plans(status);
CREATE INDEX idx_intervention_plans_agent ON analytics.intervention_plans(assigned_agent);

CREATE INDEX idx_churn_interventions_tenant ON analytics.churn_interventions(tenant_id);
CREATE INDEX idx_churn_interventions_plan ON analytics.churn_interventions(plan_id);
CREATE INDEX idx_churn_interventions_type ON analytics.churn_interventions(intervention_type);
CREATE INDEX idx_churn_interventions_status ON analytics.churn_interventions(status);
CREATE INDEX idx_churn_interventions_agent ON analytics.churn_interventions(assigned_agent);
CREATE INDEX idx_churn_interventions_scheduled ON analytics.churn_interventions(scheduled_date);

CREATE INDEX idx_intervention_outcomes_tenant ON analytics.intervention_outcomes(tenant_id);
CREATE INDEX idx_intervention_outcomes_customer ON analytics.intervention_outcomes(customer_id);
CREATE INDEX idx_intervention_outcomes_plan ON analytics.intervention_outcomes(plan_id);
CREATE INDEX idx_intervention_outcomes_prevented ON analytics.intervention_outcomes(prevented_churn);
CREATE INDEX idx_intervention_outcomes_roi ON analytics.intervention_outcomes(roi DESC);

CREATE INDEX idx_churn_model_performance_tenant ON analytics.churn_model_performance(tenant_id);
CREATE INDEX idx_churn_model_performance_model ON analytics.churn_model_performance(model_type);
CREATE INDEX idx_churn_model_performance_date ON analytics.churn_model_performance(evaluation_date DESC);

CREATE INDEX idx_churn_alerts_tenant ON analytics.churn_alerts(tenant_id);
CREATE INDEX idx_churn_alerts_customer ON analytics.churn_alerts(customer_id);
CREATE INDEX idx_churn_alerts_status ON analytics.churn_alerts(status);
CREATE INDEX idx_churn_alerts_priority ON analytics.churn_alerts(priority_score DESC);
CREATE INDEX idx_churn_alerts_risk_level ON analytics.churn_alerts(risk_level);

CREATE INDEX idx_retention_campaigns_tenant ON analytics.retention_campaigns(tenant_id);
CREATE INDEX idx_retention_campaigns_active ON analytics.retention_campaigns(is_active);
CREATE INDEX idx_retention_campaigns_type ON analytics.retention_campaigns(campaign_type);
CREATE INDEX idx_retention_campaigns_dates ON analytics.retention_campaigns(start_date, end_date);

CREATE INDEX idx_campaign_enrollments_tenant ON analytics.campaign_enrollments(tenant_id);
CREATE INDEX idx_campaign_enrollments_campaign ON analytics.campaign_enrollments(campaign_id);
CREATE INDEX idx_campaign_enrollments_customer ON analytics.campaign_enrollments(customer_id);
CREATE INDEX idx_campaign_enrollments_status ON analytics.campaign_enrollments(status);

-- Create updated_at trigger function (reuse existing one)
CREATE TRIGGER update_churn_interventions_updated_at
    BEFORE UPDATE ON analytics.churn_interventions
    FOR EACH ROW EXECUTE FUNCTION analytics.update_updated_at_column();

CREATE TRIGGER update_retention_campaigns_updated_at
    BEFORE UPDATE ON analytics.retention_campaigns
    FOR EACH ROW EXECUTE FUNCTION analytics.update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE analytics.churn_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.churn_features ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.intervention_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.churn_interventions ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.intervention_outcomes ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.churn_model_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.churn_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.retention_campaigns ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.campaign_enrollments ENABLE ROW LEVEL SECURITY;

-- RLS policies for tenant isolation
CREATE POLICY churn_predictions_tenant_isolation ON analytics.churn_predictions
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY churn_features_tenant_isolation ON analytics.churn_features
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY intervention_plans_tenant_isolation ON analytics.intervention_plans
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY churn_interventions_tenant_isolation ON analytics.churn_interventions
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY intervention_outcomes_tenant_isolation ON analytics.intervention_outcomes
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY churn_model_performance_tenant_isolation ON analytics.churn_model_performance
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY churn_alerts_tenant_isolation ON analytics.churn_alerts
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY retention_campaigns_tenant_isolation ON analytics.retention_campaigns
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY campaign_enrollments_tenant_isolation ON analytics.campaign_enrollments
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO agentsystem_api;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA analytics TO agentsystem_api;

-- Create views for analytics and reporting
CREATE VIEW analytics.churn_risk_summary AS
SELECT
    cp.tenant_id,
    cp.risk_level,
    COUNT(*) as customer_count,
    AVG(cp.churn_probability) as avg_churn_probability,
    AVG(cp.confidence_score) as avg_confidence,
    COUNT(CASE WHEN cp.time_to_churn_days <= 30 THEN 1 END) as urgent_cases,
    COUNT(CASE WHEN ip.plan_id IS NOT NULL THEN 1 END) as with_intervention_plans,
    AVG(ip.estimated_clv_impact) as avg_estimated_clv_impact
FROM analytics.churn_predictions cp
LEFT JOIN analytics.intervention_plans ip ON cp.customer_id = ip.customer_id
    AND cp.tenant_id = ip.tenant_id
WHERE cp.prediction_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY cp.tenant_id, cp.risk_level
ORDER BY cp.tenant_id,
    CASE cp.risk_level
        WHEN 'critical' THEN 1
        WHEN 'very_high' THEN 2
        WHEN 'high' THEN 3
        WHEN 'medium' THEN 4
        WHEN 'low' THEN 5
        ELSE 6
    END;

CREATE VIEW analytics.intervention_effectiveness AS
SELECT
    ci.tenant_id,
    ci.intervention_type,
    COUNT(*) as total_interventions,
    COUNT(CASE WHEN ci.status = 'completed' THEN 1 END) as completed_interventions,
    AVG(ci.engagement_score) as avg_engagement_score,
    AVG(ci.cost) as avg_cost,
    COUNT(CASE WHEN io.prevented_churn = true THEN 1 END) as successful_preventions,
    AVG(io.roi) as avg_roi,
    AVG(io.clv_impact) as avg_clv_impact
FROM analytics.churn_interventions ci
LEFT JOIN analytics.intervention_outcomes io ON ci.plan_id = io.plan_id
WHERE ci.execution_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY ci.tenant_id, ci.intervention_type
ORDER BY avg_roi DESC NULLS LAST;

CREATE VIEW analytics.customer_churn_timeline AS
SELECT
    cp.tenant_id,
    cp.customer_id,
    cp.prediction_date,
    cp.churn_probability,
    cp.risk_level,
    cp.time_to_churn_days,
    ip.plan_id,
    ip.priority_score,
    ip.estimated_success_rate,
    COUNT(ci.intervention_execution_id) as interventions_executed,
    MAX(ci.completion_date) as last_intervention_date,
    io.prevented_churn,
    io.churn_date
FROM analytics.churn_predictions cp
LEFT JOIN analytics.intervention_plans ip ON cp.customer_id = ip.customer_id
    AND cp.tenant_id = ip.tenant_id
LEFT JOIN analytics.churn_interventions ci ON ip.plan_id = ci.plan_id
LEFT JOIN analytics.intervention_outcomes io ON ip.plan_id = io.plan_id
GROUP BY cp.tenant_id, cp.customer_id, cp.prediction_date, cp.churn_probability,
         cp.risk_level, cp.time_to_churn_days, ip.plan_id, ip.priority_score,
         ip.estimated_success_rate, io.prevented_churn, io.churn_date
ORDER BY cp.tenant_id, cp.customer_id, cp.prediction_date DESC;

CREATE VIEW analytics.agent_intervention_performance AS
SELECT
    ci.tenant_id,
    ci.assigned_agent,
    COUNT(*) as total_interventions,
    COUNT(CASE WHEN ci.status = 'completed' THEN 1 END) as completed_interventions,
    AVG(ci.engagement_score) as avg_engagement_score,
    AVG(ci.duration_minutes) as avg_duration_minutes,
    COUNT(CASE WHEN io.prevented_churn = true THEN 1 END) as successful_preventions,
    AVG(io.roi) as avg_roi,
    SUM(io.clv_impact) as total_clv_impact
FROM analytics.churn_interventions ci
LEFT JOIN analytics.intervention_outcomes io ON ci.plan_id = io.plan_id
WHERE ci.assigned_agent IS NOT NULL
AND ci.execution_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY ci.tenant_id, ci.assigned_agent
ORDER BY avg_roi DESC NULLS LAST;

-- Grant permissions on views
GRANT SELECT ON analytics.churn_risk_summary TO agentsystem_api;
GRANT SELECT ON analytics.intervention_effectiveness TO agentsystem_api;
GRANT SELECT ON analytics.customer_churn_timeline TO agentsystem_api;
GRANT SELECT ON analytics.agent_intervention_performance TO agentsystem_api;

-- Create materialized view for dashboard performance
CREATE MATERIALIZED VIEW analytics.churn_dashboard_stats AS
SELECT
    cp.tenant_id,
    COUNT(DISTINCT cp.customer_id) as total_at_risk_customers,
    COUNT(CASE WHEN cp.risk_level IN ('critical', 'very_high') THEN 1 END) as high_risk_customers,
    COUNT(CASE WHEN cp.time_to_churn_days <= 30 THEN 1 END) as urgent_customers,
    AVG(cp.churn_probability) as avg_churn_probability,
    COUNT(DISTINCT ip.plan_id) as active_intervention_plans,
    COUNT(CASE WHEN ci.status = 'pending' THEN 1 END) as pending_interventions,
    COUNT(CASE WHEN io.prevented_churn = true THEN 1 END) as prevented_churns_last_30d,
    SUM(io.clv_impact) as total_clv_protected,
    AVG(io.roi) as avg_intervention_roi
FROM analytics.churn_predictions cp
LEFT JOIN analytics.intervention_plans ip ON cp.customer_id = ip.customer_id
    AND cp.tenant_id = ip.tenant_id
LEFT JOIN analytics.churn_interventions ci ON ip.plan_id = ci.plan_id
LEFT JOIN analytics.intervention_outcomes io ON ip.plan_id = io.plan_id
    AND io.evaluation_date >= CURRENT_DATE - INTERVAL '30 days'
WHERE cp.prediction_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY cp.tenant_id;

-- Create index on materialized view
CREATE INDEX idx_churn_dashboard_stats_tenant ON analytics.churn_dashboard_stats(tenant_id);

-- Grant permissions on materialized view
GRANT SELECT ON analytics.churn_dashboard_stats TO agentsystem_api;

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION analytics.refresh_churn_dashboard_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.churn_dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on function
GRANT EXECUTE ON FUNCTION analytics.refresh_churn_dashboard_stats() TO agentsystem_api;

-- Function to create churn alert
CREATE OR REPLACE FUNCTION analytics.create_churn_alert(
    p_tenant_id UUID,
    p_customer_id UUID,
    p_churn_probability DECIMAL,
    p_risk_level analytics.churn_risk_level,
    p_trigger_factors JSONB
) RETURNS UUID AS $$
DECLARE
    v_alert_id UUID;
    v_alert_message TEXT;
    v_priority_score DECIMAL;
BEGIN
    -- Generate alert message
    CASE p_risk_level
        WHEN 'critical' THEN
            v_alert_message := 'CRITICAL: Customer has ' || (p_churn_probability * 100)::TEXT || '% churn probability. Immediate intervention required.';
            v_priority_score := 100;
        WHEN 'very_high' THEN
            v_alert_message := 'HIGH RISK: Customer has ' || (p_churn_probability * 100)::TEXT || '% churn probability. Urgent intervention needed.';
            v_priority_score := 80;
        WHEN 'high' THEN
            v_alert_message := 'MODERATE RISK: Customer has ' || (p_churn_probability * 100)::TEXT || '% churn probability. Intervention recommended.';
            v_priority_score := 60;
        ELSE
            v_alert_message := 'Customer churn risk detected: ' || (p_churn_probability * 100)::TEXT || '% probability.';
            v_priority_score := 40;
    END CASE;

    -- Create alert
    INSERT INTO analytics.churn_alerts (
        tenant_id, customer_id, alert_type, risk_level, churn_probability,
        trigger_factors, alert_message, priority_score
    ) VALUES (
        p_tenant_id, p_customer_id,
        CASE WHEN p_risk_level IN ('critical', 'very_high') THEN 'critical_risk' ELSE 'high_risk' END,
        p_risk_level, p_churn_probability, p_trigger_factors, v_alert_message, v_priority_score
    ) RETURNING alert_id INTO v_alert_id;

    RETURN v_alert_id;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on alert function
GRANT EXECUTE ON FUNCTION analytics.create_churn_alert TO agentsystem_api;

-- Function to calculate intervention ROI
CREATE OR REPLACE FUNCTION analytics.calculate_intervention_roi(
    p_clv_impact DECIMAL,
    p_intervention_cost DECIMAL,
    p_churn_prevented BOOLEAN
) RETURNS DECIMAL AS $$
BEGIN
    IF p_churn_prevented AND p_intervention_cost > 0 THEN
        RETURN ROUND((p_clv_impact - p_intervention_cost) / p_intervention_cost * 100, 2);
    ELSE
        RETURN 0;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on ROI function
GRANT EXECUTE ON FUNCTION analytics.calculate_intervention_roi TO agentsystem_api;

-- Function to auto-escalate high-risk alerts
CREATE OR REPLACE FUNCTION analytics.auto_escalate_alerts()
RETURNS void AS $$
DECLARE
    alert_record RECORD;
BEGIN
    -- Escalate unacknowledged critical alerts after 2 hours
    FOR alert_record IN
        SELECT alert_id, tenant_id, customer_id, risk_level
        FROM analytics.churn_alerts
        WHERE status = 'active'
        AND risk_level = 'critical'
        AND acknowledged_at IS NULL
        AND created_at < NOW() - INTERVAL '2 hours'
        AND escalation_level < 3
    LOOP
        -- Update escalation level
        UPDATE analytics.churn_alerts
        SET escalation_level = escalation_level + 1,
            escalated_at = NOW()
        WHERE alert_id = alert_record.alert_id;

        -- TODO: Send escalation notification
        -- This would integrate with notification system
    END LOOP;

    -- Escalate very high risk alerts after 4 hours
    FOR alert_record IN
        SELECT alert_id, tenant_id, customer_id, risk_level
        FROM analytics.churn_alerts
        WHERE status = 'active'
        AND risk_level = 'very_high'
        AND acknowledged_at IS NULL
        AND created_at < NOW() - INTERVAL '4 hours'
        AND escalation_level < 2
    LOOP
        -- Update escalation level
        UPDATE analytics.churn_alerts
        SET escalation_level = escalation_level + 1,
            escalated_at = NOW()
        WHERE alert_id = alert_record.alert_id;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on escalation function
GRANT EXECUTE ON FUNCTION analytics.auto_escalate_alerts() TO agentsystem_api;

-- Create trigger to automatically calculate ROI when intervention outcomes are updated
CREATE OR REPLACE FUNCTION analytics.update_intervention_roi()
RETURNS TRIGGER AS $$
BEGIN
    NEW.roi := analytics.calculate_intervention_roi(
        NEW.clv_impact,
        NEW.intervention_cost,
        NEW.prevented_churn
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER calculate_intervention_roi_trigger
    BEFORE INSERT OR UPDATE ON analytics.intervention_outcomes
    FOR EACH ROW EXECUTE FUNCTION analytics.update_intervention_roi();

-- Insert sample churn risk thresholds for existing tenants
INSERT INTO analytics.churn_features (tenant_id, customer_id, snapshot_date)
SELECT
    t.tenant_id,
