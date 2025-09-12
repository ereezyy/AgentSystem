-- Customer Success Database Schema
-- This schema supports customer health scoring, churn prediction, and expansion opportunities

-- Customer health scores tracking
CREATE TABLE IF NOT EXISTS tenant_management.customer_health_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    calculated_date DATE NOT NULL,
    usage_score FLOAT NOT NULL DEFAULT 0,
    engagement_score FLOAT NOT NULL DEFAULT 0,
    support_score FLOAT NOT NULL DEFAULT 0,
    payment_score FLOAT NOT NULL DEFAULT 0,
    overall_health_score FLOAT NOT NULL DEFAULT 0,
    health_trend VARCHAR(50) DEFAULT 'stable',
    churn_risk_score FLOAT DEFAULT 0,
    predicted_clv FLOAT DEFAULT 0,
    factors JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(tenant_id, calculated_date)
);

-- Churn predictions
CREATE TABLE IF NOT EXISTS tenant_management.churn_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    churn_probability FLOAT NOT NULL,
    churn_risk VARCHAR(50) NOT NULL,
    predicted_churn_date TIMESTAMP WITH TIME ZONE,
    risk_factors JSONB DEFAULT '[]',
    protective_factors JSONB DEFAULT '[]',
    recommended_interventions JSONB DEFAULT '[]',
    intervention_priority VARCHAR(50),
    clv_at_risk FLOAT DEFAULT 0,
    retention_value FLOAT DEFAULT 0,
    confidence_score FLOAT DEFAULT 0,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Expansion opportunities
CREATE TABLE IF NOT EXISTS tenant_management.expansion_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    opportunity_type VARCHAR(100) NOT NULL,
    current_plan VARCHAR(100),
    recommended_plan VARCHAR(100),
    estimated_revenue_increase FLOAT DEFAULT 0,
    usage_patterns JSONB DEFAULT '{}',
    growth_indicators JSONB DEFAULT '[]',
    feature_requests JSONB DEFAULT '[]',
    readiness_score FLOAT DEFAULT 0,
    best_approach VARCHAR(255),
    recommended_timing VARCHAR(100),
    success_probability FLOAT DEFAULT 0,
    implementation_effort VARCHAR(50),
    expected_roi FLOAT DEFAULT 0,
    status VARCHAR(100) DEFAULT 'identified',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Customer interventions tracking
CREATE TABLE IF NOT EXISTS tenant_management.customer_interventions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    intervention_type VARCHAR(100) NOT NULL,
    trigger_reason VARCHAR(255),
    intervention_data JSONB DEFAULT '{}',
    status VARCHAR(100) DEFAULT 'planned',
    scheduled_at TIMESTAMP WITH TIME ZONE,
    executed_at TIMESTAMP WITH TIME ZONE,
    outcome VARCHAR(255),
    success_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Customer feedback and satisfaction scores
CREATE TABLE IF NOT EXISTS tenant_management.customer_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    feedback_type VARCHAR(100) NOT NULL, -- 'nps', 'csat', 'survey', 'support_rating'
    score INTEGER,
    feedback_text TEXT,
    category VARCHAR(100),
    sentiment VARCHAR(50), -- 'positive', 'neutral', 'negative'
    source VARCHAR(100), -- 'email', 'in_app', 'support_ticket', 'survey'
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Customer onboarding progress tracking
CREATE TABLE IF NOT EXISTS tenant_management.onboarding_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    stage VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'not_started', -- 'not_started', 'in_progress', 'completed', 'skipped'
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    completion_percentage FLOAT DEFAULT 0,
    blockers JSONB DEFAULT '[]',
    assistance_needed BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Customer success metrics aggregation
CREATE TABLE IF NOT EXISTS tenant_management.customer_success_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    metric_date DATE NOT NULL,
    time_to_value_days INTEGER,
    feature_adoption_rate FLOAT DEFAULT 0,
    support_ticket_count INTEGER DEFAULT 0,
    login_frequency FLOAT DEFAULT 0,
    api_usage_growth FLOAT DEFAULT 0,
    user_engagement_score FLOAT DEFAULT 0,
    product_stickiness_score FLOAT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(tenant_id, metric_date)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_customer_health_scores_tenant_date
ON tenant_management.customer_health_scores(tenant_id, calculated_date DESC);

CREATE INDEX IF NOT EXISTS idx_customer_health_scores_risk
ON tenant_management.customer_health_scores(churn_risk_score DESC, overall_health_score ASC);

CREATE INDEX IF NOT EXISTS idx_churn_predictions_tenant_risk
ON tenant_management.churn_predictions(tenant_id, churn_risk);

CREATE INDEX IF NOT EXISTS idx_churn_predictions_probability
ON tenant_management.churn_predictions(churn_probability DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_expansion_opportunities_tenant_status
ON tenant_management.expansion_opportunities(tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_expansion_opportunities_revenue
ON tenant_management.expansion_opportunities(estimated_revenue_increase DESC, readiness_score DESC);

CREATE INDEX IF NOT EXISTS idx_customer_interventions_tenant_type
ON tenant_management.customer_interventions(tenant_id, intervention_type);

CREATE INDEX IF NOT EXISTS idx_customer_interventions_status
ON tenant_management.customer_interventions(status, scheduled_at);

CREATE INDEX IF NOT EXISTS idx_customer_feedback_tenant_type
ON tenant_management.customer_feedback(tenant_id, feedback_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_onboarding_progress_tenant_stage
ON tenant_management.onboarding_progress(tenant_id, stage, status);

CREATE INDEX IF NOT EXISTS idx_customer_success_metrics_tenant_date
ON tenant_management.customer_success_metrics(tenant_id, metric_date DESC);

-- Views for common queries
CREATE OR REPLACE VIEW tenant_management.customer_health_dashboard AS
SELECT
    t.id as tenant_id,
    t.company_name,
    t.plan_type,
    t.created_at as tenant_since,
    chs.overall_health_score,
    chs.health_trend,
    chs.churn_risk_score,
    cp.churn_probability,
    cp.churn_risk,
    cp.predicted_churn_date,
    eo.estimated_revenue_increase as expansion_potential,
    COUNT(ci.id) as active_interventions
FROM tenant_management.tenants t
LEFT JOIN tenant_management.customer_health_scores chs ON t.id = chs.tenant_id
    AND chs.calculated_date = (
        SELECT MAX(calculated_date)
        FROM tenant_management.customer_health_scores
        WHERE tenant_id = t.id
    )
LEFT JOIN tenant_management.churn_predictions cp ON t.id = cp.tenant_id
    AND cp.created_at = (
        SELECT MAX(created_at)
        FROM tenant_management.churn_predictions
        WHERE tenant_id = t.id
    )
LEFT JOIN tenant_management.expansion_opportunities eo ON t.id = eo.tenant_id
    AND eo.status = 'identified'
LEFT JOIN tenant_management.customer_interventions ci ON t.id = ci.tenant_id
    AND ci.status IN ('planned', 'in_progress')
WHERE t.is_active = true
GROUP BY t.id, t.company_name, t.plan_type, t.created_at,
         chs.overall_health_score, chs.health_trend, chs.churn_risk_score,
         cp.churn_probability, cp.churn_risk, cp.predicted_churn_date,
         eo.estimated_revenue_increase;

-- Trigger to update customer success metrics when health scores change
CREATE OR REPLACE FUNCTION update_customer_success_metrics()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO tenant_management.customer_success_metrics (
        tenant_id, metric_date, user_engagement_score
    ) VALUES (
        NEW.tenant_id, NEW.calculated_date, NEW.engagement_score
    ) ON CONFLICT (tenant_id, metric_date) DO UPDATE SET
        user_engagement_score = EXCLUDED.user_engagement_score,
        updated_at = NOW();

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_customer_success_metrics
    AFTER INSERT OR UPDATE ON tenant_management.customer_health_scores
    FOR EACH ROW
    EXECUTE FUNCTION update_customer_success_metrics();

-- Function to calculate customer health score components
CREATE OR REPLACE FUNCTION calculate_tenant_health_components(target_tenant_id UUID)
RETURNS TABLE(
    usage_score FLOAT,
    engagement_score FLOAT,
    support_score FLOAT,
    payment_score FLOAT
) AS $$
DECLARE
    usage_data RECORD;
    engagement_data RECORD;
    support_data RECORD;
    payment_data RECORD;
BEGIN
    -- Usage score calculation (0-100)
    SELECT
        COALESCE(AVG(daily_tokens), 0) as avg_daily_tokens,
        COALESCE(COUNT(DISTINCT DATE(created_at)), 0) as active_days
    INTO usage_data
    FROM usage_tracking.usage_logs
    WHERE tenant_id = target_tenant_id
    AND created_at >= NOW() - INTERVAL '30 days';

    usage_score := LEAST(100, (usage_data.avg_daily_tokens / 1000.0) * 10 +
                         (usage_data.active_days / 30.0) * 50);

    -- Engagement score calculation (0-100)
    SELECT
        COALESCE(COUNT(DISTINCT user_id), 0) as active_users,
        COALESCE(AVG(session_duration_minutes), 0) as avg_session_duration
    INTO engagement_data
    FROM tenant_management.user_sessions
    WHERE tenant_id = target_tenant_id
    AND created_at >= NOW() - INTERVAL '30 days';

    engagement_score := LEAST(100, engagement_data.active_users * 20 +
                              LEAST(engagement_data.avg_session_duration / 60.0, 1) * 30);

    -- Support score calculation (0-100, higher is better)
    SELECT
        COALESCE(COUNT(*), 0) as ticket_count,
        COALESCE(AVG(resolution_time_hours), 24) as avg_resolution_time
    INTO support_data
    FROM tenant_management.support_tickets
    WHERE tenant_id = target_tenant_id
    AND created_at >= NOW() - INTERVAL '30 days';

    support_score := GREATEST(0, 100 - support_data.ticket_count * 5 -
                             GREATEST(0, support_data.avg_resolution_time - 24) * 2);

    -- Payment score calculation (0-100)
    SELECT
        COALESCE(COUNT(*), 0) as failed_payments,
        COALESCE(SUM(CASE WHEN status = 'paid' THEN 1 ELSE 0 END), 0) as successful_payments
    INTO payment_data
    FROM billing.payment_history
    WHERE tenant_id = target_tenant_id
    AND created_at >= NOW() - INTERVAL '90 days';

    payment_score := GREATEST(0, 100 - payment_data.failed_payments * 20);

    RETURN QUERY SELECT usage_score, engagement_score, support_score, payment_score;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE tenant_management.customer_health_scores IS 'Stores calculated health scores for each tenant with historical tracking';
COMMENT ON TABLE tenant_management.churn_predictions IS 'ML-based churn predictions with risk factors and intervention recommendations';
COMMENT ON TABLE tenant_management.expansion_opportunities IS 'Revenue expansion opportunities identified through usage analysis';
COMMENT ON TABLE tenant_management.customer_interventions IS 'Tracks customer success interventions and their outcomes';
COMMENT ON TABLE tenant_management.customer_feedback IS 'Customer feedback, ratings, and satisfaction scores';
COMMENT ON TABLE tenant_management.onboarding_progress IS 'Tracks customer onboarding progress and completion status';
COMMENT ON TABLE tenant_management.customer_success_metrics IS 'Aggregated customer success metrics for reporting and analysis';
