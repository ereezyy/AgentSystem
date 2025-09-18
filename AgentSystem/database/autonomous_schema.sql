-- AgentSystem Autonomous Operations Database Schema
-- Revolutionary self-operating AI system with autonomous decision-making

-- Autonomous decisions tracking
CREATE TABLE autonomous_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    decision_id VARCHAR(200) NOT NULL UNIQUE,
    decision_type VARCHAR(50) NOT NULL CHECK (decision_type IN ('scaling', 'optimization', 'resource_allocation', 'cost_management', 'performance_tuning', 'security_response', 'customer_intervention', 'business_strategy')),
    priority VARCHAR(20) NOT NULL CHECK (priority IN ('critical', 'high', 'medium', 'low', 'scheduled')),
    confidence_score DECIMAL(5,4) NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    reasoning TEXT NOT NULL,
    proposed_actions JSONB NOT NULL,
    expected_outcomes JSONB NOT NULL,
    risk_assessment JSONB NOT NULL,
    approval_required BOOLEAN NOT NULL DEFAULT FALSE,
    estimated_impact JSONB,
    system_state_snapshot JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_deadline TIMESTAMP,
    approved_at TIMESTAMP,
    approved_by UUID REFERENCES users(id),
    executed_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'executing', 'completed', 'failed', 'cancelled')),

    INDEX idx_autonomous_decisions_tenant (tenant_id),
    INDEX idx_autonomous_decisions_type (decision_type),
    INDEX idx_autonomous_decisions_priority (priority),
    INDEX idx_autonomous_decisions_status (status),
    INDEX idx_autonomous_decisions_created (created_at DESC)
);

-- Autonomous action executions
CREATE TABLE autonomous_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id VARCHAR(200) NOT NULL REFERENCES autonomous_decisions(decision_id),
    action_id VARCHAR(200) NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    expected_duration INTEGER, -- minutes
    rollback_plan JSONB,
    success_criteria JSONB,
    monitoring_metrics JSONB,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'executing', 'success', 'failed', 'rolled_back')),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time_minutes INTEGER,
    success_metrics JSONB,
    failure_reason TEXT,
    rollback_executed BOOLEAN DEFAULT FALSE,
    rollback_completed_at TIMESTAMP,

    INDEX idx_executions_decision (decision_id),
    INDEX idx_executions_status (status),
    INDEX idx_executions_type (action_type),
    INDEX idx_executions_started (started_at DESC)
);

-- System health monitoring for autonomous decisions
CREATE TABLE autonomous_monitoring (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    monitoring_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    current_value DECIMAL(15,6),
    baseline_value DECIMAL(15,6),
    threshold_warning DECIMAL(15,6),
    threshold_critical DECIMAL(15,6),
    deviation_percentage DECIMAL(8,4),
    trigger_autonomous_action BOOLEAN DEFAULT FALSE,
    action_triggered_at TIMESTAMP,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_monitoring_tenant_type (tenant_id, monitoring_type),
    INDEX idx_monitoring_metric (metric_name),
    INDEX idx_monitoring_trigger (trigger_autonomous_action),
    INDEX idx_monitoring_timestamp (timestamp DESC)
);

-- Autonomous decision outcomes tracking
CREATE TABLE decision_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id VARCHAR(200) NOT NULL REFERENCES autonomous_decisions(decision_id),
    outcome_type VARCHAR(100) NOT NULL,
    predicted_outcome JSONB,
    actual_outcome JSONB,
    outcome_accuracy DECIMAL(5,4),
    business_impact JSONB,
    cost_impact DECIMAL(15,2),
    performance_impact JSONB,
    customer_impact JSONB,
    lessons_learned TEXT,
    outcome_recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_outcomes_decision (decision_id),
    INDEX idx_outcomes_type (outcome_type),
    INDEX idx_outcomes_accuracy (outcome_accuracy DESC)
);

-- Autonomous system configuration
CREATE TABLE autonomous_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    autonomy_level VARCHAR(30) NOT NULL DEFAULT 'semi_autonomous' CHECK (autonomy_level IN ('monitoring', 'advisory', 'semi_autonomous', 'autonomous', 'fully_autonomous')),
    decision_confidence_threshold DECIMAL(5,4) DEFAULT 0.85,
    max_concurrent_actions INTEGER DEFAULT 10,
    auto_scaling_enabled BOOLEAN DEFAULT TRUE,
    auto_optimization_enabled BOOLEAN DEFAULT TRUE,
    auto_cost_management_enabled BOOLEAN DEFAULT TRUE,
    auto_customer_intervention_enabled BOOLEAN DEFAULT TRUE,
    approval_required_threshold DECIMAL(15,2) DEFAULT 10000, -- Dollar amount requiring approval
    emergency_override_enabled BOOLEAN DEFAULT TRUE,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(tenant_id),
    INDEX idx_autonomous_config_level (autonomy_level)
);

-- Self-healing incidents
CREATE TABLE self_healing_incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    incident_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    issue_description TEXT NOT NULL,
    detection_method VARCHAR(100),
    healing_action_taken JSONB,
    healing_status VARCHAR(20) DEFAULT 'pending' CHECK (healing_status IN ('pending', 'healing', 'healed', 'failed', 'manual_required')),
    time_to_detection_seconds INTEGER,
    time_to_healing_seconds INTEGER,
    success_probability DECIMAL(5,4),
    business_impact_prevented JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    healed_at TIMESTAMP,
    verified_at TIMESTAMP,

    INDEX idx_healing_tenant_severity (tenant_id, severity),
    INDEX idx_healing_status (healing_status),
    INDEX idx_healing_detected (detected_at DESC)
);

-- Autonomous learning from outcomes
CREATE TABLE autonomous_learning (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    learning_type VARCHAR(100) NOT NULL,
    trigger_scenario JSONB NOT NULL,
    action_taken JSONB NOT NULL,
    outcome_success BOOLEAN NOT NULL,
    outcome_data JSONB,
    confidence_adjustment DECIMAL(6,4), -- How much to adjust confidence for similar scenarios
    pattern_recognition JSONB,
    improvement_suggestions JSONB,
    applied_to_future_decisions BOOLEAN DEFAULT FALSE,
    learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_learning_tenant_type (tenant_id, learning_type),
    INDEX idx_learning_success (outcome_success),
    INDEX idx_learning_applied (applied_to_future_decisions)
);

-- Performance optimization history
CREATE TABLE autonomous_optimizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    optimization_type VARCHAR(100) NOT NULL,
    baseline_metrics JSONB NOT NULL,
    optimization_actions JSONB NOT NULL,
    post_optimization_metrics JSONB,
    improvement_percentage DECIMAL(8,4),
    cost_savings DECIMAL(15,2),
    performance_gain JSONB,
    optimization_confidence DECIMAL(5,4),
    rollback_available BOOLEAN DEFAULT TRUE,
    rollback_executed BOOLEAN DEFAULT FALSE,
    optimization_started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    optimization_completed_at TIMESTAMP,

    INDEX idx_optimizations_tenant_type (tenant_id, optimization_type),
    INDEX idx_optimizations_improvement (improvement_percentage DESC),
    INDEX idx_optimizations_savings (cost_savings DESC)
);

-- Autonomous business intelligence alerts
CREATE TABLE autonomous_bi_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('info', 'warning', 'critical', 'urgent')),
    trigger_conditions JSONB NOT NULL,
    business_context JSONB,
    recommended_autonomous_actions JSONB,
    autonomous_action_taken BOOLEAN DEFAULT FALSE,
    action_taken_at TIMESTAMP,
    human_override_required BOOLEAN DEFAULT FALSE,
    alert_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_method VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_bi_alerts_tenant_severity (tenant_id, severity),
    INDEX idx_bi_alerts_resolved (alert_resolved),
    INDEX idx_bi_alerts_autonomous (autonomous_action_taken)
);

-- Insert default autonomous configurations for different tenant tiers
INSERT INTO autonomous_config (tenant_id, autonomy_level, decision_confidence_threshold, auto_scaling_enabled, auto_optimization_enabled)
SELECT id,
       CASE
           WHEN plan = 'enterprise' THEN 'autonomous'
           WHEN plan = 'pro' THEN 'semi_autonomous'
           ELSE 'advisory'
       END,
       CASE
           WHEN plan = 'enterprise' THEN 0.80
           WHEN plan = 'pro' THEN 0.85
           ELSE 0.90
       END,
       CASE
           WHEN plan IN ('enterprise', 'pro') THEN TRUE
           ELSE FALSE
       END,
       TRUE
FROM tenants
WHERE NOT EXISTS (SELECT 1 FROM autonomous_config WHERE autonomous_config.tenant_id = tenants.id);

-- Function to trigger autonomous decision evaluation
CREATE OR REPLACE FUNCTION trigger_autonomous_evaluation(p_tenant_id UUID, p_trigger_type VARCHAR(100), p_trigger_data JSONB)
RETURNS UUID AS $$
DECLARE
    config_row autonomous_config%ROWTYPE;
    decision_uuid UUID;
BEGIN
    -- Get autonomous configuration
    SELECT * INTO config_row
    FROM autonomous_config
    WHERE tenant_id = p_tenant_id;

    -- Check if autonomous actions are enabled for this trigger type
    CASE p_trigger_type
        WHEN 'scaling' THEN
            IF NOT config_row.auto_scaling_enabled THEN
                RETURN NULL;
            END IF;
        WHEN 'optimization' THEN
            IF NOT config_row.auto_optimization_enabled THEN
                RETURN NULL;
            END IF;
        WHEN 'cost_management' THEN
            IF NOT config_row.auto_cost_management_enabled THEN
                RETURN NULL;
            END IF;
        WHEN 'customer_intervention' THEN
            IF NOT config_row.auto_customer_intervention_enabled THEN
                RETURN NULL;
            END IF;
    END CASE;

    -- Create autonomous monitoring entry
    INSERT INTO autonomous_monitoring (
        tenant_id, monitoring_type, metric_name, trigger_autonomous_action, action_triggered_at
    ) VALUES (
        p_tenant_id, p_trigger_type, 'autonomous_trigger', TRUE, CURRENT_TIMESTAMP
    );

    -- Log for processing by autonomous engine
    INSERT INTO autonomous_decisions (
        tenant_id, decision_id, decision_type, priority, confidence_score,
        reasoning, proposed_actions, expected_outcomes, risk_assessment,
        approval_required, system_state_snapshot
    ) VALUES (
        p_tenant_id,
        'trigger_' || p_trigger_type || '_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP),
        p_trigger_type,
        'medium',
        0.0, -- Will be calculated by engine
        'Triggered by system monitoring',
        '[]'::jsonb,
        '{}'::jsonb,
        '{}'::jsonb,
        config_row.autonomy_level NOT IN ('autonomous', 'fully_autonomous'),
        p_trigger_data
    ) RETURNING id INTO decision_uuid;

    RETURN decision_uuid;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate autonomous system efficiency
CREATE OR REPLACE FUNCTION calculate_autonomous_efficiency()
RETURNS TABLE(
    tenant_id UUID,
    total_decisions INTEGER,
    successful_decisions INTEGER,
    avg_confidence DECIMAL(5,4),
    avg_execution_time INTEGER,
    cost_savings_total DECIMAL(15,2),
    efficiency_score DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ad.tenant_id,
        COUNT(ad.id)::INTEGER as total_decisions,
        COUNT(CASE WHEN ad.status = 'completed' THEN 1 END)::INTEGER as successful_decisions,
        AVG(ad.confidence_score) as avg_confidence,
        AVG(EXTRACT(EPOCH FROM (ad.completed_at - ad.executed_at))/60)::INTEGER as avg_execution_time,
        COALESCE(SUM(ao.cost_savings), 0) as cost_savings_total,
        (COUNT(CASE WHEN ad.status = 'completed' THEN 1 END)::DECIMAL / NULLIF(COUNT(ad.id), 0) * 100) as efficiency_score
    FROM autonomous_decisions ad
    LEFT JOIN autonomous_optimizations ao ON ad.tenant_id = ao.tenant_id
    WHERE ad.created_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY ad.tenant_id;
END;
$$ LANGUAGE plpgsql;

-- Trigger for autonomous learning from decision outcomes
CREATE OR REPLACE FUNCTION autonomous_learning_trigger()
RETURNS TRIGGER AS $$
DECLARE
    success_rate DECIMAL(5,4);
    pattern_data JSONB;
BEGIN
    -- When a decision is completed, analyze for learning opportunities
    IF NEW.status = 'completed' AND OLD.status != 'completed' THEN
        -- Calculate recent success rate for this decision type
        SELECT
            COUNT(CASE WHEN status = 'completed' THEN 1 END)::DECIMAL / COUNT(*)
        INTO success_rate
        FROM autonomous_decisions
        WHERE decision_type = NEW.decision_type
        AND tenant_id = NEW.tenant_id
        AND created_at >= CURRENT_DATE - INTERVAL '7 days';

        -- Build pattern recognition data
        pattern_data := jsonb_build_object(
            'decision_type', NEW.decision_type,
            'confidence_score', NEW.confidence_score,
            'system_state', NEW.system_state_snapshot,
            'success_rate', success_rate
        );

        -- Store learning data
        INSERT INTO autonomous_learning (
            tenant_id, learning_type, trigger_scenario, action_taken,
            outcome_success, outcome_data, confidence_adjustment, pattern_recognition
        ) VALUES (
            NEW.tenant_id,
            'decision_outcome',
            NEW.system_state_snapshot,
            NEW.proposed_actions,
            TRUE,
            jsonb_build_object('execution_time', EXTRACT(EPOCH FROM (NEW.completed_at - NEW.executed_at))),
            CASE
                WHEN success_rate > 0.9 THEN 0.02  -- Increase confidence
                WHEN success_rate < 0.7 THEN -0.05 -- Decrease confidence
                ELSE 0.0
            END,
            pattern_data
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER autonomous_learning_outcome_trigger
    AFTER UPDATE ON autonomous_decisions
    FOR EACH ROW
    EXECUTE FUNCTION autonomous_learning_trigger();

-- View for autonomous operations dashboard
CREATE VIEW autonomous_operations_dashboard AS
SELECT
    ac.tenant_id,
    ac.autonomy_level,
    COUNT(ad.id) as total_decisions_24h,
    COUNT(CASE WHEN ad.status = 'completed' THEN 1 END) as successful_decisions_24h,
    COUNT(CASE WHEN ad.status = 'executing' THEN 1 END) as currently_executing,
    COUNT(CASE WHEN ad.approval_required = TRUE AND ad.status = 'pending' THEN 1 END) as pending_approvals,
    AVG(ad.confidence_score) as avg_confidence_24h,
    COUNT(shi.id) as healing_incidents_24h,
    COUNT(CASE WHEN shi.healing_status = 'healed' THEN 1 END) as successful_healings_24h,
    COALESCE(SUM(ao.cost_savings), 0) as cost_savings_24h
FROM autonomous_config ac
LEFT JOIN autonomous_decisions ad ON ac.tenant_id = ad.tenant_id
    AND ad.created_at >= CURRENT_DATE - INTERVAL '1 day'
LEFT JOIN self_healing_incidents shi ON ac.tenant_id = shi.tenant_id
    AND shi.detected_at >= CURRENT_DATE - INTERVAL '1 day'
LEFT JOIN autonomous_optimizations ao ON ac.tenant_id = ao.tenant_id
    AND ao.optimization_started_at >= CURRENT_DATE - INTERVAL '1 day'
GROUP BY ac.tenant_id, ac.autonomy_level;

-- Function to evaluate system readiness for full autonomy
CREATE OR REPLACE FUNCTION evaluate_autonomy_readiness(p_tenant_id UUID)
RETURNS JSONB AS $$
DECLARE
    success_rate DECIMAL(5,4);
    avg_confidence DECIMAL(5,4);
    cost_savings DECIMAL(15,2);
    incident_rate DECIMAL(8,4);
    readiness_score DECIMAL(5,2);
    recommendation VARCHAR(30);
    readiness_data JSONB;
BEGIN
    -- Calculate success rate over last 30 days
    SELECT
        COUNT(CASE WHEN status = 'completed' THEN 1 END)::DECIMAL / NULLIF(COUNT(*), 0),
        AVG(confidence_score),
        COALESCE(SUM(CASE WHEN do.cost_impact < 0 THEN ABS(do.cost_impact) ELSE 0 END), 0)
    INTO success_rate, avg_confidence, cost_savings
    FROM autonomous_decisions ad
    LEFT JOIN decision_outcomes do ON ad.decision_id = do.decision_id
    WHERE ad.tenant_id = p_tenant_id
    AND ad.created_at >= CURRENT_DATE - INTERVAL '30 days';

    -- Calculate incident rate
    SELECT COUNT(*)::DECIMAL / 30
    INTO incident_rate
    FROM self_healing_incidents
    WHERE tenant_id = p_tenant_id
    AND detected_at >= CURRENT_DATE - INTERVAL '30 days'
    AND healing_status = 'failed';

    -- Calculate readiness score
    readiness_score := (
        COALESCE(success_rate, 0) * 40 +
        COALESCE(avg_confidence, 0) * 100 * 30 +
        LEAST(1, GREATEST(0, cost_savings / 10000)) * 20 +
        GREATEST(0, 1 - incident_rate) * 10
    );

    -- Make recommendation
    IF readiness_score >= 85 THEN
        recommendation := 'fully_autonomous';
    ELSIF readiness_score >= 70 THEN
        recommendation := 'autonomous';
    ELSIF readiness_score >= 55 THEN
        recommendation := 'semi_autonomous';
    ELSE
        recommendation := 'advisory';
    END IF;

    readiness_data := jsonb_build_object(
        'readiness_score', readiness_score,
        'recommendation', recommendation,
        'success_rate', COALESCE(success_rate, 0),
        'avg_confidence', COALESCE(avg_confidence, 0),
        'cost_savings_30d', cost_savings,
        'incident_rate', incident_rate,
        'evaluation_date', CURRENT_TIMESTAMP
    );

    RETURN readiness_data;
END;
$$ LANGUAGE plpgsql;

-- Create indexes for optimal query performance
CREATE INDEX idx_decisions_tenant_created ON autonomous_decisions(tenant_id, created_at DESC);
CREATE INDEX idx_executions_decision_status ON autonomous_executions(decision_id, status);
CREATE INDEX idx_monitoring_tenant_timestamp ON autonomous_monitoring(tenant_id, timestamp DESC);
CREATE INDEX idx_outcomes_decision_accuracy ON decision_outcomes(decision_id, outcome_accuracy DESC);

COMMENT ON TABLE autonomous_decisions IS 'Autonomous AI decision tracking with full reasoning and outcomes';
COMMENT ON TABLE autonomous_executions IS 'Execution tracking for autonomous actions with rollback capabilities';
COMMENT ON TABLE autonomous_monitoring IS 'Real-time system monitoring for autonomous decision triggers';
COMMENT ON TABLE decision_outcomes IS 'Autonomous decision outcome tracking for continuous learning';
COMMENT ON TABLE autonomous_config IS 'Per-tenant autonomous operation configuration and limits';
COMMENT ON TABLE self_healing_incidents IS 'Self-healing system incident tracking and resolution';
COMMENT ON TABLE autonomous_learning IS 'Machine learning from autonomous operation outcomes';
COMMENT ON TABLE autonomous_optimizations IS 'Performance optimization tracking and results';
