-- AgentSystem Self-Optimizing Workflow Engine Database Schema
-- Machine learning-powered workflow optimization and performance tracking

-- Workflow performance metrics tracking
CREATE TABLE workflow_performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL,
    execution_time_seconds DECIMAL(10,3) NOT NULL,
    success_rate DECIMAL(5,4) NOT NULL,
    error_rate DECIMAL(5,4) NOT NULL,
    cost_per_execution DECIMAL(10,6) NOT NULL,
    throughput_per_hour DECIMAL(10,2),
    user_satisfaction_score DECIMAL(3,2),
    resource_utilization JSONB NOT NULL DEFAULT '{}',
    latency_percentiles JSONB NOT NULL DEFAULT '{}',
    analysis_period_start TIMESTAMP NOT NULL,
    analysis_period_end TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_workflow_perf_tenant_workflow (tenant_id, workflow_id),
    INDEX idx_workflow_perf_execution_time (execution_time_seconds),
    INDEX idx_workflow_perf_success_rate (success_rate DESC),
    INDEX idx_workflow_perf_created (created_at DESC)
);

-- Optimization opportunities identification
CREATE TABLE optimization_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    opportunity_id VARCHAR(200) NOT NULL UNIQUE,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL,
    optimization_type VARCHAR(50) NOT NULL CHECK (optimization_type IN ('performance', 'cost', 'reliability', 'user_experience', 'resource_efficiency', 'execution_time')),
    strategy VARCHAR(50) NOT NULL CHECK (strategy IN ('parallel_execution', 'caching', 'batch_processing', 'intelligent_routing', 'resource_pooling', 'predictive_scaling', 'dynamic_configuration')),
    potential_improvement JSONB NOT NULL,
    implementation_effort VARCHAR(20) CHECK (implementation_effort IN ('low', 'medium', 'high')),
    confidence_score DECIMAL(5,4) NOT NULL,
    business_impact JSONB NOT NULL,
    recommended_actions JSONB NOT NULL DEFAULT '[]',
    status VARCHAR(20) DEFAULT 'identified' CHECK (status IN ('identified', 'approved', 'implementing', 'completed', 'rejected')),
    identified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    implemented_at TIMESTAMP,

    INDEX idx_opportunities_tenant_workflow (tenant_id, workflow_id),
    INDEX idx_opportunities_type (optimization_type),
    INDEX idx_opportunities_confidence (confidence_score DESC),
    INDEX idx_opportunities_status (status)
);

-- Optimization implementations tracking
CREATE TABLE workflow_optimizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    optimization_id VARCHAR(200) NOT NULL UNIQUE,
    opportunity_id VARCHAR(200) NOT NULL REFERENCES optimization_opportunities(opportunity_id),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL,
    optimization_strategy VARCHAR(50) NOT NULL,
    baseline_metrics JSONB NOT NULL,
    implementation_details JSONB NOT NULL,
    post_optimization_metrics JSONB,
    actual_improvement JSONB,
    success_metrics JSONB,
    rollback_available BOOLEAN DEFAULT TRUE,
    rollback_executed BOOLEAN DEFAULT FALSE,
    rollback_reason TEXT,
    implementation_started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    implementation_completed_at TIMESTAMP,
    monitoring_until TIMESTAMP,

    INDEX idx_optimizations_tenant_workflow (tenant_id, workflow_id),
    INDEX idx_optimizations_strategy (optimization_strategy),
    INDEX idx_optimizations_started (implementation_started_at DESC)
);

-- Workflow execution resource usage
CREATE TABLE workflow_resource_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL,
    execution_id UUID NOT NULL,
    step_id VARCHAR(100),
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(10,2), -- MB
    io_operations INTEGER,
    network_bytes BIGINT,
    ai_tokens_used INTEGER,
    ai_cost DECIMAL(10,6),
    execution_duration_seconds DECIMAL(10,3),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_resource_usage_tenant_workflow (tenant_id, workflow_id),
    INDEX idx_resource_usage_execution (execution_id),
    INDEX idx_resource_usage_timestamp (timestamp DESC)
);

-- Workflow feedback for satisfaction tracking
CREATE TABLE workflow_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL,
    execution_id UUID,
    user_id UUID REFERENCES users(id),
    satisfaction_score INTEGER CHECK (satisfaction_score BETWEEN 1 AND 5),
    feedback_text TEXT,
    improvement_suggestions TEXT,
    execution_speed_rating INTEGER CHECK (execution_speed_rating BETWEEN 1 AND 5),
    accuracy_rating INTEGER CHECK (accuracy_rating BETWEEN 1 AND 5),
    ease_of_use_rating INTEGER CHECK (ease_of_use_rating BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_workflow_feedback_workflow (workflow_id),
    INDEX idx_workflow_feedback_satisfaction (satisfaction_score),
    INDEX idx_workflow_feedback_created (created_at DESC)
);

-- Optimization learning and patterns
CREATE TABLE optimization_learning (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    optimization_pattern VARCHAR(200) NOT NULL,
    workflow_characteristics JSONB NOT NULL,
    strategy_applied VARCHAR(50) NOT NULL,
    outcome_success BOOLEAN NOT NULL,
    improvement_achieved JSONB,
    confidence_level DECIMAL(5,4),
    pattern_frequency INTEGER DEFAULT 1,
    success_rate DECIMAL(5,4),
    learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_applied TIMESTAMP,

    INDEX idx_learning_pattern (optimization_pattern),
    INDEX idx_learning_strategy (strategy_applied),
    INDEX idx_learning_success (outcome_success)
);

-- Workflow bottleneck analysis
CREATE TABLE workflow_bottlenecks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL,
    step_id VARCHAR(100) NOT NULL,
    bottleneck_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    impact_metrics JSONB NOT NULL,
    suggested_optimizations JSONB NOT NULL DEFAULT '[]',
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution_method TEXT,

    INDEX idx_bottlenecks_workflow_step (workflow_id, step_id),
    INDEX idx_bottlenecks_severity (severity),
    INDEX idx_bottlenecks_detected (detected_at DESC)
);

-- Performance baselines for comparison
CREATE TABLE workflow_performance_baselines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL,
    baseline_type VARCHAR(50) NOT NULL,
    baseline_metrics JSONB NOT NULL,
    calculation_period TSRANGE NOT NULL,
    samples_count INTEGER NOT NULL,
    confidence_interval JSONB,
    established_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,

    UNIQUE(tenant_id, workflow_id, baseline_type),
    INDEX idx_baselines_workflow (workflow_id),
    INDEX idx_baselines_active (is_active)
);

-- Real-time optimization monitoring
CREATE TABLE optimization_monitoring (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    optimization_id VARCHAR(200) NOT NULL REFERENCES workflow_optimizations(optimization_id),
    monitoring_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    performance_metrics JSONB NOT NULL,
    improvement_metrics JSONB,
    anomalies_detected JSONB DEFAULT '[]',
    optimization_health_score DECIMAL(5,2),
    rollback_recommended BOOLEAN DEFAULT FALSE,
    monitoring_notes TEXT,

    INDEX idx_optimization_monitoring_id (optimization_id),
    INDEX idx_optimization_monitoring_timestamp (monitoring_timestamp DESC),
    INDEX idx_optimization_monitoring_health (optimization_health_score DESC)
);

-- Insert default performance baselines
INSERT INTO workflow_performance_baselines (tenant_id, workflow_id, baseline_type, baseline_metrics, calculation_period, samples_count)
SELECT DISTINCT
    tenant_id,
    workflow_id,
    'execution_time',
    jsonb_build_object(
        'avg_execution_time', AVG(execution_time_seconds),
        'p95_execution_time', PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_seconds),
        'success_rate', COUNT(CASE WHEN status = 'completed' THEN 1 END)::FLOAT / COUNT(*)
    ),
    tsrange(CURRENT_DATE - INTERVAL '30 days', CURRENT_DATE),
    COUNT(*)
FROM workflow_executions
WHERE executed_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY tenant_id, workflow_id
HAVING COUNT(*) >= 10;  -- Minimum sample size

-- Function to calculate workflow optimization score
CREATE OR REPLACE FUNCTION calculate_optimization_score(p_workflow_id UUID)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    current_performance RECORD;
    baseline_performance RECORD;
    optimization_score DECIMAL(5,2);
    performance_ratio DECIMAL(6,4);
    cost_ratio DECIMAL(6,4);
    satisfaction_ratio DECIMAL(6,4);
BEGIN
    -- Get current performance
    SELECT
        AVG(execution_time_seconds) as avg_time,
        AVG(cost_per_execution) as avg_cost,
        AVG(user_satisfaction_score) as avg_satisfaction
    INTO current_performance
    FROM workflow_performance_metrics
    WHERE workflow_id = p_workflow_id
    AND created_at >= CURRENT_DATE - INTERVAL '7 days';

    -- Get baseline performance
    SELECT
        (baseline_metrics->>'avg_execution_time')::DECIMAL as baseline_time,
        (baseline_metrics->>'avg_cost')::DECIMAL as baseline_cost,
        (baseline_metrics->>'avg_satisfaction')::DECIMAL as baseline_satisfaction
    INTO baseline_performance
    FROM workflow_performance_baselines
    WHERE workflow_id = p_workflow_id
    AND baseline_type = 'execution_time'
    AND is_active = TRUE;

    -- Calculate ratios (lower is better for time/cost, higher is better for satisfaction)
    performance_ratio := COALESCE(baseline_performance.baseline_time / NULLIF(current_performance.avg_time, 0), 1);
    cost_ratio := COALESCE(baseline_performance.baseline_cost / NULLIF(current_performance.avg_cost, 0), 1);
    satisfaction_ratio := COALESCE(current_performance.avg_satisfaction / NULLIF(baseline_performance.baseline_satisfaction, 0), 1);

    -- Calculate weighted optimization score
    optimization_score := (
        performance_ratio * 40 +  -- 40% weight on performance
        cost_ratio * 35 +         -- 35% weight on cost
        satisfaction_ratio * 25   -- 25% weight on satisfaction
    );

    RETURN LEAST(100, GREATEST(0, optimization_score));
END;
$$ LANGUAGE plpgsql;

-- Function to detect workflow bottlenecks
CREATE OR REPLACE FUNCTION detect_workflow_bottlenecks(p_workflow_id UUID)
RETURNS void AS $$
DECLARE
    step_performance RECORD;
    avg_step_time DECIMAL(10,3);
    bottleneck_threshold DECIMAL(10,3);
BEGIN
    -- Calculate average step execution time
    SELECT AVG(execution_duration_seconds) INTO avg_step_time
    FROM workflow_resource_usage
    WHERE workflow_id = p_workflow_id
    AND timestamp >= CURRENT_DATE - INTERVAL '7 days';

    -- Set bottleneck threshold (2x average)
    bottleneck_threshold := avg_step_time * 2;

    -- Identify bottleneck steps
    FOR step_performance IN
        SELECT step_id, AVG(execution_duration_seconds) as avg_duration,
               COUNT(*) as execution_count
        FROM workflow_resource_usage
        WHERE workflow_id = p_workflow_id
        AND timestamp >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY step_id
        HAVING AVG(execution_duration_seconds) > bottleneck_threshold
    LOOP
        -- Insert or update bottleneck record
        INSERT INTO workflow_bottlenecks (
            tenant_id, workflow_id, step_id, bottleneck_type, severity,
            impact_metrics, suggested_optimizations
        )
        SELECT
            (SELECT tenant_id FROM workflows WHERE id = p_workflow_id),
            p_workflow_id,
            step_performance.step_id,
            'execution_time',
            CASE
                WHEN step_performance.avg_duration > bottleneck_threshold * 2 THEN 'critical'
                WHEN step_performance.avg_duration > bottleneck_threshold * 1.5 THEN 'high'
                ELSE 'medium'
            END,
            jsonb_build_object(
                'avg_execution_time', step_performance.avg_duration,
                'execution_count', step_performance.execution_count,
                'threshold_exceeded_by', step_performance.avg_duration - bottleneck_threshold
            ),
            jsonb_build_array(
                'parallel_execution',
                'caching_optimization',
                'resource_allocation_tuning'
            )
        ON CONFLICT (workflow_id, step_id) DO UPDATE SET
            impact_metrics = EXCLUDED.impact_metrics,
            detected_at = CURRENT_TIMESTAMP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to auto-generate optimization opportunities
CREATE OR REPLACE FUNCTION generate_optimization_opportunities(p_workflow_id UUID)
RETURNS INTEGER AS $$
DECLARE
    workflow_metrics RECORD;
    opportunities_created INTEGER := 0;
    baseline_metrics RECORD;
BEGIN
    -- Get current workflow metrics
    SELECT * INTO workflow_metrics
    FROM workflow_performance_metrics
    WHERE workflow_id = p_workflow_id
    ORDER BY created_at DESC
    LIMIT 1;

    -- Get baseline for comparison
    SELECT * INTO baseline_metrics
    FROM workflow_performance_baselines
    WHERE workflow_id = p_workflow_id
    AND baseline_type = 'execution_time'
    AND is_active = TRUE;

    -- Execution time optimization opportunity
    IF workflow_metrics.execution_time_seconds > (baseline_metrics.baseline_metrics->>'avg_execution_time')::DECIMAL * 1.3 THEN
        INSERT INTO optimization_opportunities (
            opportunity_id, tenant_id, workflow_id, optimization_type, strategy,
            potential_improvement, implementation_effort, confidence_score, business_impact,
            recommended_actions
        ) VALUES (
            'exec_time_' || p_workflow_id || '_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP),
            workflow_metrics.tenant_id,
            p_workflow_id,
            'execution_time',
            'parallel_execution',
            jsonb_build_object('execution_time_reduction', 0.35, 'throughput_increase', 0.5),
            'medium',
            0.82,
            jsonb_build_object('time_savings_per_day', 3600, 'cost_savings_monthly', 1500),
            jsonb_build_array('Implement parallel step execution', 'Optimize resource allocation', 'Monitor performance impact')
        ) ON CONFLICT (opportunity_id) DO NOTHING;

        opportunities_created := opportunities_created + 1;
    END IF;

    -- Cost optimization opportunity
    IF workflow_metrics.cost_per_execution > 0.05 THEN
        INSERT INTO optimization_opportunities (
            opportunity_id, tenant_id, workflow_id, optimization_type, strategy,
            potential_improvement, implementation_effort, confidence_score, business_impact,
            recommended_actions
        ) VALUES (
            'cost_' || p_workflow_id || '_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP),
            workflow_metrics.tenant_id,
            p_workflow_id,
            'cost',
            'intelligent_routing',
            jsonb_build_object('cost_reduction', 0.4, 'performance_maintained', 0.95),
            'low',
            0.88,
            jsonb_build_object('monthly_savings', 2000, 'annual_savings', 24000),
            jsonb_build_array('Implement AI provider routing', 'Enable intelligent caching', 'Monitor cost metrics')
        ) ON CONFLICT (opportunity_id) DO NOTHING;

        opportunities_created := opportunities_created + 1;
    END IF;

    -- Reliability optimization opportunity
    IF workflow_metrics.success_rate < 0.95 THEN
        INSERT INTO optimization_opportunities (
            opportunity_id, tenant_id, workflow_id, optimization_type, strategy,
            potential_improvement, implementation_effort, confidence_score, business_impact,
            recommended_actions
        ) VALUES (
            'reliability_' || p_workflow_id || '_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP),
            workflow_metrics.tenant_id,
            p_workflow_id,
            'reliability',
            'dynamic_configuration',
            jsonb_build_object('success_rate_improvement', 0.08, 'error_reduction', 0.6),
            'medium',
            0.85,
            jsonb_build_object('reduced_failures', 200, 'customer_satisfaction_increase', 0.3),
            jsonb_build_array('Implement retry logic', 'Add error handling', 'Enable graceful degradation')
        ) ON CONFLICT (opportunity_id) DO NOTHING;

        opportunities_created := opportunities_created + 1;
    END IF;

    RETURN opportunities_created;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-detect optimization opportunities
CREATE OR REPLACE FUNCTION workflow_optimization_trigger()
RETURNS TRIGGER AS $$
BEGIN
    -- Generate optimization opportunities when new performance metrics are recorded
    PERFORM generate_optimization_opportunities(NEW.workflow_id);

    -- Detect bottlenecks
    PERFORM detect_workflow_bottlenecks(NEW.workflow_id);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER workflow_performance_optimization_trigger
    AFTER INSERT ON workflow_performance_metrics
    FOR EACH ROW
    EXECUTE FUNCTION workflow_optimization_trigger();

-- View for optimization dashboard
CREATE VIEW workflow_optimization_dashboard AS
SELECT
    wpm.tenant_id,
    wpm.workflow_id,
    wpm.execution_time_seconds,
    wpm.success_rate,
    wpm.cost_per_execution,
    wpm.user_satisfaction_score,
    COUNT(oo.id) as optimization_opportunities,
    COUNT(wo.id) as active_optimizations,
    COALESCE(SUM((wo.actual_improvement->>'cost_reduction')::DECIMAL), 0) as total_cost_savings,
    COALESCE(AVG((wo.actual_improvement->>'performance_improvement')::DECIMAL), 0) as avg_performance_gain,
    calculate_optimization_score(wpm.workflow_id) as optimization_score
FROM workflow_performance_metrics wpm
LEFT JOIN optimization_opportunities oo ON wpm.workflow_id = oo.workflow_id AND oo.status = 'identified'
LEFT JOIN workflow_optimizations wo ON wpm.workflow_id = wo.workflow_id AND wo.implementation_completed_at IS NOT NULL
WHERE wpm.created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY wpm.tenant_id, wpm.workflow_id, wpm.execution_time_seconds, wpm.success_rate,
         wpm.cost_per_execution, wpm.user_satisfaction_score;

-- Function to refresh optimization recommendations
CREATE OR REPLACE FUNCTION refresh_optimization_recommendations()
RETURNS void AS $$
DECLARE
    workflow_record RECORD;
BEGIN
    -- Refresh optimization opportunities for all active workflows
    FOR workflow_record IN
        SELECT DISTINCT workflow_id
        FROM workflow_executions
        WHERE executed_at >= CURRENT_DATE - INTERVAL '1 day'
    LOOP
        PERFORM generate_optimization_opportunities(workflow_record.workflow_id);
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Materialized view for optimization analytics
CREATE MATERIALIZED VIEW optimization_analytics AS
SELECT
    DATE(wpm.created_at) as date,
    COUNT(DISTINCT wpm.workflow_id) as workflows_analyzed,
    AVG(wpm.execution_time_seconds) as avg_execution_time,
    AVG(wpm.success_rate) as avg_success_rate,
    AVG(wpm.cost_per_execution) as avg_cost_per_execution,
    COUNT(oo.id) as opportunities_identified,
    COUNT(wo.id) as optimizations_implemented,
    SUM(COALESCE((wo.actual_improvement->>'cost_reduction')::DECIMAL, 0)) as total_cost_savings,
    AVG(COALESCE((wo.actual_improvement->>'performance_improvement')::DECIMAL, 0)) as avg_performance_improvement
FROM workflow_performance_metrics wpm
LEFT JOIN optimization_opportunities oo ON wpm.workflow_id = oo.workflow_id
    AND DATE(oo.identified_at) = DATE(wpm.created_at)
LEFT JOIN workflow_optimizations wo ON wpm.workflow_id = wo.workflow_id
    AND DATE(wo.implementation_completed_at) = DATE(wpm.created_at)
WHERE wpm.created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(wpm.created_at)
ORDER BY date DESC;

CREATE INDEX idx_optimization_analytics_date ON optimization_analytics(date);

-- Schedule automatic optimization opportunity generation (requires pg_cron extension)
-- SELECT cron.schedule('generate-optimization-opportunities', '0 */6 * * *', 'SELECT refresh_optimization_recommendations();');

COMMENT ON TABLE workflow_performance_metrics IS 'Comprehensive workflow performance tracking for optimization analysis';
COMMENT ON TABLE optimization_opportunities IS 'AI-identified optimization opportunities with business impact analysis';
COMMENT ON TABLE workflow_optimizations IS 'Implemented workflow optimizations with before/after metrics';
COMMENT ON TABLE workflow_resource_usage IS 'Detailed resource usage tracking for optimization analysis';
COMMENT ON TABLE workflow_feedback IS 'User feedback on workflow performance for satisfaction optimization';
COMMENT ON TABLE optimization_learning IS 'Machine learning patterns from optimization outcomes';
COMMENT ON TABLE workflow_bottlenecks IS 'Identified workflow bottlenecks and optimization suggestions';
COMMENT ON TABLE workflow_performance_baselines IS 'Performance baselines for optimization comparison';
COMMENT ON TABLE optimization_monitoring IS 'Real-time monitoring of optimization implementations';
