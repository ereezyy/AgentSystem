-- AgentSystem Predictive Business Intelligence Database Schema
-- Advanced ML-powered predictive analytics for strategic business insights

-- Prediction models registry
CREATE TABLE prediction_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('linear_regression', 'random_forest', 'gradient_boosting', 'ensemble')),
    prediction_type VARCHAR(50) NOT NULL CHECK (prediction_type IN ('revenue_forecast', 'customer_churn', 'market_opportunity', 'resource_demand', 'competitive_threat', 'growth_potential', 'pricing_optimization')),
    model_version VARCHAR(20) NOT NULL,
    accuracy_score DECIMAL(5,4),
    feature_importance JSONB,
    hyperparameters JSONB,
    training_data_period TSRANGE,
    model_file_path VARCHAR(500),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_models_type (prediction_type),
    INDEX idx_models_active (is_active),
    INDEX idx_models_accuracy (accuracy_score DESC)
);

-- Prediction results storage
CREATE TABLE prediction_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES prediction_models(id),
    prediction_type VARCHAR(50) NOT NULL,
    predicted_value DECIMAL(15,4) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    prediction_interval_low DECIMAL(15,4),
    prediction_interval_high DECIMAL(15,4),
    factors JSONB NOT NULL,
    recommendations JSONB NOT NULL DEFAULT '[]',
    model_accuracy DECIMAL(5,4),
    prediction_date DATE NOT NULL,
    forecast_horizon_days INTEGER,
    actual_value DECIMAL(15,4), -- Filled later for accuracy tracking
    prediction_error DECIMAL(15,4), -- Calculated when actual is known
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_predictions_tenant_type (tenant_id, prediction_type),
    INDEX idx_predictions_date (prediction_date),
    INDEX idx_predictions_accuracy (model_accuracy DESC),
    INDEX idx_predictions_confidence (confidence_score DESC)
);

-- Market opportunities analysis
CREATE TABLE market_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    market_segment VARCHAR(100) NOT NULL,
    opportunity_score DECIMAL(5,2) NOT NULL,
    predicted_market_size DECIMAL(15,2),
    competitive_gaps JSONB NOT NULL DEFAULT '[]',
    entry_strategy JSONB NOT NULL DEFAULT '[]',
    confidence_level DECIMAL(5,4),
    timeline_to_market INTEGER, -- days
    investment_required DECIMAL(15,2),
    potential_roi DECIMAL(8,4),
    risk_assessment JSONB,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMP,

    INDEX idx_opportunities_tenant_score (tenant_id, opportunity_score DESC),
    INDEX idx_opportunities_segment (market_segment),
    INDEX idx_opportunities_valid (valid_until)
);

-- Customer lifetime value trends
CREATE TABLE clv_trends (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    customer_segment VARCHAR(100),
    current_clv_avg DECIMAL(15,2),
    predicted_clv_avg DECIMAL(15,2),
    clv_growth_rate DECIMAL(8,4),
    high_value_characteristics JSONB,
    optimization_strategies JSONB,
    confidence_score DECIMAL(5,4),
    analysis_period TSRANGE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_clv_trends_tenant_segment (tenant_id, customer_segment),
    INDEX idx_clv_trends_growth (clv_growth_rate DESC),
    INDEX idx_clv_trends_period (analysis_period)
);

-- Business scenario analysis
CREATE TABLE business_scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    scenario_name VARCHAR(200) NOT NULL,
    scenario_description TEXT,
    parameters JSONB NOT NULL,
    probability DECIMAL(5,4) NOT NULL,
    impact_assessment JSONB NOT NULL,
    risk_assessment JSONB,
    mitigation_strategies JSONB NOT NULL DEFAULT '[]',
    expected_value DECIMAL(15,2),
    recommendation VARCHAR(20) CHECK (recommendation IN ('pursue', 'avoid', 'monitor', 'prepare')),
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_scenarios_tenant (tenant_id),
    INDEX idx_scenarios_probability (probability DESC),
    INDEX idx_scenarios_expected_value (expected_value DESC)
);

-- Competitive intelligence forecasts
CREATE TABLE competitive_forecasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    competitor_name VARCHAR(200) NOT NULL,
    predicted_moves JSONB NOT NULL DEFAULT '[]',
    threat_level VARCHAR(20) CHECK (threat_level IN ('low', 'medium', 'high', 'critical')),
    impact_areas JSONB NOT NULL DEFAULT '[]',
    counter_strategies JSONB NOT NULL DEFAULT '[]',
    recommended_actions JSONB NOT NULL DEFAULT '[]',
    confidence_score DECIMAL(5,4),
    forecast_period TSRANGE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_competitive_tenant_competitor (tenant_id, competitor_name),
    INDEX idx_competitive_threat (threat_level),
    INDEX idx_competitive_period (forecast_period)
);

-- Feature engineering data
CREATE TABLE feature_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(15,6),
    feature_type VARCHAR(50) NOT NULL,
    data_date DATE NOT NULL,
    source_table VARCHAR(100),
    calculation_method TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(tenant_id, feature_name, data_date),
    INDEX idx_features_tenant_name_date (tenant_id, feature_name, data_date),
    INDEX idx_features_type (feature_type)
);

-- Model training metrics
CREATE TABLE model_training_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES prediction_models(id),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_samples INTEGER,
    validation_samples INTEGER,
    training_accuracy DECIMAL(5,4),
    validation_accuracy DECIMAL(5,4),
    cross_validation_score DECIMAL(5,4),
    feature_importance JSONB,
    training_time_seconds INTEGER,
    hyperparameters_used JSONB,

    INDEX idx_training_model_date (model_id, training_date DESC),
    INDEX idx_training_accuracy (validation_accuracy DESC)
);

-- Prediction accuracy tracking
CREATE TABLE prediction_accuracy_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID NOT NULL REFERENCES prediction_results(id),
    actual_outcome_date DATE,
    actual_value DECIMAL(15,4),
    prediction_error DECIMAL(15,4),
    absolute_error DECIMAL(15,4),
    percentage_error DECIMAL(8,4),
    accuracy_bucket VARCHAR(20) CHECK (accuracy_bucket IN ('excellent', 'good', 'fair', 'poor')),
    model_feedback JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_accuracy_prediction (prediction_id),
    INDEX idx_accuracy_error (absolute_error),
    INDEX idx_accuracy_bucket (accuracy_bucket)
);

-- Strategic insights cache
CREATE TABLE strategic_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    insight_type VARCHAR(100) NOT NULL,
    insight_title VARCHAR(500) NOT NULL,
    insight_description TEXT,
    confidence_score DECIMAL(5,4),
    impact_level VARCHAR(20) CHECK (impact_level IN ('low', 'medium', 'high', 'critical')),
    recommendations JSONB NOT NULL DEFAULT '[]',
    supporting_data JSONB,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'expired', 'superseded')),

    INDEX idx_insights_tenant_type (tenant_id, insight_type),
    INDEX idx_insights_impact (impact_level),
    INDEX idx_insights_expires (expires_at)
);

-- Market data integration (external sources)
CREATE TABLE market_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    data_source VARCHAR(100) NOT NULL,
    market_segment VARCHAR(100),
    data_type VARCHAR(100) NOT NULL,
    data_value DECIMAL(15,6),
    data_unit VARCHAR(50),
    geographic_region VARCHAR(100),
    time_period TSRANGE,
    confidence_level DECIMAL(5,4),
    source_url TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_market_data_source_type (data_source, data_type),
    INDEX idx_market_data_segment (market_segment),
    INDEX idx_market_data_period (time_period)
);

-- Predictive model versioning
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES prediction_models(id),
    version_number VARCHAR(20) NOT NULL,
    changes_description TEXT,
    performance_improvement DECIMAL(8,4),
    deployment_date TIMESTAMP,
    rollback_date TIMESTAMP,
    is_deployed BOOLEAN DEFAULT FALSE,
    deployment_notes TEXT,

    UNIQUE(model_id, version_number),
    INDEX idx_versions_model_deployed (model_id, is_deployed)
);

-- Business intelligence alerts
CREATE TABLE bi_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    alert_type VARCHAR(100) NOT NULL,
    alert_title VARCHAR(500) NOT NULL,
    alert_message TEXT,
    severity VARCHAR(20) CHECK (severity IN ('info', 'warning', 'critical', 'urgent')),
    trigger_conditions JSONB,
    recommended_actions JSONB,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by UUID REFERENCES users(id),
    acknowledged_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_bi_alerts_tenant_severity (tenant_id, severity),
    INDEX idx_bi_alerts_acknowledged (is_acknowledged),
    INDEX idx_bi_alerts_created (created_at DESC)
);

-- Insert default prediction models
INSERT INTO prediction_models (model_name, model_type, prediction_type, model_version, accuracy_score) VALUES
('revenue_forecast_ensemble', 'ensemble', 'revenue_forecast', '1.0', 0.85),
('churn_gradient_boost', 'gradient_boosting', 'customer_churn', '1.2', 0.82),
('market_opportunity_rf', 'random_forest', 'market_opportunity', '1.1', 0.78),
('pricing_optimization_linear', 'linear_regression', 'pricing_optimization', '1.0', 0.75),
('growth_potential_ensemble', 'ensemble', 'growth_potential', '1.3', 0.88);

-- Function to calculate prediction accuracy
CREATE OR REPLACE FUNCTION calculate_prediction_accuracy(p_prediction_id UUID, p_actual_value DECIMAL(15,4))
RETURNS DECIMAL(5,4) AS $$
DECLARE
    predicted_value DECIMAL(15,4);
    error_value DECIMAL(15,4);
    percentage_error DECIMAL(8,4);
    accuracy_score DECIMAL(5,4);
BEGIN
    -- Get predicted value
    SELECT predicted_value INTO predicted_value
    FROM prediction_results
    WHERE id = p_prediction_id;

    -- Calculate error
    error_value := ABS(p_actual_value - predicted_value);
    percentage_error := (error_value / NULLIF(p_actual_value, 0)) * 100;

    -- Calculate accuracy (inverse of percentage error, capped at 100%)
    accuracy_score := GREATEST(0, 100 - percentage_error) / 100;

    -- Update prediction result
    UPDATE prediction_results
    SET actual_value = p_actual_value,
        prediction_error = error_value
    WHERE id = p_prediction_id;

    -- Insert accuracy tracking
    INSERT INTO prediction_accuracy_tracking (
        prediction_id, actual_value, prediction_error,
        absolute_error, percentage_error, accuracy_bucket
    ) VALUES (
        p_prediction_id, p_actual_value, error_value,
        error_value, percentage_error,
        CASE
            WHEN percentage_error <= 5 THEN 'excellent'
            WHEN percentage_error <= 15 THEN 'good'
            WHEN percentage_error <= 30 THEN 'fair'
            ELSE 'poor'
        END
    );

    RETURN accuracy_score;
END;
$$ LANGUAGE plpgsql;

-- Function to generate predictive insights
CREATE OR REPLACE FUNCTION generate_predictive_insights(p_tenant_id UUID)
RETURNS void AS $$
DECLARE
    revenue_trend DECIMAL(8,4);
    churn_risk_level VARCHAR(20);
    market_opportunity_score DECIMAL(5,2);
BEGIN
    -- Analyze recent revenue predictions
    SELECT AVG(
        CASE
            WHEN prediction_date >= CURRENT_DATE - INTERVAL '30 days'
            THEN predicted_value - LAG(predicted_value) OVER (ORDER BY prediction_date)
            ELSE NULL
        END
    ) INTO revenue_trend
    FROM prediction_results
    WHERE tenant_id = p_tenant_id
    AND prediction_type = 'revenue_forecast'
    AND prediction_date >= CURRENT_DATE - INTERVAL '60 days';

    -- Generate revenue trend insight
    IF revenue_trend > 0.1 THEN
        INSERT INTO strategic_insights (
            tenant_id, insight_type, insight_title, insight_description,
            confidence_score, impact_level, recommendations
        ) VALUES (
            p_tenant_id, 'revenue_growth', 'Strong Revenue Growth Predicted',
            'Predictive models show sustained revenue growth trend of ' || ROUND(revenue_trend, 2) || '% over next 30 days',
            0.85, 'high',
            '["Scale marketing efforts", "Expand team capacity", "Explore new markets"]'::jsonb
        );
    ELSIF revenue_trend < -0.05 THEN
        INSERT INTO strategic_insights (
            tenant_id, insight_type, insight_title, insight_description,
            confidence_score, impact_level, recommendations
        ) VALUES (
            p_tenant_id, 'revenue_decline', 'Revenue Decline Risk Detected',
            'Predictive models indicate potential revenue decline of ' || ROUND(ABS(revenue_trend), 2) || '% risk',
            0.78, 'critical',
            '["Implement retention campaigns", "Review pricing strategy", "Enhance customer success"]'::jsonb
        );
    END IF;

    -- Check for high market opportunities
    SELECT MAX(opportunity_score) INTO market_opportunity_score
    FROM market_opportunities
    WHERE tenant_id = p_tenant_id
    AND analysis_date >= CURRENT_DATE - INTERVAL '7 days';

    IF market_opportunity_score > 80 THEN
        INSERT INTO strategic_insights (
            tenant_id, insight_type, insight_title, insight_description,
            confidence_score, impact_level, recommendations
        ) VALUES (
            p_tenant_id, 'market_opportunity', 'High-Value Market Opportunity Identified',
            'Market analysis reveals opportunity with score of ' || market_opportunity_score || '/100',
            0.82, 'high',
            '["Conduct market research", "Develop go-to-market strategy", "Allocate resources"]'::jsonb
        );
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-generate insights when new predictions are created
CREATE OR REPLACE FUNCTION trigger_insight_generation()
RETURNS TRIGGER AS $$
BEGIN
    -- Schedule insight generation for this tenant
    PERFORM generate_predictive_insights(NEW.tenant_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER prediction_insight_trigger
    AFTER INSERT ON prediction_results
    FOR EACH ROW
    EXECUTE FUNCTION trigger_insight_generation();

-- View for model performance dashboard
CREATE VIEW model_performance_summary AS
SELECT
    pm.model_name,
    pm.prediction_type,
    pm.accuracy_score as current_accuracy,
    COUNT(pr.id) as total_predictions,
    AVG(pr.confidence_score) as avg_confidence,
    COUNT(pat.id) as predictions_with_outcomes,
    AVG(pat.percentage_error) as avg_error_rate,
    COUNT(CASE WHEN pat.accuracy_bucket = 'excellent' THEN 1 END) as excellent_predictions,
    pm.updated_at as last_updated
FROM prediction_models pm
LEFT JOIN prediction_results pr ON pm.id = pr.model_id
LEFT JOIN prediction_accuracy_tracking pat ON pr.id = pat.prediction_id
WHERE pm.is_active = TRUE
GROUP BY pm.id, pm.model_name, pm.prediction_type, pm.accuracy_score, pm.updated_at;

-- Function to refresh model performance
CREATE OR REPLACE FUNCTION refresh_model_performance()
RETURNS void AS $$
BEGIN
    -- Update model accuracy scores based on recent performance
    UPDATE prediction_models pm
    SET accuracy_score = (
        SELECT AVG(100 - pat.percentage_error) / 100
        FROM prediction_results pr
        JOIN prediction_accuracy_tracking pat ON pr.id = pat.prediction_id
        WHERE pr.model_id = pm.id
        AND pat.actual_outcome_date >= CURRENT_DATE - INTERVAL '30 days'
    )
    WHERE EXISTS (
        SELECT 1 FROM prediction_results pr
        JOIN prediction_accuracy_tracking pat ON pr.id = pat.prediction_id
        WHERE pr.model_id = pm.id
        AND pat.actual_outcome_date >= CURRENT_DATE - INTERVAL '30 days'
    );
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE prediction_models IS 'Registry of ML models for predictive analytics';
COMMENT ON TABLE prediction_results IS 'Results from predictive model executions';
COMMENT ON TABLE market_opportunities IS 'Identified market opportunities with scoring';
COMMENT ON TABLE clv_trends IS 'Customer lifetime value trend analysis';
COMMENT ON TABLE business_scenarios IS 'What-if scenario analysis results';
COMMENT ON TABLE competitive_forecasts IS 'Competitive intelligence predictions';
COMMENT ON TABLE strategic_insights IS 'AI-generated strategic business insights';
COMMENT ON TABLE model_training_metrics IS 'ML model training performance metrics';
COMMENT ON TABLE prediction_accuracy_tracking IS 'Tracking accuracy of predictions vs actual outcomes';
