-- AgentSystem Automated Onboarding Database Schema
-- Customer success and onboarding journey management

-- Customer journeys tracking
CREATE TABLE customer_journeys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    current_stage VARCHAR(50) NOT NULL,
    completed_steps JSONB NOT NULL DEFAULT '[]',
    health_score DECIMAL(5,2) NOT NULL DEFAULT 50.0,
    health_status VARCHAR(20) NOT NULL DEFAULT 'at_risk',
    last_activity TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    engagement_score DECIMAL(5,2) NOT NULL DEFAULT 0.0,
    risk_factors JSONB NOT NULL DEFAULT '[]',
    success_milestones JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(tenant_id),
    INDEX idx_journeys_stage (current_stage),
    INDEX idx_journeys_health (health_status),
    INDEX idx_journeys_activity (last_activity),
    INDEX idx_journeys_score (health_score)
);

-- Onboarding steps definitions
CREATE TABLE onboarding_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    step_id VARCHAR(100) NOT NULL UNIQUE,
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    stage VARCHAR(50) NOT NULL,
    required BOOLEAN NOT NULL DEFAULT TRUE,
    estimated_time_minutes INTEGER NOT NULL DEFAULT 10,
    completion_criteria JSONB NOT NULL,
    help_resources JSONB NOT NULL DEFAULT '[]',
    automation_triggers JSONB NOT NULL DEFAULT '[]',
    sort_order INTEGER NOT NULL DEFAULT 0,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_steps_stage (stage),
    INDEX idx_steps_order (sort_order),
    INDEX idx_steps_active (active)
);

-- Customer step completions
CREATE TABLE step_completions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    step_id VARCHAR(100) NOT NULL REFERENCES onboarding_steps(step_id),
    completed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completion_data JSONB,
    time_spent_minutes INTEGER,
    assistance_used BOOLEAN DEFAULT FALSE,
    satisfaction_score INTEGER CHECK (satisfaction_score BETWEEN 1 AND 5),

    UNIQUE(tenant_id, step_id),
    INDEX idx_completions_tenant (tenant_id),
    INDEX idx_completions_step (step_id),
    INDEX idx_completions_date (completed_at)
);

-- Customer success interventions
CREATE TABLE success_interventions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    intervention_type VARCHAR(50) NOT NULL,
    trigger_reason VARCHAR(100) NOT NULL,
    context JSONB,
    actions_taken JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
    scheduled_at TIMESTAMP,
    executed_at TIMESTAMP,
    completed_at TIMESTAMP,
    outcome JSONB,
    effectiveness_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_interventions_tenant (tenant_id),
    INDEX idx_interventions_type (intervention_type),
    INDEX idx_interventions_status (status),
    INDEX idx_interventions_scheduled (scheduled_at)
);

-- Customer health history
CREATE TABLE health_score_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    health_score DECIMAL(5,2) NOT NULL,
    health_status VARCHAR(20) NOT NULL,
    health_factors JSONB NOT NULL,
    risk_factors JSONB NOT NULL DEFAULT '[]',
    recommendations JSONB NOT NULL DEFAULT '[]',
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_health_tenant_date (tenant_id, calculated_at),
    INDEX idx_health_score (health_score),
    INDEX idx_health_status (health_status)
);

-- Onboarding templates for different customer segments
CREATE TABLE onboarding_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    target_segment VARCHAR(50) NOT NULL,
    industry VARCHAR(50),
    company_size VARCHAR(20),
    use_case VARCHAR(100),
    steps JSONB NOT NULL,
    customizations JSONB,
    success_criteria JSONB,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_templates_segment (target_segment),
    INDEX idx_templates_industry (industry),
    INDEX idx_templates_active (active)
);

-- Customer success metrics
CREATE TABLE success_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    benchmark_value DECIMAL(15,4),
    target_value DECIMAL(15,4),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_metrics_tenant_name (tenant_id, metric_name),
    INDEX idx_metrics_type (metric_type),
    INDEX idx_metrics_period (period_start, period_end)
);

-- Automated communication logs
CREATE TABLE communication_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    communication_type VARCHAR(50) NOT NULL,
    channel VARCHAR(20) NOT NULL CHECK (channel IN ('email', 'in_app', 'sms', 'phone', 'chat')),
    template_id VARCHAR(100),
    subject VARCHAR(200),
    content TEXT,
    personalization_data JSONB,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMP,
    opened_at TIMESTAMP,
    clicked_at TIMESTAMP,
    responded_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'sent' CHECK (status IN ('sent', 'delivered', 'opened', 'clicked', 'responded', 'failed')),

    INDEX idx_comms_tenant_type (tenant_id, communication_type),
    INDEX idx_comms_channel (channel),
    INDEX idx_comms_status (status),
    INDEX idx_comms_sent (sent_at)
);

-- Customer feedback and surveys
CREATE TABLE customer_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    feedback_type VARCHAR(50) NOT NULL,
    survey_id VARCHAR(100),
    question VARCHAR(500),
    response TEXT,
    rating INTEGER CHECK (rating BETWEEN 1 AND 10),
    sentiment VARCHAR(20) CHECK (sentiment IN ('positive', 'neutral', 'negative')),
    category VARCHAR(50),
    tags JSONB,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    action_taken JSONB,

    INDEX idx_feedback_tenant_type (tenant_id, feedback_type),
    INDEX idx_feedback_rating (rating),
    INDEX idx_feedback_sentiment (sentiment),
    INDEX idx_feedback_submitted (submitted_at)
);

-- Success milestone definitions
CREATE TABLE success_milestones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    milestone_id VARCHAR(100) NOT NULL UNIQUE,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL,
    criteria JSONB NOT NULL,
    reward_type VARCHAR(50),
    reward_data JSONB,
    celebration_template VARCHAR(100),
    sort_order INTEGER NOT NULL DEFAULT 0,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_milestones_category (category),
    INDEX idx_milestones_active (active)
);

-- Customer milestone achievements
CREATE TABLE milestone_achievements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    milestone_id VARCHAR(100) NOT NULL REFERENCES success_milestones(milestone_id),
    achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    achievement_data JSONB,
    celebration_sent BOOLEAN DEFAULT FALSE,
    reward_claimed BOOLEAN DEFAULT FALSE,

    UNIQUE(tenant_id, milestone_id),
    INDEX idx_achievements_tenant (tenant_id),
    INDEX idx_achievements_milestone (milestone_id),
    INDEX idx_achievements_date (achieved_at)
);

-- Insert default onboarding steps
INSERT INTO onboarding_steps (step_id, title, description, stage, required, estimated_time_minutes, completion_criteria, help_resources, automation_triggers, sort_order) VALUES
('welcome_email', 'Welcome to AgentSystem', 'Send personalized welcome email with getting started guide', 'signup', TRUE, 2, '{"email_sent": true, "email_opened": true}', '["getting_started_guide", "video_tour"]', '["user_signup"]', 1),
('email_verification', 'Verify Your Email', 'Confirm email address to activate account', 'email_verification', TRUE, 5, '{"email_verified": true}', '["email_verification_help"]', '["verification_reminder_24h", "verification_reminder_72h"]', 2),
('profile_setup', 'Complete Your Profile', 'Set up company information and preferences', 'profile_setup', TRUE, 10, '{"company_name": true, "industry": true, "team_size": true}', '["profile_setup_guide"]', '["profile_incomplete_reminder"]', 3),
('first_agent_creation', 'Create Your First AI Agent', 'Build your first AI agent to see the power of automation', 'first_agent_creation', TRUE, 15, '{"agent_created": true, "agent_configured": true}', '["agent_creation_tutorial", "agent_templates"]', '["agent_creation_reminder", "guided_agent_setup"]', 4),
('first_workflow', 'Build Your First Workflow', 'Create an automated workflow to streamline your processes', 'first_workflow', TRUE, 20, '{"workflow_created": true, "workflow_executed": true}', '["workflow_builder_guide", "workflow_templates"]', '["workflow_creation_assistance"]', 5),
('integration_setup', 'Connect Your Tools', 'Integrate with your existing tools and platforms', 'integration_setup', FALSE, 25, '{"integration_connected": true}', '["integration_guides", "api_documentation"]', '["integration_suggestions"]', 6),
('first_success', 'Achieve First Success', 'Complete your first successful automation', 'first_success', TRUE, 30, '{"automation_success": true, "value_realized": true}', '["success_stories", "optimization_tips"]', '["success_celebration", "expansion_opportunities"]', 7);

-- Insert default success milestones
INSERT INTO success_milestones (milestone_id, title, description, category, criteria, reward_type, celebration_template, sort_order) VALUES
('first_agent', 'First AI Agent Created', 'Successfully created your first AI agent', 'onboarding', '{"agents_created": 1}', 'badge', 'first_agent_celebration', 1),
('first_workflow', 'First Workflow Built', 'Built and executed your first automated workflow', 'onboarding', '{"workflows_created": 1, "workflows_executed": 1}', 'badge', 'first_workflow_celebration', 2),
('first_integration', 'First Integration Connected', 'Connected your first external tool or platform', 'onboarding', '{"integrations_connected": 1}', 'badge', 'first_integration_celebration', 3),
('power_user', 'Power User', 'Created 10+ agents and workflows', 'engagement', '{"agents_created": 10, "workflows_created": 10}', 'feature_unlock', 'power_user_celebration', 4),
('automation_master', 'Automation Master', 'Achieved 100+ successful automations', 'success', '{"successful_automations": 100}', 'discount', 'automation_master_celebration', 5);

-- Insert default onboarding templates
INSERT INTO onboarding_templates (template_name, description, target_segment, industry, company_size, steps, success_criteria) VALUES
('startup_tech', 'Onboarding for tech startups', 'startup', 'technology', 'small', '["welcome_email", "email_verification", "profile_setup", "first_agent_creation", "first_workflow"]', '{"time_to_first_value": 24, "activation_score": 70}'),
('enterprise_sales', 'Onboarding for enterprise sales teams', 'enterprise', 'sales', 'large', '["welcome_email", "email_verification", "profile_setup", "first_agent_creation", "integration_setup", "first_workflow", "first_success"]', '{"time_to_first_value": 48, "activation_score": 80}'),
('marketing_agency', 'Onboarding for marketing agencies', 'agency', 'marketing', 'medium', '["welcome_email", "email_verification", "profile_setup", "first_agent_creation", "first_workflow", "integration_setup"]', '{"time_to_first_value": 36, "activation_score": 75}');

-- Function to calculate customer health score
CREATE OR REPLACE FUNCTION calculate_customer_health_score(p_tenant_id UUID)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    onboarding_score DECIMAL(5,2);
    engagement_score DECIMAL(5,2);
    usage_score DECIMAL(5,2);
    overall_score DECIMAL(5,2);
    total_steps INTEGER;
    completed_steps INTEGER;
    days_since_activity INTEGER;
    recent_usage INTEGER;
BEGIN
    -- Calculate onboarding progress score
    SELECT COUNT(*) INTO total_steps FROM onboarding_steps WHERE active = TRUE;
    SELECT COUNT(*) INTO completed_steps FROM step_completions sc
    JOIN onboarding_steps os ON sc.step_id = os.step_id
    WHERE sc.tenant_id = p_tenant_id AND os.active = TRUE;

    onboarding_score := (completed_steps::DECIMAL / total_steps::DECIMAL) * 100;

    -- Calculate engagement score based on recent activity
    SELECT EXTRACT(DAYS FROM (CURRENT_TIMESTAMP - last_activity)) INTO days_since_activity
    FROM customer_journeys WHERE tenant_id = p_tenant_id;

    engagement_score := GREATEST(0, 100 - (days_since_activity * 10));

    -- Calculate usage score from recent API calls
    SELECT COALESCE(SUM(api_calls), 0) INTO recent_usage
    FROM usage_tracking
    WHERE tenant_id = p_tenant_id
    AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '7 days';

    IF recent_usage = 0 THEN
        usage_score := 0;
    ELSIF recent_usage < 100 THEN
        usage_score := 30;
    ELSIF recent_usage < 500 THEN
        usage_score := 60;
    ELSIF recent_usage < 1000 THEN
        usage_score := 80;
    ELSE
        usage_score := 100;
    END IF;

    -- Calculate weighted overall score
    overall_score := (onboarding_score * 0.4) + (engagement_score * 0.3) + (usage_score * 0.3);

    RETURN LEAST(100, GREATEST(0, overall_score));
END;
$$ LANGUAGE plpgsql;

-- Function to update customer journey health
CREATE OR REPLACE FUNCTION update_customer_health()
RETURNS TRIGGER AS $$
BEGIN
    -- Update health score when step is completed
    IF TG_TABLE_NAME = 'step_completions' THEN
        UPDATE customer_journeys
        SET health_score = calculate_customer_health_score(NEW.tenant_id),
            last_activity = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE tenant_id = NEW.tenant_id;
    END IF;

    -- Update health score when usage is tracked
    IF TG_TABLE_NAME = 'usage_tracking' THEN
        UPDATE customer_journeys
        SET health_score = calculate_customer_health_score(NEW.tenant_id),
            last_activity = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE tenant_id = NEW.tenant_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic health score updates
CREATE TRIGGER step_completion_health_trigger
    AFTER INSERT ON step_completions
    FOR EACH ROW
    EXECUTE FUNCTION update_customer_health();

CREATE TRIGGER usage_health_trigger
    AFTER INSERT ON usage_tracking
    FOR EACH ROW
    EXECUTE FUNCTION update_customer_health();

-- Function to check milestone achievements
CREATE OR REPLACE FUNCTION check_milestone_achievements(p_tenant_id UUID)
RETURNS void AS $$
DECLARE
    milestone RECORD;
    criteria JSONB;
    achieved BOOLEAN;
BEGIN
    -- Check each milestone
    FOR milestone IN SELECT * FROM success_milestones WHERE active = TRUE LOOP
        -- Check if already achieved
        IF EXISTS (SELECT 1 FROM milestone_achievements WHERE tenant_id = p_tenant_id AND milestone_id = milestone.milestone_id) THEN
            CONTINUE;
        END IF;

        criteria := milestone.criteria;
        achieved := TRUE;

        -- Check agents created criteria
        IF criteria ? 'agents_created' THEN
            IF (SELECT COUNT(*) FROM agents WHERE tenant_id = p_tenant_id) < (criteria->>'agents_created')::INTEGER THEN
                achieved := FALSE;
            END IF;
        END IF;

        -- Check workflows created criteria
        IF criteria ? 'workflows_created' THEN
            IF (SELECT COUNT(*) FROM workflows WHERE tenant_id = p_tenant_id) < (criteria->>'workflows_created')::INTEGER THEN
                achieved := FALSE;
            END IF;
        END IF;

        -- Check workflows executed criteria
        IF criteria ? 'workflows_executed' THEN
            IF (SELECT COUNT(*) FROM workflow_executions WHERE tenant_id = p_tenant_id AND status = 'completed') < (criteria->>'workflows_executed')::INTEGER THEN
                achieved := FALSE;
            END IF;
        END IF;

        -- If milestone achieved, record it
        IF achieved THEN
            INSERT INTO milestone_achievements (tenant_id, milestone_id, achievement_data)
            VALUES (p_tenant_id, milestone.milestone_id, '{}');
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Trigger to check milestones on relevant activities
CREATE OR REPLACE FUNCTION milestone_check_trigger()
RETURNS TRIGGER AS $$
BEGIN
    -- Check milestones when agents, workflows, or executions are created
    PERFORM check_milestone_achievements(NEW.tenant_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create milestone check triggers (would need to be created after agents/workflows tables exist)
-- CREATE TRIGGER agent_milestone_trigger AFTER INSERT ON agents FOR EACH ROW EXECUTE FUNCTION milestone_check_trigger();
-- CREATE TRIGGER workflow_milestone_trigger AFTER INSERT ON workflows FOR EACH ROW EXECUTE FUNCTION milestone_check_trigger();

COMMENT ON TABLE customer_journeys IS 'Customer onboarding journey tracking and health scoring';
COMMENT ON TABLE onboarding_steps IS 'Configurable onboarding step definitions';
COMMENT ON TABLE step_completions IS 'Customer completion tracking for onboarding steps';
COMMENT ON TABLE success_interventions IS 'Automated customer success interventions and outcomes';
COMMENT ON TABLE health_score_history IS 'Historical customer health score tracking';
COMMENT ON TABLE onboarding_templates IS 'Customizable onboarding flows for different segments';
COMMENT ON TABLE success_metrics IS 'Customer success KPI tracking and benchmarking';
COMMENT ON TABLE communication_logs IS 'Automated communication tracking and engagement metrics';
COMMENT ON TABLE customer_feedback IS 'Customer feedback collection and sentiment analysis';
COMMENT ON TABLE success_milestones IS 'Achievement milestone definitions and rewards';
COMMENT ON TABLE milestone_achievements IS 'Customer milestone achievement tracking';
