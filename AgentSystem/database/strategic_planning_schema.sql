-- Strategic Planning and Decision Support Database Schema
-- Stores strategic plans, objectives, scenarios, and decision analyses

-- Strategic plans table - main strategic plan entities
CREATE TABLE strategic_plans (
    plan_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    time_horizon INT DEFAULT 36, -- months
    status ENUM('draft', 'under_review', 'approved', 'active', 'completed', 'paused', 'cancelled') DEFAULT 'draft',
    created_by VARCHAR(255) NOT NULL,
    approved_by VARCHAR(255),
    approval_date TIMESTAMP NULL,
    start_date DATE,
    end_date DATE,
    total_budget DECIMAL(15,2) DEFAULT 0.00,
    budget_used DECIMAL(15,2) DEFAULT 0.00,
    overall_progress DECIMAL(5,2) DEFAULT 0.00,
    success_probability DECIMAL(3,2) DEFAULT 0.50,
    last_review_date TIMESTAMP NULL,
    next_review_date TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_status (status),
    INDEX idx_created_by (created_by),
    INDEX idx_created_at (created_at)
);

-- Strategic objectives table
CREATE TABLE strategic_objectives (
    objective_id VARCHAR(255) PRIMARY KEY,
    plan_id VARCHAR(255) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    objective_type ENUM('revenue_growth', 'market_expansion', 'cost_optimization', 'innovation',
                       'customer_acquisition', 'operational_excellence', 'digital_transformation', 'sustainability') NOT NULL,
    target_value DECIMAL(15,2) DEFAULT 0.00,
    current_value DECIMAL(15,2) DEFAULT 0.00,
    target_date DATE,
    priority INT DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    owner VARCHAR(255) NOT NULL,
    budget DECIMAL(15,2) DEFAULT 0.00,
    budget_used DECIMAL(15,2) DEFAULT 0.00,
    progress DECIMAL(5,2) DEFAULT 0.00,
    status ENUM('not_started', 'planning', 'in_progress', 'on_hold', 'completed', 'cancelled') DEFAULT 'not_started',
    weight DECIMAL(3,2) DEFAULT 1.00, -- strategic importance weight
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES strategic_plans(plan_id) ON DELETE CASCADE,
    INDEX idx_plan_id (plan_id),
    INDEX idx_objective_type (objective_type),
    INDEX idx_priority (priority),
    INDEX idx_status (status),
    INDEX idx_target_date (target_date)
);

-- Objective KPIs
CREATE TABLE objective_kpis (
    id INT AUTO_INCREMENT PRIMARY KEY,
    objective_id VARCHAR(255) NOT NULL,
    kpi_name VARCHAR(255) NOT NULL,
    kpi_description TEXT,
    measurement_unit VARCHAR(100),
    target_value DECIMAL(15,2),
    current_value DECIMAL(15,2) DEFAULT 0.00,
    measurement_frequency ENUM('daily', 'weekly', 'monthly', 'quarterly', 'annually') DEFAULT 'monthly',
    data_source VARCHAR(255),
    last_measured TIMESTAMP NULL,
    trend ENUM('improving', 'stable', 'declining', 'unknown') DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (objective_id) REFERENCES strategic_objectives(objective_id) ON DELETE CASCADE,
    INDEX idx_objective_id (objective_id),
    INDEX idx_kpi_name (kpi_name)
);

-- Objective dependencies
CREATE TABLE objective_dependencies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    objective_id VARCHAR(255) NOT NULL,
    depends_on_objective_id VARCHAR(255) NOT NULL,
    dependency_type ENUM('prerequisite', 'parallel', 'input_required', 'resource_shared') DEFAULT 'prerequisite',
    dependency_strength ENUM('weak', 'moderate', 'strong', 'critical') DEFAULT 'moderate',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (objective_id) REFERENCES strategic_objectives(objective_id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_objective_id) REFERENCES strategic_objectives(objective_id) ON DELETE CASCADE,
    UNIQUE KEY unique_dependency (objective_id, depends_on_objective_id),
    INDEX idx_objective_id (objective_id),
    INDEX idx_depends_on (depends_on_objective_id)
);

-- Strategic decisions table
CREATE TABLE strategic_decisions (
    decision_id VARCHAR(255) PRIMARY KEY,
    plan_id VARCHAR(255),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    decision_type ENUM('strategic', 'tactical', 'operational', 'investment', 'resource_allocation',
                      'market_entry', 'product_development', 'partnership') NOT NULL,
    urgency ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    impact_level ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    timeline_days INT DEFAULT 30,
    approval_required BOOLEAN DEFAULT TRUE,
    approved_by VARCHAR(255),
    approved_date TIMESTAMP NULL,
    decision_date TIMESTAMP NULL,
    selected_option_id INT,
    confidence_score DECIMAL(3,2) DEFAULT 0.50,
    status ENUM('pending_analysis', 'under_review', 'approved', 'implemented', 'rejected', 'cancelled') DEFAULT 'pending_analysis',
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES strategic_plans(plan_id) ON DELETE SET NULL,
    INDEX idx_plan_id (plan_id),
    INDEX idx_decision_type (decision_type),
    INDEX idx_status (status),
    INDEX idx_urgency (urgency),
    INDEX idx_created_at (created_at)
);

-- Decision options
CREATE TABLE decision_options (
    option_id INT AUTO_INCREMENT PRIMARY KEY,
    decision_id VARCHAR(255) NOT NULL,
    option_name VARCHAR(255) NOT NULL,
    description TEXT,
    estimated_cost DECIMAL(15,2) DEFAULT 0.00,
    expected_roi DECIMAL(5,2) DEFAULT 0.00,
    implementation_time_weeks INT DEFAULT 4,
    risk_level ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    feasibility_score DECIMAL(3,2) DEFAULT 0.50,
    strategic_alignment_score DECIMAL(3,2) DEFAULT 0.50,
    overall_score DECIMAL(3,2) DEFAULT 0.50,
    pros TEXT,
    cons TEXT,
    assumptions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (decision_id) REFERENCES strategic_decisions(decision_id) ON DELETE CASCADE,
    INDEX idx_decision_id (decision_id),
    INDEX idx_overall_score (overall_score)
);

-- Decision criteria and weights
CREATE TABLE decision_criteria (
    id INT AUTO_INCREMENT PRIMARY KEY,
    decision_id VARCHAR(255) NOT NULL,
    criteria_name VARCHAR(255) NOT NULL,
    criteria_description TEXT,
    weight DECIMAL(3,2) DEFAULT 0.20,
    measurement_method ENUM('quantitative', 'qualitative', 'mixed') DEFAULT 'mixed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (decision_id) REFERENCES strategic_decisions(decision_id) ON DELETE CASCADE,
    INDEX idx_decision_id (decision_id)
);

-- Decision option scores per criteria
CREATE TABLE option_criteria_scores (
    id INT AUTO_INCREMENT PRIMARY KEY,
    option_id INT NOT NULL,
    criteria_id INT NOT NULL,
    score DECIMAL(3,2) DEFAULT 0.50,
    rationale TEXT,
    confidence_level ENUM('low', 'medium', 'high') DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (option_id) REFERENCES decision_options(option_id) ON DELETE CASCADE,
    FOREIGN KEY (criteria_id) REFERENCES decision_criteria(id) ON DELETE CASCADE,
    UNIQUE KEY unique_option_criteria (option_id, criteria_id),
    INDEX idx_option_id (option_id),
    INDEX idx_criteria_id (criteria_id)
);

-- Decision stakeholders
CREATE TABLE decision_stakeholders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    decision_id VARCHAR(255) NOT NULL,
    stakeholder_name VARCHAR(255) NOT NULL,
    stakeholder_role VARCHAR(255),
    influence_level ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    support_level ENUM('strongly_oppose', 'oppose', 'neutral', 'support', 'strongly_support') DEFAULT 'neutral',
    involvement_type ENUM('informed', 'consulted', 'responsible', 'accountable') DEFAULT 'informed',
    contact_info VARCHAR(500),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (decision_id) REFERENCES strategic_decisions(decision_id) ON DELETE CASCADE,
    INDEX idx_decision_id (decision_id),
    INDEX idx_influence_level (influence_level)
);

-- Scenario analyses table
CREATE TABLE scenario_analyses (
    scenario_id VARCHAR(255) PRIMARY KEY,
    plan_id VARCHAR(255),
    name VARCHAR(500) NOT NULL,
    description TEXT,
    scenario_type ENUM('optimistic', 'realistic', 'pessimistic', 'crisis', 'disruption', 'growth') NOT NULL,
    probability DECIMAL(3,2) DEFAULT 0.33,
    impact_score DECIMAL(3,2) DEFAULT 0.50,
    confidence_level DECIMAL(3,2) DEFAULT 0.70,
    time_horizon_months INT DEFAULT 36,
    created_by VARCHAR(255) NOT NULL,
    review_date TIMESTAMP NULL,
    status ENUM('draft', 'under_review', 'approved', 'active', 'archived') DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES strategic_plans(plan_id) ON DELETE SET NULL,
    INDEX idx_plan_id (plan_id),
    INDEX idx_scenario_type (scenario_type),
    INDEX idx_probability (probability),
    INDEX idx_impact_score (impact_score)
);

-- Scenario assumptions
CREATE TABLE scenario_assumptions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scenario_id VARCHAR(255) NOT NULL,
    assumption_text TEXT NOT NULL,
    assumption_type ENUM('market', 'economic', 'technological', 'competitive', 'regulatory', 'internal') NOT NULL,
    likelihood ENUM('low', 'medium', 'high') DEFAULT 'medium',
    impact_if_wrong ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    validation_method VARCHAR(255),
    last_validated TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scenario_id) REFERENCES scenario_analyses(scenario_id) ON DELETE CASCADE,
    INDEX idx_scenario_id (scenario_id),
    INDEX idx_assumption_type (assumption_type)
);

-- Scenario market conditions
CREATE TABLE scenario_market_conditions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scenario_id VARCHAR(255) NOT NULL,
    condition_name VARCHAR(255) NOT NULL,
    condition_value VARCHAR(500),
    condition_type ENUM('economic_indicator', 'market_size', 'growth_rate', 'competition_level',
                       'regulatory_environment', 'technology_adoption') NOT NULL,
    baseline_value VARCHAR(500),
    impact_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scenario_id) REFERENCES scenario_analyses(scenario_id) ON DELETE CASCADE,
    INDEX idx_scenario_id (scenario_id),
    INDEX idx_condition_type (condition_type)
);

-- Scenario expected outcomes
CREATE TABLE scenario_outcomes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scenario_id VARCHAR(255) NOT NULL,
    outcome_name VARCHAR(255) NOT NULL,
    outcome_value DECIMAL(15,2),
    outcome_unit VARCHAR(100),
    outcome_type ENUM('financial', 'operational', 'market', 'strategic', 'risk') NOT NULL,
    confidence_level DECIMAL(3,2) DEFAULT 0.70,
    measurement_method VARCHAR(255),
    baseline_value DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scenario_id) REFERENCES scenario_analyses(scenario_id) ON DELETE CASCADE,
    INDEX idx_scenario_id (scenario_id),
    INDEX idx_outcome_type (outcome_type)
);

-- Scenario risks and opportunities
CREATE TABLE scenario_risks_opportunities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scenario_id VARCHAR(255) NOT NULL,
    item_type ENUM('risk', 'opportunity') NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    probability ENUM('low', 'medium', 'high') DEFAULT 'medium',
    impact ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    category ENUM('market', 'financial', 'operational', 'strategic', 'technological', 'regulatory') NOT NULL,
    mitigation_strategy TEXT,
    response_plan TEXT,
    owner VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scenario_id) REFERENCES scenario_analyses(scenario_id) ON DELETE CASCADE,
    INDEX idx_scenario_id (scenario_id),
    INDEX idx_item_type (item_type),
    INDEX idx_category (category)
);

-- Resource allocation table
CREATE TABLE resource_allocations (
    allocation_id VARCHAR(255) PRIMARY KEY,
    plan_id VARCHAR(255) NOT NULL,
    allocation_name VARCHAR(255) NOT NULL,
    allocation_type ENUM('budget', 'personnel', 'technology', 'facilities', 'equipment') NOT NULL,
    total_amount DECIMAL(15,2) DEFAULT 0.00,
    allocated_amount DECIMAL(15,2) DEFAULT 0.00,
    remaining_amount DECIMAL(15,2) DEFAULT 0.00,
    unit VARCHAR(100),
    allocation_period ENUM('monthly', 'quarterly', 'annually', 'project_based') DEFAULT 'quarterly',
    start_date DATE,
    end_date DATE,
    optimization_score DECIMAL(3,2) DEFAULT 0.50,
    efficiency_rating ENUM('poor', 'fair', 'good', 'excellent') DEFAULT 'good',
    last_reviewed TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES strategic_plans(plan_id) ON DELETE CASCADE,
    INDEX idx_plan_id (plan_id),
    INDEX idx_allocation_type (allocation_type),
    INDEX idx_optimization_score (optimization_score)
);

-- Resource allocation details (per objective/initiative)
CREATE TABLE allocation_details (
    id INT AUTO_INCREMENT PRIMARY KEY,
    allocation_id VARCHAR(255) NOT NULL,
    objective_id VARCHAR(255),
    initiative_name VARCHAR(255),
    allocated_amount DECIMAL(15,2) DEFAULT 0.00,
    percentage_of_total DECIMAL(5,2) DEFAULT 0.00,
    justification TEXT,
    expected_roi DECIMAL(5,2) DEFAULT 0.00,
    risk_factor DECIMAL(3,2) DEFAULT 0.50,
    priority_score DECIMAL(3,2) DEFAULT 0.50,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (allocation_id) REFERENCES resource_allocations(allocation_id) ON DELETE CASCADE,
    FOREIGN KEY (objective_id) REFERENCES strategic_objectives(objective_id) ON DELETE SET NULL,
    INDEX idx_allocation_id (allocation_id),
    INDEX idx_objective_id (objective_id)
);

-- Strategic initiatives table
CREATE TABLE strategic_initiatives (
    initiative_id VARCHAR(255) PRIMARY KEY,
    plan_id VARCHAR(255) NOT NULL,
    objective_id VARCHAR(255),
    name VARCHAR(500) NOT NULL,
    description TEXT,
    initiative_type ENUM('project', 'program', 'capability_building', 'process_improvement',
                        'technology_implementation', 'market_initiative') NOT NULL,
    status ENUM('planning', 'approved', 'in_progress', 'on_hold', 'completed', 'cancelled') DEFAULT 'planning',
    priority ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    start_date DATE,
    end_date DATE,
    budget DECIMAL(15,2) DEFAULT 0.00,
    budget_used DECIMAL(15,2) DEFAULT 0.00,
    progress DECIMAL(5,2) DEFAULT 0.00,
    owner VARCHAR(255) NOT NULL,
    team_members TEXT,
    expected_outcomes TEXT,
    success_criteria TEXT,
    risks TEXT,
    dependencies TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES strategic_plans(plan_id) ON DELETE CASCADE,
    FOREIGN KEY (objective_id) REFERENCES strategic_objectives(objective_id) ON DELETE SET NULL,
    INDEX idx_plan_id (plan_id),
    INDEX idx_objective_id (objective_id),
    INDEX idx_status (status),
    INDEX idx_priority (priority)
);

-- Initiative milestones
CREATE TABLE initiative_milestones (
    milestone_id VARCHAR(255) PRIMARY KEY,
    initiative_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    target_date DATE,
    actual_date DATE,
    status ENUM('not_started', 'in_progress', 'completed', 'delayed', 'cancelled') DEFAULT 'not_started',
    completion_percentage DECIMAL(5,2) DEFAULT 0.00,
    deliverables TEXT,
    success_criteria TEXT,
    owner VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (initiative_id) REFERENCES strategic_initiatives(initiative_id) ON DELETE CASCADE,
    INDEX idx_initiative_id (initiative_id),
    INDEX idx_target_date (target_date),
    INDEX idx_status (status)
);

-- Performance tracking table
CREATE TABLE strategic_performance (
    performance_id VARCHAR(255) PRIMARY KEY,
    plan_id VARCHAR(255) NOT NULL,
    objective_id VARCHAR(255),
    initiative_id VARCHAR(255),
    measurement_date DATE NOT NULL,
    kpi_name VARCHAR(255) NOT NULL,
    target_value DECIMAL(15,2),
    actual_value DECIMAL(15,2),
    variance_percentage DECIMAL(5,2),
    performance_rating ENUM('poor', 'below_target', 'on_target', 'above_target', 'excellent') DEFAULT 'on_target',
    trend ENUM('improving', 'stable', 'declining') DEFAULT 'stable',
    notes TEXT,
    data_source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES strategic_plans(plan_id) ON DELETE CASCADE,
    FOREIGN KEY (objective_id) REFERENCES strategic_objectives(objective_id) ON DELETE SET NULL,
    FOREIGN KEY (initiative_id) REFERENCES strategic_initiatives(initiative_id) ON DELETE SET NULL,
    INDEX idx_plan_id (plan_id),
    INDEX idx_objective_id (objective_id),
    INDEX idx_measurement_date (measurement_date),
    INDEX idx_kpi_name (kpi_name)
);

-- Strategic reviews table
CREATE TABLE strategic_reviews (
    review_id VARCHAR(255) PRIMARY KEY,
    plan_id VARCHAR(255) NOT NULL,
    review_type ENUM('monthly', 'quarterly', 'annual', 'ad_hoc', 'milestone') NOT NULL,
    review_date DATE NOT NULL,
    conducted_by VARCHAR(255) NOT NULL,
    attendees TEXT,
    overall_rating ENUM('poor', 'fair', 'good', 'excellent') DEFAULT 'good',
    key_achievements TEXT,
    major_challenges TEXT,
    course_corrections TEXT,
    action_items TEXT,
    next_review_date DATE,
    review_summary TEXT,
    recommendations TEXT,
    approved BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR(255),
    approval_date TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES strategic_plans(plan_id) ON DELETE CASCADE,
    INDEX idx_plan_id (plan_id),
    INDEX idx_review_date (review_date),
    INDEX idx_review_type (review_type)
);

-- Business situation analysis table
CREATE TABLE business_situation_analysis (
    analysis_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    plan_id VARCHAR(255),
    analysis_date DATE NOT NULL,
    analysis_type ENUM('swot', 'pestle', 'porters_five_forces', 'market_position', 'financial_health') NOT NULL,
    strengths TEXT,
    weaknesses TEXT,
    opportunities TEXT,
    threats TEXT,
    market_position_score DECIMAL(3,2) DEFAULT 0.50,
    competitive_advantage TEXT,
    growth_potential_score DECIMAL(3,2) DEFAULT 0.50,
    financial_health_score DECIMAL(3,2) DEFAULT 0.50,
    overall_assessment TEXT,
    key_insights TEXT,
    recommendations TEXT,
    data_sources TEXT,
    conducted_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES strategic_plans(plan_id) ON DELETE SET NULL,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_plan_id (plan_id),
    INDEX idx_analysis_date (analysis_date),
    INDEX idx_analysis_type (analysis_type)
);

-- AI analysis results for strategic planning
CREATE TABLE ai_strategic_analysis (
    analysis_id VARCHAR(255) PRIMARY KEY,
    plan_id VARCHAR(255),
    decision_id VARCHAR(255),
    scenario_id VARCHAR(255),
    analysis_type ENUM('objective_generation', 'scenario_analysis', 'decision_support',
                      'risk_assessment', 'opportunity_identification', 'resource_optimization') NOT NULL,
    ai_provider ENUM('openai', 'anthropic', 'azure', 'google') NOT NULL,
    model_used VARCHAR(100),
    input_data_size INT DEFAULT 0,
    processing_time_seconds INT DEFAULT 0,
    token_usage INT DEFAULT 0,
    cost DECIMAL(8,4) DEFAULT 0.0000,
    confidence_score DECIMAL(3,2) DEFAULT 0.50,
    quality_rating ENUM('poor', 'fair', 'good', 'excellent') DEFAULT 'good',
    key_insights TEXT,
    recommendations TEXT,
    raw_response TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES strategic_plans(plan_id) ON DELETE SET NULL,
    FOREIGN KEY (decision_id) REFERENCES strategic_decisions(decision_id) ON DELETE SET NULL,
    FOREIGN KEY (scenario_id) REFERENCES scenario_analyses(scenario_id) ON DELETE SET NULL,
    INDEX idx_plan_id (plan_id),
    INDEX idx_analysis_type (analysis_type),
    INDEX idx_ai_provider (ai_provider),
    INDEX idx_created_at (created_at)
);

-- Create views for common strategic planning queries

-- Strategic plan overview
CREATE VIEW strategic_plan_overview AS
SELECT
    sp.plan_id,
    sp.tenant_id,
    sp.name,
    sp.status,
    sp.time_horizon,
    sp.total_budget,
    sp.budget_used,
    sp.overall_progress,
    sp.success_probability,
    COUNT(DISTINCT so.objective_id) as total_objectives,
    COUNT(CASE WHEN so.status = 'completed' THEN 1 END) as completed_objectives,
    COUNT(DISTINCT si.initiative_id) as total_initiatives,
    COUNT(CASE WHEN si.status = 'completed' THEN 1 END) as completed_initiatives,
    COUNT(DISTINCT sa.scenario_id) as scenario_count,
    AVG(so.progress) as avg_objective_progress,
    sp.created_at,
    sp.updated_at
FROM strategic_plans sp
LEFT JOIN strategic_objectives so ON sp.plan_id = so.plan_id
LEFT JOIN strategic_initiatives si ON sp.plan_id = si.plan_id
LEFT JOIN scenario_analyses sa ON sp.plan_id = sa.plan_id
GROUP BY sp.plan_id;

-- Objective performance view
CREATE VIEW objective_performance AS
SELECT
    so.objective_id,
    so.plan_id,
    so.title,
    so.objective_type,
    so.target_value,
    so.current_value,
    so.progress,
    so.status,
    so.priority,
    CASE
        WHEN so.target_value > 0 THEN (so.current_value / so.target_value) * 100
        ELSE 0
    END as achievement_percentage,
    COUNT(DISTINCT ok.id) as kpi_count,
    COUNT(DISTINCT si.initiative_id) as initiative_count,
    AVG(sp.actual_value) as avg_kpi_performance,
    so.target_date,
    DATEDIFF(so.target_date, CURDATE()) as days_to_target
FROM strategic_objectives so
LEFT JOIN objective_kpis ok ON so.objective_id = ok.objective_id
LEFT JOIN strategic_initiatives si ON so.objective_id = si.objective_id
LEFT JOIN strategic_performance sp ON so.objective_id = sp.objective_id
GROUP BY so.objective_id;

-- Decision analysis summary
CREATE VIEW decision_analysis_summary AS
SELECT
    sd.decision_id,
    sd.plan_id,
    sd.title,
    sd.decision_type,
    sd.status,
    sd.urgency,
    sd.impact_level,
    sd.confidence_score,
    COUNT(DISTINCT do.option_id) as option_count,
    COUNT(DISTINCT dc.id) as criteria_count,
    COUNT(DISTINCT ds.id) as stakeholder_count,
    MAX(do.overall_score) as best_option_score,
    sd.created_at,
    DATEDIFF(CURDATE(), sd.created_at) as days_pending
FROM strategic_decisions sd
LEFT JOIN decision_options do ON sd.decision_id = do.decision_id
LEFT JOIN decision_criteria dc ON sd.decision_id = dc.decision_id
LEFT JOIN decision_stakeholders ds ON sd.decision_id = ds.decision_id
GROUP BY sd.decision_id;

-- Resource utilization view
CREATE VIEW resource_utilization AS
SELECT
    ra.allocation_id,
    ra.plan_id,
    ra.allocation_type,
    ra.total_amount,
    ra.allocated_amount,
    ra.remaining_amount,
    CASE
        WHEN ra.total_amount > 0 THEN (ra.allocated_amount / ra.total_amount) * 100
        ELSE 0
    END as utilization_percentage,
    COUNT(DISTINCT ad.id) as allocation_details_count,
    AVG(ad.expected_roi) as avg_expected_roi,
    ra.optimization_score,
    ra.efficiency_rating
FROM resource_allocations ra
LEFT JOIN allocation_details ad ON ra.allocation_id = ad.allocation_id
GROUP BY ra.allocation_id;

-- Strategic performance dashboard
CREATE VIEW strategic_dashboard AS
SELECT
    sp.plan_id,
    sp.tenant_id,
    sp.name as plan_name,
    sp.status as plan_status,
    sp.overall_progress,
    sp.success_probability,
    COUNT(DISTINCT so.objective_id) as total_objectives,
    SUM(CASE WHEN so.status = 'completed' THEN 1 ELSE 0 END) as completed_objectives,
    AVG(so.progress) as avg_objective_progress,
    COUNT(DISTINCT si.initiative_id) as total_initiatives,
    SUM(CASE WHEN si.status = 'completed' THEN 1 ELSE 0 END) as completed_initiatives,
    sp.total_budget,
    sp.budget_used,
    CASE
        WHEN sp.total_budget > 0 THEN (sp.budget_used / sp.total_budget) * 100
        ELSE 0
    END as budget_utilization,
    COUNT(DISTINCT sd.decision_id) as pending_decisions,
    COUNT(DISTINCT sa.scenario_id) as scenario_analyses,
    MAX(sr.review_date) as last_review_date
FROM strategic_plans sp
LEFT JOIN strategic_objectives so ON sp.plan_id = so.plan_id
LEFT JOIN strategic_initiatives si ON sp.plan_id = si.plan_id
LEFT JOIN strategic_decisions sd ON sp.plan_id = sd.plan_id AND sd.status IN ('pending_analysis', 'under_review')
LEFT JOIN scenario_analyses sa ON sp.plan_id = sa.plan_id
LEFT JOIN strategic_reviews sr ON sp.plan_id = sr.plan_id
GROUP BY sp.plan_id;
