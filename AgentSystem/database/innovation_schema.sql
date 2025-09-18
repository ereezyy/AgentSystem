-- Innovation and Opportunity Discovery Database Schema
-- Stores opportunity insights, market trends, and innovation patterns

-- Opportunities table - stores discovered business opportunities
CREATE TABLE opportunities (
    opportunity_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    opportunity_type ENUM('market_gap', 'technology_trend', 'customer_need', 'competitive_advantage',
                         'partnership', 'product_innovation', 'process_optimization', 'revenue_stream') NOT NULL,
    priority ENUM('critical', 'high', 'medium', 'low') NOT NULL,
    innovation_category ENUM('disruptive', 'incremental', 'architectural', 'radical') NOT NULL,
    market_size DECIMAL(15,2) DEFAULT 0.00,
    implementation_effort INT DEFAULT 5 CHECK (implementation_effort BETWEEN 1 AND 10),
    time_to_market INT DEFAULT 12, -- months
    revenue_potential DECIMAL(15,2) DEFAULT 0.00,
    confidence_score DECIMAL(3,2) DEFAULT 0.50 CHECK (confidence_score BETWEEN 0.00 AND 1.00),
    status ENUM('identified', 'evaluating', 'planning', 'implementing', 'completed', 'rejected') DEFAULT 'identified',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_priority (priority),
    INDEX idx_opportunity_type (opportunity_type),
    INDEX idx_status (status),
    INDEX idx_confidence_score (confidence_score),
    INDEX idx_created_at (created_at)
);

-- Opportunity data sources - tracks where opportunity data came from
CREATE TABLE opportunity_data_sources (
    id INT AUTO_INCREMENT PRIMARY KEY,
    opportunity_id VARCHAR(255) NOT NULL,
    source_name VARCHAR(255) NOT NULL,
    source_type ENUM('news_api', 'market_research', 'patent_db', 'social_media', 'financial_data',
                    'customer_feedback', 'competitive_intel', 'internal_data') NOT NULL,
    data_quality_score DECIMAL(3,2) DEFAULT 0.50,
    reliability_score DECIMAL(3,2) DEFAULT 0.50,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id) ON DELETE CASCADE,
    INDEX idx_opportunity_id (opportunity_id),
    INDEX idx_source_type (source_type)
);

-- Opportunity insights - key insights for each opportunity
CREATE TABLE opportunity_insights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    opportunity_id VARCHAR(255) NOT NULL,
    insight_text TEXT NOT NULL,
    insight_type ENUM('market_analysis', 'competitive_advantage', 'risk_assessment', 'success_factor',
                     'implementation_note', 'financial_projection') NOT NULL,
    confidence_level ENUM('high', 'medium', 'low') DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id) ON DELETE CASCADE,
    INDEX idx_opportunity_id (opportunity_id),
    INDEX idx_insight_type (insight_type)
);

-- Recommended actions for opportunities
CREATE TABLE opportunity_actions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    opportunity_id VARCHAR(255) NOT NULL,
    action_description TEXT NOT NULL,
    action_type ENUM('research', 'prototype', 'market_test', 'partnership', 'investment',
                    'team_building', 'technology_development', 'marketing') NOT NULL,
    priority ENUM('critical', 'high', 'medium', 'low') DEFAULT 'medium',
    estimated_effort_hours INT DEFAULT 0,
    estimated_cost DECIMAL(12,2) DEFAULT 0.00,
    timeline_weeks INT DEFAULT 4,
    dependencies TEXT,
    status ENUM('pending', 'in_progress', 'completed', 'blocked') DEFAULT 'pending',
    assigned_to VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id) ON DELETE CASCADE,
    INDEX idx_opportunity_id (opportunity_id),
    INDEX idx_action_type (action_type),
    INDEX idx_priority (priority),
    INDEX idx_status (status)
);

-- Opportunity risks and mitigation strategies
CREATE TABLE opportunity_risks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    opportunity_id VARCHAR(255) NOT NULL,
    risk_description TEXT NOT NULL,
    risk_type ENUM('market', 'technical', 'financial', 'competitive', 'regulatory', 'operational') NOT NULL,
    probability ENUM('high', 'medium', 'low') DEFAULT 'medium',
    impact ENUM('high', 'medium', 'low') DEFAULT 'medium',
    mitigation_strategy TEXT,
    mitigation_cost DECIMAL(12,2) DEFAULT 0.00,
    status ENUM('identified', 'mitigating', 'mitigated', 'accepted') DEFAULT 'identified',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id) ON DELETE CASCADE,
    INDEX idx_opportunity_id (opportunity_id),
    INDEX idx_risk_type (risk_type),
    INDEX idx_probability (probability),
    INDEX idx_impact (impact)
);

-- Success metrics for opportunities
CREATE TABLE opportunity_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    opportunity_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_type ENUM('revenue', 'cost_savings', 'market_share', 'customer_acquisition',
                    'efficiency_gain', 'time_savings', 'quality_improvement') NOT NULL,
    target_value DECIMAL(15,2),
    current_value DECIMAL(15,2) DEFAULT 0.00,
    unit VARCHAR(100),
    measurement_frequency ENUM('daily', 'weekly', 'monthly', 'quarterly', 'annually') DEFAULT 'monthly',
    baseline_date DATE,
    target_date DATE,
    status ENUM('not_started', 'tracking', 'achieved', 'missed') DEFAULT 'not_started',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id) ON DELETE CASCADE,
    INDEX idx_opportunity_id (opportunity_id),
    INDEX idx_metric_type (metric_type),
    INDEX idx_status (status)
);

-- Market trends table
CREATE TABLE market_trends (
    trend_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    growth_rate DECIMAL(5,2) DEFAULT 0.00, -- percentage
    market_size DECIMAL(15,2) DEFAULT 0.00,
    adoption_stage ENUM('emerging', 'early_adopter', 'early_majority', 'late_majority', 'laggards') DEFAULT 'emerging',
    confidence_score DECIMAL(3,2) DEFAULT 0.50 CHECK (confidence_score BETWEEN 0.00 AND 1.00),
    geographic_scope ENUM('local', 'regional', 'national', 'global') DEFAULT 'global',
    industry_impact ENUM('low', 'medium', 'high', 'transformative') DEFAULT 'medium',
    time_horizon ENUM('short_term', 'medium_term', 'long_term') DEFAULT 'medium_term', -- 1-2, 3-5, 5+ years
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_adoption_stage (adoption_stage),
    INDEX idx_growth_rate (growth_rate),
    INDEX idx_market_size (market_size),
    INDEX idx_industry_impact (industry_impact),
    INDEX idx_created_at (created_at)
);

-- Trend key players
CREATE TABLE trend_key_players (
    id INT AUTO_INCREMENT PRIMARY KEY,
    trend_id VARCHAR(255) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    player_type ENUM('leader', 'challenger', 'follower', 'niche_player') NOT NULL,
    market_share DECIMAL(5,2) DEFAULT 0.00,
    investment_amount DECIMAL(15,2) DEFAULT 0.00,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trend_id) REFERENCES market_trends(trend_id) ON DELETE CASCADE,
    INDEX idx_trend_id (trend_id),
    INDEX idx_player_type (player_type)
);

-- Trend technologies
CREATE TABLE trend_technologies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    trend_id VARCHAR(255) NOT NULL,
    technology_name VARCHAR(255) NOT NULL,
    maturity_level ENUM('research', 'development', 'pilot', 'deployment', 'mature') DEFAULT 'development',
    adoption_rate DECIMAL(5,2) DEFAULT 0.00,
    impact_score DECIMAL(3,2) DEFAULT 0.50,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trend_id) REFERENCES market_trends(trend_id) ON DELETE CASCADE,
    INDEX idx_trend_id (trend_id),
    INDEX idx_maturity_level (maturity_level)
);

-- Trend geographic regions
CREATE TABLE trend_regions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    trend_id VARCHAR(255) NOT NULL,
    region_name VARCHAR(255) NOT NULL,
    adoption_level ENUM('low', 'medium', 'high') DEFAULT 'medium',
    market_size DECIMAL(15,2) DEFAULT 0.00,
    growth_rate DECIMAL(5,2) DEFAULT 0.00,
    regulatory_environment ENUM('supportive', 'neutral', 'restrictive') DEFAULT 'neutral',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trend_id) REFERENCES market_trends(trend_id) ON DELETE CASCADE,
    INDEX idx_trend_id (trend_id),
    INDEX idx_adoption_level (adoption_level)
);

-- Innovation patterns table
CREATE TABLE innovation_patterns (
    pattern_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    frequency INT DEFAULT 0,
    success_rate DECIMAL(5,2) DEFAULT 0.00,
    pattern_type ENUM('technology_adoption', 'business_model', 'market_entry', 'disruption',
                     'collaboration', 'scaling') NOT NULL,
    confidence_score DECIMAL(3,2) DEFAULT 0.50 CHECK (confidence_score BETWEEN 0.00 AND 1.00),
    time_to_success_months INT DEFAULT 24,
    average_investment DECIMAL(15,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_pattern_type (pattern_type),
    INDEX idx_success_rate (success_rate),
    INDEX idx_frequency (frequency),
    INDEX idx_created_at (created_at)
);

-- Pattern industries
CREATE TABLE pattern_industries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pattern_id VARCHAR(255) NOT NULL,
    industry_name VARCHAR(255) NOT NULL,
    applicability_score DECIMAL(3,2) DEFAULT 0.50,
    success_examples INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pattern_id) REFERENCES innovation_patterns(pattern_id) ON DELETE CASCADE,
    INDEX idx_pattern_id (pattern_id),
    INDEX idx_industry_name (industry_name)
);

-- Pattern technologies
CREATE TABLE pattern_technologies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pattern_id VARCHAR(255) NOT NULL,
    technology_name VARCHAR(255) NOT NULL,
    importance_level ENUM('critical', 'important', 'supporting') DEFAULT 'important',
    adoption_difficulty ENUM('low', 'medium', 'high') DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pattern_id) REFERENCES innovation_patterns(pattern_id) ON DELETE CASCADE,
    INDEX idx_pattern_id (pattern_id),
    INDEX idx_importance_level (importance_level)
);

-- Pattern business models
CREATE TABLE pattern_business_models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pattern_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_type ENUM('subscription', 'marketplace', 'freemium', 'enterprise', 'advertising',
                   'transaction', 'licensing', 'hybrid') NOT NULL,
    success_rate DECIMAL(5,2) DEFAULT 0.00,
    revenue_potential ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pattern_id) REFERENCES innovation_patterns(pattern_id) ON DELETE CASCADE,
    INDEX idx_pattern_id (pattern_id),
    INDEX idx_model_type (model_type)
);

-- Pattern key success factors
CREATE TABLE pattern_success_factors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pattern_id VARCHAR(255) NOT NULL,
    factor_name VARCHAR(255) NOT NULL,
    factor_type ENUM('technical', 'market', 'financial', 'organizational', 'strategic') NOT NULL,
    importance_level ENUM('critical', 'important', 'nice_to_have') DEFAULT 'important',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pattern_id) REFERENCES innovation_patterns(pattern_id) ON DELETE CASCADE,
    INDEX idx_pattern_id (pattern_id),
    INDEX idx_factor_type (factor_type),
    INDEX idx_importance_level (importance_level)
);

-- Pattern examples
CREATE TABLE pattern_examples (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pattern_id VARCHAR(255) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    example_description TEXT,
    outcome ENUM('success', 'failure', 'mixed', 'ongoing') NOT NULL,
    timeline_months INT DEFAULT 0,
    investment_amount DECIMAL(15,2) DEFAULT 0.00,
    revenue_impact DECIMAL(15,2) DEFAULT 0.00,
    lessons_learned TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pattern_id) REFERENCES innovation_patterns(pattern_id) ON DELETE CASCADE,
    INDEX idx_pattern_id (pattern_id),
    INDEX idx_outcome (outcome)
);

-- Opportunity trend relationships
CREATE TABLE opportunity_trends (
    id INT AUTO_INCREMENT PRIMARY KEY,
    opportunity_id VARCHAR(255) NOT NULL,
    trend_id VARCHAR(255) NOT NULL,
    relationship_type ENUM('driven_by', 'enables', 'competes_with', 'complements') NOT NULL,
    strength ENUM('weak', 'moderate', 'strong') DEFAULT 'moderate',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id) ON DELETE CASCADE,
    FOREIGN KEY (trend_id) REFERENCES market_trends(trend_id) ON DELETE CASCADE,
    UNIQUE KEY unique_opportunity_trend (opportunity_id, trend_id),
    INDEX idx_opportunity_id (opportunity_id),
    INDEX idx_trend_id (trend_id)
);

-- Opportunity pattern relationships
CREATE TABLE opportunity_patterns (
    id INT AUTO_INCREMENT PRIMARY KEY,
    opportunity_id VARCHAR(255) NOT NULL,
    pattern_id VARCHAR(255) NOT NULL,
    applicability_score DECIMAL(3,2) DEFAULT 0.50,
    adaptation_required ENUM('none', 'minor', 'moderate', 'major') DEFAULT 'minor',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id) ON DELETE CASCADE,
    FOREIGN KEY (pattern_id) REFERENCES innovation_patterns(pattern_id) ON DELETE CASCADE,
    UNIQUE KEY unique_opportunity_pattern (opportunity_id, pattern_id),
    INDEX idx_opportunity_id (opportunity_id),
    INDEX idx_pattern_id (pattern_id)
);

-- Discovery sessions - track opportunity discovery runs
CREATE TABLE discovery_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    session_type ENUM('full_discovery', 'trend_analysis', 'pattern_analysis', 'focused_search') NOT NULL,
    focus_areas JSON,
    data_sources_used JSON,
    opportunities_found INT DEFAULT 0,
    trends_identified INT DEFAULT 0,
    patterns_discovered INT DEFAULT 0,
    session_duration_minutes INT DEFAULT 0,
    quality_score DECIMAL(3,2) DEFAULT 0.50,
    status ENUM('running', 'completed', 'failed', 'cancelled') DEFAULT 'running',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_session_type (session_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Data source performance tracking
CREATE TABLE data_source_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_name VARCHAR(255) NOT NULL,
    source_type ENUM('news_api', 'market_research', 'patent_db', 'social_media', 'financial_data') NOT NULL,
    requests_made INT DEFAULT 0,
    successful_requests INT DEFAULT 0,
    average_response_time_ms INT DEFAULT 0,
    data_quality_score DECIMAL(3,2) DEFAULT 0.50,
    cost_per_request DECIMAL(8,4) DEFAULT 0.0000,
    insights_generated INT DEFAULT 0,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status ENUM('active', 'inactive', 'error', 'rate_limited') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_source_type (source_type),
    INDEX idx_status (status),
    INDEX idx_last_used (last_used)
);

-- AI analysis results
CREATE TABLE ai_analysis_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    ai_provider ENUM('openai', 'anthropic', 'google', 'azure') NOT NULL,
    analysis_type ENUM('opportunity_identification', 'trend_analysis', 'pattern_recognition', 'risk_assessment') NOT NULL,
    input_data_size INT DEFAULT 0,
    processing_time_seconds INT DEFAULT 0,
    token_usage INT DEFAULT 0,
    cost DECIMAL(8,4) DEFAULT 0.0000,
    confidence_score DECIMAL(3,2) DEFAULT 0.50,
    insights_generated INT DEFAULT 0,
    quality_rating ENUM('excellent', 'good', 'fair', 'poor') DEFAULT 'good',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES discovery_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_ai_provider (ai_provider),
    INDEX idx_analysis_type (analysis_type)
);

-- Create views for common queries

-- Opportunity summary view
CREATE VIEW opportunity_summary AS
SELECT
    o.opportunity_id,
    o.tenant_id,
    o.title,
    o.opportunity_type,
    o.priority,
    o.innovation_category,
    o.market_size,
    o.revenue_potential,
    o.confidence_score,
    o.status,
    COUNT(DISTINCT oa.id) as action_count,
    COUNT(DISTINCT or_table.id) as risk_count,
    COUNT(DISTINCT om.id) as metric_count,
    o.created_at,
    o.updated_at
FROM opportunities o
LEFT JOIN opportunity_actions oa ON o.opportunity_id = oa.opportunity_id
LEFT JOIN opportunity_risks or_table ON o.opportunity_id = or_table.opportunity_id
LEFT JOIN opportunity_metrics om ON o.opportunity_id = om.opportunity_id
GROUP BY o.opportunity_id;

-- Trend analysis view
CREATE VIEW trend_analysis AS
SELECT
    mt.trend_id,
    mt.name,
    mt.growth_rate,
    mt.market_size,
    mt.adoption_stage,
    mt.industry_impact,
    COUNT(DISTINCT tkp.id) as key_players_count,
    COUNT(DISTINCT tt.id) as technologies_count,
    COUNT(DISTINCT tr.id) as regions_count,
    COUNT(DISTINCT ot.opportunity_id) as related_opportunities,
    mt.created_at
FROM market_trends mt
LEFT JOIN trend_key_players tkp ON mt.trend_id = tkp.trend_id
LEFT JOIN trend_technologies tt ON mt.trend_id = tt.trend_id
LEFT JOIN trend_regions tr ON mt.trend_id = tr.trend_id
LEFT JOIN opportunity_trends ot ON mt.trend_id = ot.trend_id
GROUP BY mt.trend_id;

-- Pattern success view
CREATE VIEW pattern_success AS
SELECT
    ip.pattern_id,
    ip.name,
    ip.pattern_type,
    ip.frequency,
    ip.success_rate,
    COUNT(DISTINCT pi.id) as applicable_industries,
    COUNT(DISTINCT pt.id) as required_technologies,
    COUNT(DISTINCT pbm.id) as business_models,
    COUNT(DISTINCT pe.id) as examples,
    AVG(CASE WHEN pe.outcome = 'success' THEN 1 ELSE 0 END) * 100 as actual_success_rate,
    ip.created_at
FROM innovation_patterns ip
LEFT JOIN pattern_industries pi ON ip.pattern_id = pi.pattern_id
LEFT JOIN pattern_technologies pt ON ip.pattern_id = pt.pattern_id
LEFT JOIN pattern_business_models pbm ON ip.pattern_id = pbm.pattern_id
LEFT JOIN pattern_examples pe ON ip.pattern_id = pe.pattern_id
GROUP BY ip.pattern_id;

-- Discovery performance view
CREATE VIEW discovery_performance AS
SELECT
    ds.session_id,
    ds.tenant_id,
    ds.session_type,
    ds.opportunities_found,
    ds.trends_identified,
    ds.patterns_discovered,
    ds.session_duration_minutes,
    ds.quality_score,
    AVG(aar.confidence_score) as avg_ai_confidence,
    SUM(aar.cost) as total_ai_cost,
    SUM(aar.token_usage) as total_tokens,
    ds.created_at,
    ds.completed_at
FROM discovery_sessions ds
LEFT JOIN ai_analysis_results aar ON ds.session_id = aar.session_id
WHERE ds.status = 'completed'
GROUP BY ds.session_id;
