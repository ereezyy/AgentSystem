-- Global Scaling and Localization Database Schema
-- Stores regional deployments, localization data, and global performance metrics

-- Supported locales configuration
CREATE TABLE supported_locales (
    locale_code VARCHAR(10) PRIMARY KEY,
    language_code VARCHAR(5) NOT NULL,
    country_code VARCHAR(5) NOT NULL,
    region ENUM('north_america', 'south_america', 'europe', 'asia_pacific', 'middle_east', 'africa', 'oceania') NOT NULL,
    currency_code VARCHAR(3) NOT NULL,
    timezone VARCHAR(50) NOT NULL,
    date_format VARCHAR(20) DEFAULT '%Y-%m-%d',
    number_format VARCHAR(20) DEFAULT '#,##0.00',
    rtl_support BOOLEAN DEFAULT FALSE,
    decimal_separator VARCHAR(1) DEFAULT '.',
    thousands_separator VARCHAR(1) DEFAULT ',',
    supported_by_ai BOOLEAN DEFAULT FALSE,
    translation_quality DECIMAL(3,2) DEFAULT 0.50,
    localization_status ENUM('not_started', 'in_progress', 'completed', 'needs_review', 'approved') DEFAULT 'not_started',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_language_code (language_code),
    INDEX idx_country_code (country_code),
    INDEX idx_region (region),
    INDEX idx_localization_status (localization_status)
);

-- Compliance requirements mapping
CREATE TABLE locale_compliance_requirements (
    id INT AUTO_INCREMENT PRIMARY KEY,
    locale_code VARCHAR(10) NOT NULL,
    compliance_regime ENUM('gdpr', 'ccpa', 'pipeda', 'lgpd', 'pdpa_sg', 'pdpa_th', 'dpa_uk', 'sox', 'hipaa') NOT NULL,
    mandatory BOOLEAN DEFAULT TRUE,
    implementation_status ENUM('not_implemented', 'in_progress', 'implemented', 'verified') DEFAULT 'not_implemented',
    compliance_score DECIMAL(5,2) DEFAULT 0.00,
    last_audit_date DATE,
    next_audit_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (locale_code) REFERENCES supported_locales(locale_code) ON DELETE CASCADE,
    UNIQUE KEY unique_locale_compliance (locale_code, compliance_regime),
    INDEX idx_locale_code (locale_code),
    INDEX idx_compliance_regime (compliance_regime)
);

-- Regional deployments
CREATE TABLE regional_deployments (
    deployment_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    region ENUM('north_america', 'south_america', 'europe', 'asia_pacific', 'middle_east', 'africa', 'oceania') NOT NULL,
    deployment_mode ENUM('single_region', 'multi_region', 'global', 'edge_distributed') NOT NULL,
    deployment_status ENUM('planning', 'deploying', 'active', 'maintenance', 'failed', 'terminated') DEFAULT 'planning',
    health_status ENUM('healthy', 'degraded', 'unhealthy', 'unknown') DEFAULT 'unknown',
    primary_data_center VARCHAR(100),
    failover_data_center VARCHAR(100),
    edge_locations JSON,
    supported_locales JSON,
    latency_target_ms INT DEFAULT 100,
    availability_target DECIMAL(5,3) DEFAULT 99.900,
    capacity_cpu_cores INT DEFAULT 0,
    capacity_memory_gb INT DEFAULT 0,
    capacity_storage_gb INT DEFAULT 0,
    capacity_bandwidth_mbps INT DEFAULT 0,
    auto_scaling_enabled BOOLEAN DEFAULT TRUE,
    auto_failover_enabled BOOLEAN DEFAULT TRUE,
    cost_per_hour DECIMAL(8,4) DEFAULT 0.0000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    activated_at TIMESTAMP NULL,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_region (region),
    INDEX idx_deployment_status (deployment_status),
    INDEX idx_health_status (health_status)
);

-- Regional deployment compliance
CREATE TABLE deployment_compliance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    deployment_id VARCHAR(255) NOT NULL,
    compliance_regime ENUM('gdpr', 'ccpa', 'pipeda', 'lgpd', 'pdpa_sg', 'pdpa_th', 'dpa_uk', 'sox', 'hipaa') NOT NULL,
    data_residency_required BOOLEAN DEFAULT FALSE,
    data_residency_location VARCHAR(100),
    encryption_at_rest BOOLEAN DEFAULT TRUE,
    encryption_in_transit BOOLEAN DEFAULT TRUE,
    access_logging_enabled BOOLEAN DEFAULT TRUE,
    audit_trail_enabled BOOLEAN DEFAULT TRUE,
    compliance_status ENUM('compliant', 'non_compliant', 'partial', 'unknown') DEFAULT 'unknown',
    last_compliance_check TIMESTAMP NULL,
    compliance_score DECIMAL(5,2) DEFAULT 0.00,
    violations_count INT DEFAULT 0,
    remediation_plan TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (deployment_id) REFERENCES regional_deployments(deployment_id) ON DELETE CASCADE,
    UNIQUE KEY unique_deployment_compliance (deployment_id, compliance_regime),
    INDEX idx_deployment_id (deployment_id),
    INDEX idx_compliance_regime (compliance_regime),
    INDEX idx_compliance_status (compliance_status)
);

-- Translation jobs
CREATE TABLE translation_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    job_name VARCHAR(255) NOT NULL,
    source_language VARCHAR(10) NOT NULL,
    target_languages JSON NOT NULL,
    content_type ENUM('ui', 'documentation', 'marketing', 'legal', 'technical', 'support') NOT NULL,
    priority ENUM('low', 'standard', 'high', 'urgent') DEFAULT 'standard',
    translation_method ENUM('ai', 'human', 'hybrid', 'crowd_sourced') NOT NULL,
    word_count INT DEFAULT 0,
    estimated_cost DECIMAL(10,2) DEFAULT 0.00,
    actual_cost DECIMAL(10,2) DEFAULT 0.00,
    status ENUM('queued', 'in_progress', 'review', 'completed', 'cancelled', 'failed') DEFAULT 'queued',
    progress DECIMAL(5,2) DEFAULT 0.00,
    quality_score DECIMAL(3,2) DEFAULT 0.00,
    assigned_translator VARCHAR(255),
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    estimated_completion TIMESTAMP NULL,
    deadline TIMESTAMP NULL,
    notes TEXT,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_source_language (source_language),
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_created_at (created_at)
);

-- Translation content items
CREATE TABLE translation_content_items (
    item_id INT AUTO_INCREMENT PRIMARY KEY,
    job_id VARCHAR(255) NOT NULL,
    content_key VARCHAR(500) NOT NULL,
    source_text TEXT NOT NULL,
    context_description TEXT,
    character_limit INT,
    content_category ENUM('button', 'label', 'message', 'title', 'description', 'instruction', 'error') NOT NULL,
    requires_review BOOLEAN DEFAULT FALSE,
    sensitive_content BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES translation_jobs(job_id) ON DELETE CASCADE,
    INDEX idx_job_id (job_id),
    INDEX idx_content_key (content_key),
    INDEX idx_content_category (content_category)
);

-- Translation results
CREATE TABLE translation_results (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    item_id INT NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    translated_text TEXT NOT NULL,
    translation_method ENUM('ai', 'human', 'hybrid', 'crowd_sourced') NOT NULL,
    translator_id VARCHAR(255),
    quality_score DECIMAL(3,2) DEFAULT 0.00,
    confidence_score DECIMAL(3,2) DEFAULT 0.00,
    review_status ENUM('pending', 'approved', 'rejected', 'needs_revision') DEFAULT 'pending',
    reviewer_id VARCHAR(255),
    review_notes TEXT,
    alternative_translations JSON,
    cultural_adaptations TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES translation_content_items(item_id) ON DELETE CASCADE,
    INDEX idx_item_id (item_id),
    INDEX idx_target_language (target_language),
    INDEX idx_translation_method (translation_method),
    INDEX idx_review_status (review_status)
);

-- Localization quality assessments
CREATE TABLE localization_quality_assessments (
    assessment_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    locale_code VARCHAR(10) NOT NULL,
    assessment_type ENUM('linguistic', 'cultural', 'functional', 'visual', 'comprehensive') NOT NULL,
    overall_score DECIMAL(5,2) DEFAULT 0.00,
    linguistic_accuracy DECIMAL(5,2) DEFAULT 0.00,
    cultural_appropriateness DECIMAL(5,2) DEFAULT 0.00,
    functional_correctness DECIMAL(5,2) DEFAULT 0.00,
    visual_layout DECIMAL(5,2) DEFAULT 0.00,
    user_experience DECIMAL(5,2) DEFAULT 0.00,
    assessor_type ENUM('automated', 'human_expert', 'crowd_sourced', 'native_speaker') NOT NULL,
    assessor_id VARCHAR(255),
    sample_size INT DEFAULT 0,
    issues_found INT DEFAULT 0,
    critical_issues INT DEFAULT 0,
    recommendations TEXT,
    assessment_date DATE NOT NULL,
    next_assessment_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_locale_code (locale_code),
    INDEX idx_assessment_type (assessment_type),
    INDEX idx_overall_score (overall_score),
    INDEX idx_assessment_date (assessment_date)
);

-- Global performance metrics
CREATE TABLE global_performance_metrics (
    metric_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    measurement_timestamp TIMESTAMP NOT NULL,
    total_regions INT DEFAULT 0,
    active_locales INT DEFAULT 0,
    global_users_count INT DEFAULT 0,
    regional_user_distribution JSON,
    average_latency_ms DECIMAL(7,2) DEFAULT 0.00,
    p95_latency_ms DECIMAL(7,2) DEFAULT 0.00,
    p99_latency_ms DECIMAL(7,2) DEFAULT 0.00,
    availability_percentage DECIMAL(5,3) DEFAULT 0.000,
    error_rate_percentage DECIMAL(5,3) DEFAULT 0.000,
    throughput_rps DECIMAL(10,2) DEFAULT 0.00,
    data_transfer_gb DECIMAL(12,2) DEFAULT 0.00,
    total_cost_usd DECIMAL(12,2) DEFAULT 0.00,
    cost_per_region JSON,
    compliance_score DECIMAL(5,2) DEFAULT 0.00,
    translation_coverage_percentage DECIMAL(5,2) DEFAULT 0.00,
    localization_quality_score DECIMAL(5,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_measurement_timestamp (measurement_timestamp),
    INDEX idx_availability_percentage (availability_percentage),
    INDEX idx_average_latency_ms (average_latency_ms)
);

-- Regional performance metrics
CREATE TABLE regional_performance_metrics (
    metric_id VARCHAR(255) PRIMARY KEY,
    deployment_id VARCHAR(255) NOT NULL,
    region ENUM('north_america', 'south_america', 'europe', 'asia_pacific', 'middle_east', 'africa', 'oceania') NOT NULL,
    measurement_timestamp TIMESTAMP NOT NULL,
    users_count INT DEFAULT 0,
    active_sessions INT DEFAULT 0,
    average_latency_ms DECIMAL(7,2) DEFAULT 0.00,
    availability_percentage DECIMAL(5,3) DEFAULT 0.000,
    error_rate_percentage DECIMAL(5,3) DEFAULT 0.000,
    throughput_rps DECIMAL(10,2) DEFAULT 0.00,
    cpu_utilization_percentage DECIMAL(5,2) DEFAULT 0.00,
    memory_utilization_percentage DECIMAL(5,2) DEFAULT 0.00,
    storage_utilization_percentage DECIMAL(5,2) DEFAULT 0.00,
    network_utilization_mbps DECIMAL(10,2) DEFAULT 0.00,
    cost_per_hour DECIMAL(8,4) DEFAULT 0.0000,
    auto_scaling_events INT DEFAULT 0,
    failover_events INT DEFAULT 0,
    health_check_status ENUM('healthy', 'degraded', 'unhealthy') DEFAULT 'healthy',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (deployment_id) REFERENCES regional_deployments(deployment_id) ON DELETE CASCADE,
    INDEX idx_deployment_id (deployment_id),
    INDEX idx_region (region),
    INDEX idx_measurement_timestamp (measurement_timestamp),
    INDEX idx_health_check_status (health_check_status)
);

-- Data sovereignty tracking
CREATE TABLE data_sovereignty_records (
    record_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    data_type ENUM('user_data', 'transactional_data', 'analytics_data', 'backup_data', 'log_data') NOT NULL,
    data_classification ENUM('public', 'internal', 'confidential', 'restricted') NOT NULL,
    storage_region ENUM('north_america', 'south_america', 'europe', 'asia_pacific', 'middle_east', 'africa', 'oceania') NOT NULL,
    storage_location VARCHAR(100) NOT NULL,
    compliance_requirements JSON,
    encryption_status BOOLEAN DEFAULT TRUE,
    access_restrictions JSON,
    retention_period_days INT DEFAULT 2555, -- 7 years default
    data_size_gb DECIMAL(12,2) DEFAULT 0.00,
    created_date DATE NOT NULL,
    last_accessed TIMESTAMP NULL,
    scheduled_deletion DATE NULL,
    compliance_status ENUM('compliant', 'non_compliant', 'under_review') DEFAULT 'compliant',
    audit_trail TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_data_type (data_type),
    INDEX idx_storage_region (storage_region),
    INDEX idx_compliance_status (compliance_status),
    INDEX idx_created_date (created_date)
);

-- Global routing configuration
CREATE TABLE global_routing_config (
    config_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    routing_strategy ENUM('geo_proximity', 'latency_based', 'load_based', 'cost_optimized', 'hybrid') NOT NULL,
    primary_region ENUM('north_america', 'south_america', 'europe', 'asia_pacific', 'middle_east', 'africa', 'oceania') NOT NULL,
    failover_regions JSON,
    health_check_enabled BOOLEAN DEFAULT TRUE,
    health_check_interval_seconds INT DEFAULT 30,
    circuit_breaker_enabled BOOLEAN DEFAULT TRUE,
    circuit_breaker_threshold INT DEFAULT 5,
    session_affinity BOOLEAN DEFAULT TRUE,
    ssl_termination_enabled BOOLEAN DEFAULT TRUE,
    compression_enabled BOOLEAN DEFAULT TRUE,
    caching_enabled BOOLEAN DEFAULT TRUE,
    cache_ttl_seconds INT DEFAULT 3600,
    traffic_splitting_rules JSON,
    maintenance_mode BOOLEAN DEFAULT FALSE,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_routing_strategy (routing_strategy),
    INDEX idx_primary_region (primary_region),
    INDEX idx_active (active)
);

-- CDN and edge cache configuration
CREATE TABLE cdn_edge_config (
    config_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    edge_location VARCHAR(100) NOT NULL,
    region ENUM('north_america', 'south_america', 'europe', 'asia_pacific', 'middle_east', 'africa', 'oceania') NOT NULL,
    cache_size_gb INT DEFAULT 100,
    cache_hit_ratio DECIMAL(5,2) DEFAULT 0.00,
    bandwidth_limit_mbps INT DEFAULT 1000,
    compression_types JSON,
    cache_rules JSON,
    origin_server VARCHAR(255),
    failover_origin VARCHAR(255),
    ssl_certificate_id VARCHAR(255),
    custom_domain VARCHAR(255),
    security_headers JSON,
    rate_limiting_enabled BOOLEAN DEFAULT TRUE,
    rate_limit_rps INT DEFAULT 1000,
    ddos_protection_enabled BOOLEAN DEFAULT TRUE,
    waf_enabled BOOLEAN DEFAULT TRUE,
    status ENUM('active', 'inactive', 'maintenance') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_edge_location (edge_location),
    INDEX idx_region (region),
    INDEX idx_status (status)
);

-- Scaling policies and rules
CREATE TABLE auto_scaling_policies (
    policy_id VARCHAR(255) PRIMARY KEY,
    deployment_id VARCHAR(255) NOT NULL,
    policy_name VARCHAR(255) NOT NULL,
    metric_type ENUM('cpu', 'memory', 'network', 'requests_per_second', 'response_time', 'custom') NOT NULL,
    scale_up_threshold DECIMAL(7,2) NOT NULL,
    scale_down_threshold DECIMAL(7,2) NOT NULL,
    scale_up_adjustment INT DEFAULT 1,
    scale_down_adjustment INT DEFAULT -1,
    min_instances INT DEFAULT 1,
    max_instances INT DEFAULT 10,
    cooldown_period_seconds INT DEFAULT 300,
    evaluation_period_seconds INT DEFAULT 60,
    datapoints_to_alarm INT DEFAULT 2,
    scaling_enabled BOOLEAN DEFAULT TRUE,
    notification_topic VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (deployment_id) REFERENCES regional_deployments(deployment_id) ON DELETE CASCADE,
    INDEX idx_deployment_id (deployment_id),
    INDEX idx_metric_type (metric_type),
    INDEX idx_scaling_enabled (scaling_enabled)
);

-- Scaling events log
CREATE TABLE scaling_events (
    event_id VARCHAR(255) PRIMARY KEY,
    policy_id VARCHAR(255) NOT NULL,
    deployment_id VARCHAR(255) NOT NULL,
    event_type ENUM('scale_up', 'scale_down', 'scale_out', 'scale_in') NOT NULL,
    trigger_metric VARCHAR(100) NOT NULL,
    trigger_value DECIMAL(10,2) NOT NULL,
    threshold_value DECIMAL(10,2) NOT NULL,
    instances_before INT NOT NULL,
    instances_after INT NOT NULL,
    scaling_reason TEXT,
    execution_status ENUM('initiated', 'in_progress', 'completed', 'failed') DEFAULT 'initiated',
    execution_time_seconds INT DEFAULT 0,
    cost_impact DECIMAL(8,4) DEFAULT 0.0000,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_timestamp TIMESTAMP NULL,
    FOREIGN KEY (policy_id) REFERENCES auto_scaling_policies(policy_id) ON DELETE CASCADE,
    FOREIGN KEY (deployment_id) REFERENCES regional_deployments(deployment_id) ON DELETE CASCADE,
    INDEX idx_policy_id (policy_id),
    INDEX idx_deployment_id (deployment_id),
    INDEX idx_event_type (event_type),
    INDEX idx_event_timestamp (event_timestamp)
);

-- Create views for global monitoring and reporting

-- Global deployment overview
CREATE VIEW global_deployment_overview AS
SELECT
    rd.tenant_id,
    COUNT(DISTINCT rd.deployment_id) as total_deployments,
    COUNT(DISTINCT rd.region) as regions_deployed,
    COUNT(CASE WHEN rd.deployment_status = 'active' THEN 1 END) as active_deployments,
    COUNT(CASE WHEN rd.health_status = 'healthy' THEN 1 END) as healthy_deployments,
    AVG(rd.latency_target_ms) as avg_latency_target,
    SUM(rd.capacity_cpu_cores) as total_cpu_cores,
    SUM(rd.capacity_memory_gb) as total_memory_gb,
    SUM(rd.cost_per_hour) as total_cost_per_hour,
    MAX(rd.updated_at) as last_updated
FROM regional_deployments rd
GROUP BY rd.tenant_id;

-- Localization status overview
CREATE VIEW localization_status_overview AS
SELECT
    sl.region,
    COUNT(*) as total_locales,
    COUNT(CASE WHEN sl.localization_status = 'completed' THEN 1 END) as completed_locales,
    COUNT(CASE WHEN sl.localization_status = 'in_progress' THEN 1 END) as in_progress_locales,
    COUNT(CASE WHEN sl.localization_status = 'approved' THEN 1 END) as approved_locales,
    AVG(sl.translation_quality) as avg_translation_quality,
    COUNT(CASE WHEN sl.supported_by_ai = TRUE THEN 1 END) as ai_supported_locales
FROM supported_locales sl
WHERE sl.active = TRUE
GROUP BY sl.region;

-- Translation job statistics
CREATE VIEW translation_job_statistics AS
SELECT
    tj.tenant_id,
    tj.source_language,
    COUNT(*) as total_jobs,
    COUNT(CASE WHEN tj.status = 'completed' THEN 1 END) as completed_jobs,
    COUNT(CASE WHEN tj.status = 'in_progress' THEN 1 END) as in_progress_jobs,
    AVG(tj.quality_score) as avg_quality_score,
    SUM(tj.word_count) as total_words_translated,
    SUM(tj.actual_cost) as total_translation_cost,
    AVG(TIMESTAMPDIFF(HOUR, tj.started_at, tj.completed_at)) as avg_completion_time_hours
FROM translation_jobs tj
WHERE tj.status IN ('completed', 'in_progress')
GROUP BY tj.tenant_id, tj.source_language;

-- Global performance dashboard
CREATE VIEW global_performance_dashboard AS
SELECT
    gpm.tenant_id,
    gpm.measurement_timestamp,
    gpm.total_regions,
    gpm.active_locales,
    gpm.global_users_count,
    gpm.average_latency_ms,
    gpm.availability_percentage,
    gpm.error_rate_percentage,
    gpm.throughput_rps,
    gpm.total_cost_usd,
    gpm.compliance_score,
    gpm.translation_coverage_percentage,
    gpm.localization_quality_score,
    ROW_NUMBER() OVER (PARTITION BY gpm.tenant_id ORDER BY gpm.measurement_timestamp DESC) as rn
FROM global_performance_metrics gpm;

-- Regional health status
CREATE VIEW regional_health_status AS
SELECT
    rd.tenant_id,
    rd.region,
    rd.deployment_status,
    rd.health_status,
    rpm.average_latency_ms,
    rpm.availability_percentage,
    rpm.error_rate_percentage,
    rpm.cpu_utilization_percentage,
    rpm.memory_utilization_percentage,
    rpm.health_check_status,
    rpm.measurement_timestamp
FROM regional_deployments rd
LEFT JOIN regional_performance_metrics rpm ON rd.deployment_id = rpm.deployment_id
WHERE rpm.measurement_timestamp = (
    SELECT MAX(rpm2.measurement_timestamp)
    FROM regional_performance_metrics rpm2
    WHERE rpm2.deployment_id = rd.deployment_id
)
OR rpm.measurement_timestamp IS NULL;

-- Compliance status summary
CREATE VIEW compliance_status_summary AS
SELECT
    dc.deployment_id,
    rd.tenant_id,
    rd.region,
    COUNT(*) as total_compliance_requirements,
    COUNT(CASE WHEN dc.compliance_status = 'compliant' THEN 1 END) as compliant_requirements,
    COUNT(CASE WHEN dc.compliance_status = 'non_compliant' THEN 1 END) as non_compliant_requirements,
    AVG(dc.compliance_score) as avg_compliance_score,
    SUM(dc.violations_count) as total_violations
FROM deployment_compliance dc
JOIN regional_deployments rd ON dc.deployment_id = rd.deployment_id
GROUP BY dc.deployment_id, rd.tenant_id, rd.region;
