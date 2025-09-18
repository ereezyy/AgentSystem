-- Automated Testing and Deployment Pipeline Database Schema
-- Stores pipeline configurations, executions, test results, and deployment history

-- Pipeline configurations
CREATE TABLE pipeline_configurations (
    pipeline_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    pipeline_name VARCHAR(255) NOT NULL,
    repository_url VARCHAR(500) NOT NULL,
    branch VARCHAR(100) NOT NULL DEFAULT 'main',
    environment VARCHAR(50) NOT NULL,
    deployment_strategy ENUM('rolling', 'blue_green', 'canary', 'recreate', 'a_b_testing') DEFAULT 'rolling',
    quality_gates JSON NOT NULL,
    test_configurations JSON NOT NULL,
    security_policies JSON,
    deployment_config JSON NOT NULL,
    notification_config JSON,
    rollback_config JSON,
    webhook_triggers JSON,
    enabled BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_pipeline_name (pipeline_name),
    INDEX idx_environment (environment),
    INDEX idx_enabled (enabled),
    INDEX idx_created_at (created_at)
);

-- Pipeline executions
CREATE TABLE pipeline_executions (
    execution_id VARCHAR(255) PRIMARY KEY,
    pipeline_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    status ENUM('running', 'completed', 'failed', 'cancelled', 'pending_approval') DEFAULT 'running',
    current_stage ENUM('source_checkout', 'dependency_install', 'lint_check', 'unit_tests', 'integration_tests',
                      'security_scan', 'performance_test', 'code_coverage', 'build_artifacts', 'container_build',
                      'vulnerability_scan', 'quality_gate', 'staging_deploy', 'e2e_tests', 'production_deploy',
                      'smoke_tests', 'monitoring_validation', 'rollback') NOT NULL,
    stages_completed JSON,
    stages_failed JSON,
    test_results JSON,
    security_scan_results JSON,
    quality_metrics JSON,
    deployment_details JSON,
    artifacts_generated JSON,
    execution_time_seconds DECIMAL(10,3) DEFAULT 0.000,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    triggered_by VARCHAR(255) NOT NULL,
    trigger_data JSON,
    commit_hash VARCHAR(100),
    branch_name VARCHAR(100),
    workspace_path VARCHAR(500),
    error_message TEXT,
    logs_location VARCHAR(500),
    FOREIGN KEY (pipeline_id) REFERENCES pipeline_configurations(pipeline_id) ON DELETE CASCADE,
    INDEX idx_pipeline_id (pipeline_id),
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_status (status),
    INDEX idx_current_stage (current_stage),
    INDEX idx_started_at (started_at),
    INDEX idx_triggered_by (triggered_by),
    INDEX idx_commit_hash (commit_hash)
);

-- Test execution results
CREATE TABLE test_execution_results (
    result_id VARCHAR(255) PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL,
    test_type ENUM('unit', 'integration', 'e2e', 'performance', 'security', 'smoke', 'regression', 'load') NOT NULL,
    test_framework VARCHAR(100) NOT NULL,
    total_tests INT DEFAULT 0,
    passed_tests INT DEFAULT 0,
    failed_tests INT DEFAULT 0,
    skipped_tests INT DEFAULT 0,
    pass_rate DECIMAL(5,2) DEFAULT 0.00,
    duration_seconds DECIMAL(10,3) DEFAULT 0.000,
    coverage_percentage DECIMAL(5,2) DEFAULT 0.00,
    test_output TEXT,
    test_report_path VARCHAR(500),
    failed_test_details JSON,
    performance_metrics JSON,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions(execution_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_test_type (test_type),
    INDEX idx_test_framework (test_framework),
    INDEX idx_pass_rate (pass_rate),
    INDEX idx_started_at (started_at)
);

-- Security scan results
CREATE TABLE security_scan_results (
    scan_id VARCHAR(255) PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL,
    scan_type ENUM('sast', 'dependency_scan', 'secret_scan', 'license_scan', 'container_scan') NOT NULL,
    scan_tool VARCHAR(100) NOT NULL,
    overall_score DECIMAL(3,1) DEFAULT 0.0,
    vulnerabilities_found INT DEFAULT 0,
    critical_vulnerabilities INT DEFAULT 0,
    high_vulnerabilities INT DEFAULT 0,
    medium_vulnerabilities INT DEFAULT 0,
    low_vulnerabilities INT DEFAULT 0,
    scan_duration_seconds DECIMAL(10,3) DEFAULT 0.000,
    scan_output TEXT,
    scan_report_path VARCHAR(500),
    vulnerability_details JSON,
    remediation_suggestions JSON,
    compliance_status JSON,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions(execution_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_scan_type (scan_type),
    INDEX idx_overall_score (overall_score),
    INDEX idx_vulnerabilities_found (vulnerabilities_found),
    INDEX idx_started_at (started_at)
);

-- Deployment history
CREATE TABLE deployment_history (
    deployment_id VARCHAR(255) PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    pipeline_id VARCHAR(255) NOT NULL,
    environment VARCHAR(50) NOT NULL,
    deployment_strategy VARCHAR(50) NOT NULL,
    container_image VARCHAR(500),
    commit_hash VARCHAR(100) NOT NULL,
    deployment_status ENUM('deploying', 'deployed', 'failed', 'rolled_back') DEFAULT 'deploying',
    deployment_url VARCHAR(500),
    instances_deployed INT DEFAULT 1,
    traffic_percentage DECIMAL(5,2) DEFAULT 100.00,
    health_check_passed BOOLEAN DEFAULT FALSE,
    deployment_time_seconds DECIMAL(10,3) DEFAULT 0.000,
    rollback_executed BOOLEAN DEFAULT FALSE,
    rollback_reason TEXT,
    deployment_config JSON,
    deployment_logs TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions(execution_id) ON DELETE CASCADE,
    FOREIGN KEY (pipeline_id) REFERENCES pipeline_configurations(pipeline_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_pipeline_id (pipeline_id),
    INDEX idx_environment (environment),
    INDEX idx_deployment_status (deployment_status),
    INDEX idx_commit_hash (commit_hash),
    INDEX idx_started_at (started_at)
);

-- Quality gate evaluations
CREATE TABLE quality_gate_evaluations (
    evaluation_id VARCHAR(255) PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL,
    gate_name VARCHAR(255) NOT NULL,
    gate_type ENUM('code_coverage', 'security_score', 'test_pass_rate', 'performance', 'lint_score', 'vulnerability_count') NOT NULL,
    threshold_value DECIMAL(10,4) NOT NULL,
    actual_value DECIMAL(10,4) NOT NULL,
    passed BOOLEAN NOT NULL,
    weight DECIMAL(3,2) DEFAULT 1.00,
    gate_config JSON,
    evaluation_details JSON,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions(execution_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_gate_type (gate_type),
    INDEX idx_passed (passed),
    INDEX idx_evaluated_at (evaluated_at)
);

-- Pipeline artifacts
CREATE TABLE pipeline_artifacts (
    artifact_id VARCHAR(255) PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL,
    artifact_name VARCHAR(255) NOT NULL,
    artifact_type ENUM('source_code', 'compiled_binary', 'container_image', 'test_report', 'security_report',
                      'coverage_report', 'documentation', 'configuration', 'deployment_manifest') NOT NULL,
    artifact_path VARCHAR(500) NOT NULL,
    artifact_size_bytes BIGINT DEFAULT 0,
    artifact_hash VARCHAR(100),
    storage_location VARCHAR(500),
    retention_days INT DEFAULT 90,
    download_count INT DEFAULT 0,
    last_downloaded TIMESTAMP NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NULL,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions(execution_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_artifact_type (artifact_type),
    INDEX idx_created_at (created_at),
    INDEX idx_expires_at (expires_at)
);

-- Pipeline notifications
CREATE TABLE pipeline_notifications (
    notification_id VARCHAR(255) PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL,
    notification_type ENUM('started', 'completed', 'failed', 'cancelled', 'approval_required', 'deployed') NOT NULL,
    channel_type ENUM('email', 'slack', 'webhook', 'teams', 'sms', 'pagerduty') NOT NULL,
    recipient VARCHAR(255) NOT NULL,
    subject VARCHAR(500),
    message TEXT,
    delivery_status ENUM('pending', 'sent', 'delivered', 'failed') DEFAULT 'pending',
    delivery_timestamp TIMESTAMP NULL,
    failure_reason TEXT,
    retry_count INT DEFAULT 0,
    notification_config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions(execution_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_notification_type (notification_type),
    INDEX idx_channel_type (channel_type),
    INDEX idx_delivery_status (delivery_status),
    INDEX idx_created_at (created_at)
);

-- Pipeline approvals
CREATE TABLE pipeline_approvals (
    approval_id VARCHAR(255) PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL,
    approval_type ENUM('manual', 'automated', 'conditional') NOT NULL,
    approval_stage ENUM('production_deploy', 'security_override', 'quality_override', 'emergency_deploy') NOT NULL,
    approval_status ENUM('pending', 'approved', 'rejected', 'expired') DEFAULT 'pending',
    approver_id VARCHAR(255),
    approver_name VARCHAR(255),
    approval_reason TEXT,
    rejection_reason TEXT,
    approval_criteria JSON,
    approval_deadline TIMESTAMP NULL,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    responded_at TIMESTAMP NULL,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions(execution_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_approval_type (approval_type),
    INDEX idx_approval_status (approval_status),
    INDEX idx_approver_id (approver_id),
    INDEX idx_requested_at (requested_at)
);

-- Pipeline metrics and analytics
CREATE TABLE pipeline_metrics (
    metric_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    pipeline_id VARCHAR(255),
    execution_id VARCHAR(255),
    metric_name VARCHAR(255) NOT NULL,
    metric_type ENUM('execution_time', 'success_rate', 'failure_rate', 'test_coverage', 'security_score',
                    'deployment_frequency', 'lead_time', 'mttr', 'change_failure_rate') NOT NULL,
    metric_value DECIMAL(20,6) NOT NULL,
    metric_unit VARCHAR(50),
    measurement_period_start TIMESTAMP NOT NULL,
    measurement_period_end TIMESTAMP NOT NULL,
    aggregation_level ENUM('execution', 'daily', 'weekly', 'monthly') NOT NULL,
    tags JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_pipeline_id (pipeline_id),
    INDEX idx_execution_id (execution_id),
    INDEX idx_metric_name (metric_name),
    INDEX idx_metric_type (metric_type),
    INDEX idx_measurement_period_start (measurement_period_start)
);

-- Pipeline environments
CREATE TABLE pipeline_environments (
    environment_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    environment_name VARCHAR(100) NOT NULL,
    environment_type ENUM('development', 'testing', 'staging', 'production', 'sandbox') NOT NULL,
    deployment_target VARCHAR(255) NOT NULL, -- k8s cluster, server group, etc.
    configuration JSON NOT NULL,
    health_check_url VARCHAR(500),
    monitoring_enabled BOOLEAN DEFAULT TRUE,
    auto_deploy_enabled BOOLEAN DEFAULT FALSE,
    approval_required BOOLEAN DEFAULT TRUE,
    resource_limits JSON,
    environment_variables JSON,
    secrets_config JSON,
    active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_environment_name (environment_name),
    INDEX idx_environment_type (environment_type),
    INDEX idx_active (active)
);

-- Pipeline webhooks
CREATE TABLE pipeline_webhooks (
    webhook_id VARCHAR(255) PRIMARY KEY,
    pipeline_id VARCHAR(255) NOT NULL,
    webhook_url VARCHAR(500) NOT NULL,
    webhook_secret VARCHAR(255),
    trigger_events JSON NOT NULL, -- ['push', 'pull_request', 'tag', etc.]
    branch_filters JSON, -- ['main', 'develop', 'release/*']
    path_filters JSON, -- ['src/**', '!docs/**']
    enabled BOOLEAN DEFAULT TRUE,
    last_triggered TIMESTAMP NULL,
    trigger_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipeline_configurations(pipeline_id) ON DELETE CASCADE,
    INDEX idx_pipeline_id (pipeline_id),
    INDEX idx_enabled (enabled),
    INDEX idx_last_triggered (last_triggered)
);

-- Pipeline schedules
CREATE TABLE pipeline_schedules (
    schedule_id VARCHAR(255) PRIMARY KEY,
    pipeline_id VARCHAR(255) NOT NULL,
    schedule_name VARCHAR(255) NOT NULL,
    cron_expression VARCHAR(100) NOT NULL,
    timezone VARCHAR(50) DEFAULT 'UTC',
    enabled BOOLEAN DEFAULT TRUE,
    next_run_time TIMESTAMP NULL,
    last_run_time TIMESTAMP NULL,
    run_count INT DEFAULT 0,
    max_concurrent_executions INT DEFAULT 1,
    schedule_config JSON,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipeline_configurations(pipeline_id) ON DELETE CASCADE,
    INDEX idx_pipeline_id (pipeline_id),
    INDEX idx_enabled (enabled),
    INDEX idx_next_run_time (next_run_time),
    INDEX idx_last_run_time (last_run_time)
);

-- Pipeline dependencies
CREATE TABLE pipeline_dependencies (
    dependency_id VARCHAR(255) PRIMARY KEY,
    pipeline_id VARCHAR(255) NOT NULL,
    depends_on_pipeline_id VARCHAR(255) NOT NULL,
    dependency_type ENUM('sequential', 'parallel', 'conditional') DEFAULT 'sequential',
    dependency_condition JSON, -- conditions for conditional dependencies
    wait_for_completion BOOLEAN DEFAULT TRUE,
    timeout_minutes INT DEFAULT 60,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipeline_configurations(pipeline_id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_pipeline_id) REFERENCES pipeline_configurations(pipeline_id) ON DELETE CASCADE,
    INDEX idx_pipeline_id (pipeline_id),
    INDEX idx_depends_on_pipeline_id (depends_on_pipeline_id),
    INDEX idx_dependency_type (dependency_type)
);

-- Pipeline templates
CREATE TABLE pipeline_templates (
    template_id VARCHAR(255) PRIMARY KEY,
    template_name VARCHAR(255) NOT NULL,
    template_description TEXT,
    template_category VARCHAR(100) NOT NULL, -- 'web_app', 'api', 'mobile', 'data_pipeline', etc.
    technology_stack JSON NOT NULL, -- ['python', 'react', 'docker', etc.]
    template_config JSON NOT NULL,
    default_quality_gates JSON NOT NULL,
    default_test_config JSON NOT NULL,
    default_deployment_config JSON NOT NULL,
    usage_count INT DEFAULT 0,
    rating DECIMAL(3,2) DEFAULT 0.00,
    public_template BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_template_category (template_category),
    INDEX idx_public_template (public_template),
    INDEX idx_usage_count (usage_count),
    INDEX idx_rating (rating)
);

-- Pipeline stage logs
CREATE TABLE pipeline_stage_logs (
    log_id VARCHAR(255) PRIMARY KEY,
    execution_id VARCHAR(255) NOT NULL,
    stage_name VARCHAR(100) NOT NULL,
    log_level ENUM('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') NOT NULL,
    log_message TEXT NOT NULL,
    log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_component VARCHAR(100),
    additional_data JSON,
    FOREIGN KEY (execution_id) REFERENCES pipeline_executions(execution_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_stage_name (stage_name),
    INDEX idx_log_level (log_level),
    INDEX idx_log_timestamp (log_timestamp)
);

-- Create views for pipeline analytics

-- Pipeline execution summary
CREATE VIEW pipeline_execution_summary AS
SELECT
    pe.tenant_id,
    pe.pipeline_id,
    pc.pipeline_name,
    pc.environment,
    COUNT(*) as total_executions,
    COUNT(CASE WHEN pe.status = 'completed' THEN 1 END) as successful_executions,
    COUNT(CASE WHEN pe.status = 'failed' THEN 1 END) as failed_executions,
    COUNT(CASE WHEN pe.status = 'running' THEN 1 END) as running_executions,
    AVG(pe.execution_time_seconds) as avg_execution_time_seconds,
    MIN(pe.execution_time_seconds) as min_execution_time_seconds,
    MAX(pe.execution_time_seconds) as max_execution_time_seconds,
    AVG(CASE WHEN pe.status = 'completed' THEN pe.execution_time_seconds END) as avg_successful_execution_time,
    COUNT(CASE WHEN pe.status = 'completed' THEN 1 END) / COUNT(*) * 100 as success_rate_percentage,
    MAX(pe.started_at) as last_execution_time,
    MIN(pe.started_at) as first_execution_time
FROM pipeline_executions pe
JOIN pipeline_configurations pc ON pe.pipeline_id = pc.pipeline_id
WHERE pe.started_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY pe.tenant_id, pe.pipeline_id, pc.pipeline_name, pc.environment;

-- Test results summary
CREATE VIEW test_results_summary AS
SELECT
    ter.execution_id,
    pe.tenant_id,
    pe.pipeline_id,
    ter.test_type,
    SUM(ter.total_tests) as total_tests,
    SUM(ter.passed_tests) as passed_tests,
    SUM(ter.failed_tests) as failed_tests,
    AVG(ter.pass_rate) as avg_pass_rate,
    AVG(ter.duration_seconds) as avg_duration_seconds,
    AVG(ter.coverage_percentage) as avg_coverage_percentage,
    COUNT(*) as test_executions
FROM test_execution_results ter
JOIN pipeline_executions pe ON ter.execution_id = pe.execution_id
WHERE ter.started_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY ter.execution_id, pe.tenant_id, pe.pipeline_id, ter.test_type;

-- Security scan summary
CREATE VIEW security_scan_summary AS
SELECT
    ssr.execution_id,
    pe.tenant_id,
    pe.pipeline_id,
    ssr.scan_type,
    AVG(ssr.overall_score) as avg_security_score,
    SUM(ssr.vulnerabilities_found) as total_vulnerabilities,
    SUM(ssr.critical_vulnerabilities) as total_critical_vulnerabilities,
    SUM(ssr.high_vulnerabilities) as total_high_vulnerabilities,
    AVG(ssr.scan_duration_seconds) as avg_scan_duration_seconds,
    COUNT(*) as scan_executions
FROM security_scan_results ssr
JOIN pipeline_executions pe ON ssr.execution_id = pe.execution_id
WHERE ssr.started_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY ssr.execution_id, pe.tenant_id, pe.pipeline_id, ssr.scan_type;

-- Deployment frequency metrics
CREATE VIEW deployment_frequency_metrics AS
SELECT
    dh.tenant_id,
    dh.pipeline_id,
    dh.environment,
    COUNT(*) as total_deployments,
    COUNT(CASE WHEN dh.deployment_status = 'deployed' THEN 1 END) as successful_deployments,
    COUNT(CASE WHEN dh.deployment_status = 'failed' THEN 1 END) as failed_deployments,
    COUNT(CASE WHEN dh.rollback_executed = TRUE THEN 1 END) as rollbacks_executed,
    AVG(dh.deployment_time_seconds) as avg_deployment_time_seconds,
    AVG(dh.traffic_percentage) as avg_traffic_percentage,
    COUNT(*) / 30 as deployments_per_day, -- last 30 days
    DATE(dh.started_at) as deployment_date
FROM deployment_history dh
WHERE dh.started_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY dh.tenant_id, dh.pipeline_id, dh.environment, DATE(dh.started_at);

-- Quality trends view
CREATE VIEW quality_trends AS
SELECT
    pe.tenant_id,
    pe.pipeline_id,
    DATE(pe.started_at) as execution_date,
    AVG(JSON_EXTRACT(pe.quality_metrics, '$.code_coverage.overall_coverage')) as avg_code_coverage,
    AVG(JSON_EXTRACT(pe.security_scan_results, '$.overall_score')) as avg_security_score,
    AVG(JSON_EXTRACT(pe.test_results, '$.unit.pass_rate')) as avg_unit_test_pass_rate,
    AVG(JSON_EXTRACT(pe.test_results, '$.performance.average_response_time_ms')) as avg_response_time_ms,
    COUNT(CASE WHEN pe.status = 'completed' THEN 1 END) as successful_executions,
    COUNT(CASE WHEN pe.status = 'failed' THEN 1 END) as failed_executions
FROM pipeline_executions pe
WHERE pe.started_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
AND pe.status IN ('completed', 'failed')
GROUP BY pe.tenant_id, pe.pipeline_id, DATE(pe.started_at)
ORDER BY execution_date DESC;

-- Pipeline performance dashboard
CREATE VIEW pipeline_performance_dashboard AS
SELECT
    pc.tenant_id,
    pc.pipeline_id,
    pc.pipeline_name,
    pc.environment,
    pc.deployment_strategy,
    COUNT(pe.execution_id) as total_executions_7d,
    COUNT(CASE WHEN pe.status = 'completed' THEN 1 END) as successful_executions_7d,
    COUNT(CASE WHEN pe.status = 'failed' THEN 1 END) as failed_executions_7d,
    AVG(pe.execution_time_seconds) as avg_execution_time_7d,
    AVG(JSON_EXTRACT(pe.quality_metrics, '$.code_coverage.overall_coverage')) as avg_coverage_7d,
    AVG(JSON_EXTRACT(pe.security_scan_results, '$.overall_score')) as avg_security_score_7d,
    COUNT(CASE WHEN pe.status = 'completed' THEN 1 END) / COUNT(*) * 100 as success_rate_7d,
    MAX(pe.started_at) as last_execution_time,
    COUNT(CASE WHEN dh.environment = 'production' AND dh.deployment_status = 'deployed' THEN 1 END) as prod_deployments_7d
FROM pipeline_configurations pc
LEFT JOIN pipeline_executions pe ON pc.pipeline_id = pe.pipeline_id
    AND pe.started_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
LEFT JOIN deployment_history dh ON pe.execution_id = dh.execution_id
    AND dh.started_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
WHERE pc.enabled = TRUE
GROUP BY pc.tenant_id, pc.pipeline_id, pc.pipeline_name, pc.environment, pc.deployment_strategy;
