-- Comprehensive Monitoring and Alerting Database Schema
-- Stores metrics, alerts, health checks, and monitoring configurations

-- Monitoring sessions - track active monitoring configurations
CREATE TABLE monitoring_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    session_name VARCHAR(255) NOT NULL,
    monitoring_scope ENUM('global', 'regional', 'tenant', 'service', 'component') NOT NULL,
    status ENUM('active', 'paused', 'stopped', 'error') DEFAULT 'active',
    metrics_configured INT DEFAULT 0,
    alerts_configured INT DEFAULT 0,
    health_checks_configured INT DEFAULT 0,
    notification_channels_configured INT DEFAULT 0,
    data_retention_days INT DEFAULT 90,
    sampling_interval_seconds INT DEFAULT 30,
    created_by VARCHAR(255) NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    stopped_at TIMESTAMP NULL,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    configuration JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_status (status),
    INDEX idx_monitoring_scope (monitoring_scope),
    INDEX idx_started_at (started_at)
);

-- Metrics table - stores all collected metrics
CREATE TABLE monitoring_metrics (
    metric_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_type ENUM('system', 'application', 'business', 'security', 'performance', 'availability', 'custom') NOT NULL,
    metric_value DECIMAL(20,6) NOT NULL,
    metric_unit VARCHAR(50),
    metric_labels JSON,
    threshold_warning DECIMAL(20,6),
    threshold_critical DECIMAL(20,6),
    baseline_value DECIMAL(20,6),
    trend ENUM('increasing', 'decreasing', 'stable', 'volatile', 'unknown') DEFAULT 'unknown',
    anomaly_score DECIMAL(5,4) DEFAULT 0.0000,
    collection_timestamp TIMESTAMP NOT NULL,
    source_system VARCHAR(100) NOT NULL,
    region VARCHAR(50),
    component VARCHAR(100),
    tags JSON,
    retention_days INT DEFAULT 90,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES monitoring_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_metric_name (metric_name),
    INDEX idx_metric_type (metric_type),
    INDEX idx_collection_timestamp (collection_timestamp),
    INDEX idx_source_system (source_system),
    INDEX idx_anomaly_score (anomaly_score)
);

-- Alert rules - define conditions for generating alerts
CREATE TABLE alert_rules (
    rule_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,
    description TEXT,
    metric_name VARCHAR(255) NOT NULL,
    condition_operator ENUM('>', '<', '>=', '<=', '==', '!=', 'contains', 'not_contains') NOT NULL,
    threshold_value DECIMAL(20,6) NOT NULL,
    severity ENUM('info', 'warning', 'error', 'critical', 'emergency') NOT NULL,
    evaluation_window_seconds INT DEFAULT 300,
    evaluation_frequency_seconds INT DEFAULT 60,
    minimum_occurrences INT DEFAULT 1,
    consecutive_breaches_required INT DEFAULT 1,
    recovery_threshold DECIMAL(20,6),
    recovery_occurrences INT DEFAULT 1,
    enabled BOOLEAN DEFAULT TRUE,
    suppression_enabled BOOLEAN DEFAULT FALSE,
    suppression_start_time TIME,
    suppression_end_time TIME,
    suppression_days JSON, -- ['monday', 'tuesday', etc.]
    escalation_policy_id VARCHAR(255),
    runbook_url VARCHAR(500),
    tags JSON,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_metric_name (metric_name),
    INDEX idx_severity (severity),
    INDEX idx_enabled (enabled),
    INDEX idx_created_at (created_at)
);

-- Alert rule notification channels
CREATE TABLE alert_rule_notifications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rule_id VARCHAR(255) NOT NULL,
    notification_channel ENUM('email', 'slack', 'sms', 'webhook', 'pagerduty', 'teams', 'discord') NOT NULL,
    channel_config JSON NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    delay_seconds INT DEFAULT 0,
    repeat_interval_seconds INT DEFAULT 0,
    max_notifications INT DEFAULT 0, -- 0 = unlimited
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rule_id) REFERENCES alert_rules(rule_id) ON DELETE CASCADE,
    INDEX idx_rule_id (rule_id),
    INDEX idx_notification_channel (notification_channel)
);

-- Alerts table - stores generated alerts
CREATE TABLE alerts (
    alert_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    rule_id VARCHAR(255) NOT NULL,
    alert_name VARCHAR(255) NOT NULL,
    alert_description TEXT,
    severity ENUM('info', 'warning', 'error', 'critical', 'emergency') NOT NULL,
    status ENUM('open', 'acknowledged', 'investigating', 'resolved', 'suppressed') DEFAULT 'open',
    metric_name VARCHAR(255) NOT NULL,
    current_value DECIMAL(20,6) NOT NULL,
    threshold_value DECIMAL(20,6) NOT NULL,
    source_system VARCHAR(100) NOT NULL,
    affected_components JSON,
    runbook_url VARCHAR(500),
    escalation_policy VARCHAR(255),
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP NULL,
    resolved_by VARCHAR(255),
    resolved_at TIMESTAMP NULL,
    resolution_notes TEXT,
    suppressed_until TIMESTAMP NULL,
    suppression_reason TEXT,
    notification_count INT DEFAULT 0,
    last_notification_sent TIMESTAMP NULL,
    correlation_group VARCHAR(255),
    parent_alert_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (rule_id) REFERENCES alert_rules(rule_id) ON DELETE CASCADE,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_rule_id (rule_id),
    INDEX idx_severity (severity),
    INDEX idx_status (status),
    INDEX idx_metric_name (metric_name),
    INDEX idx_created_at (created_at),
    INDEX idx_correlation_group (correlation_group)
);

-- Alert timeline - tracks alert state changes
CREATE TABLE alert_timeline (
    timeline_id VARCHAR(255) PRIMARY KEY,
    alert_id VARCHAR(255) NOT NULL,
    event_type ENUM('created', 'acknowledged', 'escalated', 'resolved', 'suppressed', 'reopened', 'updated') NOT NULL,
    previous_status ENUM('open', 'acknowledged', 'investigating', 'resolved', 'suppressed'),
    new_status ENUM('open', 'acknowledged', 'investigating', 'resolved', 'suppressed'),
    performed_by VARCHAR(255),
    event_description TEXT,
    automated BOOLEAN DEFAULT FALSE,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE,
    INDEX idx_alert_id (alert_id),
    INDEX idx_event_type (event_type),
    INDEX idx_event_timestamp (event_timestamp)
);

-- Health checks configuration
CREATE TABLE health_checks (
    check_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    service_name VARCHAR(255) NOT NULL,
    check_name VARCHAR(255) NOT NULL,
    check_type ENUM('http', 'tcp', 'database', 'redis', 'custom', 'dependency') NOT NULL,
    endpoint_url VARCHAR(500),
    expected_response TEXT,
    expected_status_code INT DEFAULT 200,
    timeout_seconds INT DEFAULT 30,
    check_interval_seconds INT DEFAULT 60,
    consecutive_failures_threshold INT DEFAULT 3,
    enabled BOOLEAN DEFAULT TRUE,
    critical_check BOOLEAN DEFAULT FALSE,
    check_configuration JSON,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_service_name (service_name),
    INDEX idx_check_type (check_type),
    INDEX idx_enabled (enabled),
    INDEX idx_critical_check (critical_check)
);

-- Health check results
CREATE TABLE health_check_results (
    result_id VARCHAR(255) PRIMARY KEY,
    check_id VARCHAR(255) NOT NULL,
    check_timestamp TIMESTAMP NOT NULL,
    status ENUM('healthy', 'degraded', 'unhealthy', 'timeout', 'error') NOT NULL,
    response_time_ms DECIMAL(10,3) DEFAULT 0.000,
    status_code INT,
    response_body TEXT,
    error_message TEXT,
    check_duration_ms DECIMAL(10,3) DEFAULT 0.000,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (check_id) REFERENCES health_checks(check_id) ON DELETE CASCADE,
    INDEX idx_check_id (check_id),
    INDEX idx_check_timestamp (check_timestamp),
    INDEX idx_status (status),
    INDEX idx_response_time_ms (response_time_ms)
);

-- Service health status summary
CREATE TABLE service_health_status (
    status_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    service_name VARCHAR(255) NOT NULL,
    overall_status ENUM('healthy', 'degraded', 'unhealthy', 'unknown') NOT NULL,
    health_score DECIMAL(5,2) DEFAULT 100.00,
    total_checks INT DEFAULT 0,
    healthy_checks INT DEFAULT 0,
    degraded_checks INT DEFAULT 0,
    unhealthy_checks INT DEFAULT 0,
    average_response_time_ms DECIMAL(10,3) DEFAULT 0.000,
    availability_percentage DECIMAL(5,3) DEFAULT 100.000,
    last_incident TIMESTAMP NULL,
    incident_count_24h INT DEFAULT 0,
    uptime_percentage_24h DECIMAL(5,3) DEFAULT 100.000,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_service_name (service_name),
    INDEX idx_overall_status (overall_status),
    INDEX idx_health_score (health_score),
    INDEX idx_last_updated (last_updated)
);

-- Escalation policies
CREATE TABLE escalation_policies (
    policy_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    policy_name VARCHAR(255) NOT NULL,
    description TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_enabled (enabled)
);

-- Escalation policy steps
CREATE TABLE escalation_policy_steps (
    step_id VARCHAR(255) PRIMARY KEY,
    policy_id VARCHAR(255) NOT NULL,
    step_order INT NOT NULL,
    step_name VARCHAR(255) NOT NULL,
    delay_minutes INT DEFAULT 0,
    notification_channels JSON NOT NULL,
    escalation_targets JSON NOT NULL, -- users, groups, external systems
    repeat_notifications BOOLEAN DEFAULT FALSE,
    repeat_interval_minutes INT DEFAULT 30,
    max_repeats INT DEFAULT 3,
    conditions JSON, -- conditions for this step to trigger
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (policy_id) REFERENCES escalation_policies(policy_id) ON DELETE CASCADE,
    INDEX idx_policy_id (policy_id),
    INDEX idx_step_order (step_order)
);

-- Notification channels configuration
CREATE TABLE notification_channels (
    channel_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    channel_name VARCHAR(255) NOT NULL,
    channel_type ENUM('email', 'slack', 'sms', 'webhook', 'pagerduty', 'teams', 'discord') NOT NULL,
    channel_config JSON NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    rate_limit_per_hour INT DEFAULT 100,
    current_hour_count INT DEFAULT 0,
    last_reset_hour TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    test_status ENUM('not_tested', 'passed', 'failed') DEFAULT 'not_tested',
    last_test_date TIMESTAMP NULL,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_channel_type (channel_type),
    INDEX idx_enabled (enabled)
);

-- Notification history
CREATE TABLE notification_history (
    notification_id VARCHAR(255) PRIMARY KEY,
    alert_id VARCHAR(255) NOT NULL,
    channel_id VARCHAR(255) NOT NULL,
    notification_type ENUM('alert', 'escalation', 'resolution', 'test') NOT NULL,
    recipient VARCHAR(255) NOT NULL,
    subject VARCHAR(500),
    message TEXT,
    delivery_status ENUM('pending', 'sent', 'delivered', 'failed', 'bounced') DEFAULT 'pending',
    delivery_timestamp TIMESTAMP NULL,
    failure_reason TEXT,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    external_message_id VARCHAR(255),
    cost DECIMAL(8,4) DEFAULT 0.0000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE,
    FOREIGN KEY (channel_id) REFERENCES notification_channels(channel_id) ON DELETE CASCADE,
    INDEX idx_alert_id (alert_id),
    INDEX idx_channel_id (channel_id),
    INDEX idx_delivery_status (delivery_status),
    INDEX idx_created_at (created_at)
);

-- Anomaly detection results
CREATE TABLE monitoring_anomalies (
    anomaly_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    anomaly_type ENUM('statistical', 'ml_based', 'pattern_based', 'threshold_based') NOT NULL,
    anomaly_score DECIMAL(5,4) NOT NULL,
    confidence_level DECIMAL(3,2) DEFAULT 0.50,
    baseline_period_start TIMESTAMP,
    baseline_period_end TIMESTAMP,
    anomaly_start_time TIMESTAMP NOT NULL,
    anomaly_end_time TIMESTAMP,
    affected_metrics JSON,
    anomaly_description TEXT,
    potential_causes JSON,
    recommended_actions JSON,
    investigation_status ENUM('detected', 'investigating', 'explained', 'false_positive') DEFAULT 'detected',
    analyst_assigned VARCHAR(255),
    investigation_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_metric_name (metric_name),
    INDEX idx_anomaly_type (anomaly_type),
    INDEX idx_anomaly_score (anomaly_score),
    INDEX idx_anomaly_start_time (anomaly_start_time),
    INDEX idx_investigation_status (investigation_status)
);

-- Predictive analysis results
CREATE TABLE predictive_analysis (
    prediction_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    prediction_type ENUM('performance_degradation', 'capacity_exhaustion', 'system_failure', 'security_incident', 'cost_overrun') NOT NULL,
    target_metric VARCHAR(255) NOT NULL,
    prediction_model VARCHAR(100) NOT NULL,
    prediction_confidence DECIMAL(3,2) DEFAULT 0.50,
    predicted_value DECIMAL(20,6),
    predicted_timestamp TIMESTAMP NOT NULL,
    prediction_horizon_hours INT NOT NULL,
    risk_level ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    impact_assessment TEXT,
    recommended_actions JSON,
    prevention_strategies JSON,
    model_accuracy DECIMAL(5,2) DEFAULT 0.00,
    feature_importance JSON,
    prediction_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_validated BOOLEAN DEFAULT FALSE,
    actual_outcome VARCHAR(255),
    validation_date TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_prediction_type (prediction_type),
    INDEX idx_target_metric (target_metric),
    INDEX idx_predicted_timestamp (predicted_timestamp),
    INDEX idx_risk_level (risk_level),
    INDEX idx_prediction_confidence (prediction_confidence)
);

-- SLA definitions and tracking
CREATE TABLE sla_definitions (
    sla_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    sla_name VARCHAR(255) NOT NULL,
    service_name VARCHAR(255) NOT NULL,
    sla_type ENUM('availability', 'performance', 'response_time', 'error_rate', 'throughput') NOT NULL,
    target_value DECIMAL(10,4) NOT NULL,
    measurement_unit VARCHAR(50) NOT NULL,
    measurement_period ENUM('hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'annually') NOT NULL,
    penalty_threshold DECIMAL(10,4),
    penalty_amount DECIMAL(12,2) DEFAULT 0.00,
    grace_period_hours INT DEFAULT 0,
    exclusions JSON, -- maintenance windows, etc.
    active BOOLEAN DEFAULT TRUE,
    start_date DATE NOT NULL,
    end_date DATE,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_service_name (service_name),
    INDEX idx_sla_type (sla_type),
    INDEX idx_active (active),
    INDEX idx_start_date (start_date)
);

-- SLA performance tracking
CREATE TABLE sla_performance (
    performance_id VARCHAR(255) PRIMARY KEY,
    sla_id VARCHAR(255) NOT NULL,
    measurement_period_start TIMESTAMP NOT NULL,
    measurement_period_end TIMESTAMP NOT NULL,
    actual_value DECIMAL(10,4) NOT NULL,
    target_value DECIMAL(10,4) NOT NULL,
    performance_percentage DECIMAL(5,2) NOT NULL,
    sla_met BOOLEAN NOT NULL,
    breach_duration_minutes INT DEFAULT 0,
    penalty_incurred DECIMAL(12,2) DEFAULT 0.00,
    contributing_incidents JSON,
    exclusion_time_minutes INT DEFAULT 0,
    notes TEXT,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sla_id) REFERENCES sla_definitions(sla_id) ON DELETE CASCADE,
    INDEX idx_sla_id (sla_id),
    INDEX idx_measurement_period_start (measurement_period_start),
    INDEX idx_sla_met (sla_met),
    INDEX idx_performance_percentage (performance_percentage)
);

-- Monitoring dashboards configuration
CREATE TABLE monitoring_dashboards (
    dashboard_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    dashboard_name VARCHAR(255) NOT NULL,
    dashboard_type ENUM('overview', 'system', 'application', 'business', 'security', 'custom') NOT NULL,
    layout_config JSON NOT NULL,
    widget_configs JSON NOT NULL,
    refresh_interval_seconds INT DEFAULT 30,
    auto_refresh BOOLEAN DEFAULT TRUE,
    public_access BOOLEAN DEFAULT FALSE,
    access_permissions JSON,
    theme VARCHAR(50) DEFAULT 'default',
    created_by VARCHAR(255) NOT NULL,
    last_accessed TIMESTAMP NULL,
    access_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_dashboard_type (dashboard_type),
    INDEX idx_created_by (created_by),
    INDEX idx_last_accessed (last_accessed)
);

-- Dashboard widgets
CREATE TABLE dashboard_widgets (
    widget_id VARCHAR(255) PRIMARY KEY,
    dashboard_id VARCHAR(255) NOT NULL,
    widget_name VARCHAR(255) NOT NULL,
    widget_type ENUM('metric_chart', 'alert_list', 'health_status', 'sla_summary', 'custom_query', 'text') NOT NULL,
    position_x INT DEFAULT 0,
    position_y INT DEFAULT 0,
    width INT DEFAULT 4,
    height INT DEFAULT 3,
    widget_config JSON NOT NULL,
    data_source VARCHAR(255),
    refresh_interval_seconds INT DEFAULT 60,
    visible BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (dashboard_id) REFERENCES monitoring_dashboards(dashboard_id) ON DELETE CASCADE,
    INDEX idx_dashboard_id (dashboard_id),
    INDEX idx_widget_type (widget_type),
    INDEX idx_visible (visible)
);

-- Incident management
CREATE TABLE monitoring_incidents (
    incident_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    incident_title VARCHAR(500) NOT NULL,
    incident_description TEXT,
    severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    priority ENUM('p1', 'p2', 'p3', 'p4') NOT NULL,
    status ENUM('open', 'investigating', 'identified', 'monitoring', 'resolved', 'closed') DEFAULT 'open',
    affected_services JSON,
    root_cause TEXT,
    resolution_summary TEXT,
    lessons_learned TEXT,
    incident_commander VARCHAR(255),
    response_team JSON,
    customer_impact ENUM('none', 'minimal', 'moderate', 'significant', 'severe') DEFAULT 'none',
    business_impact_usd DECIMAL(12,2) DEFAULT 0.00,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP NULL,
    resolved_at TIMESTAMP NULL,
    closed_at TIMESTAMP NULL,
    mttr_minutes INT DEFAULT 0, -- Mean Time To Resolution
    post_mortem_required BOOLEAN DEFAULT FALSE,
    post_mortem_completed BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_severity (severity),
    INDEX idx_status (status),
    INDEX idx_detected_at (detected_at),
    INDEX idx_incident_commander (incident_commander)
);

-- Incident alert relationships
CREATE TABLE incident_alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    incident_id VARCHAR(255) NOT NULL,
    alert_id VARCHAR(255) NOT NULL,
    relationship_type ENUM('root_cause', 'contributing_factor', 'symptom', 'related') NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    added_by VARCHAR(255),
    FOREIGN KEY (incident_id) REFERENCES monitoring_incidents(incident_id) ON DELETE CASCADE,
    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE,
    UNIQUE KEY unique_incident_alert (incident_id, alert_id),
    INDEX idx_incident_id (incident_id),
    INDEX idx_alert_id (alert_id)
);

-- Create views for monitoring and alerting

-- Real-time monitoring overview
CREATE VIEW monitoring_overview AS
SELECT
    ms.tenant_id,
    ms.session_id,
    ms.session_name,
    ms.status as monitoring_status,
    COUNT(DISTINCT mm.metric_id) as total_metrics_collected,
    COUNT(DISTINCT ar.rule_id) as total_alert_rules,
    COUNT(DISTINCT a.alert_id) as total_alerts,
    COUNT(CASE WHEN a.status = 'open' THEN 1 END) as open_alerts,
    COUNT(CASE WHEN a.severity = 'critical' THEN 1 END) as critical_alerts,
    COUNT(DISTINCT hc.check_id) as total_health_checks,
    COUNT(CASE WHEN shs.overall_status = 'healthy' THEN 1 END) as healthy_services,
    COUNT(CASE WHEN shs.overall_status = 'unhealthy' THEN 1 END) as unhealthy_services,
    AVG(shs.health_score) as avg_health_score,
    ms.started_at,
    ms.last_activity
FROM monitoring_sessions ms
LEFT JOIN monitoring_metrics mm ON ms.session_id = mm.session_id AND mm.collection_timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
LEFT JOIN alert_rules ar ON ms.tenant_id = ar.tenant_id AND ar.enabled = TRUE
LEFT JOIN alerts a ON ar.rule_id = a.rule_id AND a.created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
LEFT JOIN health_checks hc ON ms.tenant_id = hc.tenant_id AND hc.enabled = TRUE
LEFT JOIN service_health_status shs ON ms.tenant_id = shs.tenant_id
WHERE ms.status = 'active'
GROUP BY ms.session_id;

-- Alert summary view
CREATE VIEW alert_summary AS
SELECT
    a.tenant_id,
    a.severity,
    a.status,
    COUNT(*) as alert_count,
    AVG(TIMESTAMPDIFF(MINUTE, a.created_at, COALESCE(a.resolved_at, NOW()))) as avg_resolution_time_minutes,
    COUNT(CASE WHEN a.acknowledged_at IS NOT NULL THEN 1 END) as acknowledged_count,
    COUNT(CASE WHEN a.resolved_at IS NOT NULL THEN 1 END) as resolved_count,
    MIN(a.created_at) as oldest_alert,
    MAX(a.created_at) as newest_alert
FROM alerts a
WHERE a.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY a.tenant_id, a.severity, a.status;

-- Service health overview
CREATE VIEW service_health_overview AS
SELECT
    shs.tenant_id,
    shs.service_name,
    shs.overall_status,
    shs.health_score,
    shs.availability_percentage,
    shs.average_response_time_ms,
    shs.incident_count_24h,
    COUNT(DISTINCT hc.check_id) as total_checks,
    COUNT(CASE WHEN hcr.status = 'healthy' THEN 1 END) as healthy_checks,
    COUNT(CASE WHEN hcr.status = 'unhealthy' THEN 1 END) as unhealthy_checks,
    AVG(hcr.response_time_ms) as avg_check_response_time,
    MAX(hcr.check_timestamp) as last_check_time
FROM service_health_status shs
LEFT JOIN health_checks hc ON shs.tenant_id = hc.tenant_id AND shs.service_name = hc.service_name
LEFT JOIN health_check_results hcr ON hc.check_id = hcr.check_id AND hcr.check_timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
GROUP BY shs.tenant_id, shs.service_name;

-- SLA performance summary
CREATE VIEW sla_performance_summary AS
SELECT
    sd.tenant_id,
    sd.service_name,
    sd.sla_type,
    sd.target_value,
    sp.actual_value,
    sp.performance_percentage,
    sp.sla_met,
    COUNT(*) as measurement_periods,
    COUNT(CASE WHEN sp.sla_met = TRUE THEN 1 END) as periods_met,
    AVG(sp.performance_percentage) as avg_performance,
    SUM(sp.penalty_incurred) as total_penalties,
    MIN(sp.measurement_period_start) as tracking_start,
    MAX(sp.measurement_period_end) as tracking_end
FROM sla_definitions sd
LEFT JOIN sla_performance sp ON sd.sla_id = sp.sla_id
WHERE sd.active = TRUE
AND sp.measurement_period_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY sd.tenant_id, sd.service_name, sd.sla_type;

-- Monitoring performance metrics
CREATE VIEW monitoring_performance_metrics AS
SELECT
    mm.tenant_id,
    mm.metric_type,
    mm.source_system,
    COUNT(*) as metrics_collected,
    AVG(mm.metric_value) as avg_metric_value,
    MIN(mm.metric_value) as min_metric_value,
    MAX(mm.metric_value) as max_metric_value,
    STDDEV(mm.metric_value) as metric_stddev,
    COUNT(CASE WHEN mm.anomaly_score > 0.5 THEN 1 END) as anomalous_metrics,
    DATE(mm.collection_timestamp) as collection_date
FROM monitoring_metrics mm
WHERE mm.collection_timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY mm.tenant_id, mm.metric_type, mm.source_system, DATE(mm.collection_timestamp);
