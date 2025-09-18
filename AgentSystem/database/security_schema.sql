
-- Advanced Security and Compliance Database Schema
-- Stores security events, compliance audits, vulnerabilities, and policies

-- Security events table - main security event tracking
CREATE TABLE security_events (
    event_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    event_type ENUM('login_attempt', 'failed_login', 'suspicious_activity', 'data_access', 'api_abuse',
                   'privilege_escalation', 'data_breach', 'malware_detection', 'vulnerability_found', 'compliance_violation') NOT NULL,
    threat_level ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    source_ip VARCHAR(45) NOT NULL,
    user_id VARCHAR(255),
    user_agent TEXT,
    resource_accessed VARCHAR(500),
    action_attempted VARCHAR(255),
    risk_score DECIMAL(3,2) DEFAULT 0.00 CHECK (risk_score BETWEEN 0.00 AND 1.00),
    investigation_status ENUM('open', 'investigating', 'resolved', 'false_positive', 'auto_resolved') DEFAULT 'open',
    severity_score INT DEFAULT 0 CHECK (severity_score BETWEEN 0 AND 100),
    automated_response_executed BOOLEAN DEFAULT FALSE,
    manual_review_required BOOLEAN DEFAULT FALSE,
    incident_id VARCHAR(255),
    geolocation JSON,
    device_fingerprint VARCHAR(255),
    session_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP NULL,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_event_type (event_type),
    INDEX idx_threat_level (threat_level),
    INDEX idx_source_ip (source_ip),
    INDEX idx_user_id (user_id),
    INDEX idx_risk_score (risk_score),
    INDEX idx_created_at (created_at),
    INDEX idx_investigation_status (investigation_status)
);

-- Security event details - additional context for events
CREATE TABLE security_event_details (
    id INT AUTO_INCREMENT PRIMARY KEY,
    event_id VARCHAR(255) NOT NULL,
    detail_type ENUM('request_headers', 'request_body', 'response_data', 'system_logs', 'network_traffic', 'file_access') NOT NULL,
    detail_key VARCHAR(255) NOT NULL,
    detail_value TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,
    encrypted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES security_events(event_id) ON DELETE CASCADE,
    INDEX idx_event_id (event_id),
    INDEX idx_detail_type (detail_type)
);

-- Automated security responses
CREATE TABLE security_responses (
    response_id VARCHAR(255) PRIMARY KEY,
    event_id VARCHAR(255) NOT NULL,
    response_type ENUM('block_ip', 'disable_user', 'require_mfa', 'alert_admin', 'quarantine_file',
                      'reset_password', 'audit_log', 'escalate_incident') NOT NULL,
    response_status ENUM('pending', 'executed', 'failed', 'cancelled') DEFAULT 'pending',
    execution_time TIMESTAMP NULL,
    response_details JSON,
    effectiveness_score DECIMAL(3,2) DEFAULT 0.50,
    rollback_available BOOLEAN DEFAULT FALSE,
    rollback_executed BOOLEAN DEFAULT FALSE,
    created_by ENUM('system', 'admin', 'ai_engine') DEFAULT 'system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES security_events(event_id) ON DELETE CASCADE,
    INDEX idx_event_id (event_id),
    INDEX idx_response_type (response_type),
    INDEX idx_response_status (response_status)
);

-- Threat intelligence data
CREATE TABLE threat_intelligence (
    intel_id VARCHAR(255) PRIMARY KEY,
    intel_type ENUM('ip_reputation', 'domain_reputation', 'malware_signature', 'attack_pattern', 'ioc') NOT NULL,
    indicator_value VARCHAR(500) NOT NULL,
    threat_category ENUM('malware', 'phishing', 'botnet', 'tor', 'proxy', 'scanner', 'spam', 'suspicious') NOT NULL,
    confidence_level ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    severity ENUM('info', 'low', 'medium', 'high', 'critical') DEFAULT 'medium',
    source VARCHAR(255) NOT NULL,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expiry_date TIMESTAMP NULL,
    description TEXT,
    additional_context JSON,
    false_positive_rate DECIMAL(3,2) DEFAULT 0.05,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_intel_type (intel_type),
    INDEX idx_indicator_value (indicator_value),
    INDEX idx_threat_category (threat_category),
    INDEX idx_confidence_level (confidence_level),
    INDEX idx_active (active),
    INDEX idx_expiry_date (expiry_date)
);

-- Compliance frameworks and rules
CREATE TABLE compliance_frameworks (
    framework_id VARCHAR(255) PRIMARY KEY,
    framework_name VARCHAR(255) NOT NULL,
    framework_code ENUM('soc2', 'gdpr', 'hipaa', 'pci_dss', 'iso27001', 'nist', 'ccpa', 'sox') NOT NULL,
    version VARCHAR(50),
    description TEXT,
    authority VARCHAR(255),
    geographic_scope VARCHAR(255),
    industry_scope VARCHAR(255),
    mandatory BOOLEAN DEFAULT FALSE,
    effective_date DATE,
    review_frequency_months INT DEFAULT 12,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_framework_code (framework_code),
    INDEX idx_mandatory (mandatory)
);

-- Compliance controls and requirements
CREATE TABLE compliance_controls (
    control_id VARCHAR(255) PRIMARY KEY,
    framework_id VARCHAR(255) NOT NULL,
    control_number VARCHAR(100) NOT NULL,
    control_name VARCHAR(500) NOT NULL,
    control_description TEXT,
    control_type ENUM('preventive', 'detective', 'corrective', 'compensating') NOT NULL,
    control_category ENUM('access_control', 'data_protection', 'network_security', 'incident_response',
                         'business_continuity', 'risk_management', 'audit_logging', 'encryption') NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    automated_check BOOLEAN DEFAULT FALSE,
    check_frequency ENUM('continuous', 'daily', 'weekly', 'monthly', 'quarterly', 'annually') DEFAULT 'monthly',
    implementation_guidance TEXT,
    testing_procedures TEXT,
    evidence_requirements TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (framework_id) REFERENCES compliance_frameworks(framework_id) ON DELETE CASCADE,
    INDEX idx_framework_id (framework_id),
    INDEX idx_control_type (control_type),
    INDEX idx_control_category (control_category),
    INDEX idx_automated_check (automated_check)
);

-- Compliance assessments and audits
CREATE TABLE compliance_assessments (
    assessment_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    framework_id VARCHAR(255) NOT NULL,
    assessment_type ENUM('self_assessment', 'internal_audit', 'external_audit', 'continuous_monitoring') NOT NULL,
    assessment_scope TEXT,
    assessor_name VARCHAR(255),
    assessor_organization VARCHAR(255),
    start_date DATE NOT NULL,
    end_date DATE,
    overall_score DECIMAL(5,2) DEFAULT 0.00,
    compliance_percentage DECIMAL(5,2) DEFAULT 0.00,
    total_controls INT DEFAULT 0,
    compliant_controls INT DEFAULT 0,
    non_compliant_controls INT DEFAULT 0,
    partially_compliant_controls INT DEFAULT 0,
    not_applicable_controls INT DEFAULT 0,
    critical_findings INT DEFAULT 0,
    high_findings INT DEFAULT 0,
    medium_findings INT DEFAULT 0,
    low_findings INT DEFAULT 0,
    status ENUM('planning', 'in_progress', 'completed', 'cancelled') DEFAULT 'planning',
    next_assessment_date DATE,
    certification_status ENUM('not_certified', 'certified', 'expired', 'suspended') DEFAULT 'not_certified',
    certification_expiry DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (framework_id) REFERENCES compliance_frameworks(framework_id) ON DELETE RESTRICT,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_framework_id (framework_id),
    INDEX idx_assessment_type (assessment_type),
    INDEX idx_status (status),
    INDEX idx_start_date (start_date)
);

-- Compliance control assessments
CREATE TABLE compliance_control_assessments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    assessment_id VARCHAR(255) NOT NULL,
    control_id VARCHAR(255) NOT NULL,
    compliance_status ENUM('compliant', 'non_compliant', 'partially_compliant', 'not_applicable') NOT NULL,
    implementation_status ENUM('not_implemented', 'planned', 'in_progress', 'implemented', 'needs_improvement') NOT NULL,
    effectiveness_rating ENUM('ineffective', 'partially_effective', 'largely_effective', 'effective') DEFAULT 'effective',
    test_results TEXT,
    evidence_collected TEXT,
    findings TEXT,
    recommendations TEXT,
    remediation_plan TEXT,
    remediation_due_date DATE,
    remediation_owner VARCHAR(255),
    remediation_status ENUM('not_started', 'in_progress', 'completed', 'overdue') DEFAULT 'not_started',
    assessor_notes TEXT,
    last_tested DATE,
    next_test_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (assessment_id) REFERENCES compliance_assessments(assessment_id) ON DELETE CASCADE,
    FOREIGN KEY (control_id) REFERENCES compliance_controls(control_id) ON DELETE CASCADE,
    INDEX idx_assessment_id (assessment_id),
    INDEX idx_control_id (control_id),
    INDEX idx_compliance_status (compliance_status),
    INDEX idx_remediation_status (remediation_status)
);

-- Vulnerability assessments
CREATE TABLE vulnerability_assessments (
    assessment_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    scan_name VARCHAR(255) NOT NULL,
    scan_type ENUM('network', 'web_application', 'database', 'configuration', 'comprehensive') NOT NULL,
    scan_scope TEXT,
    target_systems JSON,
    scan_engine VARCHAR(255),
    scan_version VARCHAR(100),
    scan_start_time TIMESTAMP NOT NULL,
    scan_end_time TIMESTAMP,
    scan_duration_seconds INT DEFAULT 0,
    scan_status ENUM('queued', 'running', 'completed', 'failed', 'cancelled') DEFAULT 'queued',
    total_vulnerabilities INT DEFAULT 0,
    critical_count INT DEFAULT 0,
    high_count INT DEFAULT 0,
    medium_count INT DEFAULT 0,
    low_count INT DEFAULT 0,
    info_count INT DEFAULT 0,
    false_positive_count INT DEFAULT 0,
    risk_score DECIMAL(5,2) DEFAULT 0.00,
    baseline_scan BOOLEAN DEFAULT FALSE,
    previous_assessment_id VARCHAR(255),
    next_scan_date TIMESTAMP,
    scan_configuration JSON,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_scan_type (scan_type),
    INDEX idx_scan_status (scan_status),
    INDEX idx_scan_start_time (scan_start_time),
    INDEX idx_risk_score (risk_score)
);

-- Vulnerability findings
CREATE TABLE vulnerability_findings (
    finding_id VARCHAR(255) PRIMARY KEY,
    assessment_id VARCHAR(255) NOT NULL,
    vulnerability_id VARCHAR(255),
    cve_id VARCHAR(50),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    severity ENUM('info', 'low', 'medium', 'high', 'critical') NOT NULL,
    cvss_score DECIMAL(3,1) DEFAULT 0.0,
    cvss_vector VARCHAR(255),
    affected_asset VARCHAR(255),
    asset_type ENUM('server', 'workstation', 'network_device', 'web_application', 'database', 'mobile_device') NOT NULL,
    port_number INT,
    protocol VARCHAR(20),
    service_name VARCHAR(100),
    vulnerability_category ENUM('injection', 'authentication', 'authorization', 'cryptography', 'configuration',
                               'input_validation', 'session_management', 'information_disclosure') NOT NULL,
    exploit_available BOOLEAN DEFAULT FALSE,
    exploit_complexity ENUM('low', 'medium', 'high') DEFAULT 'medium',
    attack_vector ENUM('network', 'adjacent', 'local', 'physical') DEFAULT 'network',
    remediation_effort ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    business_impact ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    false_positive BOOLEAN DEFAULT FALSE,
    status ENUM('open', 'confirmed', 'remediated', 'accepted_risk', 'false_positive') DEFAULT 'open',
    first_discovered TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    remediation_deadline DATE,
    remediation_owner VARCHAR(255),
    remediation_notes TEXT,
    verification_status ENUM('not_verified', 'verified', 'retest_required') DEFAULT 'not_verified',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (assessment_id) REFERENCES vulnerability_assessments(assessment_id) ON DELETE CASCADE,
    INDEX idx_assessment_id (assessment_id),
    INDEX idx_severity (severity),
    INDEX idx_status (status),
    INDEX idx_cve_id (cve_id),
    INDEX idx_affected_asset (affected_asset),
    INDEX idx_remediation_deadline (remediation_deadline)
);

-- Security policies
CREATE TABLE security_policies (
    policy_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    policy_name VARCHAR(255) NOT NULL,
    policy_type ENUM('access_control', 'data_protection', 'network_security', 'incident_response',
                    'acceptable_use', 'password_policy', 'encryption_policy', 'backup_policy') NOT NULL,
    policy_version VARCHAR(50) DEFAULT '1.0',
    policy_description TEXT,
    policy_scope TEXT,
    enforcement_level ENUM('advisory', 'enforced', 'strict') DEFAULT 'enforced',
    auto_remediation BOOLEAN DEFAULT TRUE,
    policy_owner VARCHAR(255) NOT NULL,
    approver VARCHAR(255),
    approval_date TIMESTAMP NULL,
    effective_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    review_date TIMESTAMP,
    expiry_date TIMESTAMP,
    status ENUM('draft', 'pending_approval', 'active', 'suspended', 'expired', 'superseded') DEFAULT 'draft',
    compliance_frameworks JSON,
    exceptions_allowed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_policy_type (policy_type),
    INDEX idx_status (status),
    INDEX idx_effective_date (effective_date),
    INDEX idx_review_date (review_date)
);

-- Security policy rules
CREATE TABLE security_policy_rules (
    rule_id VARCHAR(255) PRIMARY KEY,
    policy_id VARCHAR(255) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,
    rule_description TEXT,
    rule_type ENUM('allow', 'deny', 'require', 'restrict', 'monitor') NOT NULL,
    rule_conditions JSON NOT NULL,
    rule_actions JSON,
    priority INT DEFAULT 100,
    enabled BOOLEAN DEFAULT TRUE,
    rule_scope ENUM('global', 'user_group', 'resource_type', 'specific_resource') DEFAULT 'global',
    target_entities JSON,
    exceptions JSON,
    monitoring_enabled BOOLEAN DEFAULT TRUE,
    violation_severity ENUM('info', 'low', 'medium', 'high', 'critical') DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (policy_id) REFERENCES security_policies(policy_id) ON DELETE CASCADE,
    INDEX idx_policy_id (policy_id),
    INDEX idx_rule_type (rule_type),
    INDEX idx_priority (priority),
    INDEX idx_enabled (enabled)
);

-- Security policy violations
CREATE TABLE security_policy_violations (
    violation_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    policy_id VARCHAR(255) NOT NULL,
    rule_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    resource_id VARCHAR(255),
    violation_type ENUM('access_denied', 'unauthorized_access', 'policy_breach', 'configuration_drift') NOT NULL,
    severity ENUM('info', 'low', 'medium', 'high', 'critical') NOT NULL,
    violation_details JSON,
    source_ip VARCHAR(45),
    user_agent TEXT,
    detected_by ENUM('policy_engine', 'monitoring_system', 'manual_review') DEFAULT 'policy_engine',
    auto_remediated BOOLEAN DEFAULT FALSE,
    remediation_actions JSON,
    status ENUM('open', 'investigating', 'resolved', 'accepted', 'false_positive') DEFAULT 'open',
    assigned_to VARCHAR(255),
    resolution_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP NULL,
    FOREIGN KEY (policy_id) REFERENCES security_policies(policy_id) ON DELETE CASCADE,
    FOREIGN KEY (rule_id) REFERENCES security_policy_rules(rule_id) ON DELETE CASCADE,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_policy_id (policy_id),
    INDEX idx_rule_id (rule_id),
    INDEX idx_user_id (user_id),
    INDEX idx_severity (severity),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Security incidents
CREATE TABLE security_incidents (
    incident_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    incident_title VARCHAR(500) NOT NULL,
    incident_description TEXT,
    incident_type ENUM('data_breach', 'malware_infection', 'unauthorized_access', 'denial_of_service',
                      'phishing_attack', 'insider_threat', 'system_compromise', 'policy_violation') NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    priority ENUM('p1', 'p2', 'p3', 'p4') NOT NULL,
    status ENUM('new', 'assigned', 'investigating', 'containment', 'eradication', 'recovery', 'closed') DEFAULT 'new',
    source_event_id VARCHAR(255),
    affected_systems JSON,
    affected_users JSON,
    affected_data_types JSON,
    estimated_impact TEXT,
    business_impact_score DECIMAL(3,2) DEFAULT 0.00,
    incident_commander VARCHAR(255),
    assigned_team VARCHAR(255),
    escalation_level INT DEFAULT 1,
    regulatory_notification_required BOOLEAN DEFAULT FALSE,
    customer_notification_required BOOLEAN DEFAULT FALSE,
    media_attention BOOLEAN DEFAULT FALSE,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP NULL,
    contained_at TIMESTAMP NULL,
    resolved_at TIMESTAMP NULL,
    lessons_learned TEXT,
    post_incident_actions JSON,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (source_event_id) REFERENCES security_events(event_id) ON DELETE SET NULL,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_incident_type (incident_type),
    INDEX idx_severity (severity),
    INDEX idx_status (status),
    INDEX idx_detected_at (detected_at),
    INDEX idx_incident_commander (incident_commander)
);

-- Security incident timeline
CREATE TABLE security_incident_timeline (
    timeline_id VARCHAR(255) PRIMARY KEY,
    incident_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action_type ENUM('detection', 'analysis', 'containment', 'communication', 'remediation', 'recovery', 'documentation') NOT NULL,
    action_description TEXT NOT NULL,
    performed_by VARCHAR(255) NOT NULL,
    automated BOOLEAN DEFAULT FALSE,
    evidence_collected TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (incident_id) REFERENCES security_incidents(incident_id) ON DELETE CASCADE,
    INDEX idx_incident_id (incident_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_action_type (action_type)
);

-- Anomaly detection results
CREATE TABLE anomaly_detections (
    anomaly_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    detection_model VARCHAR(255) NOT NULL,
    anomaly_type ENUM('behavioral', 'statistical', 'pattern_based', 'ml_based') NOT NULL,
    anomaly_category ENUM('user_behavior', 'network_traffic', 'system_performance', 'data_access', 'api_usage') NOT NULL,
    anomaly_score DECIMAL(5,4) NOT NULL,
    confidence_level DECIMAL(3,2) DEFAULT 0.50,
    baseline_period_start TIMESTAMP,
    baseline_period_end TIMESTAMP,
    detection_period_start TIMESTAMP,
    detection_period_end TIMESTAMP,
    affected_entity VARCHAR(255),
    entity_type ENUM('user', 'system', 'application', 'network', 'data') NOT NULL,
    anomaly_details JSON,
    features_analyzed JSON,
    deviation_metrics JSON,
    false_positive_probability DECIMAL(3,2) DEFAULT 0.10,
    investigation_priority ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    status ENUM('detected', 'investigating', 'confirmed', 'false_positive', 'resolved') DEFAULT 'detected',
    analyst_assigned VARCHAR(255),
    investigation_notes TEXT,
    resolution_action TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_anomaly_type (anomaly_type),
    INDEX idx_anomaly_category (anomaly_category),
    INDEX idx_anomaly_score (anomaly_score),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Security metrics and KPIs
CREATE TABLE security_metrics (
    metric_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_category ENUM('threat_detection', 'incident_response', 'compliance', 'vulnerability_management',
                        'access_control', 'data_protection', 'security_awareness') NOT NULL,
    metric_type ENUM('count', 'percentage', 'ratio', 'time_duration', 'score') NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    target_value DECIMAL(15,4),
    threshold_warning DECIMAL(15,4),
    threshold_critical DECIMAL(15,4),
    measurement_unit VARCHAR(50),
    measurement_period ENUM('real_time', 'hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'annually') NOT NULL,
    measurement_date DATE NOT NULL,
    trend ENUM('improving', 'stable', 'declining', 'unknown') DEFAULT 'unknown',
    benchmark_comparison DECIMAL(5,2),
    data_source VARCHAR(255),
    calculation_method TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_metric_category (metric_category),
    INDEX idx_measurement_date (measurement_date),
    INDEX idx_metric_name (metric_name)
);

-- Security audit logs
CREATE TABLE security_audit_logs (
    log_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    log_source ENUM('application', 'system', 'network', 'database', 'security_tool') NOT NULL,
    event_category ENUM('authentication', 'authorization', 'data_access', 'configuration_change',
                       'system_activity', 'network_activity', 'file_activity') NOT NULL,
    event_action VARCHAR(255) NOT NULL,
    event_result ENUM('success', 'failure', 'error', 'warning') NOT NULL,
    user_id VARCHAR(255),
    source_ip VARCHAR(45),
    target_resource VARCHAR(500),
    event_details JSON,
    risk_level ENUM('info', 'low', 'medium', 'high', 'critical') DEFAULT 'info',
    retention_period_days INT DEFAULT 2555, -- 7 years default
    archived BOOLEAN DEFAULT FALSE,
    hash_value VARCHAR(256),
    digital_signature TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_log_source (log_source),
    INDEX idx_event_category (event_category),
    INDEX idx_user_id (user_id),
    INDEX idx_source_ip (source_ip),
    INDEX idx_created_at (created_at),
    INDEX idx_risk_level (risk_level)
);

-- Create views for security reporting and dashboards

-- Security dashboard overview
CREATE VIEW security_dashboard AS
SELECT
    se.tenant_id,
    COUNT(DISTINCT se.event_id) as total_events_24h,
    COUNT(DISTINCT CASE WHEN se.threat_level = 'critical' THEN se.event_id END) as critical_events_24h,
    COUNT(DISTINCT CASE WHEN se.threat_level = 'high' THEN se.event_id END) as high_events_24h,
    COUNT(DISTINCT si.incident_id) as active_incidents,
    COUNT(DISTINCT CASE WHEN si.severity = 'critical' THEN si.incident_id END) as critical_incidents,
    COUNT(DISTINCT vf.finding_id) as open_vulnerabilities,
    COUNT(DISTINCT CASE WHEN vf.severity = 'critical' THEN vf.finding_id END) as critical_vulnerabilities,
    COUNT(DISTINCT spv.violation_id) as policy_violations_24h,
    AVG(se.risk_score) as avg_risk_score_24h
FROM security_events se
LEFT JOIN security_incidents si ON se.tenant_id = si.tenant_id AND si.status NOT IN ('closed')
LEFT JOIN vulnerability_findings vf ON se.tenant_id = vf.assessment_id AND vf.status = 'open'
LEFT JOIN security_policy_violations spv ON se.tenant_id = spv.tenant_id AND spv.created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
WHERE se.created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY se.tenant_id;

-- Compliance status overview
CREATE VIEW compliance_status_overview AS
SELECT
    ca.tenant_id,
    cf.framework_code,
    cf.framework_name,
    ca.assessment_id,
    ca.overall_score,
    ca.compliance_percentage,
    ca.total_controls,
    ca.compliant_controls,
    ca.non_compliant_controls,
    ca.critical_findings,
    ca.high_findings,
    ca.status as assessment_status,
    ca.certification_status,
    ca.certification_expiry,
    ca.next_assessment_date
FROM compliance_assessments ca
JOIN compliance_frameworks cf ON ca.framework_id = cf.framework_id
WHERE ca.status = 'completed'
AND ca.end_date = (
    SELECT MAX(ca2.end_date)
    FROM compliance_assessments ca2
    WHERE ca2.tenant_id = ca.tenant_id
    AND ca2.framework_id = ca.framework_id
    AND ca2.status = 'completed'
);

-- Vulnerability risk assessment
CREATE VIEW vulnerability_risk_assessment AS
SELECT
    va.tenant_id,
    va.assessment_id,
    va.scan_type,
    va.total_vulnerabilities,
    va.critical_count,
    va.high_count,
    va.medium_count,
    va.low_count,
    va.risk_score,
    COUNT(DISTINCT vf.finding_id) as open_findings,
    COUNT(DISTINCT CASE WHEN vf.severity = 'critical' AND vf.remediation_deadline < CURDATE() THEN vf.finding_id END) as overdue_critical,
    COUNT(DISTINCT CASE WHEN vf.severity = 'high' AND vf.remediation_deadline < CURDATE() THEN vf.finding_id END) as overdue_high,
    AVG(vf.cvss_score) as avg_cvss_score,
    va.scan_start_time,
    va.next_scan_date
FROM vulnerability_assessments va
LEFT JOIN vulnerability_findings vf ON va.assessment_id = vf.assessment_id AND vf.status = 'open'
WHERE va.scan_status = 'completed'
GROUP BY va.assessment_id;

-- Security incident metrics
CREATE VIEW security_incident_metrics AS
SELECT
    si.tenant_id,
    COUNT(DISTINCT si.incident_id) as total_incidents,
    COUNT(DISTINCT CASE WHEN si.severity = 'critical' THEN si.incident_id END) as critical_incidents,
    COUNT(DISTINCT CASE WHEN si.severity = 'high' THEN si.incident_id END) as high_incidents,
    COUNT(DISTINCT CASE WHEN si.status = 'closed' THEN si.incident_id END) as resolved_incidents,
    AVG(TIMESTAMPDIFF(HOUR, si.detected_at, si.acknowledged_at)) as avg_acknowledgment_time_hours
