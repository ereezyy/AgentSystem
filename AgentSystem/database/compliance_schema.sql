-- Compliance Framework Database Schema - AgentSystem Profit Machine
-- SOC2, GDPR, HIPAA compliance tracking and automation

-- Create compliance schema
CREATE SCHEMA IF NOT EXISTS compliance;

-- Compliance standards enum
CREATE TYPE compliance.compliance_standard AS ENUM (
    'soc2_type1', 'soc2_type2', 'gdpr', 'hipaa', 'pci_dss', 'iso_27001', 'ccpa', 'nist'
);

-- Compliance status enum
CREATE TYPE compliance.compliance_status AS ENUM (
    'compliant', 'non_compliant', 'in_progress', 'not_applicable', 'needs_review'
);

-- Risk level enum
CREATE TYPE compliance.risk_level AS ENUM (
    'low', 'medium', 'high', 'critical'
);

-- Data classification enum
CREATE TYPE compliance.data_classification AS ENUM (
    'public', 'internal', 'confidential', 'restricted', 'pii', 'phi', 'pci'
);

-- Compliance controls table
CREATE TABLE compliance.controls (
    control_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    standard compliance.compliance_standard NOT NULL,
    control_number VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    implementation_guidance TEXT,
    testing_procedures JSONB DEFAULT '[]',
    evidence_requirements JSONB DEFAULT '[]',
    automation_possible BOOLEAN DEFAULT false,
    risk_level compliance.risk_level NOT NULL,
    frequency VARCHAR(50) NOT NULL, -- daily, weekly, monthly, quarterly, annually
    responsible_team VARCHAR(100) NOT NULL,
    status compliance.compliance_status DEFAULT 'in_progress',
    last_assessed TIMESTAMPTZ,
    next_assessment TIMESTAMPTZ,
    findings JSONB DEFAULT '[]',
    remediation_actions JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Compliance assessments table
CREATE TABLE compliance.assessments (
    assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    standard compliance.compliance_standard NOT NULL,
    assessment_type VARCHAR(50) NOT NULL, -- self, third_party, audit
    assessor VARCHAR(200) NOT NULL,
    scope TEXT NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'planning', -- planning, in_progress, completed, cancelled
    overall_score DECIMAL(5,2),
    findings_count INTEGER DEFAULT 0,
    critical_findings INTEGER DEFAULT 0,
    high_findings INTEGER DEFAULT 0,
    medium_findings INTEGER DEFAULT 0,
    low_findings INTEGER DEFAULT 0,
    recommendations JSONB DEFAULT '[]',
    evidence_collected JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Data processing activities table (GDPR Article 30)
CREATE TABLE compliance.data_processing_activities (
    activity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(300) NOT NULL,
    purpose TEXT NOT NULL,
    legal_basis VARCHAR(100) NOT NULL, -- GDPR legal basis
    data_categories compliance.data_classification[] NOT NULL,
    data_subjects TEXT[] NOT NULL,
    recipients TEXT[] DEFAULT '{}',
    retention_period VARCHAR(200) NOT NULL,
    cross_border_transfers BOOLEAN DEFAULT false,
    transfer_safeguards TEXT,
    automated_decision_making BOOLEAN DEFAULT false,
    profiling BOOLEAN DEFAULT false,
    data_source VARCHAR(200) NOT NULL,
    storage_location VARCHAR(200) NOT NULL,
    encryption_status BOOLEAN DEFAULT true,
    access_controls TEXT[] DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Privacy requests table (GDPR data subject requests)
CREATE TABLE compliance.privacy_requests (
    request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    data_subject_id UUID NOT NULL,
    request_type VARCHAR(50) NOT NULL, -- access, rectification, erasure, portability, restriction, objection
    request_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    requester_email VARCHAR(255) NOT NULL,
    requester_identity_verified BOOLEAN DEFAULT false,
    status VARCHAR(50) DEFAULT 'received', -- received, processing, completed, rejected
    completion_deadline TIMESTAMPTZ NOT NULL,
    response_date TIMESTAMPTZ,
    response_method VARCHAR(50), -- email, postal, portal
    data_provided TEXT,
    actions_taken JSONB DEFAULT '[]',
    rejection_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Assessment results table
CREATE TABLE compliance.assessment_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    standards_checked TEXT[] NOT NULL,
    overall_status compliance.compliance_status NOT NULL,
    findings JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    scores JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Compliance reports table
CREATE TABLE compliance.reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    standard compliance.compliance_standard NOT NULL,
    report_type VARCHAR(50) NOT NULL, -- summary, detailed, audit
    generated_at TIMESTAMPTZ NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    executive_summary JSONB NOT NULL,
    control_assessments JSONB DEFAULT '[]',
    findings JSONB DEFAULT '[]',
    evidence JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Continuous monitoring results table
CREATE TABLE compliance.monitoring_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    checks_performed JSONB DEFAULT '[]',
    violations_found JSONB DEFAULT '[]',
    alerts_generated JSONB DEFAULT '[]',
    overall_health VARCHAR(50) NOT NULL, -- HEALTHY, ISSUES_DETECTED, ERROR
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Compliance findings table
CREATE TABLE compliance.findings (
    finding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    assessment_id UUID,
    control_id UUID,
    finding_type VARCHAR(100) NOT NULL, -- violation, deficiency, observation
    severity compliance.risk_level NOT NULL,
    title VARCHAR(300) NOT NULL,
    description TEXT NOT NULL,
    evidence TEXT,
    recommendation TEXT,
    status VARCHAR(50) DEFAULT 'open', -- open, in_progress, resolved, accepted_risk
    assigned_to VARCHAR(200),
    due_date TIMESTAMPTZ,
    resolution_notes TEXT,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (assessment_id) REFERENCES compliance.assessments(assessment_id) ON DELETE CASCADE,
    FOREIGN KEY (control_id) REFERENCES compliance.controls(control_id) ON DELETE CASCADE
);

-- Evidence vault table
CREATE TABLE compliance.evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    control_id UUID,
    assessment_id UUID,
    evidence_type VARCHAR(100) NOT NULL, -- document, screenshot, log, configuration, policy
    title VARCHAR(300) NOT NULL,
    description TEXT,
    file_path VARCHAR(500),
    file_hash VARCHAR(256),
    file_size INTEGER,
    mime_type VARCHAR(100),
    collected_by VARCHAR(200),
    collection_method VARCHAR(100), -- manual, automated, api
    retention_period VARCHAR(100),
    classification compliance.data_classification DEFAULT 'confidential',
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (control_id) REFERENCES compliance.controls(control_id) ON DELETE CASCADE,
    FOREIGN KEY (assessment_id) REFERENCES compliance.assessments(assessment_id) ON DELETE CASCADE
);

-- Compliance policies table
CREATE TABLE compliance.policies (
    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    policy_name VARCHAR(300) NOT NULL,
    policy_type VARCHAR(100) NOT NULL, -- security, privacy, data_governance, hr
    version VARCHAR(50) NOT NULL,
    effective_date TIMESTAMPTZ NOT NULL,
    review_date TIMESTAMPTZ NOT NULL,
    approval_date TIMESTAMPTZ,
    approved_by VARCHAR(200),
    policy_document TEXT NOT NULL,
    related_controls UUID[],
    applicable_standards compliance.compliance_standard[],
    status VARCHAR(50) DEFAULT 'draft', -- draft, active, superseded, retired
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Risk register table
CREATE TABLE compliance.risk_register (
    risk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    risk_title VARCHAR(300) NOT NULL,
    risk_description TEXT NOT NULL,
    risk_category VARCHAR(100) NOT NULL, -- operational, compliance, financial, strategic, technology
    likelihood compliance.risk_level NOT NULL,
    impact compliance.risk_level NOT NULL,
    inherent_risk_score INTEGER NOT NULL, -- calculated from likelihood * impact
    mitigation_strategy TEXT,
    control_effectiveness VARCHAR(50), -- effective, partially_effective, ineffective
    residual_risk_score INTEGER,
    risk_owner VARCHAR(200),
    risk_status VARCHAR(50) DEFAULT 'identified', -- identified, assessed, mitigated, monitored, closed
    last_reviewed TIMESTAMPTZ,
    next_review TIMESTAMPTZ,
    related_controls UUID[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Incident register table
CREATE TABLE compliance.incidents (
    incident_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    incident_title VARCHAR(300) NOT NULL,
    incident_type VARCHAR(100) NOT NULL, -- security, privacy, data_breach, system_failure
    severity compliance.risk_level NOT NULL,
    status VARCHAR(50) DEFAULT 'reported', -- reported, investigating, contained, resolved, closed
    reported_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reported_by VARCHAR(200),
    discovered_date TIMESTAMPTZ,
    containment_date TIMESTAMPTZ,
    resolution_date TIMESTAMPTZ,
    description TEXT NOT NULL,
    impact_assessment TEXT,
    root_cause TEXT,
    lessons_learned TEXT,
    corrective_actions JSONB DEFAULT '[]',
    notification_requirements JSONB DEFAULT '[]', -- regulatory notifications needed
    affected_data_subjects INTEGER DEFAULT 0,
    affected_records INTEGER DEFAULT 0,
    estimated_cost DECIMAL(12,2),
    related_controls UUID[],
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Training records table
CREATE TABLE compliance.training_records (
    training_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_id UUID NOT NULL,
    training_type VARCHAR(100) NOT NULL, -- security_awareness, privacy, compliance, role_specific
    training_title VARCHAR(300) NOT NULL,
    training_provider VARCHAR(200),
    completion_date TIMESTAMPTZ,
    expiry_date TIMESTAMPTZ,
    score INTEGER, -- percentage score if applicable
    status VARCHAR(50) DEFAULT 'assigned', -- assigned, in_progress, completed, overdue
    certificate_url VARCHAR(500),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Vendor assessments table
CREATE TABLE compliance.vendor_assessments (
    assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    vendor_name VARCHAR(300) NOT NULL,
    vendor_type VARCHAR(100) NOT NULL, -- technology, service, cloud, data_processor
    assessment_date TIMESTAMPTZ NOT NULL,
    assessor VARCHAR(200),
    overall_score INTEGER, -- 0-100
    security_score INTEGER,
    privacy_score INTEGER,
    compliance_score INTEGER,
    risk_rating compliance.risk_level,
    certification_status JSONB DEFAULT '{}', -- SOC2, ISO27001, etc.
    contract_review_date TIMESTAMPTZ,
    data_processing_agreement BOOLEAN DEFAULT false,
    security_questionnaire_completed BOOLEAN DEFAULT false,
    penetration_test_results VARCHAR(500),
    findings JSONB DEFAULT '[]',
    remediation_plan TEXT,
    approval_status VARCHAR(50) DEFAULT 'pending', -- pending, approved, rejected, conditional
    next_review_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX idx_compliance_controls_tenant_standard ON compliance.controls(tenant_id, standard);
CREATE INDEX idx_compliance_controls_status ON compliance.controls(status);
CREATE INDEX idx_compliance_controls_risk_level ON compliance.controls(risk_level);
CREATE INDEX idx_compliance_controls_next_assessment ON compliance.controls(next_assessment);

CREATE INDEX idx_compliance_assessments_tenant ON compliance.assessments(tenant_id);
CREATE INDEX idx_compliance_assessments_standard ON compliance.assessments(standard);
CREATE INDEX idx_compliance_assessments_status ON compliance.assessments(status);
CREATE INDEX idx_compliance_assessments_dates ON compliance.assessments(start_date, end_date);

CREATE INDEX idx_data_processing_activities_tenant ON compliance.data_processing_activities(tenant_id);
CREATE INDEX idx_data_processing_activities_active ON compliance.data_processing_activities(is_active);
CREATE INDEX idx_data_processing_activities_legal_basis ON compliance.data_processing_activities(legal_basis);

CREATE INDEX idx_privacy_requests_tenant ON compliance.privacy_requests(tenant_id);
CREATE INDEX idx_privacy_requests_status ON compliance.privacy_requests(status);
CREATE INDEX idx_privacy_requests_deadline ON compliance.privacy_requests(completion_deadline);
CREATE INDEX idx_privacy_requests_type ON compliance.privacy_requests(request_type);

CREATE INDEX idx_assessment_results_tenant ON compliance.assessment_results(tenant_id);
CREATE INDEX idx_assessment_results_timestamp ON compliance.assessment_results(timestamp);

CREATE INDEX idx_monitoring_results_tenant ON compliance.monitoring_results(tenant_id);
CREATE INDEX idx_monitoring_results_timestamp ON compliance.monitoring_results(timestamp);
CREATE INDEX idx_monitoring_results_health ON compliance.monitoring_results(overall_health);

CREATE INDEX idx_compliance_findings_tenant ON compliance.findings(tenant_id);
CREATE INDEX idx_compliance_findings_severity ON compliance.findings(severity);
CREATE INDEX idx_compliance_findings_status ON compliance.findings(status);
CREATE INDEX idx_compliance_findings_due_date ON compliance.findings(due_date);

CREATE INDEX idx_compliance_evidence_tenant ON compliance.evidence(tenant_id);
CREATE INDEX idx_compliance_evidence_type ON compliance.evidence(evidence_type);
CREATE INDEX idx_compliance_evidence_classification ON compliance.evidence(classification);

CREATE INDEX idx_compliance_policies_tenant ON compliance.policies(tenant_id);
CREATE INDEX idx_compliance_policies_status ON compliance.policies(status);
CREATE INDEX idx_compliance_policies_review_date ON compliance.policies(review_date);

CREATE INDEX idx_risk_register_tenant ON compliance.risk_register(tenant_id);
CREATE INDEX idx_risk_register_score ON compliance.risk_register(inherent_risk_score, residual_risk_score);
CREATE INDEX idx_risk_register_status ON compliance.risk_register(risk_status);
CREATE INDEX idx_risk_register_next_review ON compliance.risk_register(next_review);

CREATE INDEX idx_compliance_incidents_tenant ON compliance.incidents(tenant_id);
CREATE INDEX idx_compliance_incidents_severity ON compliance.incidents(severity);
CREATE INDEX idx_compliance_incidents_status ON compliance.incidents(status);
CREATE INDEX idx_compliance_incidents_reported_date ON compliance.incidents(reported_date);

CREATE INDEX idx_training_records_tenant ON compliance.training_records(tenant_id);
CREATE INDEX idx_training_records_user ON compliance.training_records(user_id);
CREATE INDEX idx_training_records_status ON compliance.training_records(status);
CREATE INDEX idx_training_records_expiry ON compliance.training_records(expiry_date);

CREATE INDEX idx_vendor_assessments_tenant ON compliance.vendor_assessments(tenant_id);
CREATE INDEX idx_vendor_assessments_risk_rating ON compliance.vendor_assessments(risk_rating);
CREATE INDEX idx_vendor_assessments_approval_status ON compliance.vendor_assessments(approval_status);
CREATE INDEX idx_vendor_assessments_next_review ON compliance.vendor_assessments(next_review_date);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION compliance.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers to relevant tables
CREATE TRIGGER update_controls_updated_at BEFORE UPDATE ON compliance.controls FOR EACH ROW EXECUTE FUNCTION compliance.update_updated_at_column();
CREATE TRIGGER update_assessments_updated_at BEFORE UPDATE ON compliance.assessments FOR EACH ROW EXECUTE FUNCTION compliance.update_updated_at_column();
CREATE TRIGGER update_data_processing_activities_updated_at BEFORE UPDATE ON compliance.data_processing_activities FOR EACH ROW EXECUTE FUNCTION compliance.update_updated_at_column();
CREATE TRIGGER update_privacy_requests_updated_at BEFORE UPDATE ON compliance.privacy_requests FOR EACH ROW EXECUTE FUNCTION compliance.update_updated_at_column();
CREATE TRIGGER update_findings_updated_at BEFORE UPDATE ON compliance.findings FOR EACH ROW EXECUTE FUNCTION compliance.update_updated_at_column();
CREATE TRIGGER update_policies_updated_at BEFORE UPDATE ON compliance.policies FOR EACH ROW EXECUTE FUNCTION compliance.update_updated_at_column();
CREATE TRIGGER update_risk_register_updated_at BEFORE UPDATE ON compliance.risk_register FOR EACH ROW EXECUTE FUNCTION compliance.update_updated_at_column();
CREATE TRIGGER update_incidents_updated_at BEFORE UPDATE ON compliance.incidents FOR EACH ROW EXECUTE FUNCTION compliance.update_updated_at_column();
CREATE TRIGGER update_training_records_updated_at BEFORE UPDATE ON compliance.training_records FOR EACH ROW EXECUTE FUNCTION compliance.update_updated_at_column();
CREATE TRIGGER update_vendor_assessments_updated_at BEFORE UPDATE ON compliance.vendor_assessments FOR EACH ROW EXECUTE FUNCTION compliance.update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE compliance.controls ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.data_processing_activities ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.privacy_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.assessment_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.monitoring_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.risk_register ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.incidents ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.training_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance.vendor_assessments ENABLE ROW LEVEL SECURITY;

-- RLS policies for tenant isolation
CREATE POLICY compliance_controls_tenant_isolation ON compliance.controls
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_assessments_tenant_isolation ON compliance.assessments
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_data_processing_activities_tenant_isolation ON compliance.data_processing_activities
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_privacy_requests_tenant_isolation ON compliance.privacy_requests
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_assessment_results_tenant_isolation ON compliance.assessment_results
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_reports_tenant_isolation ON compliance.reports
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_monitoring_results_tenant_isolation ON compliance.monitoring_results
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_findings_tenant_isolation ON compliance.findings
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_evidence_tenant_isolation ON compliance.evidence
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_policies_tenant_isolation ON compliance.policies
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_risk_register_tenant_isolation ON compliance.risk_register
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_incidents_tenant_isolation ON compliance.incidents
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_training_records_tenant_isolation ON compliance.training_records
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY compliance_vendor_assessments_tenant_isolation ON compliance.vendor_assessments
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant permissions
GRANT USAGE ON SCHEMA compliance TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA compliance TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA compliance TO agentsystem_api;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA compliance TO agentsystem_api;

-- Create views for common compliance reports
CREATE VIEW compliance.compliance_dashboard AS
SELECT
    t.tenant_id,
    t.name as tenant_name,
    COUNT(c.*) as total_controls,
    COUNT(CASE WHEN c.status = 'compliant' THEN 1 END) as compliant_controls,
    COUNT(CASE WHEN c.status = 'non_compliant' THEN 1 END) as non_compliant_controls,
    COUNT(CASE WHEN c.risk_level = 'critical' THEN 1 END) as critical_controls,
    COUNT(f.*) as open_findings,
    COUNT(CASE WHEN f.severity = 'critical' THEN 1 END) as critical_findings,
    COUNT(pr.*) as pending_privacy_requests,
    MAX(mr.timestamp) as last_monitoring_check
FROM billing.tenants t
LEFT JOIN compliance.controls c ON t.tenant_id = c.tenant_id
LEFT JOIN compliance.findings f ON t.tenant_id = f.tenant_id AND f.status = 'open'
LEFT JOIN compliance.privacy_requests pr ON t.tenant_id = pr.tenant_id AND pr.status IN ('received', 'processing')
LEFT JOIN compliance.monitoring_results mr ON t.tenant_id = mr.tenant_id
GROUP BY t.tenant_id, t.name;

-- Grant permissions on views
GRANT SELECT ON compliance.compliance_dashboard TO agentsystem_api;
