-- Multi-Region Deployment Infrastructure Database Schema - AgentSystem Profit Machine
-- Global deployment management with auto-scaling, failover, and compliance

-- Create infrastructure schema
CREATE SCHEMA IF NOT EXISTS infrastructure;

-- Regions table
CREATE TABLE infrastructure.regions (
    region_id VARCHAR(50) PRIMARY KEY,
    region_name VARCHAR(255) NOT NULL,
    region_code VARCHAR(50) NOT NULL,
    continent VARCHAR(100) NOT NULL,
    country VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    latitude DECIMAL(10,7) NOT NULL,
    longitude DECIMAL(10,7) NOT NULL,
    cloud_provider VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance', 'degraded', 'failed')),
    is_primary BOOLEAN DEFAULT FALSE,
    data_residency_compliant BOOLEAN DEFAULT TRUE,
    compliance_certifications JSONB DEFAULT '[]',
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_health_check TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    timezone VARCHAR(100) DEFAULT 'UTC',
    capacity_limits JSONB DEFAULT '{}',
    pricing_tier VARCHAR(50) DEFAULT 'standard'
);

-- Deployment targets table
CREATE TABLE infrastructure.deployment_targets (
    target_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id VARCHAR(50) NOT NULL REFERENCES infrastructure.regions(region_id) ON DELETE CASCADE,
    deployment_type VARCHAR(50) NOT NULL CHECK (deployment_type IN ('production', 'staging', 'development', 'canary', 'blue_green')),
    environment VARCHAR(100) NOT NULL,
    api_endpoint VARCHAR(500) NOT NULL,
    database_endpoint VARCHAR(500) NOT NULL,
    redis_endpoint VARCHAR(500) NOT NULL,
    cdn_endpoint VARCHAR(500),
    load_balancer_endpoint VARCHAR(500) NOT NULL,
    auto_scaling_config JSONB NOT NULL DEFAULT '{}',
    resource_limits JSONB DEFAULT '{}',
    monitoring_config JSONB DEFAULT '{}',
    backup_config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deployment_version VARCHAR(100),
    health_status VARCHAR(50) DEFAULT 'unknown',
    last_deployment TIMESTAMP WITH TIME ZONE,
    current_instances INTEGER DEFAULT 0,
    target_instances INTEGER DEFAULT 0
);

-- Health metrics table
CREATE TABLE infrastructure.health_metrics (
    metrics_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id VARCHAR(50) NOT NULL REFERENCES infrastructure.regions(region_id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES infrastructure.deployment_targets(target_id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    api_response_time_ms DECIMAL(10,2) DEFAULT 0,
    database_response_time_ms DECIMAL(10,2) DEFAULT 0,
    redis_response_time_ms DECIMAL(10,2) DEFAULT 0,
    cpu_utilization DECIMAL(5,2) DEFAULT 0,
    memory_utilization DECIMAL(5,2) DEFAULT 0,
    disk_utilization DECIMAL(5,2) DEFAULT 0,
    network_in_mbps DECIMAL(10,2) DEFAULT 0,
    network_out_mbps DECIMAL(10,2) DEFAULT 0,
    active_connections INTEGER DEFAULT 0,
    requests_per_second DECIMAL(10,2) DEFAULT 0,
    error_rate DECIMAL(5,2) DEFAULT 0,
    uptime_percentage DECIMAL(5,2) DEFAULT 100,
    status VARCHAR(50) DEFAULT 'healthy' CHECK (status IN ('healthy', 'warning', 'critical', 'unknown')),
    custom_metrics JSONB DEFAULT '{}'
);

-- Traffic routing rules table
CREATE TABLE infrastructure.traffic_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(255) NOT NULL,
    source_regions JSONB NOT NULL DEFAULT '[]',
    target_region VARCHAR(50) NOT NULL REFERENCES infrastructure.regions(region_id),
    traffic_percentage DECIMAL(5,2) NOT NULL DEFAULT 100.0 CHECK (traffic_percentage >= 0 AND traffic_percentage <= 100),
    conditions JSONB DEFAULT '{}',
    priority INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_applied TIMESTAMP WITH TIME ZONE,
    routing_algorithm VARCHAR(50) DEFAULT 'round_robin',
    health_check_required BOOLEAN DEFAULT TRUE,
    failover_enabled BOOLEAN DEFAULT TRUE
);

-- Backup status table
CREATE TABLE infrastructure.backup_status (
    backup_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id VARCHAR(50) NOT NULL REFERENCES infrastructure.regions(region_id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES infrastructure.deployment_targets(target_id) ON DELETE CASCADE,
    backup_type VARCHAR(50) NOT NULL CHECK (backup_type IN ('full', 'incremental', 'log', 'snapshot')),
    backup_size_gb DECIMAL(12,2) DEFAULT 0,
    backup_location VARCHAR(1000) NOT NULL,
    encryption_enabled BOOLEAN DEFAULT TRUE,
    compression_enabled BOOLEAN DEFAULT TRUE,
    retention_days INTEGER DEFAULT 30,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'expired')),
    error_message TEXT,
    verification_status VARCHAR(50) DEFAULT 'pending',
    backup_metadata JSONB DEFAULT '{}',
    cross_region_replicas JSONB DEFAULT '[]'
);

-- Deployment logs table
CREATE TABLE infrastructure.deployment_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id UUID NOT NULL REFERENCES infrastructure.deployment_targets(target_id) ON DELETE CASCADE,
    deployment_id UUID NOT NULL,
    step_name VARCHAR(255) NOT NULL,
    step_status VARCHAR(50) NOT NULL CHECK (step_status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER DEFAULT 0,
    log_message TEXT,
    error_details TEXT,
    step_output JSONB DEFAULT '{}',
    step_order INTEGER DEFAULT 0
);

-- Failover events table
CREATE TABLE infrastructure.failover_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    failed_region_id VARCHAR(50) NOT NULL REFERENCES infrastructure.regions(region_id),
    target_region_id VARCHAR(50) NOT NULL REFERENCES infrastructure.regions(region_id),
    trigger_reason VARCHAR(500) NOT NULL,
    event_type VARCHAR(50) DEFAULT 'automatic' CHECK (event_type IN ('automatic', 'manual', 'scheduled')),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    total_duration_ms INTEGER DEFAULT 0,
    affected_targets INTEGER DEFAULT 0,
    successful_failovers INTEGER DEFAULT 0,
    failed_failovers INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'completed', 'partial_failure', 'failed')),
    error_message TEXT,
    recovery_time_objective_met BOOLEAN DEFAULT FALSE,
    recovery_point_objective_met BOOLEAN DEFAULT FALSE,
    failover_details JSONB DEFAULT '{}'
);

-- Compliance audit logs table
CREATE TABLE infrastructure.compliance_audits (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id VARCHAR(50) NOT NULL REFERENCES infrastructure.regions(region_id) ON DELETE CASCADE,
    audit_type VARCHAR(100) NOT NULL,
    compliance_framework VARCHAR(100) NOT NULL,
    audit_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    auditor_name VARCHAR(255),
    audit_status VARCHAR(50) DEFAULT 'pending' CHECK (audit_status IN ('pending', 'in_progress', 'passed', 'failed', 'partial')),
    compliance_score DECIMAL(5,2) DEFAULT 0,
    findings JSONB DEFAULT '[]',
    remediation_actions JSONB DEFAULT '[]',
    next_audit_date TIMESTAMP WITH TIME ZONE,
    audit_report_url VARCHAR(1000),
    certification_valid_until TIMESTAMP WITH TIME ZONE
);

-- Auto-scaling events table
CREATE TABLE infrastructure.scaling_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id UUID NOT NULL REFERENCES infrastructure.deployment_targets(target_id) ON DELETE CASCADE,
    scaling_action VARCHAR(50) NOT NULL CHECK (scaling_action IN ('scale_up', 'scale_down', 'no_change')),
    trigger_metric VARCHAR(100) NOT NULL,
    trigger_value DECIMAL(10,2) NOT NULL,
    threshold_value DECIMAL(10,2) NOT NULL,
    instances_before INTEGER NOT NULL,
    instances_after INTEGER NOT NULL,
    event_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completion_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'triggered' CHECK (status IN ('triggered', 'in_progress', 'completed', 'failed')),
    error_message TEXT,
    cost_impact DECIMAL(10,4) DEFAULT 0,
    scaling_reason TEXT
);

-- Global configuration table
CREATE TABLE infrastructure.global_config (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    config_type VARCHAR(50) DEFAULT 'system',
    description TEXT,
    is_encrypted BOOLEAN DEFAULT FALSE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255),
    environment VARCHAR(50) DEFAULT 'production'
);

-- Performance benchmarks table
CREATE TABLE infrastructure.performance_benchmarks (
    benchmark_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id VARCHAR(50) NOT NULL REFERENCES infrastructure.regions(region_id) ON DELETE CASCADE,
    target_id UUID REFERENCES infrastructure.deployment_targets(target_id) ON DELETE CASCADE,
    benchmark_type VARCHAR(100) NOT NULL,
    test_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    test_duration_ms INTEGER NOT NULL,
    requests_per_second DECIMAL(10,2) NOT NULL,
    avg_response_time_ms DECIMAL(10,2) NOT NULL,
    p95_response_time_ms DECIMAL(10,2) NOT NULL,
    p99_response_time_ms DECIMAL(10,2) NOT NULL,
    error_rate DECIMAL(5,2) DEFAULT 0,
    throughput_mbps DECIMAL(10,2) DEFAULT 0,
    concurrent_users INTEGER DEFAULT 0,
    test_configuration JSONB DEFAULT '{}',
    baseline_comparison JSONB DEFAULT '{}'
);

-- Cost tracking table
CREATE TABLE infrastructure.cost_tracking (
    cost_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id VARCHAR(50) NOT NULL REFERENCES infrastructure.regions(region_id) ON DELETE CASCADE,
    target_id UUID REFERENCES infrastructure.deployment_targets(target_id) ON DELETE SET NULL,
    cost_date DATE NOT NULL,
    cost_type VARCHAR(100) NOT NULL,
    service_name VARCHAR(255) NOT NULL,
    cost_amount DECIMAL(12,4) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    usage_quantity DECIMAL(15,4) DEFAULT 0,
    usage_unit VARCHAR(50),
    billing_period_start DATE NOT NULL,
    billing_period_end DATE NOT NULL,
    cost_center VARCHAR(100),
    tags JSONB DEFAULT '{}',
    optimization_potential DECIMAL(12,4) DEFAULT 0
);

-- Create indexes for performance
CREATE INDEX idx_regions_status ON infrastructure.regions(status);
CREATE INDEX idx_regions_cloud_provider ON infrastructure.regions(cloud_provider);
CREATE INDEX idx_regions_compliance ON infrastructure.regions USING GIN(compliance_certifications);

CREATE INDEX idx_deployment_targets_region ON infrastructure.deployment_targets(region_id);
CREATE INDEX idx_deployment_targets_type ON infrastructure.deployment_targets(deployment_type);
CREATE INDEX idx_deployment_targets_active ON infrastructure.deployment_targets(is_active);
CREATE INDEX idx_deployment_targets_health ON infrastructure.deployment_targets(health_status);

CREATE INDEX idx_health_metrics_target ON infrastructure.health_metrics(target_id);
CREATE INDEX idx_health_metrics_timestamp ON infrastructure.health_metrics(timestamp);
CREATE INDEX idx_health_metrics_status ON infrastructure.health_metrics(status);
CREATE INDEX idx_health_metrics_region_time ON infrastructure.health_metrics(region_id, timestamp);

CREATE INDEX idx_traffic_rules_target_region ON infrastructure.traffic_rules(target_region);
CREATE INDEX idx_traffic_rules_active ON infrastructure.traffic_rules(is_active);
CREATE INDEX idx_traffic_rules_priority ON infrastructure.traffic_rules(priority);

CREATE INDEX idx_backup_status_target ON infrastructure.backup_status(target_id);
CREATE INDEX idx_backup_status_region ON infrastructure.backup_status(region_id);
CREATE INDEX idx_backup_status_type ON infrastructure.backup_status(backup_type);
CREATE INDEX idx_backup_status_date ON infrastructure.backup_status(created_date);

CREATE INDEX idx_deployment_logs_target ON infrastructure.deployment_logs(target_id);
CREATE INDEX idx_deployment_logs_deployment ON infrastructure.deployment_logs(deployment_id);
CREATE INDEX idx_deployment_logs_time ON infrastructure.deployment_logs(start_time);

CREATE INDEX idx_failover_events_failed_region ON infrastructure.failover_events(failed_region_id);
CREATE INDEX idx_failover_events_target_region ON infrastructure.failover_events(target_region_id);
CREATE INDEX idx_failover_events_time ON infrastructure.failover_events(start_time);

CREATE INDEX idx_compliance_audits_region ON infrastructure.compliance_audits(region_id);
CREATE INDEX idx_compliance_audits_framework ON infrastructure.compliance_audits(compliance_framework);
CREATE INDEX idx_compliance_audits_date ON infrastructure.compliance_audits(audit_date);

CREATE INDEX idx_scaling_events_target ON infrastructure.scaling_events(target_id);
CREATE INDEX idx_scaling_events_time ON infrastructure.scaling_events(event_time);
CREATE INDEX idx_scaling_events_action ON infrastructure.scaling_events(scaling_action);

CREATE INDEX idx_performance_benchmarks_region ON infrastructure.performance_benchmarks(region_id);
CREATE INDEX idx_performance_benchmarks_date ON infrastructure.performance_benchmarks(test_date);
CREATE INDEX idx_performance_benchmarks_type ON infrastructure.performance_benchmarks(benchmark_type);

CREATE INDEX idx_cost_tracking_region ON infrastructure.cost_tracking(region_id);
CREATE INDEX idx_cost_tracking_date ON infrastructure.cost_tracking(cost_date);
CREATE INDEX idx_cost_tracking_service ON infrastructure.cost_tracking(service_name);

-- Create composite indexes for common queries
CREATE INDEX idx_health_metrics_region_status_time ON infrastructure.health_metrics(region_id, status, timestamp);
CREATE INDEX idx_deployment_targets_region_active ON infrastructure.deployment_targets(region_id, is_active);
CREATE INDEX idx_backup_status_region_type_date ON infrastructure.backup_status(region_id, backup_type, created_date);

-- Create partial indexes for active records
CREATE INDEX idx_active_deployment_targets ON infrastructure.deployment_targets(region_id) WHERE is_active = true;
CREATE INDEX idx_active_traffic_rules ON infrastructure.traffic_rules(target_region) WHERE is_active = true;
CREATE INDEX idx_recent_health_metrics ON infrastructure.health_metrics(target_id, timestamp) WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour';

-- Create functions for infrastructure management
CREATE OR REPLACE FUNCTION infrastructure.update_target_health()
RETURNS TRIGGER AS $$
BEGIN
    -- Update deployment target health status based on latest metrics
    UPDATE infrastructure.deployment_targets
    SET
        health_status = NEW.status,
        updated_date = CURRENT_TIMESTAMP
    WHERE target_id = NEW.target_id;

    -- Auto-trigger scaling if needed
    IF NEW.status = 'critical' THEN
        INSERT INTO infrastructure.scaling_events (
            target_id, scaling_action, trigger_metric, trigger_value,
            threshold_value, instances_before, instances_after, scaling_reason
        )
        SELECT
            NEW.target_id, 'scale_up', 'health_critical', 0, 1,
            COALESCE(current_instances, 0), COALESCE(target_instances, 0) + 1,
            'Auto-scaling triggered by critical health status'
        FROM infrastructure.deployment_targets
        WHERE target_id = NEW.target_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for health status updates
CREATE TRIGGER trigger_update_target_health
    AFTER INSERT ON infrastructure.health_metrics
    FOR EACH ROW
    EXECUTE FUNCTION infrastructure.update_target_health();

-- Create function for automatic failover detection
CREATE OR REPLACE FUNCTION infrastructure.detect_failover_conditions()
RETURNS TRIGGER AS $$
DECLARE
    failed_targets INTEGER;
    region_threshold INTEGER := 2; -- Minimum failed targets to trigger region failover
BEGIN
    -- Count critical targets in the region
    SELECT COUNT(*) INTO failed_targets
    FROM infrastructure.deployment_targets dt
    JOIN infrastructure.health_metrics hm ON dt.target_id = hm.target_id
    WHERE dt.region_id = NEW.region_id
    AND hm.status = 'critical'
    AND hm.timestamp > CURRENT_TIMESTAMP - INTERVAL '5 minutes'
    AND dt.is_active = true;

    -- Trigger failover if threshold exceeded
    IF failed_targets >= region_threshold THEN
        INSERT INTO infrastructure.failover_events (
            failed_region_id, target_region_id, trigger_reason,
            event_type, affected_targets
        )
        VALUES (
            NEW.region_id,
            (SELECT region_id FROM infrastructure.regions
             WHERE region_id != NEW.region_id AND status = 'active'
             ORDER BY is_primary DESC, region_id LIMIT 1),
            'Automatic failover triggered by multiple critical health metrics',
            'automatic',
            failed_targets
        );

        -- Update region status
        UPDATE infrastructure.regions
        SET status = 'degraded', last_health_check = CURRENT_TIMESTAMP
        WHERE region_id = NEW.region_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for failover detection
CREATE TRIGGER trigger_detect_failover
    AFTER INSERT ON infrastructure.health_metrics
    FOR EACH ROW
    WHEN (NEW.status = 'critical')
    EXECUTE FUNCTION infrastructure.detect_failover_conditions();

-- Create function for cost optimization alerts
CREATE OR REPLACE FUNCTION infrastructure.analyze_cost_optimization()
RETURNS TRIGGER AS $$
DECLARE
    monthly_cost DECIMAL(12,4);
    optimization_threshold DECIMAL(12,4) := 1000.00; -- $1000 threshold
BEGIN
    -- Calculate monthly cost for the region
    SELECT SUM(cost_amount) INTO monthly_cost
    FROM infrastructure.cost_tracking
    WHERE region_id = NEW.region_id
    AND cost_date >= DATE_TRUNC('month', CURRENT_DATE)
    AND cost_date < DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month';

    -- Check if optimization potential exceeds threshold
    IF COALESCE(monthly_cost, 0) > optimization_threshold THEN
        -- Insert optimization recommendation
        INSERT INTO infrastructure.global_config (
            config_key, config_value, config_type, description
        )
        VALUES (
            'cost_optimization_alert_' || NEW.region_id || '_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP),
            json_build_object(
                'region_id', NEW.region_id,
                'monthly_cost', monthly_cost,
                'optimization_potential', NEW.optimization_potential,
                'alert_date', CURRENT_TIMESTAMP
            ),
            'alert',
            'Cost optimization opportunity detected for region ' || NEW.region_id
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for cost optimization analysis
CREATE TRIGGER trigger_cost_optimization
    AFTER INSERT OR UPDATE ON infrastructure.cost_tracking
    FOR EACH ROW
    WHEN (NEW.optimization_potential > 0)
    EXECUTE FUNCTION infrastructure.analyze_cost_optimization();

-- Insert default regions
INSERT INTO infrastructure.regions (
    region_id, region_name, region_code, continent, country, city,
    latitude, longitude, cloud_provider, status, is_primary,
    data_residency_compliant, compliance_certifications, timezone
) VALUES
(
    'us-east-1', 'US East (N. Virginia)', 'us-east-1', 'North America', 'United States', 'Virginia',
    37.5407, -77.4360, 'aws', 'active', true, true,
    '["SOC2", "HIPAA", "FedRAMP"]', 'America/New_York'
),
(
    'eu-west-1', 'EU West (Ireland)', 'eu-west-1', 'Europe', 'Ireland', 'Dublin',
    53.3498, -6.2603, 'aws', 'active', false, true,
    '["GDPR", "SOC2", "ISO27001"]', 'Europe/Dublin'
),
(
    'ap-southeast-1', 'Asia Pacific (Singapore)', 'ap-southeast-1', 'Asia', 'Singapore', 'Singapore',
    1.3521, 103.8198, 'aws', 'active', false, true,
    '["SOC2", "ISO27001"]', 'Asia/Singapore'
),
(
    'us-west-2', 'US West (Oregon)', 'us-west-2', 'North America', 'United States', 'Oregon',
    45.5152, -122.6784, 'aws', 'active', false, true,
    '["SOC2", "HIPAA", "FedRAMP"]', 'America/Los_Angeles'
),
(
    'eu-central-1', 'EU Central (Frankfurt)', 'eu-central-1', 'Europe', 'Germany', 'Frankfurt',
    50.1109, 8.6821, 'aws', 'active', false, true,
    '["GDPR", "SOC2", "ISO27001"]', 'Europe/Berlin'
);

-- Insert default global configuration
INSERT INTO infrastructure.global_config (config_key, config_value, config_type, description) VALUES
('failover_enabled', 'true', 'system', 'Enable automatic failover across regions'),
('max_failover_time_minutes', '15', 'system', 'Maximum time allowed for failover completion'),
('health_check_interval_seconds', '30', 'system', 'Interval between health checks'),
('backup_retention_days', '30', 'system', 'Default backup retention period'),
('auto_scaling_enabled', 'true', 'system', 'Enable automatic scaling'),
('cost_optimization_enabled', 'true', 'system', 'Enable cost optimization monitoring'),
('compliance_audit_interval_days', '90', 'system', 'Interval between compliance audits'),
('performance_benchmark_enabled', 'true', 'system', 'Enable performance benchmarking');

-- Grant permissions
GRANT USAGE ON SCHEMA infrastructure TO agentsystem_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA infrastructure TO agentsystem_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA infrastructure TO agentsystem_app;
