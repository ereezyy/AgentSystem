-- Customer Feedback and Feature Request Schema
-- This schema supports the storage and management of customer feedback and feature requests
-- for a multi-tenant SaaS platform.

-- Feedback table to store customer feedback submissions
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    rating INTEGER,
    sentiment VARCHAR(20) NOT NULL DEFAULT 'neutral',
    status VARCHAR(20) NOT NULL DEFAULT 'submitted',
    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
    related_feature_request VARCHAR(50),
    submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_feedback_tenant (tenant_id),
    INDEX idx_feedback_user (user_id),
    INDEX idx_feedback_category (category),
    INDEX idx_feedback_status (status),
    INDEX idx_feedback_priority (priority),
    INDEX idx_feedback_submitted (submitted_at)
);

-- Feedback notes table to store status change notes and comments
CREATE TABLE IF NOT EXISTS feedback_notes (
    note_id VARCHAR(50) PRIMARY KEY,
    feedback_id VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    note_text TEXT NOT NULL,
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (feedback_id) REFERENCES feedback(feedback_id) ON DELETE CASCADE,
    INDEX idx_feedback_notes_feedback (feedback_id),
    INDEX idx_feedback_notes_tenant (tenant_id)
);

-- Feature requests table to store customer feature requests
CREATE TABLE IF NOT EXISTS feature_requests (
    request_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    use_case TEXT,
    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
    status VARCHAR(20) NOT NULL DEFAULT 'new',
    votes INTEGER NOT NULL DEFAULT 1,
    impact_score FLOAT NOT NULL DEFAULT 0.0,
    implementation_complexity VARCHAR(20) NOT NULL DEFAULT 'medium',
    submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_feature_requests_tenant (tenant_id),
    INDEX idx_feature_requests_user (user_id),
    INDEX idx_feature_requests_status (status),
    INDEX idx_feature_requests_priority (priority),
    INDEX idx_feature_requests_votes (votes),
    INDEX idx_feature_requests_submitted (submitted_at)
);

-- Feature request votes table to track individual votes
CREATE TABLE IF NOT EXISTS feature_request_votes (
    vote_id VARCHAR(50) PRIMARY KEY,
    request_id VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    voted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (request_id) REFERENCES feature_requests(request_id) ON DELETE CASCADE,
    INDEX idx_feature_votes_request (request_id),
    INDEX idx_feature_votes_tenant (tenant_id),
    INDEX idx_feature_votes_user (user_id),
    UNIQUE KEY unique_user_request_vote (request_id, user_id)
);

-- Feature request notes table to store status change notes and comments
CREATE TABLE IF NOT EXISTS feature_request_notes (
    note_id VARCHAR(50) PRIMARY KEY,
    request_id VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    note_text TEXT NOT NULL,
    created_by VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (request_id) REFERENCES feature_requests(request_id) ON DELETE CASCADE,
    INDEX idx_feature_notes_request (request_id),
    INDEX idx_feature_notes_tenant (tenant_id)
);

-- Feedback categories table to store available feedback categories
CREATE TABLE IF NOT EXISTS feedback_categories (
    category_id VARCHAR(50) PRIMARY KEY,
    category_name VARCHAR(50) NOT NULL,
    description TEXT,
    tenant_id VARCHAR(50) NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_feedback_categories_tenant (tenant_id),
    UNIQUE KEY unique_category_name_tenant (category_name, tenant_id)
);

-- Insert default feedback categories
INSERT INTO feedback_categories (category_id, category_name, description, tenant_id, is_default)
VALUES
    ('cat_general', 'General', 'General feedback about the product or service', 'system', TRUE),
    ('cat_bug', 'Bug', 'Issues or bugs encountered while using the product', 'system', TRUE),
    ('cat_feature', 'Feature Request', 'Suggestions for new features or enhancements', 'system', TRUE),
    ('cat_ui', 'User Interface', 'Feedback related to UI/UX design', 'system', TRUE),
    ('cat_performance', 'Performance', 'Feedback related to speed and performance', 'system', TRUE),
    ('cat_support', 'Customer Support', 'Feedback about customer support experiences', 'system', TRUE);

-- Feedback trending issues table to store identified trends
CREATE TABLE IF NOT EXISTS feedback_trending_issues (
    issue_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    count INTEGER NOT NULL,
    negative_percentage FLOAT NOT NULL,
    severity_score FLOAT NOT NULL,
    category VARCHAR(50) NOT NULL,
    identified_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_trending_issues_tenant (tenant_id),
    INDEX idx_trending_issues_severity (severity_score)
);

-- Feedback analysis reports table to store periodic analysis results
CREATE TABLE IF NOT EXISTS feedback_analysis_reports (
    report_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    report_data JSON NOT NULL,
    time_range_days INTEGER NOT NULL,
    generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_feedback_reports_tenant (tenant_id),
    INDEX idx_feedback_reports_generated (generated_at)
);

-- Feature request analysis reports table to store periodic demand analysis
CREATE TABLE IF NOT EXISTS feature_analysis_reports (
    report_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    report_data JSON NOT NULL,
    generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_feature_reports_tenant (tenant_id),
    INDEX idx_feature_reports_generated (generated_at)
);
