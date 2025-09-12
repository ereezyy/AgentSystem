-- Partner Channel and Reseller Program Database Schema - AgentSystem Profit Machine
-- Comprehensive partner ecosystem for revenue sharing, white-label solutions, and market expansion

-- Create partners schema
CREATE SCHEMA IF NOT EXISTS partners;

-- Partners table
CREATE TABLE partners.partners (
    partner_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name VARCHAR(255) NOT NULL,
    contact_name VARCHAR(255) NOT NULL,
    contact_email VARCHAR(255) NOT NULL UNIQUE,
    contact_phone VARCHAR(50),
    partner_type VARCHAR(50) NOT NULL CHECK (partner_type IN ('reseller', 'system_integrator', 'technology_partner', 'referral_partner', 'managed_service_provider', 'distributor')),
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'suspended', 'terminated', 'under_review')),
    certification_level VARCHAR(50) DEFAULT 'bronze' CHECK (certification_level IN ('bronze', 'silver', 'gold', 'platinum', 'elite')),
    onboarding_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    contract_start_date TIMESTAMP WITH TIME ZONE,
    contract_end_date TIMESTAMP WITH TIME ZONE,
    territory JSONB DEFAULT '[]',
    specializations JSONB DEFAULT '[]',
    white_label_enabled BOOLEAN DEFAULT FALSE,
    custom_branding JSONB,
    commission_structure JSONB NOT NULL DEFAULT '{}',
    payment_terms JSONB DEFAULT '{"frequency": "monthly", "net_days": 30}',
    minimum_commitment JSONB,
    performance_metrics JSONB DEFAULT '{}',
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Company details
    company_size VARCHAR(50),
    industry VARCHAR(100),
    website_url VARCHAR(500),
    linkedin_url VARCHAR(500),
    annual_revenue_range VARCHAR(50),

    -- Contact details
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_province VARCHAR(100),
    postal_code VARCHAR(20),
    country VARCHAR(100),

    -- Financial details
    tax_id VARCHAR(100),
    billing_contact_name VARCHAR(255),
    billing_contact_email VARCHAR(255),
    payment_method VARCHAR(50) DEFAULT 'ach',
    bank_account_details JSONB,

    -- Additional metadata
    lead_source VARCHAR(100),
    referring_partner_id UUID REFERENCES partners.partners(partner_id),
    notes TEXT,
    tags JSONB DEFAULT '[]'
);

-- Partner deals table
CREATE TABLE partners.partner_deals (
    deal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_id UUID NOT NULL REFERENCES partners.partners(partner_id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES tenants(tenant_id),
    deal_name VARCHAR(255) NOT NULL,
    deal_value DECIMAL(12,2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    deal_stage VARCHAR(50) NOT NULL DEFAULT 'qualified' CHECK (deal_stage IN ('lead', 'qualified', 'proposal', 'negotiation', 'closed_won', 'closed_lost')),
    probability DECIMAL(5,2) DEFAULT 50.0 CHECK (probability >= 0 AND probability <= 100),
    expected_close_date TIMESTAMP WITH TIME ZONE NOT NULL,
    actual_close_date TIMESTAMP WITH TIME ZONE,
    commission_amount DECIMAL(10,2) NOT NULL DEFAULT 0,
    commission_paid BOOLEAN DEFAULT FALSE,
    commission_paid_date TIMESTAMP WITH TIME ZONE,
    deal_source VARCHAR(100) DEFAULT 'partner_generated',
    product_mix JSONB DEFAULT '{}',
    customer_segment VARCHAR(100) DEFAULT 'enterprise',
    deal_notes TEXT,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Deal tracking
    lead_qualification_date TIMESTAMP WITH TIME ZONE,
    proposal_sent_date TIMESTAMP WITH TIME ZONE,
    contract_sent_date TIMESTAMP WITH TIME ZONE,
    signed_contract_date TIMESTAMP WITH TIME ZONE,

    -- Competition and pricing
    competitors JSONB DEFAULT '[]',
    win_loss_reason TEXT,
    discount_percentage DECIMAL(5,2) DEFAULT 0,
    original_list_price DECIMAL(12,2),

    -- Customer information
    customer_company_name VARCHAR(255),
    customer_industry VARCHAR(100),
    customer_size VARCHAR(50),
    decision_maker_name VARCHAR(255),
    decision_maker_email VARCHAR(255),
    technical_contact_name VARCHAR(255),
    technical_contact_email VARCHAR(255)
);

-- Partner training table
CREATE TABLE partners.partner_training (
    training_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_id UUID NOT NULL REFERENCES partners.partners(partner_id) ON DELETE CASCADE,
    training_name VARCHAR(255) NOT NULL,
    training_type VARCHAR(100) NOT NULL CHECK (training_type IN ('onboarding', 'certification', 'product_update', 'technical', 'sales', 'marketing')),
    training_module VARCHAR(255) NOT NULL,
    completion_status VARCHAR(50) DEFAULT 'enrolled' CHECK (completion_status IN ('enrolled', 'in_progress', 'completed', 'failed', 'expired')),
    start_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completion_date TIMESTAMP WITH TIME ZONE,
    score DECIMAL(5,2),
    passing_score DECIMAL(5,2) DEFAULT 80.0,
    certification_earned VARCHAR(255),
    expiry_date TIMESTAMP WITH TIME ZONE,
    trainer_name VARCHAR(255),
    training_materials JSONB DEFAULT '[]',
    assessment_results JSONB,

    -- Training details
    duration_hours INTEGER DEFAULT 0,
    delivery_method VARCHAR(50) DEFAULT 'online' CHECK (delivery_method IN ('online', 'in_person', 'virtual_classroom', 'self_paced')),
    prerequisites JSONB DEFAULT '[]',
    learning_objectives JSONB DEFAULT '[]',

    -- Scheduling
    scheduled_date TIMESTAMP WITH TIME ZONE,
    timezone VARCHAR(100) DEFAULT 'UTC',
    session_url VARCHAR(500),
    recording_url VARCHAR(500),

    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Marketing campaigns table
CREATE TABLE partners.marketing_campaigns (
    campaign_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_id UUID NOT NULL REFERENCES partners.partners(partner_id) ON DELETE CASCADE,
    campaign_name VARCHAR(255) NOT NULL,
    campaign_type VARCHAR(100) NOT NULL CHECK (campaign_type IN ('co_marketing', 'lead_generation', 'event', 'webinar', 'content_marketing', 'paid_advertising', 'email_marketing')),
    start_date TIMESTAMP WITH TIME ZONE NOT NULL,
    end_date TIMESTAMP WITH TIME ZONE NOT NULL,
    budget_allocated DECIMAL(10,2) DEFAULT 0,
    budget_spent DECIMAL(10,2) DEFAULT 0,
    leads_generated INTEGER DEFAULT 0,
    qualified_leads INTEGER DEFAULT 0,
    opportunities_created INTEGER DEFAULT 0,
    deals_closed INTEGER DEFAULT 0,
    revenue_generated DECIMAL(12,2) DEFAULT 0,
    roi_percentage DECIMAL(8,2) DEFAULT 0,
    campaign_materials JSONB DEFAULT '[]',
    target_audience JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',

    -- Campaign details
    campaign_description TEXT,
    campaign_objectives JSONB DEFAULT '[]',
    success_metrics JSONB DEFAULT '[]',
    channels JSONB DEFAULT '[]',
    geography JSONB DEFAULT '[]',

    -- Approval and status
    approval_status VARCHAR(50) DEFAULT 'draft' CHECK (approval_status IN ('draft', 'pending_approval', 'approved', 'active', 'paused', 'completed', 'cancelled')),
    approved_by VARCHAR(255),
    approval_date TIMESTAMP WITH TIME ZONE,

    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Commission payouts table
CREATE TABLE partners.commission_payouts (
    payout_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_id UUID NOT NULL REFERENCES partners.partners(partner_id) ON DELETE CASCADE,
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    total_commission DECIMAL(12,2) NOT NULL,
    deals_count INTEGER NOT NULL DEFAULT 0,
    payment_method VARCHAR(50) DEFAULT 'ach' CHECK (payment_method IN ('ach', 'wire', 'check', 'paypal', 'stripe')),
    payment_reference VARCHAR(255),
    payout_date TIMESTAMP WITH TIME ZONE NOT NULL,
    tax_withholding DECIMAL(10,2) DEFAULT 0,
    net_amount DECIMAL(12,2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    payout_details JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'paid', 'failed', 'cancelled')),

    -- Payment processing
    payment_processor VARCHAR(100),
    transaction_id VARCHAR(255),
    processing_fee DECIMAL(8,2) DEFAULT 0,
    exchange_rate DECIMAL(10,6) DEFAULT 1.0,

    -- Reconciliation
    reconciliation_status VARCHAR(50) DEFAULT 'pending' CHECK (reconciliation_status IN ('pending', 'reconciled', 'disputed')),
    reconciliation_date TIMESTAMP WITH TIME ZONE,
    reconciliation_notes TEXT,

    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Partner portal access table
CREATE TABLE partners.portal_access (
    access_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_id UUID NOT NULL REFERENCES partners.partners(partner_id) ON DELETE CASCADE,
    user_email VARCHAR(255) NOT NULL,
    user_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'partner_user' CHECK (role IN ('partner_admin', 'partner_user', 'partner_readonly')),
    permissions JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE,
    login_count INTEGER DEFAULT 0,
    password_hash VARCHAR(255),
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    api_key VARCHAR(255) UNIQUE,
    api_key_created TIMESTAMP WITH TIME ZONE,
    api_key_expires TIMESTAMP WITH TIME ZONE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(partner_id, user_email)
);

-- Partner certifications table
CREATE TABLE partners.certifications (
    certification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_id UUID NOT NULL REFERENCES partners.partners(partner_id) ON DELETE CASCADE,
    certification_name VARCHAR(255) NOT NULL,
    certification_type VARCHAR(100) NOT NULL,
    level VARCHAR(50) NOT NULL,
    issued_date TIMESTAMP WITH TIME ZONE NOT NULL,
    expiry_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'expired', 'revoked', 'suspended')),
    issuing_authority VARCHAR(255),
    certificate_number VARCHAR(255) UNIQUE,
    verification_url VARCHAR(500),
    requirements_met JSONB DEFAULT '[]',
    continuing_education_credits INTEGER DEFAULT 0,
    renewal_requirements JSONB DEFAULT '[]',
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Partner performance metrics table
CREATE TABLE partners.performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_id UUID NOT NULL REFERENCES partners.partners(partner_id) ON DELETE CASCADE,
    metric_date DATE NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(50),
    period_type VARCHAR(50) DEFAULT 'daily' CHECK (period_type IN ('daily', 'weekly', 'monthly', 'quarterly', 'annual')),
    benchmark_value DECIMAL(15,4),
    target_value DECIMAL(15,4),
    performance_rating VARCHAR(50),
    notes TEXT,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(partner_id, metric_date, metric_type, period_type)
);

-- Partner territories table
CREATE TABLE partners.territories (
    territory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_id UUID NOT NULL REFERENCES partners.partners(partner_id) ON DELETE CASCADE,
    territory_name VARCHAR(255) NOT NULL,
    territory_type VARCHAR(50) NOT NULL CHECK (territory_type IN ('country', 'state', 'region', 'city', 'industry', 'account_list')),
    territory_value VARCHAR(255) NOT NULL,
    is_exclusive BOOLEAN DEFAULT FALSE,
    start_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_date TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    quota_amount DECIMAL(12,2),
    quota_period VARCHAR(50) DEFAULT 'annual',
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(partner_id, territory_type, territory_value)
);

-- White label configurations table
CREATE TABLE partners.white_label_configs (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_id UUID NOT NULL REFERENCES partners.partners(partner_id) ON DELETE CASCADE,
    config_name VARCHAR(255) NOT NULL,
    brand_name VARCHAR(255) NOT NULL,
    logo_url VARCHAR(500),
    favicon_url VARCHAR(500),
    primary_color VARCHAR(10),
    secondary_color VARCHAR(10),
    accent_color VARCHAR(10),
    font_family VARCHAR(100),
    custom_domain VARCHAR(255),
    ssl_certificate_url VARCHAR(500),
    custom_css TEXT,
    email_templates JSONB DEFAULT '{}',
    portal_customizations JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT FALSE,
    activation_date TIMESTAMP WITH TIME ZONE,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_partners_status ON partners.partners(status);
CREATE INDEX idx_partners_type ON partners.partners(partner_type);
CREATE INDEX idx_partners_certification ON partners.partners(certification_level);
CREATE INDEX idx_partners_territory ON partners.partners USING GIN(territory);
CREATE INDEX idx_partners_created_date ON partners.partners(created_date);

CREATE INDEX idx_partner_deals_partner ON partners.partner_deals(partner_id);
CREATE INDEX idx_partner_deals_customer ON partners.partner_deals(customer_id);
CREATE INDEX idx_partner_deals_stage ON partners.partner_deals(deal_stage);
CREATE INDEX idx_partner_deals_close_date ON partners.partner_deals(expected_close_date);
CREATE INDEX idx_partner_deals_value ON partners.partner_deals(deal_value);

CREATE INDEX idx_partner_training_partner ON partners.partner_training(partner_id);
CREATE INDEX idx_partner_training_type ON partners.partner_training(training_type);
CREATE INDEX idx_partner_training_status ON partners.partner_training(completion_status);
CREATE INDEX idx_partner_training_date ON partners.partner_training(start_date);

CREATE INDEX idx_marketing_campaigns_partner ON partners.marketing_campaigns(partner_id);
CREATE INDEX idx_marketing_campaigns_type ON partners.marketing_campaigns(campaign_type);
CREATE INDEX idx_marketing_campaigns_date ON partners.marketing_campaigns(start_date);
CREATE INDEX idx_marketing_campaigns_status ON partners.marketing_campaigns(approval_status);

CREATE INDEX idx_commission_payouts_partner ON partners.commission_payouts(partner_id);
CREATE INDEX idx_commission_payouts_date ON partners.commission_payouts(payout_date);
CREATE INDEX idx_commission_payouts_status ON partners.commission_payouts(status);
CREATE INDEX idx_commission_payouts_period ON partners.commission_payouts(period_start, period_end);

CREATE INDEX idx_portal_access_partner ON partners.portal_access(partner_id);
CREATE INDEX idx_portal_access_email ON partners.portal_access(user_email);
CREATE INDEX idx_portal_access_active ON partners.portal_access(is_active);

CREATE INDEX idx_performance_metrics_partner ON partners.performance_metrics(partner_id);
CREATE INDEX idx_performance_metrics_date ON partners.performance_metrics(metric_date);
CREATE INDEX idx_performance_metrics_type ON partners.performance_metrics(metric_type);

-- Create composite indexes for common queries
CREATE INDEX idx_partner_deals_partner_stage ON partners.partner_deals(partner_id, deal_stage);
CREATE INDEX idx_partner_deals_partner_date ON partners.partner_deals(partner_id, expected_close_date);
CREATE INDEX idx_commission_payouts_partner_period ON partners.commission_payouts(partner_id, period_start, period_end);

-- Create functions for partner management
CREATE OR REPLACE FUNCTION partners.update_partner_performance()
RETURNS TRIGGER AS $$
BEGIN
    -- Update partner performance metrics when deals are closed
    IF NEW.deal_stage = 'closed_won' AND OLD.deal_stage != 'closed_won' THEN
        -- Update total revenue
        INSERT INTO partners.performance_metrics (
            partner_id, metric_date, metric_type, metric_value, metric_unit
        ) VALUES (
            NEW.partner_id, CURRENT_DATE, 'revenue', NEW.deal_value, 'USD'
        ) ON CONFLICT (partner_id, metric_date, metric_type, period_type)
        DO UPDATE SET
            metric_value = partners.performance_metrics.metric_value + NEW.deal_value,
            updated_date = CURRENT_TIMESTAMP;

        -- Update deal count
        INSERT INTO partners.performance_metrics (
            partner_id, metric_date, metric_type, metric_value, metric_unit
        ) VALUES (
            NEW.partner_id, CURRENT_DATE, 'deals_closed', 1, 'count'
        ) ON CONFLICT (partner_id, metric_date, metric_type, period_type)
        DO UPDATE SET
            metric_value = partners.performance_metrics.metric_value + 1,
            updated_date = CURRENT_TIMESTAMP;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for performance updates
CREATE TRIGGER trigger_update_partner_performance
    AFTER UPDATE ON partners.partner_deals
    FOR EACH ROW
    EXECUTE FUNCTION partners.update_partner_performance();

-- Create function for certification level upgrades
CREATE OR REPLACE FUNCTION partners.check_certification_upgrade()
RETURNS TRIGGER AS $$
DECLARE
    total_revenue DECIMAL(12,2);
    total_deals INTEGER;
    partner_record RECORD;
BEGIN
    -- Get partner performance metrics
    SELECT
        COALESCE(SUM(deal_value), 0) as revenue,
        COUNT(*) as deals
    INTO total_revenue, total_deals
    FROM partners.partner_deals
    WHERE partner_id = NEW.partner_id
    AND deal_stage = 'closed_won'
    AND actual_close_date >= CURRENT_DATE - INTERVAL '12 months';

    -- Get current partner info
    SELECT * INTO partner_record
    FROM partners.partners
    WHERE partner_id = NEW.partner_id;

    -- Determine new certification level
    DECLARE new_level VARCHAR(50);
    BEGIN
        IF total_revenue >= 1000000 AND total_deals >= 50 THEN
            new_level := 'elite';
        ELSIF total_revenue >= 500000 AND total_deals >= 25 THEN
            new_level := 'platinum';
        ELSIF total_revenue >= 250000 AND total_deals >= 15 THEN
            new_level := 'gold';
        ELSIF total_revenue >= 100000 AND total_deals >= 10 THEN
            new_level := 'silver';
        ELSE
            new_level := 'bronze';
        END IF;

        -- Update certification level if changed
        IF partner_record.certification_level != new_level THEN
            UPDATE partners.partners
            SET
                certification_level = new_level,
                updated_date = CURRENT_TIMESTAMP
            WHERE partner_id = NEW.partner_id;

            -- Log certification change
            INSERT INTO partners.certifications (
                partner_id, certification_name, certification_type, level,
                issued_date, issuing_authority
            ) VALUES (
                NEW.partner_id,
                'Partner Performance Level',
                'performance_based',
                new_level,
                CURRENT_TIMESTAMP,
                'AgentSystem Automated System'
            );
        END IF;
    END;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for certification upgrades
CREATE TRIGGER trigger_check_certification_upgrade
    AFTER UPDATE ON partners.partner_deals
    FOR EACH ROW
    WHEN (NEW.deal_stage = 'closed_won' AND OLD.deal_stage != 'closed_won')
    EXECUTE FUNCTION partners.check_certification_upgrade();

-- Create function for commission calculations
CREATE OR REPLACE FUNCTION partners.calculate_commission_amount()
RETURNS TRIGGER AS $$
DECLARE
    partner_record RECORD;
    commission_rate DECIMAL(5,2);
    calculated_commission DECIMAL(10,2);
BEGIN
    -- Get partner commission structure
    SELECT * INTO partner_record
    FROM partners.partners
    WHERE partner_id = NEW.partner_id;

    -- Calculate commission based on structure
    IF partner_record.commission_structure->>'type' = 'percentage' THEN
        commission_rate := COALESCE((partner_record.commission_structure->>'base_rate')::DECIMAL, 15.0);
        calculated_commission := NEW.deal_value * (commission_rate / 100);
    ELSIF partner_record.commission_structure->>'type' = 'flat_fee' THEN
        calculated_commission := COALESCE((partner_record.commission_structure->>'referral_bonus')::DECIMAL, 1000.0);
    ELSE
        -- Default to 15% commission
        calculated_commission := NEW.deal_value * 0.15;
    END IF;

    -- Update the commission amount
    NEW.commission_amount := calculated_commission;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for commission calculations
CREATE TRIGGER trigger_calculate_commission
    BEFORE INSERT OR UPDATE ON partners.partner_deals
    FOR EACH ROW
    EXECUTE FUNCTION partners.calculate_commission_amount();

-- Insert default partner types and templates
INSERT INTO partners.partners (
    company_name, contact_name, contact_email, partner_type, status,
    certification_level, territory, specializations, commission_structure
) VALUES
(
    'Demo Technology Partners', 'John Smith', 'demo@techpartners.com', 'technology_partner', 'active',
    'gold', '["North America", "Europe"]', '["Healthcare", "Finance"]',
    '{"type": "percentage", "base_rate": 20.0, "tiers": [{"min_revenue": 0, "max_revenue": 100000, "rate": 20.0}, {"min_revenue": 100000, "max_revenue": 500000, "rate": 25.0}]}'
),
(
    'Global Systems Integrator', 'Sarah Johnson', 'sarah@globalsystems.com', 'system_integrator', 'active',
    'platinum', '["Global"]', '["Enterprise", "Government"]',
    '{"type": "percentage", "base_rate": 25.0, "recurring_rate": 10.0}'
),
(
    'Regional Reseller Network', 'Mike Chen', 'mike@regionalreseller.com', 'reseller', 'active',
    'silver', '["Asia Pacific"]', '["SMB", "Mid-market"]',
    '{"type": "tiered", "tiers": [{"min_revenue": 0, "max_revenue": 50000, "rate": 15.0}, {"min_revenue": 50000, "max_revenue": 250000, "rate": 20.0}]}'
);

-- Grant permissions
GRANT USAGE ON SCHEMA partners TO agentsystem_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA partners TO agentsystem_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA partners TO agentsystem_app;
