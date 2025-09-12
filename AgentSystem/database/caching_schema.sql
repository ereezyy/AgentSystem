-- Intelligent Caching Database Schema - AgentSystem Profit Machine
-- Advanced multi-level caching system for 60% AI cost reduction

-- Create caching schema
CREATE SCHEMA IF NOT EXISTS caching;

-- Cache level enum
CREATE TYPE caching.cache_level AS ENUM (
    'exact_match', 'semantic_match', 'partial_match', 'template_match'
);

-- Cache strategy enum
CREATE TYPE caching.cache_strategy AS ENUM (
    'aggressive', 'balanced', 'conservative', 'custom'
);

-- Main cache entries table
CREATE TABLE caching.cache_entries (
    cache_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    request_hash VARCHAR(64) NOT NULL,
    semantic_hash VARCHAR(32),
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    model_used VARCHAR(200) NOT NULL,
    prompt_template TEXT,
    request_content TEXT NOT NULL,
    response_content TEXT NOT NULL,
    quality_score DECIMAL(5,2) NOT NULL CHECK (quality_score >= 0 AND quality_score <= 100),
    cost_saved DECIMAL(10,6) NOT NULL,
    access_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    ttl_seconds INTEGER NOT NULL,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Cache requests tracking table
CREATE TABLE caching.cache_requests (
    request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    cache_id UUID,
    cache_level caching.cache_level,
    cache_hit BOOLEAN NOT NULL,
    similarity_score DECIMAL(5,4),
    cost_savings DECIMAL(10,6) DEFAULT 0,
    response_time_ms DECIMAL(8,2) NOT NULL,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (cache_id) REFERENCES caching.cache_entries(cache_id) ON DELETE SET NULL
);

-- Hourly cache statistics table
CREATE TABLE caching.cache_stats_hourly (
    stats_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    hour_bucket TIMESTAMPTZ NOT NULL,
    total_requests INTEGER DEFAULT 0,
    cache_hits INTEGER DEFAULT 0,
    exact_hits INTEGER DEFAULT 0,
    semantic_hits INTEGER DEFAULT 0,
    partial_hits INTEGER DEFAULT 0,
    template_hits INTEGER DEFAULT 0,
    cache_misses INTEGER DEFAULT 0,
    hit_rate DECIMAL(5,4) DEFAULT 0,
    avg_response_time_ms DECIMAL(8,2) DEFAULT 0,
    cost_savings DECIMAL(12,6) DEFAULT 0,
    storage_used_mb DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, hour_bucket)
);

-- Cache warming queue table
CREATE TABLE caching.cache_warming_queue (
    warming_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    predicted_content TEXT NOT NULL,
    predicted_model VARCHAR(200) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    expected_cost DECIMAL(10,6) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    priority INTEGER DEFAULT 1,
    attempts INTEGER DEFAULT 0,
    error_details TEXT,
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Cache configuration table
CREATE TABLE caching.cache_config (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    strategy caching.cache_strategy DEFAULT 'balanced',
    exact_match_ttl INTEGER DEFAULT 604800, -- 7 days
    semantic_match_ttl INTEGER DEFAULT 259200, -- 3 days
    partial_match_ttl INTEGER DEFAULT 86400, -- 1 day
    template_match_ttl INTEGER DEFAULT 43200, -- 12 hours
    similarity_threshold DECIMAL(5,4) DEFAULT 0.85,
    max_cache_size_mb INTEGER DEFAULT 1000,
    max_entries_per_tenant INTEGER DEFAULT 10000,
    warming_enabled BOOLEAN DEFAULT true,
    compression_enabled BOOLEAN DEFAULT true,
    auto_cleanup_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id)
);

-- Cache invalidation log table
CREATE TABLE caching.cache_invalidations (
    invalidation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    invalidation_type VARCHAR(50) NOT NULL, -- pattern, model, tags, manual, expiry
    pattern_used TEXT,
    model_filter VARCHAR(200),
    tags_filter TEXT[],
    entries_invalidated INTEGER DEFAULT 0,
    triggered_by VARCHAR(200),
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Cache performance metrics table
CREATE TABLE caching.cache_performance (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    metric_date DATE NOT NULL,
    total_requests INTEGER DEFAULT 0,
    cache_hits INTEGER DEFAULT 0,
    hit_rate DECIMAL(5,4) DEFAULT 0,
    cost_without_cache DECIMAL(12,6) DEFAULT 0,
    cost_with_cache DECIMAL(12,6) DEFAULT 0,
    total_cost_savings DECIMAL(12,6) DEFAULT 0,
    savings_percentage DECIMAL(5,2) DEFAULT 0,
    avg_response_time_ms DECIMAL(8,2) DEFAULT 0,
    storage_efficiency DECIMAL(5,4) DEFAULT 0, -- hit_rate / storage_used
    top_cached_models JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(tenant_id, metric_date)
);

-- Semantic similarity clusters table
CREATE TABLE caching.semantic_clusters (
    cluster_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    cluster_centroid VECTOR(384), -- Embedding vector for cluster center
    cluster_description TEXT,
    cache_entries UUID[] DEFAULT '{}', -- Array of cache_entry IDs in this cluster
    avg_similarity DECIMAL(5,4) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE
);

-- Cache entry embeddings table (for semantic matching)
CREATE TABLE caching.cache_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    embedding_vector VECTOR(384), -- 384-dimensional embedding from sentence-transformers
    embedding_model VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (cache_id) REFERENCES caching.cache_entries(cache_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES billing.tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE(cache_id)
);

-- Create indexes for performance
CREATE INDEX idx_cache_entries_tenant_hash ON caching.cache_entries(tenant_id, request_hash);
CREATE INDEX idx_cache_entries_semantic_hash ON caching.cache_entries(tenant_id, semantic_hash);
CREATE INDEX idx_cache_entries_model ON caching.cache_entries(tenant_id, model_used);
CREATE INDEX idx_cache_entries_tags ON caching.cache_entries USING GIN(tags);
CREATE INDEX idx_cache_entries_template ON caching.cache_entries(tenant_id, prompt_template);
CREATE INDEX idx_cache_entries_expires ON caching.cache_entries(expires_at);
CREATE INDEX idx_cache_entries_active ON caching.cache_entries(is_active);
CREATE INDEX idx_cache_entries_quality ON caching.cache_entries(quality_score DESC);
CREATE INDEX idx_cache_entries_cost ON caching.cache_entries(cost_saved DESC);
CREATE INDEX idx_cache_entries_access ON caching.cache_entries(access_count DESC, last_accessed DESC);

CREATE INDEX idx_cache_requests_tenant ON caching.cache_requests(tenant_id);
CREATE INDEX idx_cache_requests_created_at ON caching.cache_requests(created_at DESC);
CREATE INDEX idx_cache_requests_cache_hit ON caching.cache_requests(cache_hit);
CREATE INDEX idx_cache_requests_cache_level ON caching.cache_requests(cache_level);

CREATE INDEX idx_cache_stats_tenant_hour ON caching.cache_stats_hourly(tenant_id, hour_bucket DESC);

CREATE INDEX idx_cache_warming_tenant_status ON caching.cache_warming_queue(tenant_id, status);
CREATE INDEX idx_cache_warming_priority ON caching.cache_warming_queue(priority DESC, created_at ASC);

CREATE INDEX idx_cache_performance_tenant_date ON caching.cache_performance(tenant_id, metric_date DESC);

CREATE INDEX idx_semantic_clusters_tenant ON caching.semantic_clusters(tenant_id);

-- Vector similarity index for embeddings (requires pgvector extension)
CREATE INDEX idx_cache_embeddings_vector ON caching.cache_embeddings
USING hnsw (embedding_vector vector_cosine_ops);
CREATE INDEX idx_cache_embeddings_tenant ON caching.cache_embeddings(tenant_id);

-- Full-text search index for content matching
CREATE INDEX idx_cache_entries_content_fts ON caching.cache_entries
USING gin(to_tsvector('english', request_content));

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION caching.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_cache_config_updated_at BEFORE UPDATE ON caching.cache_config FOR EACH ROW EXECUTE FUNCTION caching.update_updated_at_column();
CREATE TRIGGER update_semantic_clusters_updated_at BEFORE UPDATE ON caching.semantic_clusters FOR EACH ROW EXECUTE FUNCTION caching.update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE caching.cache_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE caching.cache_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE caching.cache_stats_hourly ENABLE ROW LEVEL SECURITY;
ALTER TABLE caching.cache_warming_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE caching.cache_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE caching.cache_invalidations ENABLE ROW LEVEL SECURITY;
ALTER TABLE caching.cache_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE caching.semantic_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE caching.cache_embeddings ENABLE ROW LEVEL SECURITY;

-- RLS policies for tenant isolation
CREATE POLICY cache_entries_tenant_isolation ON caching.cache_entries
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY cache_requests_tenant_isolation ON caching.cache_requests
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY cache_stats_hourly_tenant_isolation ON caching.cache_stats_hourly
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY cache_warming_queue_tenant_isolation ON caching.cache_warming_queue
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY cache_config_tenant_isolation ON caching.cache_config
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY cache_invalidations_tenant_isolation ON caching.cache_invalidations
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY cache_performance_tenant_isolation ON caching.cache_performance
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY semantic_clusters_tenant_isolation ON caching.semantic_clusters
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY cache_embeddings_tenant_isolation ON caching.cache_embeddings
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Grant permissions
GRANT USAGE ON SCHEMA caching TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA caching TO agentsystem_api;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA caching TO agentsystem_api;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA caching TO agentsystem_api;

-- Create views for analytics and reporting
CREATE VIEW caching.cache_efficiency_summary AS
SELECT
    ce.tenant_id,
    DATE(ce.created_at) as date,
    COUNT(*) as total_entries,
    COUNT(CASE WHEN ce.access_count > 1 THEN 1 END) as reused_entries,
    COUNT(CASE WHEN ce.access_count > 1 THEN 1 END) * 100.0 / COUNT(*) as reuse_rate,
    AVG(ce.quality_score) as avg_quality,
    SUM(ce.cost_saved) as total_cost_saved,
    AVG(ce.access_count) as avg_access_count,
    SUM(LENGTH(ce.response_content)) / 1024.0 / 1024.0 as storage_mb
FROM caching.cache_entries ce
WHERE ce.is_active = true
GROUP BY ce.tenant_id, DATE(ce.created_at);

CREATE VIEW caching.cache_hit_rate_daily AS
SELECT
    cr.tenant_id,
    DATE(cr.created_at) as date,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN cr.cache_hit THEN 1 END) as cache_hits,
    COUNT(CASE WHEN cr.cache_hit THEN 1 END) * 100.0 / COUNT(*) as hit_rate,
    COUNT(CASE WHEN cr.cache_level = 'exact_match' THEN 1 END) as exact_hits,
    COUNT(CASE WHEN cr.cache_level = 'semantic_match' THEN 1 END) as semantic_hits,
    COUNT(CASE WHEN cr.cache_level = 'partial_match' THEN 1 END) as partial_hits,
    COUNT(CASE WHEN cr.cache_level = 'template_match' THEN 1 END) as template_hits,
    AVG(cr.response_time_ms) as avg_response_time,
    SUM(cr.cost_savings) as total_cost_savings
FROM caching.cache_requests cr
GROUP BY cr.tenant_id, DATE(cr.created_at);

CREATE VIEW caching.top_cached_models AS
SELECT
    ce.tenant_id,
    ce.model_used,
    COUNT(*) as cache_entries,
    SUM(ce.access_count) as total_accesses,
    AVG(ce.quality_score) as avg_quality,
    SUM(ce.cost_saved) as total_cost_saved,
    AVG(ce.access_count) as avg_reuse_rate,
    MAX(ce.last_accessed) as last_used
FROM caching.cache_entries ce
WHERE ce.is_active = true
GROUP BY ce.tenant_id, ce.model_used
ORDER BY total_accesses DESC, total_cost_saved DESC;

CREATE VIEW caching.cache_storage_usage AS
SELECT
    ce.tenant_id,
    COUNT(*) as total_entries,
    SUM(LENGTH(ce.request_content)) / 1024.0 / 1024.0 as request_storage_mb,
    SUM(LENGTH(ce.response_content)) / 1024.0 / 1024.0 as response_storage_mb,
    SUM(LENGTH(ce.request_content) + LENGTH(ce.response_content)) / 1024.0 / 1024.0 as total_storage_mb,
    AVG(LENGTH(ce.response_content)) as avg_response_size,
    COUNT(CASE WHEN ce.expires_at < NOW() THEN 1 END) as expired_entries,
    COUNT(CASE WHEN ce.access_count = 1 THEN 1 END) as unused_entries
FROM caching.cache_entries ce
WHERE ce.is_active = true
GROUP BY ce.tenant_id;

-- Grant permissions on views
GRANT SELECT ON caching.cache_efficiency_summary TO agentsystem_api;
GRANT SELECT ON caching.cache_hit_rate_daily TO agentsystem_api;
GRANT SELECT ON caching.top_cached_models TO agentsystem_api;
GRANT SELECT ON caching.cache_storage_usage TO agentsystem_api;

-- Create materialized view for fast dashboard queries
CREATE MATERIALIZED VIEW caching.cache_dashboard_stats AS
SELECT
    ce.tenant_id,
    COUNT(*) as total_entries,
    COUNT(CASE WHEN ce.access_count > 1 THEN 1 END) as reused_entries,
    AVG(ce.access_count) as avg_access_count,
    SUM(ce.cost_saved) as total_cost_saved,
    AVG(ce.quality_score) as avg_quality,
    SUM(LENGTH(ce.response_content)) / 1024.0 / 1024.0 as storage_mb,
    COUNT(DISTINCT ce.model_used) as unique_models,
    MAX(ce.last_accessed) as last_activity,
    COUNT(CASE WHEN ce.created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as entries_last_24h
FROM caching.cache_entries ce
WHERE ce.is_active = true
GROUP BY ce.tenant_id;

-- Create index on materialized view
CREATE INDEX idx_cache_dashboard_stats_tenant ON caching.cache_dashboard_stats(tenant_id);

-- Grant permissions on materialized view
GRANT SELECT ON caching.cache_dashboard_stats TO agentsystem_api;

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION caching.refresh_dashboard_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY caching.cache_dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on function
GRANT EXECUTE ON FUNCTION caching.refresh_dashboard_stats() TO agentsystem_api;

-- Function to calculate cache efficiency
CREATE OR REPLACE FUNCTION caching.calculate_cache_efficiency(p_tenant_id UUID, p_days INTEGER DEFAULT 7)
RETURNS TABLE (
    efficiency_score DECIMAL(5,2),
    hit_rate DECIMAL(5,2),
    cost_savings_percent DECIMAL(5,2),
    storage_efficiency DECIMAL(5,2),
    recommendation TEXT
) AS $$
DECLARE
    v_hit_rate DECIMAL(5,2);
    v_cost_savings DECIMAL(12,6);
    v_storage_mb DECIMAL(10,2);
    v_total_requests INTEGER;
    v_efficiency_score DECIMAL(5,2);
    v_recommendation TEXT;
BEGIN
    -- Get cache statistics
    SELECT
        COUNT(CASE WHEN cr.cache_hit THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0),
        SUM(cr.cost_savings),
        COUNT(*)
    INTO v_hit_rate, v_cost_savings, v_total_requests
    FROM caching.cache_requests cr
    WHERE cr.tenant_id = p_tenant_id
    AND cr.created_at > NOW() - INTERVAL '%s days' % p_days;

    -- Get storage usage
    SELECT SUM(LENGTH(ce.response_content)) / 1024.0 / 1024.0
    INTO v_storage_mb
    FROM caching.cache_entries ce
    WHERE ce.tenant_id = p_tenant_id AND ce.is_active = true;

    -- Calculate efficiency score
    v_efficiency_score := COALESCE(v_hit_rate, 0) * 0.4 +
                         LEAST(COALESCE(v_cost_savings, 0) / 100, 100) * 0.4 +
                         LEAST(100 - COALESCE(v_storage_mb, 0) / 10, 100) * 0.2;

    -- Generate recommendation
    IF v_hit_rate < 30 THEN
        v_recommendation := 'Low hit rate detected. Consider enabling semantic matching and cache warming.';
    ELSIF v_storage_mb > 500 THEN
        v_recommendation := 'High storage usage. Consider reducing TTL or enabling more aggressive cleanup.';
    ELSIF v_hit_rate > 70 THEN
        v_recommendation := 'Excellent cache performance. Consider expanding cache size for more savings.';
    ELSE
        v_recommendation := 'Cache performance is good. Monitor and optimize based on usage patterns.';
    END IF;

    RETURN QUERY SELECT
        v_efficiency_score,
        COALESCE(v_hit_rate, 0),
        COALESCE(v_cost_savings / NULLIF(v_total_requests, 0) * 100, 0),
        COALESCE(v_hit_rate / NULLIF(v_storage_mb, 0) * 10, 0),
        v_recommendation;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on efficiency function
GRANT EXECUTE ON FUNCTION caching.calculate_cache_efficiency TO agentsystem_api;

-- Insert default cache configurations for existing tenants
INSERT INTO caching.cache_config (tenant_id, strategy)
SELECT tenant_id, 'balanced'
FROM billing.tenants
WHERE tenant_id NOT IN (SELECT tenant_id FROM caching.cache_config);
