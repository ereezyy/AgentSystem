/*
  # Microcopy Tracking Schema

  1. New Tables
    - `microcopy_variants`
      - Stores different versions of microcopy for A/B testing
    - `microcopy_interactions`
      - Tracks user interactions with different microcopy
    - `microcopy_effectiveness`
      - Aggregated metrics on microcopy performance

  2. Security
    - Enable RLS on all tables
    - Admin-only access to microcopy management
    - Tenant-specific interaction tracking
*/

-- Microcopy variants for A/B testing
CREATE TABLE IF NOT EXISTS microcopy_variants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  key VARCHAR(100) NOT NULL,
  variant_name VARCHAR(50) NOT NULL,
  context VARCHAR(50) NOT NULL,
  type VARCHAR(30) NOT NULL,
  content JSONB NOT NULL,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(key, variant_name)
);

CREATE INDEX IF NOT EXISTS idx_microcopy_variants_key ON microcopy_variants(key);
CREATE INDEX IF NOT EXISTS idx_microcopy_variants_active ON microcopy_variants(is_active) WHERE is_active = true;

COMMENT ON TABLE microcopy_variants IS 'Stores different versions of UI text for A/B testing';
COMMENT ON COLUMN microcopy_variants.key IS 'Unique identifier for the microcopy location';
COMMENT ON COLUMN microcopy_variants.variant_name IS 'Name of this variant (e.g., control, variant_a)';
COMMENT ON COLUMN microcopy_variants.context IS 'Where this appears (e.g., button, error, tooltip)';
COMMENT ON COLUMN microcopy_variants.type IS 'Type of microcopy (e.g., cta, error, success, help)';
COMMENT ON COLUMN microcopy_variants.content IS 'JSON containing text, tooltip, aria labels, etc.';

-- User interactions with microcopy
CREATE TABLE IF NOT EXISTS microcopy_interactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  variant_id UUID REFERENCES microcopy_variants(id) ON DELETE CASCADE,
  tenant_id UUID NOT NULL,
  user_id UUID NOT NULL,
  session_id VARCHAR(100) NOT NULL,
  interaction_type VARCHAR(30) NOT NULL,
  outcome VARCHAR(30),
  metadata JSONB DEFAULT '{}',
  timestamp TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_microcopy_interactions_variant ON microcopy_interactions(variant_id);
CREATE INDEX IF NOT EXISTS idx_microcopy_interactions_tenant ON microcopy_interactions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_microcopy_interactions_timestamp ON microcopy_interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_microcopy_interactions_outcome ON microcopy_interactions(outcome);

COMMENT ON TABLE microcopy_interactions IS 'Tracks user interactions with different microcopy variants';
COMMENT ON COLUMN microcopy_interactions.interaction_type IS 'Type of interaction (view, click, complete, abandon, error)';
COMMENT ON COLUMN microcopy_interactions.outcome IS 'Result of interaction (success, failure, cancelled)';

-- Aggregated effectiveness metrics
CREATE TABLE IF NOT EXISTS microcopy_effectiveness (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  variant_id UUID REFERENCES microcopy_variants(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  total_views INTEGER DEFAULT 0,
  total_clicks INTEGER DEFAULT 0,
  total_completions INTEGER DEFAULT 0,
  total_errors INTEGER DEFAULT 0,
  total_abandons INTEGER DEFAULT 0,
  avg_time_to_action NUMERIC(10, 2),
  click_through_rate NUMERIC(5, 4),
  completion_rate NUMERIC(5, 4),
  error_rate NUMERIC(5, 4),
  calculated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(variant_id, date)
);

CREATE INDEX IF NOT EXISTS idx_microcopy_effectiveness_variant ON microcopy_effectiveness(variant_id);
CREATE INDEX IF NOT EXISTS idx_microcopy_effectiveness_date ON microcopy_effectiveness(date);

COMMENT ON TABLE microcopy_effectiveness IS 'Daily aggregated metrics for microcopy performance';

-- Function to calculate effectiveness metrics
CREATE OR REPLACE FUNCTION calculate_microcopy_effectiveness(
  p_variant_id UUID,
  p_date DATE
) RETURNS void AS $$
BEGIN
  INSERT INTO microcopy_effectiveness (
    variant_id,
    date,
    total_views,
    total_clicks,
    total_completions,
    total_errors,
    total_abandons,
    avg_time_to_action,
    click_through_rate,
    completion_rate,
    error_rate
  )
  SELECT
    p_variant_id,
    p_date,
    COUNT(*) FILTER (WHERE interaction_type = 'view'),
    COUNT(*) FILTER (WHERE interaction_type = 'click'),
    COUNT(*) FILTER (WHERE outcome = 'success'),
    COUNT(*) FILTER (WHERE outcome = 'error'),
    COUNT(*) FILTER (WHERE outcome = 'abandoned'),
    AVG(EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (PARTITION BY session_id ORDER BY timestamp)))) FILTER (WHERE interaction_type = 'click'),
    CASE
      WHEN COUNT(*) FILTER (WHERE interaction_type = 'view') > 0
      THEN COUNT(*) FILTER (WHERE interaction_type = 'click')::numeric / COUNT(*) FILTER (WHERE interaction_type = 'view')
      ELSE 0
    END,
    CASE
      WHEN COUNT(*) FILTER (WHERE interaction_type = 'click') > 0
      THEN COUNT(*) FILTER (WHERE outcome = 'success')::numeric / COUNT(*) FILTER (WHERE interaction_type = 'click')
      ELSE 0
    END,
    CASE
      WHEN COUNT(*) > 0
      THEN COUNT(*) FILTER (WHERE outcome = 'error')::numeric / COUNT(*)
      ELSE 0
    END
  FROM microcopy_interactions
  WHERE variant_id = p_variant_id
    AND DATE(timestamp) = p_date
  ON CONFLICT (variant_id, date)
  DO UPDATE SET
    total_views = EXCLUDED.total_views,
    total_clicks = EXCLUDED.total_clicks,
    total_completions = EXCLUDED.total_completions,
    total_errors = EXCLUDED.total_errors,
    total_abandons = EXCLUDED.total_abandons,
    avg_time_to_action = EXCLUDED.avg_time_to_action,
    click_through_rate = EXCLUDED.click_through_rate,
    completion_rate = EXCLUDED.completion_rate,
    error_rate = EXCLUDED.error_rate,
    calculated_at = now();
END;
$$ LANGUAGE plpgsql;

-- Enable Row Level Security
ALTER TABLE microcopy_variants ENABLE ROW LEVEL SECURITY;
ALTER TABLE microcopy_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE microcopy_effectiveness ENABLE ROW LEVEL SECURITY;

-- Policies for microcopy_variants
CREATE POLICY "Admins can manage microcopy variants"
  ON microcopy_variants
  FOR ALL
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM auth.users
      WHERE auth.uid() = id
      AND raw_app_meta_data->>'role' = 'admin'
    )
  );

CREATE POLICY "All authenticated users can view active variants"
  ON microcopy_variants
  FOR SELECT
  TO authenticated
  USING (is_active = true);

-- Policies for microcopy_interactions
CREATE POLICY "Users can insert their own interactions"
  ON microcopy_interactions
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view their own interactions"
  ON microcopy_interactions
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Admins can view all interactions"
  ON microcopy_interactions
  FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM auth.users
      WHERE auth.uid() = id
      AND raw_app_meta_data->>'role' = 'admin'
    )
  );

-- Policies for microcopy_effectiveness
CREATE POLICY "Admins can view effectiveness metrics"
  ON microcopy_effectiveness
  FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM auth.users
      WHERE auth.uid() = id
      AND raw_app_meta_data->>'role' = 'admin'
    )
  );

-- Insert sample microcopy variants
INSERT INTO microcopy_variants (key, variant_name, context, type, content) VALUES
-- Task creation button
('task_create_button', 'control', 'button', 'cta',
  '{"text": "Create New Task", "tooltip": "Create a new task for AI agents"}'::jsonb),
('task_create_button', 'variant_a', 'button', 'cta',
  '{"text": "+ New Task", "tooltip": "Start a new task"}'::jsonb),
('task_create_button', 'variant_b', 'button', 'cta',
  '{"text": "Start a Task", "tooltip": "Get your AI agents working on something new"}'::jsonb),

-- Task submission button
('task_submit_button', 'control', 'button', 'cta',
  '{"text": "Submit Task"}'::jsonb),
('task_submit_button', 'variant_a', 'button', 'cta',
  '{"text": "Launch Task Now", "icon": "🚀"}'::jsonb),

-- Error messages
('error_task_not_found', 'control', 'error', 'error',
  '{"message": "Task not found", "help": null}'::jsonb),
('error_task_not_found', 'variant_a', 'error', 'error',
  '{"message": "We couldn''t find that task", "help": "It may have been deleted or the link is incorrect. Try refreshing the page.", "action": {"label": "View All Tasks", "url": "/tasks"}}'::jsonb),

-- Success messages
('success_task_created', 'control', 'success', 'success',
  '{"message": "Task created successfully"}'::jsonb),
('success_task_created', 'variant_a', 'success', 'success',
  '{"title": "Task Created Successfully", "message": "Your AI agent will start working on it soon", "action": {"label": "Track Progress", "icon": "→"}}'::jsonb),

-- Form labels
('form_task_name', 'control', 'form', 'label',
  '{"label": "Task Name", "placeholder": null, "hint": null}'::jsonb),
('form_task_name', 'variant_a', 'form', 'label',
  '{"label": "What should we call this task?", "placeholder": "e.g., Analyze customer feedback from Q1", "hint": "Keep it short and descriptive (max 100 characters)"}'::jsonb),

-- Empty states
('empty_tasks', 'control', 'empty', 'help',
  '{"message": "No tasks"}'::jsonb),
('empty_tasks', 'variant_a', 'empty', 'help',
  '{"icon": "📋", "title": "No tasks yet", "message": "Create your first task to get your AI agents working", "action": {"label": "Create Your First Task", "onclick": "openTaskModal()"}}'::jsonb)

ON CONFLICT (key, variant_name) DO NOTHING;
