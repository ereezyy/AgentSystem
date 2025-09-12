"""
Slack Integration Configuration - AgentSystem Profit Machine
Configuration settings and environment variables for Slack bot integration
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseSettings, Field, validator
from enum import Enum

class SlackEnvironment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class SlackConfig(BaseSettings):
    """
    Slack integration configuration with environment variable support
    """

    # Slack App Credentials
    client_id: str = Field(..., env="SLACK_CLIENT_ID", description="Slack app client ID")
    client_secret: str = Field(..., env="SLACK_CLIENT_SECRET", description="Slack app client secret")
    signing_secret: str = Field(..., env="SLACK_SIGNING_SECRET", description="Slack app signing secret")

    # OAuth Configuration
    oauth_redirect_uri: str = Field(
        default="https://api.agentsystem.com/api/v1/slack/oauth/callback",
        env="SLACK_OAUTH_REDIRECT_URI",
        description="OAuth callback URL"
    )

    # Bot Configuration
    bot_name: str = Field(default="AgentSystem", env="SLACK_BOT_NAME")
    bot_emoji: str = Field(default=":robot_face:", env="SLACK_BOT_EMOJI")
    default_channel: str = Field(default="#general", env="SLACK_DEFAULT_CHANNEL")

    # Feature Flags
    ai_assistance_enabled: bool = Field(default=True, env="SLACK_AI_ASSISTANCE_ENABLED")
    auto_respond_enabled: bool = Field(default=True, env="SLACK_AUTO_RESPOND_ENABLED")
    file_processing_enabled: bool = Field(default=True, env="SLACK_FILE_PROCESSING_ENABLED")
    analytics_enabled: bool = Field(default=True, env="SLACK_ANALYTICS_ENABLED")

    # Rate Limiting
    max_requests_per_minute: int = Field(default=60, env="SLACK_MAX_REQUESTS_PER_MINUTE")
    max_ai_responses_per_hour: int = Field(default=100, env="SLACK_MAX_AI_RESPONSES_PER_HOUR")

    # Response Configuration
    response_delay_seconds: int = Field(default=2, env="SLACK_RESPONSE_DELAY_SECONDS")
    max_response_length: int = Field(default=2000, env="SLACK_MAX_RESPONSE_LENGTH")
    response_timeout_seconds: int = Field(default=30, env="SLACK_RESPONSE_TIMEOUT_SECONDS")

    # AI Configuration
    ai_model: str = Field(default="gpt-4", env="SLACK_AI_MODEL")
    ai_temperature: float = Field(default=0.7, env="SLACK_AI_TEMPERATURE")
    ai_max_tokens: int = Field(default=150, env="SLACK_AI_MAX_TOKENS")

    # Notification Settings
    notification_channels: List[str] = Field(
        default=["#alerts", "#notifications"],
        env="SLACK_NOTIFICATION_CHANNELS"
    )

    # Database Configuration
    db_connection_pool_size: int = Field(default=10, env="SLACK_DB_POOL_SIZE")
    db_connection_timeout: int = Field(default=30, env="SLACK_DB_TIMEOUT")

    # Data Retention
    message_retention_days: int = Field(default=90, env="SLACK_MESSAGE_RETENTION_DAYS")
    analytics_retention_days: int = Field(default=365, env="SLACK_ANALYTICS_RETENTION_DAYS")

    # Security Settings
    encryption_enabled: bool = Field(default=True, env="SLACK_ENCRYPTION_ENABLED")
    audit_logging_enabled: bool = Field(default=True, env="SLACK_AUDIT_LOGGING_ENABLED")
    ip_whitelist: Optional[List[str]] = Field(default=None, env="SLACK_IP_WHITELIST")

    # Performance Settings
    async_processing_enabled: bool = Field(default=True, env="SLACK_ASYNC_PROCESSING_ENABLED")
    batch_processing_size: int = Field(default=100, env="SLACK_BATCH_PROCESSING_SIZE")
    cache_ttl_seconds: int = Field(default=300, env="SLACK_CACHE_TTL_SECONDS")

    # Environment
    environment: SlackEnvironment = Field(
        default=SlackEnvironment.PRODUCTION,
        env="SLACK_ENVIRONMENT"
    )

    # Webhook Configuration
    webhook_verification_enabled: bool = Field(default=True, env="SLACK_WEBHOOK_VERIFICATION_ENABLED")
    webhook_retry_attempts: int = Field(default=3, env="SLACK_WEBHOOK_RETRY_ATTEMPTS")
    webhook_timeout_seconds: int = Field(default=10, env="SLACK_WEBHOOK_TIMEOUT_SECONDS")

    @validator('notification_channels', pre=True)
    def parse_notification_channels(cls, v):
        """Parse notification channels from string or list"""
        if isinstance(v, str):
            return [channel.strip() for channel in v.split(',') if channel.strip()]
        return v

    @validator('ip_whitelist', pre=True)
    def parse_ip_whitelist(cls, v):
        """Parse IP whitelist from string or list"""
        if isinstance(v, str):
            return [ip.strip() for ip in v.split(',') if ip.strip()]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Slack App Scopes Configuration
SLACK_SCOPES = {
    "bot": [
        "app_mentions:read",
        "channels:history",
        "channels:read",
        "channels:join",
        "chat:write",
        "chat:write.public",
        "commands",
        "files:read",
        "groups:history",
        "groups:read",
        "im:history",
        "im:read",
        "mpim:history",
        "mpim:read",
        "reactions:read",
        "reactions:write",
        "team:read",
        "users:read",
        "users:read.email",
        "workflow.steps:execute"
    ],
    "user": [
        "channels:read",
        "groups:read",
        "im:read",
        "mpim:read",
        "team:read",
        "users:read"
    ]
}

# Slack Event Types Configuration
SLACK_EVENT_SUBSCRIPTIONS = [
    "app_mention",
    "message.channels",
    "message.groups",
    "message.im",
    "message.mpim",
    "reaction_added",
    "reaction_removed",
    "file_shared",
    "member_joined_channel",
    "member_left_channel",
    "channel_created",
    "channel_deleted",
    "channel_rename",
    "team_join",
    "user_change"
]

# Slash Commands Configuration
SLACK_SLASH_COMMANDS = {
    "/analyze": {
        "description": "Analyze text for insights and sentiment",
        "usage_hint": "/analyze [your text]",
        "should_escape": False
    },
    "/summarize": {
        "description": "Create a summary of long text or conversation",
        "usage_hint": "/summarize [text or conversation]",
        "should_escape": False
    },
    "/generate": {
        "description": "Generate content based on prompt",
        "usage_hint": "/generate [content prompt]",
        "should_escape": False
    },
    "/translate": {
        "description": "Translate text to different languages",
        "usage_hint": "/translate [text] to [language]",
        "should_escape": False
    },
    "/schedule": {
        "description": "Schedule tasks and reminders",
        "usage_hint": "/schedule [task] at [time]",
        "should_escape": False
    },
    "/report": {
        "description": "Get usage and performance reports",
        "usage_hint": "/report [daily|weekly|monthly]",
        "should_escape": False
    },
    "/help": {
        "description": "Show available commands and help",
        "usage_hint": "/help [command]",
        "should_escape": False
    }
}

# Notification Types Configuration
NOTIFICATION_TEMPLATES = {
    "document_processed": {
        "icon": "📄",
        "color": "#36a64f",
        "title": "Document Processed",
        "template": "{filename} has been successfully processed and analyzed."
    },
    "workflow_completed": {
        "icon": "✅",
        "color": "#36a64f",
        "title": "Workflow Completed",
        "template": "Workflow '{workflow_name}' completed successfully."
    },
    "lead_qualified": {
        "icon": "🎯",
        "color": "#ff9800",
        "title": "New Qualified Lead",
        "template": "New qualified lead: {lead_name} from {company}"
    },
    "deal_won": {
        "icon": "💰",
        "color": "#4caf50",
        "title": "Deal Won!",
        "template": "Congratulations! Deal worth ${amount} has been won."
    },
    "churn_risk": {
        "icon": "⚠️",
        "color": "#f44336",
        "title": "Churn Risk Alert",
        "template": "Customer {customer_name} shows signs of churn risk."
    },
    "system_alert": {
        "icon": "🚨",
        "color": "#f44336",
        "title": "System Alert",
        "template": "System alert: {alert_message}"
    },
    "usage_threshold": {
        "icon": "📊",
        "color": "#2196f3",
        "title": "Usage Threshold",
        "template": "Usage threshold reached: {threshold_type} at {percentage}%"
    }
}

# Logging Configuration
SLACK_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard"
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/slack_integration.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "detailed"
        }
    },
    "loggers": {
        "slack_integration": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False
        }
    }
}

# Function to get configuration instance
def get_slack_config() -> SlackConfig:
    """Get Slack configuration instance"""
    return SlackConfig()

# Function to validate Slack configuration
def validate_slack_config(config: SlackConfig) -> Dict[str, Any]:
    """Validate Slack configuration and return validation results"""
    errors = []
    warnings = []

    # Required fields validation
    if not config.client_id:
        errors.append("SLACK_CLIENT_ID is required")

    if not config.client_secret:
        errors.append("SLACK_CLIENT_SECRET is required")

    if not config.signing_secret:
        errors.append("SLACK_SIGNING_SECRET is required")

    # URL validation
    if not config.oauth_redirect_uri.startswith(('http://', 'https://')):
        errors.append("SLACK_OAUTH_REDIRECT_URI must be a valid URL")

    # Rate limiting validation
    if config.max_requests_per_minute <= 0:
        warnings.append("max_requests_per_minute should be positive")

    if config.max_ai_responses_per_hour <= 0:
        warnings.append("max_ai_responses_per_hour should be positive")

    # Performance validation
    if config.response_timeout_seconds < 5:
        warnings.append("response_timeout_seconds should be at least 5 seconds")

    # Environment-specific validation
    if config.environment == SlackEnvironment.PRODUCTION:
        if config.oauth_redirect_uri.startswith('http://'):
            warnings.append("Production environment should use HTTPS for OAuth redirect URI")

        if not config.encryption_enabled:
            warnings.append("Encryption should be enabled in production")

        if not config.audit_logging_enabled:
            warnings.append("Audit logging should be enabled in production")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "config": config.dict()
    }

# Export configuration instance
slack_config = get_slack_config()

# Export all components
__all__ = [
    "SlackConfig",
    "SlackEnvironment",
    "SLACK_SCOPES",
    "SLACK_EVENT_SUBSCRIPTIONS",
    "SLACK_SLASH_COMMANDS",
    "NOTIFICATION_TEMPLATES",
    "SLACK_LOGGING_CONFIG",
    "get_slack_config",
    "validate_slack_config",
    "slack_config"
]
