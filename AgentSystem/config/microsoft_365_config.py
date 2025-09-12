"""
Microsoft 365 Integration Configuration - AgentSystem Profit Machine
Configuration settings and environment variables for Microsoft 365 integration
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseSettings, Field, validator
from enum import Enum

class Microsoft365Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Microsoft365Config(BaseSettings):
    """
    Microsoft 365 integration configuration with environment variable support
    """

    # Microsoft 365 App Credentials
    client_id: str = Field(..., env="M365_CLIENT_ID", description="Microsoft 365 app client ID")
    client_secret: str = Field(..., env="M365_CLIENT_SECRET", description="Microsoft 365 app client secret")
    tenant_id: str = Field(default="common", env="M365_TENANT_ID", description="Microsoft 365 tenant ID (common for multi-tenant)")

    # OAuth Configuration
    oauth_redirect_uri: str = Field(
        default="https://api.agentsystem.com/api/v1/microsoft365/oauth/callback",
        env="M365_OAUTH_REDIRECT_URI",
        description="OAuth callback URL"
    )

    # Microsoft Graph API Configuration
    graph_api_version: str = Field(default="v1.0", env="M365_GRAPH_API_VERSION")
    graph_base_url: str = Field(default="https://graph.microsoft.com", env="M365_GRAPH_BASE_URL")
    authority_base_url: str = Field(default="https://login.microsoftonline.com", env="M365_AUTHORITY_BASE_URL")

    # Feature Flags
    teams_integration_enabled: bool = Field(default=True, env="M365_TEAMS_ENABLED")
    outlook_integration_enabled: bool = Field(default=True, env="M365_OUTLOOK_ENABLED")
    sharepoint_integration_enabled: bool = Field(default=True, env="M365_SHAREPOINT_ENABLED")
    onedrive_integration_enabled: bool = Field(default=True, env="M365_ONEDRIVE_ENABLED")
    calendar_integration_enabled: bool = Field(default=True, env="M365_CALENDAR_ENABLED")
    ai_assistance_enabled: bool = Field(default=True, env="M365_AI_ASSISTANCE_ENABLED")

    # Rate Limiting
    max_requests_per_minute: int = Field(default=240, env="M365_MAX_REQUESTS_PER_MINUTE")  # Microsoft Graph limit
    max_ai_responses_per_hour: int = Field(default=200, env="M365_MAX_AI_RESPONSES_PER_HOUR")
    concurrent_requests_limit: int = Field(default=20, env="M365_CONCURRENT_REQUESTS_LIMIT")

    # Response Configuration
    response_delay_seconds: int = Field(default=3, env="M365_RESPONSE_DELAY_SECONDS")
    max_response_length: int = Field(default=3000, env="M365_MAX_RESPONSE_LENGTH")
    response_timeout_seconds: int = Field(default=45, env="M365_RESPONSE_TIMEOUT_SECONDS")

    # AI Configuration
    ai_model: str = Field(default="gpt-4", env="M365_AI_MODEL")
    ai_temperature: float = Field(default=0.7, env="M365_AI_TEMPERATURE")
    ai_max_tokens: int = Field(default=200, env="M365_AI_MAX_TOKENS")

    # Token Management
    token_refresh_buffer_minutes: int = Field(default=10, env="M365_TOKEN_REFRESH_BUFFER_MINUTES")
    token_cache_enabled: bool = Field(default=True, env="M365_TOKEN_CACHE_ENABLED")

    # Webhook Configuration
    webhook_validation_enabled: bool = Field(default=True, env="M365_WEBHOOK_VALIDATION_ENABLED")
    webhook_retry_attempts: int = Field(default=3, env="M365_WEBHOOK_RETRY_ATTEMPTS")
    webhook_timeout_seconds: int = Field(default=30, env="M365_WEBHOOK_TIMEOUT_SECONDS")
    webhook_subscription_duration_hours: int = Field(default=4320, env="M365_WEBHOOK_DURATION_HOURS")  # 180 days max

    # Database Configuration
    db_connection_pool_size: int = Field(default=15, env="M365_DB_POOL_SIZE")
    db_connection_timeout: int = Field(default=30, env="M365_DB_TIMEOUT")

    # Data Retention
    message_retention_days: int = Field(default=90, env="M365_MESSAGE_RETENTION_DAYS")
    email_retention_days: int = Field(default=365, env="M365_EMAIL_RETENTION_DAYS")
    document_retention_days: int = Field(default=730, env="M365_DOCUMENT_RETENTION_DAYS")  # 2 years
    analytics_retention_days: int = Field(default=1095, env="M365_ANALYTICS_RETENTION_DAYS")  # 3 years

    # Security Settings
    encryption_enabled: bool = Field(default=True, env="M365_ENCRYPTION_ENABLED")
    audit_logging_enabled: bool = Field(default=True, env="M365_AUDIT_LOGGING_ENABLED")
    ip_whitelist: Optional[List[str]] = Field(default=None, env="M365_IP_WHITELIST")
    require_admin_consent: bool = Field(default=False, env="M365_REQUIRE_ADMIN_CONSENT")

    # Performance Settings
    async_processing_enabled: bool = Field(default=True, env="M365_ASYNC_PROCESSING_ENABLED")
    batch_processing_size: int = Field(default=50, env="M365_BATCH_PROCESSING_SIZE")
    cache_ttl_seconds: int = Field(default=600, env="M365_CACHE_TTL_SECONDS")  # 10 minutes

    # File Processing
    max_file_size_mb: int = Field(default=250, env="M365_MAX_FILE_SIZE_MB")  # SharePoint limit
    supported_file_types: List[str] = Field(
        default=[".docx", ".xlsx", ".pptx", ".pdf", ".txt", ".md", ".csv"],
        env="M365_SUPPORTED_FILE_TYPES"
    )
    ai_file_processing_enabled: bool = Field(default=True, env="M365_AI_FILE_PROCESSING_ENABLED")

    # Environment
    environment: Microsoft365Environment = Field(
        default=Microsoft365Environment.PRODUCTION,
        env="M365_ENVIRONMENT"
    )

    @validator('supported_file_types', pre=True)
    def parse_supported_file_types(cls, v):
        """Parse supported file types from string or list"""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(',') if ext.strip()]
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

# Microsoft 365 Required Scopes Configuration
M365_SCOPES = {
    "delegated": [
        "https://graph.microsoft.com/User.Read",
        "https://graph.microsoft.com/Mail.ReadWrite",
        "https://graph.microsoft.com/Calendars.ReadWrite",
        "https://graph.microsoft.com/Files.ReadWrite.All",
        "https://graph.microsoft.com/Sites.ReadWrite.All",
        "https://graph.microsoft.com/ChannelMessage.Send",
        "https://graph.microsoft.com/Chat.ReadWrite",
        "https://graph.microsoft.com/Team.ReadBasic.All",
        "https://graph.microsoft.com/TeamMember.Read.All",
        "https://graph.microsoft.com/Directory.Read.All",
        "https://graph.microsoft.com/Group.Read.All",
        "https://graph.microsoft.com/Presence.Read.All"
    ],
    "application": [
        "https://graph.microsoft.com/User.Read.All",
        "https://graph.microsoft.com/Mail.Read",
        "https://graph.microsoft.com/Calendars.Read",
        "https://graph.microsoft.com/Files.Read.All",
        "https://graph.microsoft.com/Sites.Read.All",
        "https://graph.microsoft.com/Team.ReadBasic.All",
        "https://graph.microsoft.com/Directory.Read.All"
    ]
}

# Microsoft 365 Webhook Resource Types
M365_WEBHOOK_RESOURCES = {
    "teams": {
        "messages": "/teams/{team_id}/channels/{channel_id}/messages",
        "channels": "/teams/{team_id}/channels",
        "members": "/teams/{team_id}/members"
    },
    "outlook": {
        "messages": "/me/messages",
        "events": "/me/events",
        "mailFolder": "/me/mailFolders('Inbox')/messages"
    },
    "sharepoint": {
        "driveItems": "/sites/{site_id}/drive/root",
        "lists": "/sites/{site_id}/lists/{list_id}/items"
    },
    "onedrive": {
        "driveItems": "/me/drive/root",
        "changes": "/me/drive/root/delta"
    }
}

# Microsoft 365 Change Types for Webhooks
M365_CHANGE_TYPES = [
    "created",
    "updated",
    "deleted"
]

# Microsoft 365 AI Processing Templates
M365_AI_TEMPLATES = {
    "email_analysis": {
        "system_prompt": "Analyze this email and provide: 1) Brief summary (max 100 words), 2) Priority level (low/medium/high/urgent), 3) Category (inquiry/meeting/task/notification/other), 4) Suggested actions. Format as JSON.",
        "max_tokens": 300
    },
    "document_analysis": {
        "system_prompt": "Analyze this document and provide: 1) Summary, 2) Key topics/tags, 3) Document type classification, 4) Language detected. Format as JSON.",
        "max_tokens": 400
    },
    "teams_message_response": {
        "system_prompt": "You are an AI assistant helping with Microsoft Teams. Provide helpful, professional responses to team questions. Keep responses under 500 characters.",
        "max_tokens": 150
    },
    "calendar_preparation": {
        "system_prompt": "Analyze this calendar event and provide: 1) Meeting preparation notes, 2) Key topics to discuss, 3) Follow-up actions. Format as JSON.",
        "max_tokens": 250
    }
}

# Microsoft 365 Service Configuration
M365_SERVICE_CONFIG = {
    "teams": {
        "max_message_length": 28000,  # Teams limit
        "supported_message_types": ["text", "html"],
        "max_channels_per_team": 200,
        "webhook_events": ["channelMessage", "chatMessage"]
    },
    "outlook": {
        "max_email_size_mb": 150,  # Outlook limit
        "max_recipients": 500,
        "max_subject_length": 255,
        "webhook_events": ["messages", "events"]
    },
    "sharepoint": {
        "max_file_size_mb": 250,  # SharePoint limit
        "max_list_items": 30000000,
        "webhook_events": ["driveItem"]
    },
    "onedrive": {
        "max_file_size_gb": 250,  # OneDrive limit
        "webhook_events": ["driveItem"]
    }
}

# Notification Templates for Microsoft 365
M365_NOTIFICATION_TEMPLATES = {
    "email_processed": {
        "title": "📧 Email Processed",
        "template": "New email from {sender} has been analyzed. Priority: {priority}",
        "color": "#0078d4"
    },
    "meeting_reminder": {
        "title": "📅 Meeting Reminder",
        "template": "Meeting '{subject}' starts in {minutes} minutes",
        "color": "#ff8c00"
    },
    "document_uploaded": {
        "title": "📄 Document Uploaded",
        "template": "Document '{filename}' uploaded to {location}",
        "color": "#107c10"
    },
    "teams_mention": {
        "title": "💬 Teams Mention",
        "template": "You were mentioned in {channel} by {user}",
        "color": "#6264a7"
    },
    "calendar_conflict": {
        "title": "⚠️ Calendar Conflict",
        "template": "Meeting conflict detected for {time}",
        "color": "#d13438"
    }
}

# Logging Configuration
M365_LOGGING_CONFIG = {
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
            "filename": "logs/microsoft365_integration.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "detailed"
        }
    },
    "loggers": {
        "microsoft365_integration": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False
        }
    }
}

# Function to get configuration instance
def get_microsoft365_config() -> Microsoft365Config:
    """Get Microsoft 365 configuration instance"""
    return Microsoft365Config()

# Function to validate Microsoft 365 configuration
def validate_microsoft365_config(config: Microsoft365Config) -> Dict[str, Any]:
    """Validate Microsoft 365 configuration and return validation results"""
    errors = []
    warnings = []

    # Required fields validation
    if not config.client_id:
        errors.append("M365_CLIENT_ID is required")

    if not config.client_secret:
        errors.append("M365_CLIENT_SECRET is required")

    # URL validation
    if not config.oauth_redirect_uri.startswith(('http://', 'https://')):
        errors.append("M365_OAUTH_REDIRECT_URI must be a valid URL")

    # Rate limiting validation
    if config.max_requests_per_minute > 240:
        warnings.append("max_requests_per_minute exceeds Microsoft Graph API limits (240/min)")

    if config.max_requests_per_minute <= 0:
        warnings.append("max_requests_per_minute should be positive")

    # Performance validation
    if config.response_timeout_seconds < 10:
        warnings.append("response_timeout_seconds should be at least 10 seconds for Microsoft 365")

    # File size validation
    if config.max_file_size_mb > 250:
        warnings.append("max_file_size_mb exceeds SharePoint limits (250MB)")

    # Webhook validation
    if config.webhook_subscription_duration_hours > 4320:  # 180 days
        warnings.append("webhook_subscription_duration_hours exceeds Microsoft Graph limits (180 days)")

    # Environment-specific validation
    if config.environment == Microsoft365Environment.PRODUCTION:
        if config.oauth_redirect_uri.startswith('http://'):
            warnings.append("Production environment should use HTTPS for OAuth redirect URI")

        if not config.encryption_enabled:
            warnings.append("Encryption should be enabled in production")

        if not config.audit_logging_enabled:
            warnings.append("Audit logging should be enabled in production")

        if config.tenant_id == "common" and not config.require_admin_consent:
            warnings.append("Multi-tenant apps should consider requiring admin consent in production")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "config": config.dict()
    }

# Function to get required scopes for services
def get_required_scopes(services: List[str], delegated: bool = True) -> List[str]:
    """Get required Microsoft Graph scopes for specified services"""
    scope_type = "delegated" if delegated else "application"
    base_scopes = M365_SCOPES[scope_type].copy()

    # Add service-specific scopes if needed
    service_specific_scopes = {
        "teams": [
            "https://graph.microsoft.com/ChannelMessage.Send",
            "https://graph.microsoft.com/Chat.ReadWrite"
        ],
        "outlook": [
            "https://graph.microsoft.com/Mail.ReadWrite",
            "https://graph.microsoft.com/Calendars.ReadWrite"
        ],
        "sharepoint": [
            "https://graph.microsoft.com/Sites.ReadWrite.All"
        ],
        "onedrive": [
            "https://graph.microsoft.com/Files.ReadWrite.All"
        ]
    }

    all_scopes = set(base_scopes)
    for service in services:
        if service in service_specific_scopes:
            all_scopes.update(service_specific_scopes[service])

    return list(all_scopes)

# Export configuration instance
microsoft365_config = get_microsoft365_config()

# Export all components
__all__ = [
    "Microsoft365Config",
    "Microsoft365Environment",
    "M365_SCOPES",
    "M365_WEBHOOK_RESOURCES",
    "M365_CHANGE_TYPES",
    "M365_AI_TEMPLATES",
    "M365_SERVICE_CONFIG",
    "M365_NOTIFICATION_TEMPLATES",
    "M365_LOGGING_CONFIG",
    "get_microsoft365_config",
    "validate_microsoft365_config",
    "get_required_scopes",
    "microsoft365_config"
]
