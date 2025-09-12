"""
Test Suite for Slack Integration - AgentSystem Profit Machine
Comprehensive tests for Slack bot functionality, OAuth, and API endpoints
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncpg

from ..integrations.slack_bot import (
    SlackBot, SlackWorkspace, SlackMessage, SlackEventType,
    SlackCommandType, NotificationType
)
from ..api.slack_endpoints import router
from ..main import app

# Test fixtures
@pytest.fixture
async def mock_db_pool():
    """Mock database pool for testing"""
    pool = Mock(spec=asyncpg.Pool)
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    pool.acquire.return_value.__aexit__.return_value = None
    return pool, conn

@pytest.fixture
async def slack_bot(mock_db_pool):
    """Create SlackBot instance for testing"""
    pool, _ = mock_db_pool
    bot = SlackBot(
        db_pool=pool,
        client_id="test_client_id",
        client_secret="test_client_secret",
        signing_secret="test_signing_secret"
    )
    return bot

@pytest.fixture
def test_workspace():
    """Sample workspace data for testing"""
    return SlackWorkspace(
        workspace_id="ws_123456789",
        tenant_id="tenant_123",
        team_id="T1234567890",
        team_name="Test Workspace",
        bot_token="xoxb-test-token",
        bot_user_id="U1234567890",
        access_token="xoxp-test-access-token",
        scope="app_mentions:read,channels:history,chat:write",
        webhook_url="https://hooks.slack.com/test",
        is_active=True,
        installed_at=datetime.now(),
        updated_at=datetime.now()
    )

@pytest.fixture
def test_message():
    """Sample message data for testing"""
    return SlackMessage(
        message_id="msg_123456789",
        workspace_id="ws_123456789",
        channel_id="C1234567890",
        user_id="U9876543210",
        text="Hello, can you help me with this?",
        timestamp="1234567890.123456",
        thread_ts=None,
        message_type="message",
        ai_response=None,
        processed_at=None,
        created_at=datetime.now()
    )

class TestSlackBot:
    """Test cases for SlackBot class"""

    @pytest.mark.asyncio
    async def test_oauth_url_generation(self, slack_bot):
        """Test OAuth URL generation"""
        tenant_id = "tenant_123"
        redirect_uri = "https://example.com/callback"

        oauth_url = slack_bot.get_oauth_url(tenant_id, redirect_uri)

        assert "slack.com/oauth/v2/authorize" in oauth_url
        assert f"state={tenant_id}" in oauth_url
        assert "redirect_uri=" in oauth_url
        assert "scope=" in oauth_url

    @pytest.mark.asyncio
    async def test_token_exchange_success(self, slack_bot, mock_db_pool):
        """Test successful OAuth token exchange"""
        pool, conn = mock_db_pool

        # Mock successful API response
        mock_response_data = {
            'ok': True,
            'team': {'id': 'T1234567890', 'name': 'Test Workspace'},
            'access_token': 'xoxp-test-token',
            'bot_user_id': 'U1234567890',
            'scope': 'app_mentions:read,channels:history,chat:write'
        }

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response

            async with slack_bot:
                workspace = await slack_bot.exchange_code_for_tokens(
                    tenant_id="tenant_123",
                    code="test_code",
                    redirect_uri="https://example.com/callback"
                )

            assert workspace.team_id == "T1234567890"
            assert workspace.team_name == "Test Workspace"
            assert workspace.tenant_id == "tenant_123"
            conn.execute.assert_called_once()

    def test_request_verification_valid(self, slack_bot):
        """Test valid request signature verification"""
        import hmac
        import hashlib
        import time

        timestamp = str(int(time.time()))
        body = "test_body"

        # Generate valid signature
        sig_basestring = f"v0:{timestamp}:{body}"
        expected_signature = 'v0=' + hmac.new(
            slack_bot.signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()

        headers = {
            'X-Slack-Request-Timestamp': timestamp,
            'X-Slack-Signature': expected_signature
        }

        # Should pass verification
        result = asyncio.run(slack_bot.verify_request(headers, body))
        assert result is True

    @pytest.mark.asyncio
    async def test_analyze_command(self, slack_bot):
        """Test /analyze slash command"""
        command_data = {
            'command': '/analyze',
            'text': 'This is a sample text to analyze for sentiment and insights.',
            'user_id': 'U9876543210',
            'channel_id': 'C1234567890'
        }

        with patch.object(slack_bot, '_analyze_text', return_value="Positive sentiment detected"):
            result = await slack_bot._handle_analyze_command(command_data)

        assert result['response_type'] == 'in_channel'
        assert 'Analysis Results' in result['blocks'][0]['text']['text']

    @pytest.mark.asyncio
    async def test_help_command(self, slack_bot):
        """Test /help slash command"""
        command_data = {
            'command': '/help',
            'text': '',
            'user_id': 'U9876543210',
            'channel_id': 'C1234567890'
        }

        result = await slack_bot._handle_help_command(command_data)

        assert result['response_type'] == 'ephemeral'
        assert 'AgentSystem Bot Commands' in result['text']

    def test_should_respond_to_message(self, slack_bot):
        """Test message response trigger logic"""
        # Should respond to questions
        assert slack_bot._should_respond_to_message("How do I do this?")
        assert slack_bot._should_respond_to_message("Can you help me?")
        assert slack_bot._should_respond_to_message("What is this?")

        # Should not respond to statements
        assert not slack_bot._should_respond_to_message("This is working fine.")
        assert not slack_bot._should_respond_to_message("Good morning everyone!")

class TestSlackAPI:
    """Test cases for Slack API endpoints"""

    @pytest.fixture
    def client(self):
        """Test client for API endpoints"""
        return TestClient(app)

    def test_slack_health_check(self, client):
        """Test Slack health check endpoint"""
        response = client.get("/api/v1/slack/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "slack_integration"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
