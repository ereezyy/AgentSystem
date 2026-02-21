import pytest
from fastapi import HTTPException
from AgentSystem.api.security_endpoints import get_current_tenant
import jwt
import asyncio
from unittest.mock import MagicMock, patch

# Mock the security dependency result
class MockHTTPAuthorizationCredentials:
    def __init__(self, token):
        self.credentials = token

SECRET_KEY = "test-secret-key"
ALGORITHM = "HS256"

@pytest.mark.asyncio
async def test_get_current_tenant_valid_token():
    """Test that a valid token returns the correct tenant ID"""
    payload = {"tenant_id": "tenant_123"}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    credentials = MockHTTPAuthorizationCredentials(token)

    with patch("AgentSystem.api.security_endpoints.get_env") as mock_get_env:
        def side_effect(key, default=None, required=False):
            if key == "JWT_SECRET_KEY":
                return SECRET_KEY
            if key == "JWT_ALGORITHM":
                return ALGORITHM
            return default
        mock_get_env.side_effect = side_effect

        tenant_id = await get_current_tenant(credentials)
        assert tenant_id == "tenant_123"

@pytest.mark.asyncio
async def test_get_current_tenant_invalid_token():
    """Test that an invalid token raises 401 Unauthorized"""
    credentials = MockHTTPAuthorizationCredentials("invalid_token")

    with patch("AgentSystem.api.security_endpoints.get_env") as mock_get_env:
        def side_effect(key, default=None, required=False):
            if key == "JWT_SECRET_KEY":
                return SECRET_KEY
            if key == "JWT_ALGORITHM":
                return ALGORITHM
            return default
        mock_get_env.side_effect = side_effect

        with pytest.raises(HTTPException) as excinfo:
            await get_current_tenant(credentials)

        assert excinfo.value.status_code == 401
        assert "Invalid token" in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_get_current_tenant_missing_tenant_id():
    """Test that a token missing tenant_id raises 401 Unauthorized"""
    payload = {"user_id": "user_123"} # Missing tenant_id
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    credentials = MockHTTPAuthorizationCredentials(token)

    with patch("AgentSystem.api.security_endpoints.get_env") as mock_get_env:
        def side_effect(key, default=None, required=False):
            if key == "JWT_SECRET_KEY":
                return SECRET_KEY
            if key == "JWT_ALGORITHM":
                return ALGORITHM
            return default
        mock_get_env.side_effect = side_effect

        with pytest.raises(HTTPException) as excinfo:
            await get_current_tenant(credentials)

        assert excinfo.value.status_code == 401
        assert "Token missing tenant_id" in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_get_current_tenant_missing_env_var():
    """Test that missing JWT_SECRET_KEY raises an error (which might propagate as 500 or just crash, handled by framework)"""
    # Actually, get_env with required=True raises ValueError.
    # The current implementation catches Exception and returns 401 "Could not validate credentials".
    # This might mask configuration errors, which is debatable, but let's test the behavior.

    credentials = MockHTTPAuthorizationCredentials("some_token")

    # We don't patch get_env here, so it uses the real one which fails because env var is missing
    # But wait, imports happen at module level. No, get_env is imported at module level, but called inside function.

    # We need to make sure the real get_env raises ValueError for missing key
    # Our environment doesn't have JWT_SECRET_KEY set.

    # Since get_env raises ValueError, and the function catches Exception, it should return 401.

    with pytest.raises(HTTPException) as excinfo:
        await get_current_tenant(credentials)

    assert excinfo.value.status_code == 401
    # Check logs for "Error validating token: Required environment variable JWT_SECRET_KEY not found"

if __name__ == "__main__":
    pass
