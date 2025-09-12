"""
Security Utilities Module
------------------------
Provides security utilities for input validation, sanitization, and protection
against common security vulnerabilities.
"""

import re
import hashlib
import secrets
import hmac
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class SecurityValidator:
    """Security validation utilities"""

    # Common dangerous patterns
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',               # JavaScript URLs
        r'on\w+\s*=',                # Event handlers
        r'eval\s*\(',                # eval() calls
        r'exec\s*\(',                # exec() calls
        r'import\s+os',              # OS imports
        r'__import__',               # Dynamic imports
        r'\.\./',                    # Path traversal
        r'\\.\\.\\',                 # Windows path traversal
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
        r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
        r'(\b(OR|AND)\s+[\'"][^\'"]*[\'"])',
        r'(--|#|/\*|\*/)',
        r'(\bxp_cmdshell\b)',
    ]

    @classmethod
    def validate_input(cls, input_data: Any, max_length: int = 1000) -> Dict[str, Any]:
        """
        Validate and sanitize input data

        Args:
            input_data: Data to validate
            max_length: Maximum allowed length for string inputs

        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'sanitized': input_data,
            'issues': []
        }

        try:
            if input_data is None:
                return result

            # Convert to string for validation
            input_str = str(input_data)

            # Length check
            if len(input_str) > max_length:
                result['valid'] = False
                result['issues'].append(f'Input too long: {len(input_str)} > {max_length}')
                result['sanitized'] = input_str[:max_length]
                input_str = result['sanitized']

            # Check for dangerous patterns
            for pattern in cls.DANGEROUS_PATTERNS:
                if re.search(pattern, input_str, re.IGNORECASE):
                    result['valid'] = False
                    result['issues'].append(f'Dangerous pattern detected: {pattern}')
                    # Remove the dangerous content
                    result['sanitized'] = re.sub(pattern, '', input_str, flags=re.IGNORECASE)
                    input_str = result['sanitized']

            # Check for SQL injection patterns
            for pattern in cls.SQL_INJECTION_PATTERNS:
                if re.search(pattern, input_str, re.IGNORECASE):
                    result['valid'] = False
                    result['issues'].append(f'Potential SQL injection: {pattern}')
                    # Escape or remove SQL patterns
                    result['sanitized'] = re.sub(pattern, '', input_str, flags=re.IGNORECASE)
                    input_str = result['sanitized']

            # HTML entity encoding for remaining content
            if isinstance(input_data, str):
                result['sanitized'] = cls.html_encode(input_str)

        except Exception as e:
            logger.error(f"Input validation error: {e}")
            result['valid'] = False
            result['issues'].append(f'Validation error: {str(e)}')
            result['sanitized'] = ''

        return result

    @staticmethod
    def html_encode(text: str) -> str:
        """HTML encode text to prevent XSS"""
        if not isinstance(text, str):
            return str(text)

        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;'
        }

        for char, encoded in replacements.items():
            text = text.replace(char, encoded)

        return text

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format and safety"""
        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False

            # Check for dangerous hosts
            dangerous_hosts = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
            if parsed.hostname in dangerous_hosts:
                return False

            # Check for private IP ranges (basic check)
            if parsed.hostname:
                if (parsed.hostname.startswith('192.168.') or
                    parsed.hostname.startswith('10.') or
                    parsed.hostname.startswith('172.')):
                    return False

            return True

        except Exception:
            return False

class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """
        Initialize rate limiter

        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {client_id: [(timestamp, count), ...]}

    def is_allowed(self, client_id: str) -> Dict[str, Any]:
        """
        Check if request is allowed for client

        Args:
            client_id: Unique client identifier

        Returns:
            Dictionary with rate limit status
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds

        # Clean old entries
        if client_id in self.requests:
            self.requests[client_id] = [
                (timestamp, count) for timestamp, count in self.requests[client_id]
                if timestamp > window_start
            ]
        else:
            self.requests[client_id] = []

        # Count current requests
        current_count = sum(count for _, count in self.requests[client_id])

        # Check if allowed
        allowed = current_count < self.max_requests

        if allowed:
            # Add current request
            self.requests[client_id].append((current_time, 1))

        return {
            'allowed': allowed,
            'current_count': current_count + (1 if allowed else 0),
            'max_requests': self.max_requests,
            'window_seconds': self.window_seconds,
            'reset_time': window_start + self.window_seconds
        }

class SecureHash:
    """Secure hashing utilities"""

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """
        Hash password with salt

        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)

        Returns:
            Dictionary with hash and salt
        """
        if salt is None:
            salt = secrets.token_bytes(32)

        # Use PBKDF2 with SHA-256
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)

        return {
            'hash': hash_obj.hex(),
            'salt': salt.hex(),
            'algorithm': 'pbkdf2_sha256',
            'iterations': 100000
        }

    @staticmethod
    def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
        """
        Verify password against stored hash

        Args:
            password: Password to verify
            stored_hash: Stored hash
            stored_salt: Stored salt

        Returns:
            True if password matches
        """
        try:
            salt = bytes.fromhex(stored_salt)
            hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return hmac.compare_digest(hash_obj.hex(), stored_hash)
        except Exception:
            return False

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_data(data: Union[str, bytes]) -> str:
        """Generate SHA-256 hash of data"""
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()

class SessionManager:
    """Simple session management"""

    def __init__(self, session_timeout: int = 3600):
        """
        Initialize session manager

        Args:
            session_timeout: Session timeout in seconds
        """
        self.session_timeout = session_timeout
        self.sessions = {}  # {session_id: {data, created, last_accessed}}

    def create_session(self, user_data: Dict[str, Any]) -> str:
        """
        Create new session

        Args:
            user_data: User data to store in session

        Returns:
            Session ID
        """
        session_id = SecureHash.generate_token()
        current_time = time.time()

        self.sessions[session_id] = {
            'data': user_data,
            'created': current_time,
            'last_accessed': current_time
        }

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data

        Args:
            session_id: Session ID

        Returns:
            Session data or None if invalid/expired
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        current_time = time.time()

        # Check if expired
        if current_time - session['last_accessed'] > self.session_timeout:
            del self.sessions[session_id]
            return None

        # Update last accessed
        session['last_accessed'] = current_time
        return session['data']

    def delete_session(self, session_id: str) -> bool:
        """
        Delete session

        Args:
            session_id: Session ID

        Returns:
            True if session was deleted
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions

        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if current_time - session['last_accessed'] > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]

        return len(expired_sessions)

# Global instances
security_validator = SecurityValidator()
rate_limiter = RateLimiter()
session_manager = SessionManager()
