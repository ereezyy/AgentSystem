import unittest
import sys
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4
from datetime import datetime
import asyncio

# Mock missing dependencies
sys.modules['dotenv'] = MagicMock()
sys.modules['aioredis'] = MagicMock()
sys.modules['asyncpg'] = MagicMock()
sys.modules['stripe'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['pydantic'] = MagicMock()

# Mock AgentSystem.database.connection since it seems missing in the file tree
mock_db_connection = MagicMock()
sys.modules['AgentSystem.database'] = MagicMock()
sys.modules['AgentSystem.database.connection'] = mock_db_connection

# Import the module under test to ensure it's loaded and patchable
# This must happen after mocking sys.modules
try:
    from AgentSystem.analytics.clv_predictor import CLVPredictor
except ImportError as e:
    # If imports fail inside the module due to other dependencies, we might see it here
    print(f"Import failed: {e}")
    # Don't swallow exception if we want to debug
    # pass

# Create a dummy class to mock the return value of fetchrow which supports dict-like access
class MockRow(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        return super().__getitem__(key)

class TestCLVPredictorSupport(unittest.IsolatedAsyncioTestCase):
    @patch('AgentSystem.analytics.clv_predictor.UsageTracker')
    @patch('AgentSystem.analytics.clv_predictor.StripeService')
    async def test_extract_customer_features_support_tickets(self, MockStripe, MockUsage):
        # We can just use CLVPredictor since it's imported at top level (or we re-import)
        # But to be safe if top-level import failed partially but we want to retry or something
        from AgentSystem.analytics.clv_predictor import CLVPredictor

        predictor = CLVPredictor()
        tenant_id = uuid4()
        customer_id = uuid4()

        # Mock the database connection context manager
        mock_conn = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = None

        # Prepare mock data
        now = datetime.now()
        customer_data = MockRow({
            'created_at': now,
            'days_since_signup': 100,
            'company_size': 'small',
            'industry': 'tech',
            'country': 'US',
            'acquisition_channel': 'organic'
        })
        usage_data = MockRow({
            'total_requests': 1000,
            'total_tokens': 50000,
            'total_cost': 10.0,
            'active_days': 20,
            'total_requests': 100
        })
        subscription_data = MockRow({
            'plan_id': 'pro',
            'subscription_days': 90,
            'current_period_start': now,
            'current_period_end': now
        })
        payment_data = MockRow({
            'total_payments': 3,
            'total_revenue': 300.0,
            'avg_payment': 100.0,
            'failed_payments': 0,
            'last_payment': now
        })
        feature_data = MockRow({
            'features_used': 5
        })

        support_data = MockRow({
            'ticket_count': 7
        })

        # Correct order of fetchrow calls:
        # 1. customer_data
        # 2. usage_data
        # 3. subscription_data
        # 4. payment_data
        # 5. support_data (NEW)
        # 6. feature_data
        mock_conn.fetchrow.side_effect = [
            customer_data,
            usage_data,
            subscription_data,
            payment_data,
            support_data,
            feature_data
        ]

        # Use the imported CLVPredictor's get_db_connection reference if it was imported successfully
        # But since we mocked sys.modules['AgentSystem.database.connection'], the import inside clv_predictor
        # should have picked up our mock.
        # However, the patch needs to patch 'AgentSystem.analytics.clv_predictor.get_db_connection'.
        # Since clv_predictor imports it, it's a name in that module.

        with patch('AgentSystem.analytics.clv_predictor.get_db_connection', return_value=mock_cm):
            features = await predictor._extract_customer_features(tenant_id, customer_id)

            # Verification
            self.assertEqual(features.support_tickets, 7, f"Expected 7 support tickets, but got {features.support_tickets}")

if __name__ == '__main__':
    unittest.main()
