import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sys
import os

# Mock missing dependencies
sys.modules['dotenv'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['aioredis'] = MagicMock()
sys.modules['asyncpg'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['docker'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['yaml'] = MagicMock()

# Mock internal modules
# We need to mock AgentSystem.utils.logger and AgentSystem.monitoring.realtime_dashboard
# But we can't just set sys.modules directly if we want the actual module imports to work later for other parts.
# However, for this test, we don't need real logger or dashboard.

mock_logger_module = MagicMock()
mock_logger = MagicMock()
mock_logger_module.get_logger.return_value = mock_logger
sys.modules['AgentSystem.utils.logger'] = mock_logger_module

mock_dashboard_module = MagicMock()
sys.modules['AgentSystem.monitoring.realtime_dashboard'] = mock_dashboard_module

# Now import the class under test
from AgentSystem.scaling.auto_scaler import ServiceManager

class TestServiceManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # ServiceManager calls docker.from_env() in __init__
        self.mock_docker_module = sys.modules['docker']
        self.mock_docker_client = self.mock_docker_module.from_env.return_value
        self.mock_docker_client.containers.list.return_value = []

        self.service_manager = ServiceManager()
        self.service_manager.current_instances = {'test-service': 1}
        self.service_manager.docker_available = True

    @patch('asyncio.create_subprocess_exec')
    async def test_start_service_instance_uses_async(self, mock_create_subprocess_exec):
        # Setup mock process
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b'stdout output', b'stderr output')
        mock_process.returncode = 0
        mock_create_subprocess_exec.return_value = mock_process

        # Call method
        await self.service_manager._start_service_instance('test-service')

        # Verify call
        if not mock_create_subprocess_exec.called:
             self.fail("asyncio.create_subprocess_exec was not called! The code is likely still using subprocess.run")

        mock_create_subprocess_exec.assert_called_once()

        args = mock_create_subprocess_exec.call_args[0]
        # Verify command structure
        self.assertEqual(args[0], "docker-compose")
        self.assertEqual(args[2], "docker-compose.microservices.yml")
        self.assertEqual(args[3], "scale")

        # Find the argument that contains the service scaling
        # Note: arg[4] might be it, but let's be flexible
        scale_arg = next((arg for arg in args if arg.startswith("test-service=")), None)
        self.assertIsNotNone(scale_arg, "Scale argument not found")
        self.assertEqual(scale_arg, "test-service=2")

if __name__ == '__main__':
    unittest.main()
