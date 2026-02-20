import unittest
import sys
import os
from unittest.mock import MagicMock

# Mock dotenv before importing AgentSystem
sys.modules["dotenv"] = MagicMock()

# Add parent directory to path to handle imports if running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from AgentSystem.utils.query_builder import build_select_query

class TestQueryBuilder(unittest.TestCase):
    def test_basic_select(self):
        query, params = build_select_query(
            table="users"
        )
        expected_query = "SELECT * FROM users"
        self.assertEqual(query, expected_query)
        self.assertEqual(params, [])

    def test_with_conditions(self):
        query, params = build_select_query(
            table="users",
            conditions=[
                ("name", "=", "John"),
                ("age", ">", 20),
                ("active", "=", True),
                ("nullable_field", "=", None)
            ]
        )
        expected_query = (
            "SELECT * FROM users\n"
            "WHERE name = $1 AND age > $2 AND active = $3 AND nullable_field = $4"
        )
        self.assertEqual(query, expected_query)
        self.assertEqual(params, ["John", 20, True, None])

    def test_with_order_limit_offset(self):
        query, params = build_select_query(
            table="users",
            order_by="created_at DESC",
            limit=10,
            offset=0
        )
        expected_query = (
            "SELECT * FROM users\n"
            "ORDER BY created_at DESC\n"
            "LIMIT $1\n"
            "OFFSET $2"
        )
        self.assertEqual(query, expected_query)
        self.assertEqual(params, [10, 0])

    def test_with_columns(self):
        query, params = build_select_query(
            table="users",
            columns=["id", "name"],
            conditions=[("id", "=", 1)]
        )
        expected_query = (
            "SELECT id, name FROM users\n"
            "WHERE id = $1"
        )
        self.assertEqual(query, expected_query)
        self.assertEqual(params, [1])

if __name__ == "__main__":
    unittest.main()
