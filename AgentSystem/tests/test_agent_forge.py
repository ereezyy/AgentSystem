"""Tests for AgentForge developer framework."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from AgentSystem.modules.agent_forge import (
    AgentForgeRegistry,
    AgentForgeSDK,
    KnowledgeExchange,
    ModuleDescriptor,
)


class _StubKnowledgeManager:
    def __init__(self) -> None:
        self.facts: List[Dict[str, Any]] = []

    def add_fact(self, content: str, source: str = "", category: str = "") -> int:
        self.facts.append({"content": content, "source": source, "category": category})
        return len(self.facts)


class TestAgentForgeRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "registry"
        self.registry = AgentForgeRegistry(self.registry_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_register_and_list_modules(self) -> None:
        descriptor = ModuleDescriptor(
            name="vision-enhancer",
            version="1.0.0",
            summary="Adds advanced vision filters",
            author="tester",
            capabilities=["vision"],
            tags=["vision", "beta"],
        )
        self.registry.register(descriptor)

        modules = self.registry.list_modules()
        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0].name, "vision-enhancer")

        tagged = self.registry.list_modules(tag="vision")
        self.assertEqual(len(tagged), 1)
        self.assertEqual(tagged[0].key(), descriptor.key())

        self.assertEqual(self.registry.get("vision-enhancer"), descriptor)

    def test_register_duplicate_without_overwrite_errors(self) -> None:
        descriptor = ModuleDescriptor(
            name="vision-enhancer",
            version="1.0.0",
            summary="Adds advanced vision filters",
        )
        self.registry.register(descriptor)
        with self.assertRaises(ValueError):
            self.registry.register(descriptor)

    def test_persistence_round_trip(self) -> None:
        descriptor = ModuleDescriptor(
            name="vision-enhancer",
            version="1.0.0",
            summary="Adds advanced vision filters",
        )
        self.registry.register(descriptor)
        raw = json.loads((self.registry_path / "registry.json").read_text())
        self.assertIn("modules", raw)
        self.assertEqual(raw["modules"][0]["name"], "vision-enhancer")

        reloaded = AgentForgeRegistry(self.registry_path)
        self.assertEqual(reloaded.get("vision-enhancer"), descriptor)


class TestKnowledgeExchange(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.exchange_path = Path(self.temp_dir) / "exchange"
        self.exchange = KnowledgeExchange(self.exchange_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_publish_and_query(self) -> None:
        self.exchange.publish(
            title="Vision tips",
            content="Use histogram equalisation",
            authors=["tester"],
            tags=["vision"],
        )
        self.exchange.publish(
            title="Audio tips",
            content="Use noise reduction",
            authors=["tester"],
            tags=["audio"],
        )

        all_entries = self.exchange.query()
        self.assertEqual(len(all_entries), 2)

        vision_entries = self.exchange.query(tag="vision")
        self.assertEqual(len(vision_entries), 1)
        self.assertEqual(vision_entries[0]["title"], "Vision tips")

        limited = self.exchange.query(limit=1)
        self.assertEqual(len(limited), 1)


class TestAgentForgeSDK(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        storage = Path(self.temp_dir)
        registry = AgentForgeRegistry(storage / "registry")
        exchange = KnowledgeExchange(storage / "exchange")
        self.knowledge_manager = _StubKnowledgeManager()
        self.sdk = AgentForgeSDK(
            registry=registry,
            exchange=exchange,
            knowledge_manager=self.knowledge_manager,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_publish_module_and_share_knowledge(self) -> None:
        descriptor = ModuleDescriptor(
            name="research-booster",
            version="0.1.0",
            summary="Improves research heuristics",
            tags=["research"],
        )
        self.sdk.publish_module(descriptor)
        fetched = self.sdk.fetch_modules()
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0].name, "research-booster")

        self.sdk.share_knowledge(
            "Research handbook",
            "Always cite your sources",
            authors=["mentor"],
            tags=["research", "best-practices"],
        )
        knowledge = self.sdk.retrieve_knowledge(tag="research")
        self.assertEqual(len(knowledge), 1)
        self.assertEqual(self.knowledge_manager.facts[0]["category"], "research")


if __name__ == "__main__":  # pragma: no cover - manual execution
    unittest.main()
