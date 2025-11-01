"""AgentForge Developer Framework.

Provides registry and SDK helpers so developers can share and reuse
modules, datasets, and knowledge packages across AgentSystem deployments.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from AgentSystem.utils.logger import get_logger

logger = get_logger("modules.agent_forge")


@dataclass(frozen=True)
class ModuleDescriptor:
    """Metadata describing a distributable AgentForge module."""

    name: str
    version: str
    summary: str
    author: str = "unknown"
    capabilities: Sequence[str] = ()
    tags: Sequence[str] = ()

    def key(self) -> str:
        return f"{self.name}:{self.version}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "tags", tuple(self.tags))


class AgentForgeRegistry:
    """Lightweight registry for AgentForge modules.

    The registry persists module descriptors to disk so deployments can
    synchronise available capabilities across environments without
    requiring an external service.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self.storage_path = storage_path or Path(".agentforge")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.storage_path / "registry.json"
        self._modules: Dict[str, ModuleDescriptor] = {}
        self._load()

    def _load(self) -> None:
        if not self._registry_file.exists():
            return
        try:
            data = json.loads(self._registry_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load AgentForge registry: %s", exc)
            return
        for payload in data.get("modules", []):
            try:
                descriptor = ModuleDescriptor(**payload)
            except TypeError as exc:
                logger.debug("Skipping invalid module payload %s: %s", payload, exc)
                continue
            self._modules[descriptor.key()] = descriptor

    def _persist(self) -> None:
        payload = {"modules": [asdict(item) for item in self._modules.values()]}
        try:
            self._registry_file.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except OSError as exc:
            logger.error("Failed to persist AgentForge registry: %s", exc)

    def register(self, descriptor: ModuleDescriptor, *, overwrite: bool = False) -> ModuleDescriptor:
        """Register a module in the registry.

        Args:
            descriptor: Module metadata to persist.
            overwrite: Allow replacement when the key already exists.
        """

        key = descriptor.key()
        with self._lock:
            if not overwrite and key in self._modules:
                raise ValueError(f"Module {key} already registered")
            self._modules[key] = descriptor
            self._persist()
        logger.debug("Registered AgentForge module %s", key)
        return descriptor

    def list_modules(self, *, tag: Optional[str] = None) -> List[ModuleDescriptor]:
        modules = list(self._modules.values())
        if tag is None:
            return modules
        return [item for item in modules if tag in item.tags]

    def get(self, name: str, version: Optional[str] = None) -> Optional[ModuleDescriptor]:
        if version:
            return self._modules.get(f"{name}:{version}")
        # Return latest version lexicographically if not specified
        candidates = [item for item in self._modules.values() if item.name == name]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item.version)[-1]


class KnowledgeExchange:
    """Knowledge sharing surface for AgentForge deployments."""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self.storage_path = storage_path or Path(".agentforge")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._exchange_file = self.storage_path / "knowledge.json"
        self._entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self._exchange_file.exists():
            return
        try:
            data = json.loads(self._exchange_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load AgentForge knowledge exchange: %s", exc)
            return
        self._entries = [entry for entry in data.get("entries", []) if isinstance(entry, dict)]

    def _persist(self) -> None:
        payload = {"entries": self._entries}
        try:
            self._exchange_file.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except OSError as exc:
            logger.error("Failed to persist AgentForge knowledge exchange: %s", exc)

    def publish(self, *, title: str, content: str, authors: Sequence[str], tags: Sequence[str] = (),
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        record = {
            "title": title,
            "content": content,
            "authors": list(authors),
            "tags": list(tags),
            "metadata": dict(metadata or {}),
        }
        with self._lock:
            self._entries.append(record)
            self._persist()
        logger.debug("Published knowledge artifact %s", title)
        return record

    def query(self, *, tag: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        matches = self._entries if tag is None else [entry for entry in self._entries if tag in entry["tags"]]
        if limit is None:
            return list(matches)
        return list(matches[:limit])


class AgentForgeSDK:
    """High-level helper combining registry and knowledge exchange."""

    def __init__(
        self,
        *,
        registry: Optional[AgentForgeRegistry] = None,
        exchange: Optional[KnowledgeExchange] = None,
        knowledge_manager: Optional[Any] = None,
    ) -> None:
        self.registry = registry or AgentForgeRegistry()
        self.exchange = exchange or KnowledgeExchange()
        self.knowledge_manager = knowledge_manager

    def publish_module(self, descriptor: ModuleDescriptor, *, overwrite: bool = False) -> ModuleDescriptor:
        return self.registry.register(descriptor, overwrite=overwrite)

    def share_knowledge(
        self,
        title: str,
        content: str,
        *,
        authors: Sequence[str],
        tags: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = self.exchange.publish(
            title=title,
            content=content,
            authors=authors,
            tags=tags,
            metadata=metadata,
        )
        if self.knowledge_manager and tags:
            for tag in tags:
                fact = f"Knowledge artifact '{title}' tagged with {tag}"
                try:
                    self.knowledge_manager.add_fact(fact, source="agentforge", category=tag)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Failed to sync knowledge artifact %s into knowledge base: %s", title, exc)
        return entry

    def bootstrap(self, modules: Iterable[ModuleDescriptor]) -> None:
        for descriptor in modules:
            try:
                self.registry.register(descriptor, overwrite=False)
            except ValueError:
                continue

    def fetch_modules(self, *, tag: Optional[str] = None) -> List[ModuleDescriptor]:
        return self.registry.list_modules(tag=tag)

    def retrieve_knowledge(self, *, tag: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.exchange.query(tag=tag, limit=limit)
