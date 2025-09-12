"""
Agent Memory Management
----------------------
Manages the agent's memory, including short-term and long-term storage
"""

import os
import time
import json
import uuid
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Local imports
from AgentSystem.utils.logger import get_logger
from AgentSystem.utils.env_loader import get_env

# Get module logger
logger = get_logger("core.memory")


@dataclass
class MemoryItem:
    """A single memory item"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    type: str = "general"
    created_at: float = field(default_factory=time.time)
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary"""
        return cls(**data)


class Memory:
    """
    Agent memory system
    
    Manages different types of memory:
    - Working memory (short-term)
    - Long-term memory (persistent)
    - Episodic memory (events and experiences)
    - Procedural memory (how to perform tasks)
    """
    
    def __init__(self, storage_path: Optional[str] = None, max_working_items: int = 100):
        """
        Initialize the memory system
        
        Args:
            storage_path: Path to store persistent memory
            max_working_items: Maximum number of items in working memory
        """
        self.max_working_items = max_working_items
        self.working_memory: List[MemoryItem] = []
        self.storage_path = storage_path or get_env("MEMORY_STORAGE", "./data/memory")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize database
        self.db_path = os.path.join(self.storage_path, "memory.db")
        self._init_db()
        
        logger.debug(f"Memory system initialized with storage at {self.storage_path}")
    
    def _init_db(self) -> None:
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT,
            type TEXT,
            created_at REAL,
            importance REAL,
            metadata TEXT,
            embedding TEXT
        )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON memories (type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories (importance)')
        
        conn.commit()
        conn.close()
        
        logger.debug("Memory database initialized")
    
    def add(
        self, 
        content: Any, 
        memory_type: str = "general", 
        importance: float = 0.5, 
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        """
        Add an item to memory
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (0.0 - 1.0)
            metadata: Additional metadata
            embedding: Vector embedding of the content
            
        Returns:
            Memory ID
        """
        # Create memory item
        item = MemoryItem(
            content=content,
            type=memory_type,
            importance=importance,
            metadata=metadata or {},
            embedding=embedding
        )
        
        # Add to working memory
        self.working_memory.append(item)
        
        # Trim working memory if needed
        if len(self.working_memory) > self.max_working_items:
            # Sort by importance and recency
            self.working_memory.sort(
                key=lambda x: (x.importance, x.created_at),
                reverse=True
            )
            # Remove least important/oldest items
            self.working_memory = self.working_memory[:self.max_working_items]
        
        # For important memories, store in long-term memory
        if importance > 0.7 or memory_type in ["critical", "persistent"]:
            self._store_in_long_term(item)
        
        logger.debug(f"Added memory of type '{memory_type}' with ID {item.id}")
        
        return item.id
    
    def _store_in_long_term(self, item: MemoryItem) -> None:
        """
        Store an item in long-term memory
        
        Args:
            item: Memory item to store
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize complex data
        content = json.dumps(item.content) if isinstance(item.content, (dict, list)) else str(item.content)
        metadata = json.dumps(item.metadata)
        embedding = json.dumps(item.embedding) if item.embedding else None
        
        # Insert into database
        cursor.execute(
            'INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                item.id,
                content,
                item.type,
                item.created_at,
                item.importance,
                metadata,
                embedding
            )
        )
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Stored memory {item.id} in long-term storage")
    
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Get a memory item by ID
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory item or None
        """
        # First check working memory
        for item in self.working_memory:
            if item.id == memory_id:
                return item
        
        # Then check long-term memory
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if not row:
            return None
        
        # Deserialize data
        try:
            id, content, type, created_at, importance, metadata, embedding = row
            
            # Parse JSON fields
            content = json.loads(content) if content.startswith('{') or content.startswith('[') else content
            metadata = json.loads(metadata)
            embedding = json.loads(embedding) if embedding else None
            
            return MemoryItem(
                id=id,
                content=content,
                type=type,
                created_at=created_at,
                importance=importance,
                metadata=metadata,
                embedding=embedding
            )
        except Exception as e:
            logger.error(f"Error deserializing memory {memory_id}: {e}")
            return None
    
    def query(
        self, 
        memory_type: Optional[str] = None, 
        min_importance: float = 0.0,
        max_items: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        content_search: Optional[str] = None
    ) -> List[MemoryItem]:
        """
        Query memory items
        
        Args:
            memory_type: Type of memory to query
            min_importance: Minimum importance score
            max_items: Maximum number of items to return
            metadata_filter: Filter by metadata
            content_search: Search content for this string
            
        Returns:
            List of memory items
        """
        results: List[MemoryItem] = []
        
        # Build SQL query
        query = 'SELECT * FROM memories WHERE importance >= ?'
        params = [min_importance]
        
        if memory_type:
            query += ' AND type = ?'
            params.append(memory_type)
        
        if content_search:
            query += ' AND content LIKE ?'
            params.append(f'%{content_search}%')
        
        query += ' ORDER BY importance DESC, created_at DESC LIMIT ?'
        params.append(max_items)
        
        # Execute query
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        
        conn.close()
        
        # Process results
        for row in rows:
            try:
                id, content, type, created_at, importance, metadata, embedding = row
                
                # Parse JSON fields
                content = json.loads(content) if content.startswith('{') or content.startswith('[') else content
                metadata_dict = json.loads(metadata)
                embedding_list = json.loads(embedding) if embedding else None
                
                # Apply metadata filter if provided
                if metadata_filter:
                    skip = False
                    for key, value in metadata_filter.items():
                        if key not in metadata_dict or metadata_dict[key] != value:
                            skip = True
                            break
                    if skip:
                        continue
                
                results.append(MemoryItem(
                    id=id,
                    content=content,
                    type=type,
                    created_at=created_at,
                    importance=importance,
                    metadata=metadata_dict,
                    embedding=embedding_list
                ))
            except Exception as e:
                logger.error(f"Error parsing memory row: {e}")
        
        # Also search working memory
        for item in self.working_memory:
            # Apply filters
            if memory_type and item.type != memory_type:
                continue
            
            if item.importance < min_importance:
                continue
            
            if metadata_filter:
                skip = False
                for key, value in metadata_filter.items():
                    if key not in item.metadata or item.metadata[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            if content_search:
                content_str = str(item.content)
                if content_search.lower() not in content_str.lower():
                    continue
            
            # Add to results if not already included
            if not any(r.id == item.id for r in results):
                results.append(item)
        
        # Sort and limit results
        results.sort(key=lambda x: (x.importance, x.created_at), reverse=True)
        return results[:max_items]
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """
        Update a memory item
        
        Args:
            memory_id: Memory ID
            **kwargs: Fields to update
            
        Returns:
            Success flag
        """
        # First try to update in working memory
        for i, item in enumerate(self.working_memory):
            if item.id == memory_id:
                for key, value in kwargs.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                
                # If importance changed, may need to store in long-term
                if 'importance' in kwargs and kwargs['importance'] > 0.7:
                    self._store_in_long_term(item)
                
                return True
        
        # Then try to update in long-term memory
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First check if the item exists
        cursor.execute('SELECT id FROM memories WHERE id = ?', (memory_id,))
        if not cursor.fetchone():
            conn.close()
            return False
        
        # Build update query
        query = 'UPDATE memories SET '
        params = []
        
        for key, value in kwargs.items():
            if key in ['content', 'type', 'importance', 'metadata', 'embedding']:
                query += f'{key} = ?, '
                
                # Serialize complex data
                if key == 'content' and isinstance(value, (dict, list)):
                    params.append(json.dumps(value))
                elif key == 'metadata':
                    params.append(json.dumps(value))
                elif key == 'embedding':
                    params.append(json.dumps(value))
                else:
                    params.append(value)
        
        # Remove trailing comma and space
        query = query[:-2]
        
        query += ' WHERE id = ?'
        params.append(memory_id)
        
        cursor.execute(query, tuple(params))
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    
    def forget(self, memory_id: str) -> bool:
        """
        Remove a memory item
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Success flag
        """
        # Remove from working memory
        removed = False
        for i, item in enumerate(self.working_memory):
            if item.id == memory_id:
                self.working_memory.pop(i)
                removed = True
                break
        
        # Remove from long-term memory
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
        conn.commit()
        conn.close()
        
        removed = removed or cursor.rowcount > 0
        
        if removed:
            logger.debug(f"Removed memory {memory_id}")
        
        return removed
    
    def clear_working_memory(self) -> None:
        """Clear working memory"""
        self.working_memory = []
        logger.debug("Cleared working memory")
    
    def summarize_memories(
        self, 
        memory_type: Optional[str] = None,
        min_importance: float = 0.5,
        max_items: int = 20
    ) -> str:
        """
        Generate a summary of memories
        
        Args:
            memory_type: Type of memory to summarize
            min_importance: Minimum importance score
            max_items: Maximum number of items to include
            
        Returns:
            Summary text
        """
        memories = self.query(
            memory_type=memory_type,
            min_importance=min_importance,
            max_items=max_items
        )
        
        # TODO: Implement actual summarization logic using LLM
        # For now, just format the memories
        summary = f"Memory Summary ({len(memories)} items)\n\n"
        
        for item in memories:
            content_str = str(item.content)
            if len(content_str) > 100:
                content_str = content_str[:97] + "..."
            
            summary += f"- [{item.type}] {content_str} (importance: {item.importance:.2f})\n"
        
        return summary
