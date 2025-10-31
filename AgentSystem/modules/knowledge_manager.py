"""
Knowledge Manager Module
----------------------
Handles storage, retrieval, and maintenance of knowledge in the system.

Features:
- Fact storage and deduplication
- Document management
- Semantic search
- Knowledge pruning
"""

import sqlite3
import os
import time
import json
from typing import Dict, List, Any, Optional

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore
from datetime import datetime, timedelta
# Optional PostgreSQL support with graceful fallback
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    POSTGRES_AVAILABLE = False
# RealDictCursor import moved to conditional block above
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore
import subprocess

from AgentSystem.utils.logger import get_logger

logger = get_logger("modules.knowledge_manager")

class KnowledgeManager:
    def __init__(self, db_path: Optional[str] = None, use_postgres: bool = False,
                 pg_host: str = "localhost", pg_port: int = 5432,
                 pg_dbname: str = "knowledge_base", pg_user: str = "user", 
                 pg_password: str = "pass"):
        """Initialize knowledge manager with database connection"""
        
        # Force SQLite if PostgreSQL is not available
        if use_postgres and not POSTGRES_AVAILABLE:
            logger.warning("PostgreSQL requested but psycopg2 not available, falling back to SQLite")
            use_postgres = False
        self.db_path = db_path or ":memory:"
        self.use_postgres = use_postgres
        self.pg_config = {
            'host': pg_host,
            'port': pg_port,
            'dbname': pg_dbname,
            'user': pg_user,
            'password': pg_password
        }
        self.conn = None
        self.local_cache_limit = 10000  # Facts to cache locally on Pi 5
        self.max_retries = 3
        self.retry_delay = 1.0
        self.ram_limit = 20.0  # GB - Increased for testing
        # Memory system tuning
        self.memory_decay_half_life = 60 * 60 * 24  # one day default
        self.minimum_salience = 0.05
        self.consolidation_threshold = 0.8
        self.init_database()

    def monitor_ram(self, max_usage: float = None) -> bool:
        """Monitor RAM usage to prevent system overload"""
        if max_usage is None:
            max_usage = self.ram_limit
            
        if psutil:
            ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # GB
        else:
            ram_usage = 0.0
        if ram_usage > max_usage:
            logger.warning(f"RAM usage {ram_usage:.2f}GB exceeds limit {max_usage}GB")
            return False
        return True
    
    def check_thermal(self) -> bool:
        """Check thermal status on Raspberry Pi 5"""
        try:
            # This works on Raspberry Pi
            temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
            temp = float(temp_output.split("=")[1].split("'")[0])
            
            if temp > 80:
                logger.warning(f"CPU temperature {temp}°C exceeds limit")
                return False
            return True
        except:
            # If not on Raspberry Pi, assume thermal is OK
            return True

    def init_database(self) -> None:
        """Initialize database schema"""
        try:
            if self.use_postgres:
                # Connect to PostgreSQL on primary CPU
                self.conn = psycopg2.connect(
                    host=self.pg_config['host'],
                    port=self.pg_config['port'],
                    dbname=self.pg_config['dbname'],
                    user=self.pg_config['user'],
                    password=self.pg_config['password'],
                    sslmode='require'  # Enable TLS for security
                )
                cursor = self.conn.cursor()
                
                # Create PostgreSQL tables
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT,
                    confidence REAL DEFAULT 1.0,
                    category TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    embedding BYTEA
                )
                ''')

                cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    content TEXT NOT NULL,
                    summary TEXT,
                    url TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                )
                ''')

                cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id SERIAL PRIMARY KEY,
                    event TEXT NOT NULL,
                    outcome TEXT,
                    emotion TEXT,
                    salience REAL DEFAULT 0.5,
                    context JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding BYTEA,
                    consolidated BOOLEAN DEFAULT FALSE
                )
                ''')

                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_timestamp ON facts(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_epi_salience ON episodic_memories(salience)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_epi_timestamp ON episodic_memories(timestamp)')

            else:
                # Local SQLite for caching on Pi 5
                self.conn = sqlite3.connect(self.db_path)
                cursor = self.conn.cursor()
                
                # Create Facts table with access tracking
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT,
                    confidence REAL DEFAULT 1.0,
                    category TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed DATETIME,
                    embedding BLOB
                )
                ''')
                
                # Create Documents table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    content TEXT NOT NULL,
                    summary TEXT,
                    url TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed DATETIME
                )
                ''')

                cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT NOT NULL,
                    outcome TEXT,
                    emotion TEXT,
                    salience REAL DEFAULT 0.5,
                    context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB,
                    consolidated INTEGER DEFAULT 0
                )
                ''')

                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_access ON facts(access_count)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_epi_salience ON episodic_memories(salience)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_epi_timestamp ON episodic_memories(timestamp)')

            self.conn.commit()
            logger.info(f"Initialized {'PostgreSQL' if self.use_postgres else 'SQLite'} database")
            
        except (sqlite3.Error, psycopg2.Error) as e:
            logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
            self.conn = None

    # ------------------------------------------------------------------
    # Episodic memory interface
    # ------------------------------------------------------------------
    def add_episode(
        self,
        event: str,
        outcome: Optional[str] = None,
        emotion: Optional[str] = None,
        salience: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
        embedding: Optional[bytes] = None,
    ) -> int:
        """Persist an episodic memory with an associated salience score."""
        if not self.conn:
            self.init_database()

        normalized_salience = max(self.minimum_salience, min(salience, 1.0))
        context_payload: Optional[str]
        if context is None:
            context_payload = None
        else:
            context_payload = json.dumps(context, default=str)

        try:
            cursor = self.conn.cursor()
            if self.use_postgres:
                cursor.execute(
                    """
                    INSERT INTO episodic_memories (event, outcome, emotion, salience, context, embedding)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                    RETURNING id
                    """,
                    (event, outcome, emotion, float(normalized_salience), context_payload, embedding),
                )
                episode_id = cursor.fetchone()[0]
            else:
                cursor.execute(
                    """
                    INSERT INTO episodic_memories (event, outcome, emotion, salience, context, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (event, outcome, emotion, float(normalized_salience), context_payload, embedding),
                )
                episode_id = cursor.lastrowid

            self.conn.commit()
            return int(episode_id)
        except (sqlite3.Error, psycopg2.Error) as exc:  # type: ignore[arg-type]
            logger.error("Failed to persist episodic memory: %s", exc)
            return -1

    def decay_memories(self, half_life: Optional[float] = None) -> None:
        """Apply exponential decay to episodic salience, pruning stale entries."""
        if not self.conn:
            self.init_database()

        half_life = half_life or self.memory_decay_half_life
        if half_life <= 0:
            return

        decay_factor = 0.5 ** (self.retry_delay / half_life)

        try:
            cursor = self.conn.cursor()
            if self.use_postgres:
                cursor.execute(
                    """
                    UPDATE episodic_memories
                    SET salience = GREATEST(%s, salience * %s)
                    """,
                    (self.minimum_salience, decay_factor),
                )
                cursor.execute(
                    "DELETE FROM episodic_memories WHERE salience <= %s",
                    (self.minimum_salience,),
                )
            else:
                cursor.execute(
                    """
                    UPDATE episodic_memories
                    SET salience = MAX(?, salience * ?)
                    """,
                    (self.minimum_salience, decay_factor),
                )
                cursor.execute(
                    "DELETE FROM episodic_memories WHERE salience <= ?",
                    (self.minimum_salience,),
                )
            self.conn.commit()
        except (sqlite3.Error, psycopg2.Error) as exc:  # type: ignore[arg-type]
            logger.error("Failed to decay episodic memories: %s", exc)

    def consolidate_memories(self, limit: int = 10) -> List[int]:
        """Elevate highly salient episodes into long-term factual knowledge."""
        if not self.conn:
            self.init_database()

        promoted: List[int] = []
        try:
            cursor = self.conn.cursor()
            if self.use_postgres:
                cursor.execute(
                    """
                    SELECT id, event, outcome, emotion
                    FROM episodic_memories
                    WHERE consolidated = FALSE AND salience >= %s
                    ORDER BY salience DESC, timestamp DESC
                    LIMIT %s
                    """,
                    (self.consolidation_threshold, limit),
                )
                rows = cursor.fetchall()
            else:
                cursor.execute(
                    """
                    SELECT id, event, outcome, emotion
                    FROM episodic_memories
                    WHERE consolidated = 0 AND salience >= ?
                    ORDER BY salience DESC, timestamp DESC
                    LIMIT ?
                    """,
                    (self.consolidation_threshold, limit),
                )
                rows = cursor.fetchall()

            for row in rows:
                episode_id, event, outcome, emotion = row
                summary_parts = [event]
                if outcome:
                    summary_parts.append(f"Outcome: {outcome}")
                if emotion:
                    summary_parts.append(f"Emotion: {emotion}")
                fact_text = " | ".join(summary_parts)
                fact_id = self.add_fact(
                    content=fact_text,
                    source="episodic_memory",
                    confidence=0.9,
                    category="experience",
                )
                promoted.append(int(fact_id))
                if self.use_postgres:
                    cursor.execute(
                        "UPDATE episodic_memories SET consolidated = TRUE WHERE id = %s",
                        (episode_id,),
                    )
                else:
                    cursor.execute(
                        "UPDATE episodic_memories SET consolidated = 1 WHERE id = ?",
                        (episode_id,),
                    )

            self.conn.commit()
        except (sqlite3.Error, psycopg2.Error) as exc:  # type: ignore[arg-type]
            logger.error("Failed to consolidate episodic memories: %s", exc)

        return promoted

    def contextual_recall(self, cue: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve episodic memories related to the provided cue."""
        if not self.conn:
            self.init_database()

        try:
            cursor = self.conn.cursor()
            if self.use_postgres:
                cursor.execute(
                    """
                    SELECT id, event, outcome, emotion, salience, context::text, timestamp
                    FROM episodic_memories
                    WHERE event ILIKE %s OR outcome ILIKE %s
                    ORDER BY salience DESC, timestamp DESC
                    LIMIT %s
                    """,
                    (f"%{cue}%", f"%{cue}%", limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, event, outcome, emotion, salience, context, timestamp
                    FROM episodic_memories
                    WHERE event LIKE ? OR outcome LIKE ?
                    ORDER BY salience DESC, timestamp DESC
                    LIMIT ?
                    """,
                    (f"%{cue}%", f"%{cue}%", limit),
                )
            rows = cursor.fetchall()
            memories: List[Dict[str, Any]] = []
            for row in rows:
                context_blob = row[5]
                parsed_context = None
                if context_blob:
                    try:
                        parsed_context = json.loads(context_blob)
                    except json.JSONDecodeError:
                        parsed_context = {"raw": context_blob}
                memories.append(
                    {
                        "id": row[0],
                        "event": row[1],
                        "outcome": row[2],
                        "emotion": row[3],
                        "salience": row[4],
                        "context": parsed_context,
                        "timestamp": row[6],
                    }
                )
            return memories
        except (sqlite3.Error, psycopg2.Error) as exc:  # type: ignore[arg-type]
            logger.error("Failed contextual recall: %s", exc)
            return []

    def fusion_search(self, query: str, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Combine structured facts and episodic memories for richer recall."""
        facts = self.search_facts(query, limit=limit)
        episodes = self.contextual_recall(query, limit=max(1, limit // 2))
        return {"facts": facts, "episodes": episodes}

    def synthesize_knowledge(self, topic: str, limit: int = 15) -> Dict[str, Any]:
        """Build a lightweight semantic graph for the requested topic."""
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        facts = self.search_facts(topic, limit=limit)
        episodes = self.contextual_recall(topic, limit=max(3, limit // 3))

        for fact in facts:
            node_id = f"fact-{fact['id']}"
            nodes[node_id] = {
                "id": node_id,
                "label": fact["content"],
                "type": "fact",
                "confidence": fact.get("confidence", 1.0),
            }

        for episode in episodes:
            node_id = f"episode-{episode['id']}"
            nodes[node_id] = {
                "id": node_id,
                "label": episode["event"],
                "type": "episode",
                "salience": episode.get("salience"),
            }

        combined = list(nodes.values())
        for idx, source in enumerate(combined):
            for target in combined[idx + 1 : idx + 4]:
                edges.append(
                    {
                        "source": source["id"],
                        "target": target["id"],
                        "weight": 0.5,
                        "relation": "related",
                    }
                )

        return {"topic": topic, "nodes": list(nodes.values()), "edges": edges}

    def generate_hypotheses(self, topic: str, limit: int = 5) -> List[str]:
        """Draft simple hypotheses based on existing facts and episodes."""
        fused = self.fusion_search(topic, limit=limit * 2)
        hypotheses: List[str] = []
        for fact in fused["facts"][:limit]:
            hypotheses.append(f"If {fact['content']}, then exploring more about {topic} may reveal deeper causes.")
        for episode in fused["episodes"][:limit]:
            hypotheses.append(
                f"When {episode['event']} occurs, outcome {episode.get('outcome')} could influence future {topic} tasks."
            )
        return hypotheses[:limit]

    def verify_claim(self, claim: str, min_sources: int = 2) -> Dict[str, Any]:
        """Cross-check a claim across multiple stored sources."""
        facts = self.search_facts(claim, limit=10)
        supporting = [fact for fact in facts if claim.lower() in fact["content"].lower()]
        verdict = "unknown"
        if len(supporting) >= min_sources:
            verdict = "supported"
        elif facts:
            verdict = "partial"
        return {"claim": claim, "verdict": verdict, "sources": supporting[:min_sources]}

    def add_fact(self, content: str, source: Optional[str] = None, 
                confidence: float = 1.0, category: Optional[str] = None,
                embedding: Optional[bytes] = None) -> int:
        """Add a fact to the knowledge base"""
        if not self.conn:
            self.init_database()
            
        # Check RAM before adding
        if not self.monitor_ram():
            logger.warning("RAM limit exceeded, cannot add fact")
            return -1
            
        try:
            cursor = self.conn.cursor()
            
            if self.use_postgres:
                # PostgreSQL query
                cursor.execute("SELECT id FROM facts WHERE content = %s", (content,))
                existing = cursor.fetchone()
                if existing:
                    return existing[0]
                
                cursor.execute(
                    "INSERT INTO facts (content, source, confidence, category, embedding, last_accessed) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP) RETURNING id",
                    (content, source, confidence, category, embedding)
                )
                fact_id = cursor.fetchone()[0]
            else:
                # SQLite query
                cursor.execute("SELECT id FROM facts WHERE content = ?", (content,))
                existing = cursor.fetchone()
                if existing:
                    return existing[0]
                
                cursor.execute(
                    "INSERT INTO facts (content, source, confidence, category, embedding, last_accessed) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (content, source, confidence, category, embedding)
                )
                fact_id = cursor.lastrowid
            
            self.conn.commit()
            return fact_id
            
        except (sqlite3.Error, psycopg2.Error) as e:
            logger.error(f"Error adding fact: {e}")
            return -1
    
    def add_facts_batch(self, facts: List[Dict[str, Any]]) -> List[int]:
        """Add multiple facts in batch to reduce network overhead"""
        if not self.conn:
            self.init_database()
            
        # Check RAM before batch operation
        if not self.monitor_ram():
            logger.warning("RAM limit exceeded, cannot add facts batch")
            return []
            
        fact_ids = []
        try:
            cursor = self.conn.cursor()
            
            if self.use_postgres:
                # PostgreSQL batch insert
                for fact in facts:
                    cursor.execute(
                        "INSERT INTO facts (content, source, confidence, category, last_accessed) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP) RETURNING id",
                        (fact['content'], fact.get('source'), fact.get('confidence', 1.0), fact.get('category'))
                    )
                    fact_ids.append(cursor.fetchone()[0])
            else:
                # SQLite batch insert
                cursor.executemany(
                    "INSERT INTO facts (content, source, confidence, category, last_accessed) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    [(f['content'], f.get('source'), f.get('confidence', 1.0), f.get('category')) for f in facts]
                )
                # Get inserted IDs
                cursor.execute("SELECT last_insert_rowid()")
                last_id = cursor.fetchone()[0]
                fact_ids = list(range(last_id - len(facts) + 1, last_id + 1))
            
            self.conn.commit()
            logger.info(f"Added {len(facts)} facts in batch")
            
        except (sqlite3.Error, psycopg2.Error) as e:
            logger.error(f"Error adding facts batch: {e}")
            self.conn.rollback()
            
        return fact_ids
    
    def cache_facts(self, fact_ids: List[int]) -> None:
        """Update access count and timestamp for cached facts"""
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            
            if self.use_postgres:
                cursor.execute(
                    "UPDATE facts SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = ANY(%s)",
                    (fact_ids,)
                )
            else:
                placeholders = ','.join('?' * len(fact_ids))
                cursor.execute(
                    f"UPDATE facts SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id IN ({placeholders})",
                    fact_ids
                )
            
            self.conn.commit()
            
        except (sqlite3.Error, psycopg2.Error) as e:
            logger.error(f"Error caching facts: {e}")
    
    def prune_knowledge(self, max_age_days: int = 30, min_confidence: float = 0.2) -> None:
        """Remove old or low confidence facts"""
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            
            # Delete old facts
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cursor.execute(
                "DELETE FROM facts WHERE timestamp < ? OR confidence < ?",
                (cutoff_date, min_confidence)
            )
            
            self.conn.commit()
            logger.info(f"Pruned knowledge base: removed facts older than {max_age_days} days or confidence < {min_confidence}")
            
        except sqlite3.Error as e:
            logger.error(f"Error pruning knowledge: {e}")
    
    def search_facts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for facts in the knowledge base"""
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            
            # Simple text search (can be enhanced with full-text search)
            cursor.execute(
                """
                SELECT id, content, source, confidence, category, timestamp
                FROM facts 
                WHERE content LIKE ? OR category LIKE ?
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit)
            )
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "source": row[2],
                    "confidence": row[3],
                    "category": row[4],
                    "timestamp": row[5]
                })
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error searching facts: {e}")
            return []
    
    def get_fact(self, fact_id: int) -> Optional[Dict[str, Any]]:
        """Get a fact by ID"""
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, content, source, confidence, category, timestamp
                FROM facts 
                WHERE id = ?
                """,
                (fact_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
                
            return {
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "confidence": row[3],
                "category": row[4],
                "timestamp": row[5]
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error getting fact: {e}")
            return None
    
    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
