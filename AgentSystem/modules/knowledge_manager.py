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
import numpy as np
from typing import Dict, List, Any, Optional
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
import psutil
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
        self.init_database()

    def monitor_ram(self, max_usage: float = None) -> bool:
        """Monitor RAM usage to prevent system overload"""
        if max_usage is None:
            max_usage = self.ram_limit
            
        ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # GB
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
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_timestamp ON facts(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence)')
                
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
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_access ON facts(access_count)')
            
            self.conn.commit()
            logger.info(f"Initialized {'PostgreSQL' if self.use_postgres else 'SQLite'} database")
            
        except (sqlite3.Error, psycopg2.Error) as e:
            logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
            self.conn = None
    
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
