"""
Continuous Learning Module
-------------------------
Enables continuous learning, adaptation, and autonomous research capabilities for AI agents.
This module provides a framework for agents to autonomously acquire, store, and retrieve knowledge, and modify its own code.

Key Features:
- Knowledge Base: Persistent storage of facts and documents with full-text search capabilities.
- Web Research: Autonomous web searching and content extraction with robust error handling.
- Fact Extraction: Advanced NLP-based fact extraction from text content with fallback mechanisms.
- Continuous Learning: Background thread for ongoing research and knowledge updates.
- Self-Modification: Ability for the agent to modify its own code to improve performance or add new features.

Note: Full-text search functionality may be limited if SQLite FTS5 extension is not available.
      Fallback mechanisms ensure basic functionality even without advanced dependencies.
"""

import os
import time
import json
import threading
import queue
import sqlite3
import tempfile
import hashlib
import re
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import urllib.parse
from collections import OrderedDict

# Local imports
from AgentSystem.utils.logger import get_logger
from AgentSystem.modules.code_editor import CodeEditor  # Import CodeEditor

# Get module logger
logger = get_logger("modules.continuous_learning")

try:
    import requests
    from bs4 import BeautifulSoup
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    LEARNING_IMPORTS_AVAILABLE = True
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
except ImportError:
    logger.warning("Continuous learning dependencies not available. Install with: pip install requests beautifulsoup4 nltk scikit-learn")
    logger.info("Fallback mechanisms will be used for basic functionality.")
    LEARNING_IMPORTS_AVAILABLE = False
from AgentSystem.services.ai import ai_service, AIMessage

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_IMPORTS_AVAILABLE = True
except ImportError:
    logger.warning("Sentence Transformers not available. Install with: pip install sentence-transformers")
    EMBEDDING_IMPORTS_AVAILABLE = False

if EMBEDDING_IMPORTS_AVAILABLE:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
else:
    embedding_model = None

class KnowledgeBase:
    """Knowledge base for storing and retrieving information"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the knowledge base
        
        Args:
            db_path: Path to SQLite database file (default: in-memory database)
        """
        self.db_path = db_path or ":memory:"
        self.conn = None
        self.init_database()
        self.code_editor = CodeEditor()  # Initialize CodeEditor
        
    def init_database(self) -> None:
        """Initialize the database schema"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            
            # Facts table for storing discrete pieces of information
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT,
                confidence REAL,
                category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed DATETIME,
                access_count INTEGER DEFAULT 0,
                embedding_id INTEGER,
                FOREIGN KEY (embedding_id) REFERENCES embeddings (id) ON DELETE CASCADE
            )
            ''')
            
            # Documents table for storing larger texts
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT NOT NULL,
                summary TEXT,
                url TEXT,
                source TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed DATETIME,
                access_count INTEGER DEFAULT 0
            )
            ''')
            
            # Document chunks for semantic search
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER,
                embedding_id INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
            ''')
            
            # Embeddings table for vector representations
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector BLOB NOT NULL,
                dimension INTEGER,
                model TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Relationships table for connecting facts/documents
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                target_id INTEGER NOT NULL,
                target_type TEXT NOT NULL,
                relation_type TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Activity Log table for tracking research activities
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                activity_type TEXT NOT NULL,
                details TEXT
            )
            ''')
            
            # Queries table for tracking research queries
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                source TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                results_count INTEGER DEFAULT 0
            )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_category ON facts (category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_embedding_id ON facts (embedding_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks (document_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_id ON document_chunks (embedding_id)')
            
            # Create full-text search virtual tables
            cursor.execute('CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(content, source, category)')
            cursor.execute('CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(title, content, summary, url, source)')
            
            self.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
            self.conn = None

    def search_similar_facts(self, content: str, threshold: float = 0.9, limit: int = 5) -> List[Dict[str, Any]]:
        if not EMBEDDING_IMPORTS_AVAILABLE:
            return []
        embedding = embedding_model.encode(content).tobytes()
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, content, vector FROM facts JOIN embeddings ON facts.embedding_id = embeddings.id")
        similar_facts = []
        query_vector = np.frombuffer(embedding)
        for row in cursor.fetchall():
            fact_vector = np.frombuffer(row[2])
            similarity = cosine_similarity([query_vector], [fact_vector])[0][0]
            if similarity >= threshold:
                similar_facts.append({"id": row[0], "content": row[1], "similarity": similarity})
        return sorted(similar_facts, key=lambda x: x["similarity"], reverse=True)[:limit]

    def close(self) -> None:
        """Close the database connection"""
        if self.conn:
            self.conn.close()
        self.conn = None

    def log_activity(self, activity_type: str, details: str) -> int:
        """
        Log an activity to the activity log

        Args:
            activity_type: Type of activity
            details: Details of the activity

        Returns:
            ID of the inserted activity log entry
        """
        if not self.conn:
            self.init_database()

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO activity_log (activity_type, details) VALUES (?, ?)",
                (activity_type, details)
            )

            activity_id = cursor.lastrowid
            self.conn.commit()
            return activity_id

        except sqlite3.Error as e:
            logger.error(f"Error logging activity: {e}")
            return -1

    def add_embedding(self, vector: bytes, dimension: int, model: str) -> int:
        """
        Add an embedding vector to the embeddings table.

        Args:
            vector: The embedding vector as a bytes object.
            dimension: The dimension of the vector.
            model: The name of the embedding model.

        Returns:
            The ID of the inserted embedding.
        """
        if not self.conn:
            self.init_database()

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO embeddings (vector, dimension, model) VALUES (?, ?, ?)",
                (vector, dimension, model)
            )

            embedding_id = cursor.lastrowid
            self.conn.commit()
            return embedding_id

        except sqlite3.Error as e:
            logger.error(f"Error adding embedding: {e}")
            return -1

    def add_fact(self, content: str, source: Optional[str] = None,
                confidence: float = 1.0, category: Optional[str] = None) -> int:
        """
        Add a fact to the knowledge base
        
        Args:
            content: The fact content
            source: Source of the fact
            confidence: Confidence score (0.0-1.0)
            category: Category for organizing facts
            
        Returns:
            ID of the inserted fact
        """
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            
            # Check if similar fact already exists
            similar_facts = self.search_similar_facts(content, threshold=0.9)
            if similar_facts:
                logger.info(f"Similar fact found, merging: {content}")
                fact_id = similar_facts[0]["id"]
                cursor.execute(
                    "UPDATE facts SET confidence = MAX(confidence, ?), last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id = ?",
                    (confidence, fact_id)
                )
                self.conn.commit()
                return fact_id
            
            # Insert new fact
            cursor.execute(
                "INSERT INTO facts (content, source, confidence, category, last_accessed) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (content, source, confidence, category)
            )
            
            fact_id = cursor.lastrowid
            
            # Add to full-text search
            cursor.execute(
                "INSERT INTO facts_fts (rowid, content, source, category) VALUES (?, ?, ?, ?)",
                (fact_id, content, source or "", category or "")
            )
            
            self.conn.commit()
            return fact_id
            
        except sqlite3.Error as e:
            logger.error(f"Error adding fact: {e}")
            return -1
    
    def add_document(self, content: str, title: Optional[str] = None, 
                    summary: Optional[str] = None, url: Optional[str] = None, 
                    source: Optional[str] = None, chunk_size: int = 500) -> int:
        """
        Add a document to the knowledge base
        
        Args:
            content: Document content
            title: Document title
            summary: Document summary
            url: Source URL
            source: Source name
            chunk_size: Size of chunks for splitting document
            
        Returns:
            ID of the inserted document
        """
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            
            # Insert document
            cursor.execute(
                "INSERT INTO documents (title, content, summary, url, source, last_accessed) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (title, content, summary, url, source)
            )
            
            document_id = cursor.lastrowid
            
            # Add to full-text search
            cursor.execute(
                "INSERT INTO documents_fts (rowid, title, content, summary, url, source) VALUES (?, ?, ?, ?, ?, ?)",
                (document_id, title or "", content, summary or "", url or "", source or "")
            )
            
            # Split document into chunks for semantic search
            if LEARNING_IMPORTS_AVAILABLE:
                # Use NLTK to split into sentences
                sentences = sent_tokenize(content)
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        chunks.append(current_chunk)
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Insert chunks
                for i, chunk in enumerate(chunks):
                    cursor.execute(
                        "INSERT INTO document_chunks (document_id, chunk_text, chunk_index) VALUES (?, ?, ?)",
                        (document_id, chunk, i)
                    )
            
            self.conn.commit()
            return document_id
            
        except sqlite3.Error as e:
            logger.error(f"Error adding document: {e}")
            return -1
    
    def search_facts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for facts in the knowledge base
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching facts
        """
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            
            # Use full-text search
            cursor.execute(
                """
                SELECT f.id, f.content, f.source, f.confidence, f.category, f.timestamp 
                FROM facts_fts fts 
                JOIN facts f ON fts.rowid = f.id 
                WHERE fts.content MATCH ? OR fts.category MATCH ?
                ORDER BY f.confidence DESC, f.access_count DESC
                LIMIT ?
                """,
                (query, query, limit)
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
            
            # Update access stats
            if results:
                fact_ids = [r["id"] for r in results]
                placeholders = ", ".join(["?"] * len(fact_ids))
                cursor.execute(
                    f"UPDATE facts SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id IN ({placeholders})",
                    fact_ids
                )
                self.conn.commit()
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error searching facts: {e}")
            return []
    
    def search_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents in the knowledge base
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            
            # Use full-text search
            cursor.execute(
                """
                SELECT d.id, d.title, d.summary, d.url, d.source, d.timestamp 
                FROM documents_fts fts 
                JOIN documents d ON fts.rowid = d.id 
                WHERE fts.content MATCH ? OR fts.title MATCH ? OR fts.summary MATCH ?
                ORDER BY d.access_count DESC
                LIMIT ?
                """,
                (query, query, query, limit)
            )
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "title": row[1],
                    "summary": row[2],
                    "url": row[3],
                    "source": row[4],
                    "timestamp": row[5]
                })
            
            # Update access stats
            if results:
                doc_ids = [r["id"] for r in results]
                placeholders = ", ".join(["?"] * len(doc_ids))
                cursor.execute(
                    f"UPDATE documents SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id IN ({placeholders})",
                    doc_ids
                )
                self.conn.commit()
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_fact(self, fact_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a fact by ID
        
        Args:
            fact_id: Fact ID
            
        Returns:
            Fact information or None if not found
        """
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
                
            # Update access stats
            cursor.execute(
                "UPDATE facts SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id = ?",
                (fact_id,)
            )
            self.conn.commit()
            
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
    
    def get_document(self, document_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID
        
        Args:
            document_id: Document ID
            
        Returns:
            Document information or None if not found
        """
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, title, content, summary, url, source, timestamp
                FROM documents 
                WHERE id = ?
                """,
                (document_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
                
            # Update access stats
            cursor.execute(
                "UPDATE documents SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id = ?",
                (document_id,)
            )
            self.conn.commit()
            
            return {
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "summary": row[3],
                "url": row[4],
                "source": row[5],
                "timestamp": row[6]
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error getting document: {e}")
            return None
    
    def add_relationship(self, source_id: int, source_type: str, 
                        target_id: int, target_type: str, 
                        relation_type: Optional[str] = None, 
                        confidence: float = 1.0) -> int:
        """
        Add a relationship between two entities
        
        Args:
            source_id: ID of source entity
            source_type: Type of source entity ('fact' or 'document')
            target_id: ID of target entity
            target_type: Type of target entity ('fact' or 'document')
            relation_type: Type of relationship
            confidence: Confidence score (0.0-1.0)
            
        Returns:
            ID of the inserted relationship
        """
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            
            # Check if relationship already exists
            cursor.execute(
                "SELECT id, confidence FROM relationships WHERE source_id = ? AND source_type = ? AND target_id = ? AND target_type = ?",
                (source_id, source_type, target_id, target_type)
            )
            
            existing = cursor.fetchone()
            if existing:
                rel_id, existing_confidence = existing
                
                # Update confidence if new confidence is higher
                if confidence > existing_confidence:
                    cursor.execute(
                        "UPDATE relationships SET confidence = ?, relation_type = ? WHERE id = ?",
                        (confidence, relation_type, rel_id)
                    )
                    self.conn.commit()
                
                return rel_id
            
            # Insert new relationship
            cursor.execute(
                "INSERT INTO relationships (source_id, source_type, target_id, target_type, relation_type, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, source_type, target_id, target_type, relation_type, confidence)
            )
            
            rel_id = cursor.lastrowid
            self.conn.commit()
            return rel_id
            
        except sqlite3.Error as e:
            logger.error(f"Error adding relationship: {e}")
            return -1
    
    def get_related_items(self, item_id: int, item_type: str) -> List[Dict[str, Any]]:
        """
        Get items related to the specified item
        
        Args:
            item_id: ID of the item
            item_id: ID of the item
            item_type: Type of the item ('fact' or 'document')
            
        Returns:
            List of related items
        """
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            
            # Get items where the specified item is the source
            cursor.execute(
                """
                SELECT r.target_id, r.target_type, r.relation_type, r.confidence
                FROM relationships r
                WHERE r.source_id = ? AND r.source_type = ?
                """,
                (item_id, item_type)
            )
            
            outgoing = []
            for row in cursor.fetchall():
                outgoing.append({
                    "id": row[0],
                    "type": row[1],
                    "relation": row[2],
                    "confidence": row[3],
                    "direction": "outgoing"
                })
            
            # Get items where the specified item is the target
            cursor.execute(
                """
                SELECT r.source_id, r.source_type, r.relation_type, r.confidence
                FROM relationships r
                WHERE r.target_id = ? AND r.target_type = ?
                """,
                (item_id, item_type)
            )
            
            incoming = []
            for row in cursor.fetchall():
                incoming.append({
                    "id": row[0],
                    "type": row[1],
                    "relation": row[2],
                    "confidence": row[3],
                    "direction": "incoming"
                })
            
            return outgoing + incoming
            
        except sqlite3.Error as e:
            logger.error(f"Error getting related items: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base
        
        Returns:
            Dictionary with statistics
        """
        if not self.conn:
            self.init_database()
            
        try:
            cursor = self.conn.cursor()
            
            # Count facts
            cursor.execute("SELECT COUNT(*) FROM facts")
            fact_count = cursor.fetchone()[0]
            
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            # Count relationships
            cursor.execute("SELECT COUNT(*) FROM relationships")
            rel_count = cursor.fetchone()[0]
            
            # Get fact categories
            cursor.execute("SELECT category, COUNT(*) FROM facts GROUP BY category")
            categories = {}
            for row in cursor.fetchall():
                if row[0]:  # Skip None category
                    categories[row[0]] = row[1]
            
            # Get recently added facts
            cursor.execute("SELECT COUNT(*) FROM facts WHERE timestamp > datetime('now', '-24 hours')")
            recent_facts = cursor.fetchone()[0]
            
            # Get recently added documents
            cursor.execute("SELECT COUNT(*) FROM documents WHERE timestamp > datetime('now', '-24 hours')")
            recent_docs = cursor.fetchone()[0]
            
            return {
                "fact_count": fact_count,
                "document_count": doc_count,
                "relationship_count": rel_count,
                "categories": categories,
                "recent_facts_24h": recent_facts,
                "recent_documents_24h": recent_docs
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

            
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search and return results
            
        Args:
            query: Search query
            num_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if not LEARNING_IMPORTS_AVAILABLE:
            logger.error("Web research dependencies not available")
            logger.info("Using fallback mechanism for search with a placeholder result.")
            return [{"title": "Fallback Result", "url": "", "snippet": "Dependencies not available. Install required libraries for full functionality.", "source": "fallback"}]
            
        # Check cache first
        cache_key = f"{query}:{num_results}"
        if cache_key in self.search_cache:
            # Move key to end to mark as recently used
            self.search_cache.move_to_end(cache_key)
            cache_entry = self.search_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_expiry:
                return cache_entry["results"]
        
        # Normalize and encode query
        normalized_query = query.strip().lower()
        encoded_query = urllib.parse.quote_plus(normalized_query)
        
        # Construct search URL (using DuckDuckGo Lite)
        url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
        
        try:
            # Apply rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.request_delay:
                time.sleep(self.request_delay - time_since_last_request)
            
            # Make request
            response = self.session.get(url, headers=self.headers, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code != 200:
                logger.error(f"Search request failed with status {response.status_code}")
                return []
            
            # Parse results
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # DuckDuckGo Lite format
            for i, result in enumerate(soup.select('table.result-link')):
                if i >= num_results:
                    break
                    
                link_element = result.select_one('a')
                if not link_element:
                    continue
                
                title = link_element.get_text(strip=True)
                url = link_element.get('href')
                
                # Extract snippet from the next table
                snippet = ""
                snippet_table = result.find_next('table', {'class': 'result-snippet'})
                if snippet_table:
                    snippet = snippet_table.get_text(strip=True)
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "duckduckgo"
                })
            
            # Cache results (LRU logic)
            self.search_cache[cache_key] = {
                "results": results,
                "timestamp": time.time()
            }
            self.search_cache.move_to_end(cache_key)
            if len(self.search_cache) > self.cache_max_size:
                # Remove least recently used item
                self.search_cache.popitem(last=False)
                
            return results
                
        except Exception as e:
            logger.error(f"Error during web search: {e}")
            return []


# Import WebResearcher from the separate module
try:
    from AgentSystem.modules.web_researcher import WebResearcher
except ImportError:
    logger.warning("WebResearcher not available")
    WebResearcher = None


class ContinuousLearningModule:
    """
    Continuous Learning Module
    
    A high-level interface for continuous learning capabilities including
    knowledge management, web research, and fact extraction.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the continuous learning module
        
        Args:
            db_path: Path to SQLite database file (default: in-memory database)
        """
        self.knowledge_base = KnowledgeBase(db_path)
        
        # Initialize web researcher if available
        if WebResearcher:
            try:
                # Import KnowledgeManager for WebResearcher
                from AgentSystem.modules.knowledge_manager import KnowledgeManager
                self.knowledge_manager = KnowledgeManager()
                self.web_researcher = WebResearcher(self.knowledge_manager)
            except ImportError:
                logger.warning("KnowledgeManager not available for WebResearcher")
                self.web_researcher = None
        else:
            self.web_researcher = None
    
    def add_fact(self, content: str, source: Optional[str] = None,
                confidence: float = 1.0, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a fact to the knowledge base
        
        Args:
            content: The fact content
            source: Source of the fact
            confidence: Confidence score (0.0-1.0)
            category: Category for organizing facts
            
        Returns:
            Dictionary with success status and fact ID
        """
        try:
            fact_id = self.knowledge_base.add_fact(content, source, confidence, category)
            if fact_id > 0:
                return {"success": True, "fact_id": fact_id}
            else:
                return {"success": False, "error": "Failed to add fact"}
        except Exception as e:
            logger.error(f"Error adding fact: {e}")
            return {"success": False, "error": str(e)}
    
    def add_document(self, content: str, title: Optional[str] = None,
                    summary: Optional[str] = None, url: Optional[str] = None,
                    source: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a document to the knowledge base
        
        Args:
            content: Document content
            title: Document title
            summary: Document summary
            url: Source URL
            source: Source name
            
        Returns:
            Dictionary with success status and document ID
        """
        try:
            doc_id = self.knowledge_base.add_document(content, title, summary, url, source)
            if doc_id > 0:
                return {"success": True, "document_id": doc_id}
            else:
                return {"success": False, "error": "Failed to add document"}
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return {"success": False, "error": str(e)}
    
    def search_knowledge(self, query: str, max_facts: int = 10, max_docs: int = 5) -> Dict[str, Any]:
        """
        Search for knowledge in the knowledge base
        
        Args:
            query: Search query
            max_facts: Maximum number of facts to return
            max_docs: Maximum number of documents to return
            
        Returns:
            Dictionary with search results
        """
        try:
            facts = self.knowledge_base.search_facts(query, max_facts)
            documents = self.knowledge_base.search_documents(query, max_docs)
            
            return {
                "success": True,
                "facts": facts,
                "documents": documents,
                "fact_count": len(facts),
                "document_count": len(documents)
            }
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return {"success": False, "error": str(e)}
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = self.knowledge_base.get_statistics()
            return {"success": True, "statistics": stats}
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"success": False, "error": str(e)}
    
    def extract_facts_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract facts from text using NLP
        
        Args:
            text: Text to extract facts from
            
        Returns:
            Dictionary with extracted facts
        """
        try:
            if not LEARNING_IMPORTS_AVAILABLE:
                return {
                    "success": False,
                    "error": "NLP dependencies not available",
                    "facts": [],
                    "facts_found": 0
                }
            
            # Simple fact extraction using sentence tokenization
            sentences = sent_tokenize(text)
            facts = []
            
            for sentence in sentences:
                # Skip very short sentences
                if len(sentence.strip()) < 20:
                    continue
                
                # Simple heuristics for fact-like sentences
                sentence = sentence.strip()
                if (sentence.endswith('.') and 
                    not sentence.startswith('The ') and
                    not sentence.startswith('This ') and
                    not sentence.startswith('That ')):
                    
                    facts.append({
                        "content": sentence,
                        "confidence": 0.7,
                        "source": "text_extraction"
                    })
            
            return {
                "success": True,
                "facts": facts,
                "facts_found": len(facts)
            }
            
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return {"success": False, "error": str(e), "facts": [], "facts_found": 0}
    
    def research_topic(self, topic: str, depth: int = 1, max_results: int = 5) -> Dict[str, Any]:
        """
        Research a topic using web search
        
        Args:
            topic: Topic to research
            depth: Research depth (1-3)
            max_results: Maximum number of search results to process
            
        Returns:
            Dictionary with research results
        """
        try:
            if not self.web_researcher:
                return {
                    "success": False,
                    "error": "Web researcher not available",
                    "facts_found": 0,
                    "pages_processed": 0
                }
            
            # Use web researcher to research the topic
            results = self.web_researcher.research_topic(topic, depth, max_results)
            
            return {
                "success": True,
                "facts_found": results.get("facts_found", 0),
                "pages_processed": results.get("pages_processed", 0),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error researching topic: {e}")
            return {
                "success": False,
                "error": str(e),
                "facts_found": 0,
                "pages_processed": 0
            }
    
    def close(self):
        """Close the knowledge base connection"""
        if self.knowledge_base:
            self.knowledge_base.close()
