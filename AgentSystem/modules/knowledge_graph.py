"""
Knowledge Graph Module
--------------------
Provides advanced knowledge management with hierarchical representation,
concept relationships, knowledge consolidation, and reasoning capabilities.
"""

import os
import time
import json
import sqlite3
import numpy as np
import threading
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter

# Local imports
from AgentSystem.utils.logger import get_logger
from AgentSystem.modules.continuous_learning import KnowledgeBase

# Get module logger
logger = get_logger("modules.knowledge_graph")

try:
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        
    GRAPH_IMPORTS_AVAILABLE = True
except ImportError:
    logger.warning("Knowledge graph dependencies not available. Install with: pip install networkx scikit-learn nltk")
    GRAPH_IMPORTS_AVAILABLE = False


class ConceptNode:
    """Represents a concept in the knowledge graph"""
    
    def __init__(self, name: str, node_type: str = "concept", 
                description: Optional[str] = None, 
                attributes: Optional[Dict[str, Any]] = None):
        """
        Initialize a concept node
        
        Args:
            name: Name of the concept
            node_type: Type of node (concept, entity, event, etc.)
            description: Description of the concept
            attributes: Additional attributes
        """
        self.name = name
        self.node_type = node_type
        self.description = description
        self.attributes = attributes or {}
        
        # Tracking
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.confidence = 1.0
        self.source_ids = set()  # IDs of facts/documents that support this concept
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.node_type,
            "description": self.description,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "confidence": self.confidence,
            "source_count": len(self.source_ids)
        }
    
    def __str__(self) -> str:
        return f"ConceptNode({self.name}, {self.node_type})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ConceptRelation:
    """Represents a relation between concepts in the knowledge graph"""
    
    def __init__(self, source: str, target: str, relation_type: str, 
                weight: float = 1.0, attributes: Optional[Dict[str, Any]] = None):
        """
        Initialize a concept relation
        
        Args:
            source: Source concept name
            target: Target concept name
            relation_type: Type of relation
            weight: Relation weight/strength
            attributes: Additional attributes
        """
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.weight = weight
        self.attributes = attributes or {}
        
        # Tracking
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.confidence = 1.0
        self.source_ids = set()  # IDs of facts/documents that support this relation
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "confidence": self.confidence,
            "source_count": len(self.source_ids)
        }
    
    def __str__(self) -> str:
        return f"ConceptRelation({self.source} --[{self.relation_type}]--> {self.target})"
    
    def __repr__(self) -> str:
        return self.__str__()


class KnowledgeGraph:
    """Advanced knowledge representation with graph-based structure"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize the knowledge graph
        
        Args:
            knowledge_base: Underlying knowledge base
        """
        self.kb = knowledge_base
        self.concepts = {}  # name -> ConceptNode
        self.relations = {}  # (source, target, type) -> ConceptRelation
        
        # Graph (if networkx is available)
        self.graph = nx.DiGraph() if GRAPH_IMPORTS_AVAILABLE else None
        
        # Concept embeddings
        self.concept_embeddings = {}  # name -> embedding vector
        self.vectorizer = TfidfVectorizer(stop_words='english') if GRAPH_IMPORTS_AVAILABLE else None
        
        # Lemmatizer for normalization
        self.lemmatizer = WordNetLemmatizer() if GRAPH_IMPORTS_AVAILABLE else None
        
        # Memory management
        self.memory_decay_rate = 0.1  # Rate at which unused concepts decay
        self.memory_threshold = 0.2  # Threshold for forgetting concepts
        self.last_cleanup = datetime.now()
        self.cleanup_interval = 86400  # 24 hours
        
        # Schema/ontology
        self.concept_types = {
            "concept": "Abstract or general concept",
            "entity": "Specific named entity",
            "event": "Something that happened at a specific time",
            "attribute": "Property or characteristic",
            "process": "Series of actions or steps",
            "location": "Physical or virtual location",
            "time": "Temporal concept"
        }
        
        self.relation_types = {
            "is_a": "Hierarchical relationship (subtype)",
            "part_of": "Composition relationship",
            "has_part": "Inverse of part_of",
            "related_to": "General relationship",
            "causes": "Causal relationship",
            "follows": "Temporal sequence",
            "precedes": "Temporal sequence (inverse of follows)",
            "located_in": "Spatial relationship",
            "attribute_of": "Property relationship",
            "opposed_to": "Contrasting relationship",
            "similar_to": "Similarity relationship",
            "instance_of": "Instance relationship",
            "derives_from": "Derivation relationship",
            "used_for": "Purpose relationship"
        }
        
        # Initialize database tables
        self._init_database()
        
        # Load existing concepts and relations from DB
        self._load_from_database()
        
        logger.info(f"Initialized KnowledgeGraph with {len(self.concepts)} concepts and {len(self.relations)} relations")
    
    def _init_database(self) -> None:
        """Initialize database tables for storing graph data"""
        if not self.kb.conn:
            self.kb.init_database()
            
        try:
            cursor = self.kb.conn.cursor()
            
            # Create concepts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                description TEXT,
                attributes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed DATETIME,
                access_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 1.0,
                embedding_id INTEGER
            )
            ''')
            
            # Create concept_sources table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS concept_sources (
                concept_id INTEGER NOT NULL,
                source_id INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                PRIMARY KEY (concept_id, source_id, source_type),
                FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
            ''')
            
            # Create relations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS concept_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_concept_id INTEGER NOT NULL,
                target_concept_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                attributes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed DATETIME,
                access_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (source_concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
                FOREIGN KEY (target_concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
                UNIQUE (source_concept_id, target_concept_id, relation_type)
            )
            ''')
            
            # Create relation_sources table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS relation_sources (
                relation_id INTEGER NOT NULL,
                source_id INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                PRIMARY KEY (relation_id, source_id, source_type),
                FOREIGN KEY (relation_id) REFERENCES concept_relations (id) ON DELETE CASCADE
            )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts (name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts (type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_relations_source ON concept_relations (source_concept_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_relations_target ON concept_relations (target_concept_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_relations_type ON concept_relations (relation_type)')
            
            # Create virtual table for full-text search
            cursor.execute('CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(name, description)')
            
            self.kb.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
    
    def _load_from_database(self) -> None:
        """Load concepts and relations from the database"""
        if not self.kb.conn:
            return
            
        try:
            cursor = self.kb.conn.cursor()
            
            # Load concepts
            cursor.execute('''
            SELECT id, name, type, description, attributes, created_at, last_accessed, access_count, confidence
            FROM concepts
            ''')
            
            for row in cursor.fetchall():
                concept_id, name, node_type, description, attributes_json, created_at, last_accessed, access_count, confidence = row
                
                # Parse attributes
                attributes = json.loads(attributes_json) if attributes_json else {}
                
                # Create concept
                concept = ConceptNode(name, node_type, description, attributes)
                concept.created_at = datetime.fromisoformat(created_at) if created_at else datetime.now()
                concept.last_accessed = datetime.fromisoformat(last_accessed) if last_accessed else datetime.now()
                concept.access_count = access_count
                concept.confidence = confidence
                
                # Add to memory
                self.concepts[name] = concept
                
                # Add to graph
                if self.graph is not None:
                    self.graph.add_node(name, 
                                      type=node_type, 
                                      description=description,
                                      attributes=attributes,
                                      created_at=created_at,
                                      last_accessed=last_accessed,
                                      access_count=access_count,
                                      confidence=confidence)
                
                # Load concept sources
                cursor.execute('''
                SELECT source_id, source_type FROM concept_sources
                WHERE concept_id = ?
                ''', (concept_id,))
                
                for source_row in cursor.fetchall():
                    source_id, source_type = source_row
                    concept.source_ids.add((source_id, source_type))
            
            # Load relations
            cursor.execute('''
            SELECT r.id, c1.name, c2.name, r.relation_type, r.weight, r.attributes, 
                   r.created_at, r.last_accessed, r.access_count, r.confidence
            FROM concept_relations r
            JOIN concepts c1 ON r.source_concept_id = c1.id
            JOIN concepts c2 ON r.target_concept_id = c2.id
            ''')
            
            for row in cursor.fetchall():
                rel_id, source, target, relation_type, weight, attributes_json, created_at, last_accessed, access_count, confidence = row
                
                # Parse attributes
                attributes = json.loads(attributes_json) if attributes_json else {}
                
                # Create relation
                relation = ConceptRelation(source, target, relation_type, weight, attributes)
                relation.created_at = datetime.fromisoformat(created_at) if created_at else datetime.now()
                relation.last_accessed = datetime.fromisoformat(last_accessed) if last_accessed else datetime.now()
                relation.access_count = access_count
                relation.confidence = confidence
                
                # Add to memory
                relation_key = (source, target, relation_type)
                self.relations[relation_key] = relation
                
                # Add to graph
                if self.graph is not None:
                    self.graph.add_edge(source, target, 
                                      type=relation_type,
                                      weight=weight,
                                      attributes=attributes,
                                      created_at=created_at,
                                      last_accessed=last_accessed,
                                      access_count=access_count,
                                      confidence=confidence)
                
                # Load relation sources
                cursor.execute('''
                SELECT source_id, source_type FROM relation_sources
                WHERE relation_id = ?
                ''', (rel_id,))
                
                for source_row in cursor.fetchall():
                    source_id, source_type = source_row
                    relation.source_ids.add((source_id, source_type))
                    
        except sqlite3.Error as e:
            logger.error(f"Error loading from database: {e}")
    
    def _save_concept_to_db(self, concept: ConceptNode) -> int:
        """
        Save a concept to the database
        
        Args:
            concept: Concept node to save
            
        Returns:
            Concept ID
        """
        if not self.kb.conn:
            return -1
            
        try:
            cursor = self.kb.conn.cursor()
            
            # Check if concept already exists
            cursor.execute(
                "SELECT id FROM concepts WHERE name = ?",
                (concept.name,)
            )
            
            row = cursor.fetchone()
            if row:
                concept_id = row[0]
                
                # Update existing concept
                cursor.execute(
                    """
                    UPDATE concepts SET
                    type = ?, description = ?, attributes = ?,
                    last_accessed = ?, access_count = ?, confidence = ?
                    WHERE id = ?
                    """,
                    (
                        concept.node_type,
                        concept.description,
                        json.dumps(concept.attributes),
                        concept.last_accessed.isoformat(),
                        concept.access_count,
                        concept.confidence,
                        concept_id
                    )
                )
            else:
                # Insert new concept
                cursor.execute(
                    """
                    INSERT INTO concepts
                    (name, type, description, attributes, created_at, last_accessed, access_count, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        concept.name,
                        concept.node_type,
                        concept.description,
                        json.dumps(concept.attributes),
                        concept.created_at.isoformat(),
                        concept.last_accessed.isoformat(),
                        concept.access_count,
                        concept.confidence
                    )
                )
                
                concept_id = cursor.lastrowid
                
                # Add to full-text search
                cursor.execute(
                    "INSERT INTO concepts_fts (rowid, name, description) VALUES (?, ?, ?)",
                    (concept_id, concept.name, concept.description or "")
                )
            
            # Update concept sources
            # First delete existing sources
            cursor.execute(
                "DELETE FROM concept_sources WHERE concept_id = ?",
                (concept_id,)
            )
            
            # Then insert current sources
            for source_id, source_type in concept.source_ids:
                cursor.execute(
                    "INSERT INTO concept_sources (concept_id, source_id, source_type) VALUES (?, ?, ?)",
                    (concept_id, source_id, source_type)
                )
            
            self.kb.conn.commit()
            return concept_id
            
        except sqlite3.Error as e:
            logger.error(f"Error saving concept to database: {e}")
            return -1
    
    def _save_relation_to_db(self, relation: ConceptRelation) -> int:
        """
        Save a relation to the database
        
        Args:
            relation: Concept relation to save
            
        Returns:
            Relation ID
        """
        if not self.kb.conn:
            return -1
            
        try:
            cursor = self.kb.conn.cursor()
            
            # Get concept IDs
            cursor.execute("SELECT id FROM concepts WHERE name = ?", (relation.source,))
            source_row = cursor.fetchone()
            
            cursor.execute("SELECT id FROM concepts WHERE name = ?", (relation.target,))
            target_row = cursor.fetchone()
            
            if not source_row or not target_row:
                logger.error(f"Cannot save relation: source or target concept not found")
                return -1
                
            source_id = source_row[0]
            target_id = target_row[0]
            
            # Check if relation already exists
            cursor.execute(
                """
                SELECT id FROM concept_relations 
                WHERE source_concept_id = ? AND target_concept_id = ? AND relation_type = ?
                """,
                (source_id, target_id, relation.relation_type)
            )
            
            row = cursor.fetchone()
            if row:
                relation_id = row[0]
                
                # Update existing relation
                cursor.execute(
                    """
                    UPDATE concept_relations SET
                    weight = ?, attributes = ?,
                    last_accessed = ?, access_count = ?, confidence = ?
                    WHERE id = ?
                    """,
                    (
                        relation.weight,
                        json.dumps(relation.attributes),
                        relation.last_accessed.isoformat(),
                        relation.access_count,
                        relation.confidence,
                        relation_id
                    )
                )
            else:
                # Insert new relation
                cursor.execute(
                    """
                    INSERT INTO concept_relations
                    (source_concept_id, target_concept_id, relation_type, weight, attributes, 
                     created_at, last_accessed, access_count, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        target_id,
                        relation.relation_type,
                        relation.weight,
                        json.dumps(relation.attributes),
                        relation.created_at.isoformat(),
                        relation.last_accessed.isoformat(),
                        relation.access_count,
                        relation.confidence
                    )
                )
                
                relation_id = cursor.lastrowid
            
            # Update relation sources
            # First delete existing sources
            cursor.execute(
                "DELETE FROM relation_sources WHERE relation_id = ?",
                (relation_id,)
            )
            
            # Then insert current sources
            for source_id, source_type in relation.source_ids:
                cursor.execute(
                    "INSERT INTO relation_sources (relation_id, source_id, source_type) VALUES (?, ?, ?)",
                    (relation_id, source_id, source_type)
                )
            
            self.kb.conn.commit()
            return relation_id
            
        except sqlite3.Error as e:
            logger.error(f"Error saving relation to database: {e}")
            return -1
    
    def add_concept(self, name: str, node_type: str = "concept", 
                   description: Optional[str] = None, 
                   attributes: Optional[Dict[str, Any]] = None,
                   source_id: Optional[int] = None,
                   source_type: Optional[str] = None) -> ConceptNode:
        """
        Add a concept to the knowledge graph
        
        Args:
            name: Name of the concept
            node_type: Type of node
            description: Description of the concept
            attributes: Additional attributes
            source_id: ID of the source fact or document
            source_type: Type of the source ('fact' or 'document')
            
        Returns:
            Added concept node
        """
        # Normalize concept name
        name = name.strip().lower()
        
        # Check if concept already exists
        if name in self.concepts:
            concept = self.concepts[name]
            concept.last_accessed = datetime.now()
            concept.access_count += 1
            
            # Update description if new one is provided
            if description and not concept.description:
                concept.description = description
            
            # Update attributes
            if attributes:
                concept.attributes.update(attributes)
            
            # Add source if provided
            if source_id and source_type:
                concept.source_ids.add((source_id, source_type))
                concept.confidence = min(1.0, concept.confidence + 0.1)  # Increase confidence
                
        else:
            # Create new concept
            concept = ConceptNode(name, node_type, description, attributes)
            
            if source_id and source_type:
                concept.source_ids.add((source_id, source_type))
            
            self.concepts[name] = concept
            
            # Add to graph
            if self.graph is not None:
                self.graph.add_node(name, 
                                  type=node_type, 
                                  description=description,
                                  attributes=attributes)
        
        # Update embeddings
        if GRAPH_IMPORTS_AVAILABLE and self.vectorizer is not None:
            text = f"{name} {description or ''}"
            try:
                # Transform single document
                if not hasattr(self.vectorizer, 'vocabulary_'):
                    # First time, fit on this document
                    matrix = self.vectorizer.fit_transform([text])
                else:
                    # Already fitted, just transform
                    matrix = self.vectorizer.transform([text])
                
                # Extract the vector
                self.concept_embeddings[name] = matrix.toarray()[0]
            except Exception as e:
                logger.error(f"Error creating concept embedding: {e}")
        
        # Save to database
        self._save_concept_to_db(concept)
        
        return concept
    
    def add_relation(self, source: str, target: str, relation_type: str,
                    weight: float = 1.0, attributes: Optional[Dict[str, Any]] = None,
                    source_id: Optional[int] = None, 
                    source_type: Optional[str] = None) -> ConceptRelation:
        """
        Add a relation between concepts
        
        Args:
            source: Source concept name
            target: Target concept name
            relation_type: Type of relation
            weight: Relation weight/strength
            attributes: Additional attributes
            source_id: ID of the source fact or document
            source_type: Type of the source ('fact' or 'document')
            
        Returns:
            Added relation
        """
        # Normalize concept names
        source = source.strip().lower()
        target = target.strip().lower()
        
        # Ensure concepts exist
        if source not in self.concepts:
            self.add_concept(source)
        
        if target not in self.concepts:
            self.add_concept(target)
        
        # Check if relation already exists
        relation_key = (source, target, relation_type)
        if relation_key in self.relations:
            relation = self.relations[relation_key]
            relation.last_accessed = datetime.now()
            relation.access_count += 1
            
            # Update weight (use max)
            relation.weight = max(relation.weight, weight)
            
            # Update attributes
            if attributes:
                relation.attributes.update(attributes)
            
            # Add source if provided
            if source_id and source_type:
                relation.source_ids.add((source_id, source_type))
                relation.confidence = min(1.0, relation.confidence + 0.1)  # Increase confidence
                
        else:
            # Create new relation
            relation = ConceptRelation(source, target, relation_type, weight, attributes)
            
            if source_id and source_type:
                relation.source_ids.add((source_id, source_type))
            
            self.relations[relation_key] = relation
            
            # Add to graph
            if self.graph is not None:
                self.graph.add_edge(source, target, 
                                  type=relation_type,
                                  weight=weight,
                                  attributes=attributes)
        
        # Save to database
        self._save_relation_to_db(relation)
        
        return relation
    
    def get_concept(self, name: str) -> Optional[ConceptNode]:
        """
        Get a concept by name
        
        Args:
            name: Concept name
            
        Returns:
            Concept node or None if not found
        """
        # Normalize concept name
        name = name.strip().lower()
        
        if name in self.concepts:
            concept = self.concepts[name]
            concept.last_accessed = datetime.now()
            concept.access_count += 1
            return concept
        
        return None
    
    def get_relation(self, source: str, target: str, relation_type: Optional[str] = None) -> Optional[ConceptRelation]:
        """
        Get a relation between concepts
        
        Args:
            source: Source concept name
            target: Target concept name
            relation_type: Type of relation (optional)
            
        Returns:
            Concept relation or None if not found
        """
        # Normalize concept names
        source = source.strip().lower()
        target = target.strip().lower()
        
        if relation_type:
            relation_key = (source, target, relation_type)
            if relation_key in self.relations:
                relation = self.relations[relation_key]
                relation.last_accessed = datetime.now()
                relation.access_count += 1
                return relation
        else:
            # Find any relation between source and target
            for key, relation in self.relations.items():
                if key[0] == source and key[1] == target:
                    relation.last_accessed = datetime.now()
                    relation.access_count += 1
                    return relation
        
        return None
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for concepts
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        if not self.kb.conn:
            return []
            
        try:
            cursor = self.kb.conn.cursor()
            
            # Use full-text search
            cursor.execute(
                """
                SELECT c.id, c.name, c.type, c.description, c.confidence, c.access_count
                FROM concepts_fts fts
                JOIN concepts c ON fts.rowid = c.id
                WHERE fts.name MATCH ? OR fts.description MATCH ?
                ORDER BY c.confidence DESC, c.access_count DESC
                LIMIT ?
                """,
                (query, query, limit)
            )
            
            results = []
            for row in cursor.fetchall():
                concept_id, name, node_type, description, confidence, access_count = row
                
                if name in self.concepts:
                    # Update access stats
                    concept = self.concepts[name]
                    concept.last_accessed = datetime.now()
                    concept.access_count += 1
                    
                    results.append(concept.to_dict())
            
            # Update database
            if results:
                concept_ids = [row[0] for row in cursor.fetchall()]
                placeholders = ", ".join(["?"] * len(concept_ids))
                cursor.execute(
                    f"UPDATE concepts SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id IN ({placeholders})",
                    concept_ids
                )
                self.kb.conn.commit()
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error searching concepts: {e}")
            return []
    
    def get_concept_neighborhood(self, concept_name: str, max_distance: int = 2) -> Dict[str, Any]:
        """
        Get the neighborhood of a concept
        
        Args:
            concept_name: Name of the central concept
            max_distance: Maximum distance from the central concept
            
        Returns:
            Dictionary with neighborhood information
        """
        # Normalize concept name
        concept_name = concept_name.strip().lower()
        
        if concept_name not in self.concepts:
            return {
                "concept": None,
                "neighbors": [],
                "error": f"Concept '{concept_name}' not found"
            }
        
        concept = self.concepts[concept_name]
        concept.last_accessed = datetime.now()
        concept.access_count += 1
        
        # Get neighborhood
        neighbors = []
        
        if self.graph is not None:
            # Use networkx for efficient neighborhood extraction
            try:
                # Get subgraph within max_distance
                neighborhood = nx.ego_graph(self.graph, concept_name, radius=max_distance)
                
                for node in neighborhood.nodes():
                    if node == concept_name:
                        continue
                    
                    # Find the shortest path
                    path = nx.shortest_path(neighborhood, concept_name, node)
                    distance = len(path) - 1
                    
                    # Get edge data along the path
                    edges = []
                    for i in range(len(path) - 1):
                        src, tgt = path[i], path[i + 1]
                        edge_data = neighborhood.get_edge_data(src, tgt)
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "type": edge_data.get('type', 'related_to'),
                            "weight": edge_data.get('weight', 1.0)
                        })
                    
                    # Get node data
                    node_data = neighborhood.nodes[node]
                    
                    neighbors.append({
                        "name": node,
                        "type": node_data.get('type', 'concept'),
                        "description": node_data.get('description', ''),
                        "distance": distance,
                        "path": path,
                        "path_edges": edges
                    })
                
                # Sort by distance
                neighbors.sort(key=lambda x: x['distance'])
                
            except Exception as e:
                logger.error(f"Error getting neighborhood: {e}")
        else:
            # Manual neighborhood search
            for relation_key, relation in self.relations.items():
                src, tgt, rel_type = relation_key
                
                # Direct neighbors
                if src == concept_name and max_distance >= 1:
                    if tgt not in [n['name'] for n in neighbors]:
                        neighbors.append({
                            "name": tgt,
                            "type": self.concepts[tgt].node_type,
                            "description": self.concepts[tgt].description,
                            "distance": 1,
                            "path": [concept_name, tgt],
                            "path_edges": [{
                                "source": src,
                                "target": tgt,
                                "type": rel_type,
                                "weight": relation.weight
                            }]
                        })
                
                # Incoming edges
                if tgt == concept_name and max_distance >= 1:
                    if src not in [n['name'] for n in neighbors]:
                        neighbors.append({
                            "name": src,
                            "type": self.concepts[src].node_type,
                            "description": self.concepts[src].description,
                            "distance": 1,
                            "path": [concept_name, src],
                            "path_edges": [{
                                "source": tgt,
                                "target": src,
                                "type": rel_type,
                                "weight": relation.weight
                            }]
                        })
        
        return {
            "concept": concept.to_dict(),
            "neighbors": neighbors,
            "total_neighbors": len(neighbors)
        }
    
    def find_paths(self, source: str, target: str, max_length: int = 4) -> List[Dict[str, Any]]:
        """
        Find paths between two concepts
        
        Args:
            source: Source concept name
            target: Target concept name
            max_length: Maximum path length
            
        Returns:
            List of paths between concepts
        """
        # Normalize concept names
        source = source.strip().lower()
        target = target.strip().lower()
        
        if source not in self.concepts or target not in self.concepts:
            return []
        
        # Update access stats
        self.concepts[source].last_accessed = datetime.now()
        self.concepts[source].access_count += 1
        self.concepts[target].last_accessed = datetime.now()
        self.concepts[target].access_count += 1
        
        # Find paths
        paths = []
        
        if self.graph is not None:
            try:
                # Use networkx to find paths
                all_paths = list(nx.all_simple_paths(
                    self.graph, source, target, cutoff=max_length
                ))
                
                for path in all_paths:
                    # Get edge data along the path
                    edges = []
                    for i in range(len(path) - 1):
                        src, tgt = path[i], path[i + 1]
                        edge_data = self.graph.get_edge_data(src, tgt)
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "type": edge_data.get('type', 'related_to'),
                            "weight": edge_data.get('weight', 1.0)
                        })
                    
                    paths.append({
                        "path": path,
                        "length": len(path) - 1,
                        "edges": edges
                    })
                
                # Sort by path length
                paths.sort(key=lambda x: x['length'])
                
            except Exception as e:
                logger.error(f"Error finding paths: {e}")
        else:
            # Simple BFS search
            # This is a simplified version and won't find all paths
            queue = [(source, [source], [])]
            visited = set()
            
            while queue:
                current, path, edges = queue.pop(0)
                
                if current == target:
                    paths.append({
                        "path": path,
                        "length": len(path) - 1,
                        "edges": edges
                    })
                    continue
                
                if len(path) > max_length or current in visited:
                    continue
                
                visited.add(current)
                
                # Find outgoing relations
                for relation_key, relation in self.relations.items():
                    src, tgt, rel_type = relation_key
                    
                    if src == current and tgt not in path:
                        new_path = path + [tgt]
                        new_edges = edges + [{
                            "source": src,
                            "target": tgt,
                            "type": rel_type,
                            "weight": relation.weight
                        }]
                        queue.append((tgt, new_path, new_edges))
        
        return paths
    
    def get_related_concepts(self, concept_name: str, relation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get concepts related to the specified concept
        
        Args:
            concept_name: Name of the concept
            relation_type: Type of relation to filter by (optional)
            
        Returns:
            Dictionary with related concepts
        """
        # Normalize concept name
        concept_name = concept_name.strip().lower()
        
        if concept_name not in self.concepts:
            return {
                "concept": None,
                "outgoing_relations": [],
                "incoming_relations": [],
                "error": f"Concept '{concept_name}' not found"
            }
        
        concept = self.concepts[concept_name]
        concept.last_accessed = datetime.now()
        concept.access_count += 1
        
        # Get related concepts
        outgoing_relations = []
        incoming_relations = []
        
        for relation_key, relation in self.relations.items():
            src, tgt, rel_type = relation_key
            
            # Filter by relation type if specified
            if relation_type and rel_type != relation_type:
                continue
            
            # Outgoing relations
            if src == concept_name:
                outgoing_relations.append({
                    "target": tgt,
                    "relation_type": rel_type,
                    "weight": relation.weight,
                    "confidence": relation.confidence,
                    "target_type": self.concepts[tgt].node_type if tgt in self.concepts else "unknown"
                })
            
            # Incoming relations
            if tgt == concept_name:
                incoming_relations.append({
                    "source": src,
                    "relation_type": rel_type,
                    "weight": relation.weight,
                    "confidence": relation.confidence,
                    "source_type": self.concepts[src].node_type if src in self.concepts else "unknown"
                })
        
        return {
            "concept": concept.to_dict(),
            "outgoing_relations": outgoing_relations,
            "incoming_relations": incoming_relations,
            "total_relations": len(outgoing_relations) + len(incoming_relations)
        }
    
    def find_similar_concepts(self, concept_name: str, max_results: int = 10, 
                             threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find concepts similar to the specified concept
        
        Args:
            concept_name: Name of the concept
            max_results: Maximum number of results
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            List of similar concepts
        """
        # Normalize concept name
        concept_name = concept_name.strip().lower()
        
        if concept_name not in self.concepts:
            return []
        
        concept = self.concepts[concept_name]
        concept.last_accessed = datetime.now()
        concept.access_count += 1
        
        # Check if we have embeddings
        if not GRAPH_IMPORTS_AVAILABLE or concept_name not in self.concept_embeddings:
            # Fall back to relation-based similarity
            return self._find_similar_by_relations(concept_name, max_results)
        
        # Get embedding for the query concept
        query_embedding = self.concept_embeddings[concept_name]
        
        # Calculate similarity with all other concepts
        similarities = []
        
        for name, embedding in self.concept_embeddings.items():
            if name == concept_name:
                continue
            
            # Calculate cosine similarity
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            
            if similarity >= threshold:
                similarities.append({
                    "name": name,
                    "similarity": float(similarity),
                    "type": self.concepts[name].node_type,
                    "description": self.concepts[name].description
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:max_results]
    
    def _find_similar_by_relations(self, concept_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find similar concepts based on shared relations
        
        Args:
            concept_name: Name of the concept
            max_results: Maximum number of results
            
        Returns:
            List of similar concepts with shared relations
        """
        # Get all relations involving the concept
        concept_relations = set()
        relation_concepts = set()
        
        for relation_key in self.relations:
            src, tgt, rel_type = relation_key
            
            if src == concept_name:
                concept_relations.add((src, tgt, rel_type))
                relation_concepts.add(tgt)
            
            if tgt == concept_name:
                concept_relations.add((src, tgt, rel_type))
                relation_concepts.add(src)
        
        # Find concepts with shared relations
        similarities = []
        
        for other_concept in self.concepts:
            if other_concept == concept_name:
                continue
            
            # Skip concepts not involved in any relations with the target concept
            if other_concept not in relation_concepts:
                continue
            
            # Count shared relations
            shared_relations = 0
            total_relations = 0
            
            for relation_key in self.relations:
                src, tgt, rel_type = relation_key
                
                if src == other_concept or tgt == other_concept:
                    total_relations += 1
                    
                    # Check if this relation involves the query concept
                    if src == concept_name or tgt == concept_name:
                        shared_relations += 1
            
            if total_relations > 0:
                similarity = shared_relations / total_relations
                
                similarities.append({
                    "name": other_concept,
                    "similarity": similarity,
                    "shared_relations": shared_relations,
                    "total_relations": total_relations,
                    "type": self.concepts[other_concept].node_type,
                    "description": self.concepts[other_concept].description
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:max_results]
    
    def extract_concepts_from_text(self, text: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Extract concepts and relations from text
        
        Args:
            text: Text to process
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with extracted concepts and relations
        """
        if not GRAPH_IMPORTS_AVAILABLE:
            return {
                "error": "NLP dependencies not available",
                "concepts": [],
                "relations": []
            }
        
        try:
            # Extract concepts
            extracted_concepts = []
            extracted_relations = []
            
            # Tokenize text
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                # Process sentence
                words = word_tokenize(sentence.lower())
                pos_tags = nltk.pos_tag(words)
                
                # Extract noun phrases as potential concepts
                noun_phrases = []
                current_phrase = []
                
                for word, tag in pos_tags:
                    # Skip stopwords and punctuation
                    if word in stopwords.words('english') or not word.isalnum():
                        if current_phrase:
                            noun_phrases.append(' '.join(current_phrase))
                            current_phrase = []
                        continue
                    
                    # Add nouns and adjectives to current phrase
                    if tag.startswith('NN') or tag.startswith('JJ'):
                        current_phrase.append(word)
                    else:
                        if current_phrase:
                            noun_phrases.append(' '.join(current_phrase))
                            current_phrase = []
                
                if current_phrase:
                    noun_phrases.append(' '.join(current_phrase))
                
                # Add extracted concepts
                for phrase in noun_phrases:
                    if len(phrase.split()) > 0:  # Ensure non-empty
                        # Normalize concept name
                        concept_name = phrase.strip().lower()
                        
                        # Skip if too short
                        if len(concept_name) < 3:
                            continue
                        
                        # Add to extracted concepts
                        if concept_name not in [c["name"] for c in extracted_concepts]:
                            extracted_concepts.append({
                                "name": concept_name,
                                "type": "concept",
                                "confidence": min_confidence,
                                "source_text": sentence
                            })
                
                # Try to extract relations between concepts
                if len(noun_phrases) >= 2:
                    for i in range(len(noun_phrases) - 1):
                        source = noun_phrases[i].strip().lower()
                        target = noun_phrases[i+1].strip().lower()
                        
                        # Skip if either concept is too short
                        if len(source) < 3 or len(target) < 3:
                            continue
                        
                        # Determine relation type
                        relation_type = "related_to"  # Default
                        
                        # Check for specific relation indicators
                        lower_sentence = sentence.lower()
                        
                        if " is a " in lower_sentence or " are a " in lower_sentence:
                            relation_type = "is_a"
                        elif " part of " in lower_sentence or " contains " in lower_sentence:
                            relation_type = "part_of"
                        elif " causes " in lower_sentence or " leads to " in lower_sentence:
                            relation_type = "causes"
                        elif " before " in lower_sentence or " after " in lower_sentence:
                            relation_type = "follows"
                        elif " used for " in lower_sentence or " purpose " in lower_sentence:
                            relation_type = "used_for"
                        
                        # Add to extracted relations
                        relation_key = (source, target, relation_type)
                        if relation_key not in [(r["source"], r["target"], r["type"]) for r in extracted_relations]:
                            extracted_relations.append({
                                "source": source,
                                "target": target,
                                "type": relation_type,
                                "confidence": min_confidence * 0.8,  # Lower confidence for relations
                                "source_text": sentence
                            })
            
            return {
                "concepts": extracted_concepts,
                "relations": extracted_relations,
                "concept_count": len(extracted_concepts),
                "relation_count": len(extracted_relations)
            }
            
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return {
                "error": str(e),
                "concepts": [],
                "relations": []
            }
    
    def add_extracted_concepts(self, extraction_result: Dict[str, Any], 
                              source_id: Optional[int] = None,
                              source_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Add extracted concepts and relations to the knowledge graph
        
        Args:
            extraction_result: Result from extract_concepts_from_text
            source_id: ID of the source fact or document
            source_type: Type of the source ('fact' or 'document')
            
        Returns:
            Dictionary with result information
        """
        if "error" in extraction_result:
            return {
                "success": False,
                "error": extraction_result["error"]
            }
        
        # Add concepts
        added_concepts = []
        for concept_data in extraction_result["concepts"]:
            concept = self.add_concept(
                name=concept_data["name"],
                node_type=concept_data.get("type", "concept"),
                description=concept_data.get("source_text", ""),
                source_id=source_id,
                source_type=source_type
            )
            
            added_concepts.append(concept.name)
        
        # Add relations
        added_relations = []
        for relation_data in extraction_result["relations"]:
            relation = self.add_relation(
                source=relation_data["source"],
                target=relation_data["target"],
                relation_type=relation_data["type"],
                weight=relation_data.get("confidence", 0.5),
                source_id=source_id,
                source_type=source_type
            )
            
            added_relations.append((relation.source, relation.target, relation.relation_type))
        
        return {
            "success": True,
            "added_concepts": added_concepts,
            "added_relations": added_relations,
            "concept_count": len(added_concepts),
            "relation_count": len(added_relations)
        }
    
    def perform_memory_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform memory cleanup (forgetting)
        
        Args:
            force: Force cleanup even if interval hasn't elapsed
            
        Returns:
            Dictionary with cleanup results
        """
        # Check if cleanup is needed
        now = datetime.now()
        time_since_cleanup = (now - self.last_cleanup).total_seconds()
        
        if not force and time_since_cleanup < self.cleanup_interval:
            return {
                "success": True,
                "message": "Cleanup not needed yet",
                "next_cleanup_in": self.cleanup_interval - time_since_cleanup
            }
        
        # Identify concepts to forget
        forgotten_concepts = []
        forgotten_relations = []
        
        # Calculate memory strength for concepts
        for name, concept in list(self.concepts.items()):
            # Skip concepts with sources
            if concept.source_ids:
                continue
                
            # Calculate time factor (decay based on time since last access)
            time_factor = (now - concept.last_accessed).total_seconds() / (86400 * 30)  # 30 days
            
            # Calculate memory strength
            memory_strength = concept.confidence * (1.0 - self.memory_decay_rate * time_factor)
            
            # Adjust for access count
            memory_strength *= (1.0 + 0.1 * min(concept.access_count, 10))
            
            # Check if below threshold
            if memory_strength < self.memory_threshold:
                # Forget concept
                forgotten_concepts.append(name)
                del self.concepts[name]
                
                # Remove from graph
                if self.graph is not None and name in self.graph:
                    self.graph.remove_node(name)
                
                # Remove from database
                if self.kb.conn:
                    try:
                        cursor = self.kb.conn.cursor()
                        cursor.execute("DELETE FROM concepts WHERE name = ?", (name,))
                        self.kb.conn.commit()
                    except sqlite3.Error as e:
                        logger.error(f"Error deleting concept from database: {e}")
        
        # Calculate memory strength for relations
        for relation_key, relation in list(self.relations.items()):
            # Skip relations with sources
            if relation.source_ids:
                continue
                
            # Calculate time factor (decay based on time since last access)
            time_factor = (now - relation.last_accessed).total_seconds() / (86400 * 30)  # 30 days
            
            # Calculate memory strength
            memory_strength = relation.confidence * (1.0 - self.memory_decay_rate * time_factor)
            
            # Adjust for access count
            memory_strength *= (1.0 + 0.1 * min(relation.access_count, 10))
            
            # Check if below threshold
            if memory_strength < self.memory_threshold:
                # Forget relation
                forgotten_relations.append(relation_key)
                del self.relations[relation_key]
                
                # Remove from graph
                if self.graph is not None:
                    source, target, rel_type = relation_key
                    if self.graph.has_edge(source, target):
                        self.graph.remove_edge(source, target)
                
                # Remove from database
                if self.kb.conn:
                    try:
                        cursor = self.kb.conn.cursor()
                        cursor.execute("""
                            DELETE FROM concept_relations 
                            WHERE source_concept_id = (SELECT id FROM concepts WHERE name = ?)
                            AND target_concept_id = (SELECT id FROM concepts WHERE name = ?)
                            AND relation_type = ?
                        """, (relation_key[0], relation_key[1], relation_key[2]))
                        self.kb.conn.commit()
                    except sqlite3.Error as e:
                        logger.error(f"Error deleting relation from database: {e}")
        
        # Update last cleanup time
        self.last_cleanup = now
        
        return {
            "success": True,
            "forgotten_concepts": forgotten_concepts,
            "forgotten_relations": forgotten_relations,
            "concept_count": len(forgotten_concepts),
            "relation_count": len(forgotten_relations),
            "memory_threshold": self.memory_threshold,
            "memory_decay_rate": self.memory_decay_rate
        }
    
    def merge_similar_concepts(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Merge similar concepts to consolidate knowledge
        
        Args:
            threshold: Similarity threshold for merging (0.0-1.0)
            
        Returns:
            Dictionary with merge results
        """
        if not GRAPH_IMPORTS_AVAILABLE:
            return {
                "success": False,
                "error": "NLP dependencies not available"
            }
        
        # Find similar concept pairs
        merge_candidates = []
        
        # Use embeddings if available
        if self.concept_embeddings:
            concept_names = list(self.concept_embeddings.keys())
            
            for i in range(len(concept_names)):
                for j in range(i + 1, len(concept_names)):
                    name1 = concept_names[i]
                    name2 = concept_names[j]
                    
                    # Skip if either concept doesn't exist anymore
                    if name1 not in self.concepts or name2 not in self.concepts:
                        continue
                    
                    # Calculate similarity
                    similarity = cosine_similarity(
                        [self.concept_embeddings[name1]], 
                        [self.concept_embeddings[name2]]
                    )[0][0]
                    
                    if similarity >= threshold:
                        merge_candidates.append({
                            "concept1": name1,
                            "concept2": name2,
                            "similarity": float(similarity),
                            "method": "embedding"
                        })
        
        # Also check for string similarity
        for name1 in self.concepts:
            for name2 in self.concepts:
                if name1 >= name2:  # Skip duplicates and self-comparisons
                    continue
                
                # Skip if already a merge candidate
                if any(c["concept1"] == name1 and c["concept2"] == name2 for c in merge_candidates):
                    continue
                
                # Calculate string similarity
                # Simple method: check if one is contained in the other
                if name1 in name2 or name2 in name1:
                    similarity = len(set(name1.split()) & set(name2.split())) / max(len(name1.split()), len(name2.split()))
                    
                    if similarity >= threshold:
                        merge_candidates.append({
                            "concept1": name1,
                            "concept2": name2,
                            "similarity": similarity,
                            "method": "string"
                        })
        
        # Sort by similarity (descending)
        merge_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Perform merges
        merged_concepts = []
        
        for candidate in merge_candidates:
            name1 = candidate["concept1"]
            name2 = candidate["concept2"]
            
            # Skip if either concept has already been merged
            if name1 not in self.concepts or name2 not in self.concepts:
                continue
            
            # Merge concepts
            merged_name = self._merge_concepts(name1, name2)
            
            merged_concepts.append({
                "primary": merged_name,
                "merged": [name1, name2] if merged_name != name1 else [name2],
                "similarity": candidate["similarity"],
                "method": candidate["method"]
            })
        
        return {
            "success": True,
            "merged_concepts": merged_concepts,
            "merge_count": len(merged_concepts),
            "threshold": threshold
        }
    
    def _merge_concepts(self, name1: str, name2: str) -> str:
        """
        Merge two concepts
        
        Args:
            name1: First concept name
            name2: Second concept name
            
        Returns:
            Name of the primary concept after merging
        """
        # Determine primary concept (the one with higher confidence or more sources)
        concept1 = self.concepts[name1]
        concept2 = self.concepts[name2]
        
        # Use the concept with higher confidence as primary
        if concept1.confidence >= concept2.confidence:
            primary = name1
            secondary = name2
            primary_concept = concept1
            secondary_concept = concept2
        else:
            primary = name2
            secondary = name1
            primary_concept = concept2
            secondary_concept = concept1
        
        # Update primary concept
        # Merge descriptions
        if not primary_concept.description and secondary_concept.description:
            primary_concept.description = secondary_concept.description
        elif primary_concept.description and secondary_concept.description:
            primary_concept.description = f"{primary_concept.description}. {secondary_concept.description}"
        
        # Merge attributes
        primary_concept.attributes.update(secondary_concept.attributes)
        
        # Merge sources
        primary_concept.source_ids.update(secondary_concept.source_ids)
        
        # Update confidence (use max)
        primary_concept.confidence = max(primary_concept.confidence, secondary_concept.confidence)
        
        # Merge access stats
        primary_concept.access_count += secondary_concept.access_count
        
        # Update database
        self._save_concept_to_db(primary_concept)
        
        # Update relations
        for relation_key, relation in list(self.relations.items()):
            source, target, rel_type = relation_key
            
            # Update source
            if source == secondary:
                # Create new relation with primary as source
                new_key = (primary, target, rel_type)
                
                if new_key not in self.relations:
                    self.relations[new_key] = ConceptRelation(
                        primary, target, rel_type, relation.weight, relation.attributes
                    )
                    self.relations[new_key].source_ids.update(relation.source_ids)
                    self.relations[new_key].confidence = relation.confidence
                    self.relations[new_key].access_count = relation.access_count
                else:
                    # Merge with existing relation
                    existing = self.relations[new_key]
                    existing.weight = max(existing.weight, relation.weight)
                    existing.attributes.update(relation.attributes)
                    existing.source_ids.update(relation.source_ids)
                    existing.confidence = max(existing.confidence, relation.confidence)
                    existing.access_count += relation.access_count
                
                # Update graph
                if self.graph is not None and self.graph.has_edge(source, target):
                    edge_data = self.graph.get_edge_data(source, target)
                    self.graph.add_edge(primary, target, **edge_data)
                    self.graph.remove_edge(source, target)
                
                # Remove old relation
                del self.relations[relation_key]
            
            # Update target
            if target == secondary:
                # Create new relation with primary as target
                new_key = (source, primary, rel_type)
                
                if new_key not in self.relations:
                    self.relations[new_key] = ConceptRelation(
                        source, primary, rel_type, relation.weight, relation.attributes
                    )
                    self.relations[new_key].source_ids.update(relation.source_ids)
                    self.relations[new_key].confidence = relation.confidence
                    self.relations[new_key].access_count = relation.access_count
                else:
                    # Merge with existing relation
                    existing = self.relations[new_key]
                    existing.weight = max(existing.weight, relation.weight)
                    existing.attributes.update(relation.attributes)
                    existing.source_ids.update(relation.source_ids)
                    existing.confidence = max(existing.confidence, relation.confidence)
                    existing.access_count += relation.access_count
                
                # Update graph
                if self.graph is not None and self.graph.has_edge(source, target):
                    edge_data = self.graph.get_edge_data(source, target)
                    self.graph.add_edge(source, primary, **edge_data)
                    self.graph.remove_edge(source, target)
                
                # Remove old relation
                del self.relations[relation_key]
        
        # Remove secondary concept
        del self.concepts[secondary]
        
        # Remove from graph
        if self.graph is not None and secondary in self.graph:
            self.graph.remove_node(secondary)
        
        # Remove from database
        if self.kb.conn:
            try:
                cursor = self.kb.conn.cursor()
                cursor.execute("DELETE FROM concepts WHERE name = ?", (secondary,))
                self.kb.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Error deleting secondary concept from database: {e}")
        
        return primary
    
    def infer_new_relations(self) -> Dict[str, Any]:
        """
        Infer new relations based on existing knowledge
        
        Returns:
            Dictionary with inference results
        """
        if not self.graph:
            return {
                "success": False,
                "error": "Graph functionality not available"
            }
        
        # Inference patterns
        inferred_relations = []
        
        # Pattern 1: Transitive is_a relations
        # If A is_a B and B is_a C, then A is_a C
        for a_name in self.concepts:
            for b_relation_key, b_relation in self.relations.items():
                if b_relation_key[0] == a_name and b_relation_key[2] == "is_a":
                    b_name = b_relation_key[1]
                    
                    for c_relation_key, c_relation in self.relations.items():
                        if c_relation_key[0] == b_name and c_relation_key[2] == "is_a":
                            c_name = c_relation_key[1]
                            
                            # Check if A is_a C already exists
                            new_key = (a_name, c_name, "is_a")
                            if new_key not in self.relations:
                                # Calculate confidence
                                confidence = b_relation.confidence * c_relation.confidence * 0.9
                                
                                # Add inferred relation
                                relation = self.add_relation(
                                    source=a_name,
                                    target=c_name,
                                    relation_type="is_a",
                                    weight=min(b_relation.weight, c_relation.weight) * 0.9,
                                    attributes={"inferred": True}
                                )
                                
                                inferred_relations.append({
                                    "source": a_name,
                                    "target": c_name,
                                    "type": "is_a",
                                    "confidence": confidence,
                                    "pattern": "transitive_is_a"
                                })
        
        # Pattern 2: Part-whole transitivity
        # If A part_of B and B part_of C, then A part_of C
        for a_name in self.concepts:
            for b_relation_key, b_relation in self.relations.items():
                if b_relation_key[0] == a_name and b_relation_key[2] == "part_of":
                    b_name = b_relation_key[1]
                    
                    for c_relation_key, c_relation in self.relations.items():
                        if c_relation_key[0] == b_name and c_relation_key[2] == "part_of":
                            c_name = c_relation_key[1]
                            
                            # Check if A part_of C already exists
                            new_key = (a_name, c_name, "part_of")
                            if new_key not in self.relations:
                                # Calculate confidence
                                confidence = b_relation.confidence * c_relation.confidence * 0.8
                                
                                # Add inferred relation
                                relation = self.add_relation(
                                    source=a_name,
                                    target=c_name,
                                    relation_type="part_of",
                                    weight=min(b_relation.weight, c_relation.weight) * 0.8,
                                    attributes={"inferred": True}
                                )
                                
                                inferred_relations.append({
                                    "source": a_name,
                                    "target": c_name,
                                    "type": "part_of",
                                    "confidence": confidence,
                                    "pattern": "transitive_part_of"
                                })
        
        # Pattern 3: Inverse relations
        # If A part_of B, then B has_part A
        for relation_key, relation in self.relations.items():
            source, target, rel_type = relation_key
            
            if rel_type == "part_of":
                inverse_key = (target, source, "has_part")
                if inverse_key not in self.relations:
                    # Add inverse relation
                    inverse_relation = self.add_relation(
                        source=target,
                        target=source,
                        relation_type="has_part",
                        weight=relation.weight,
                        attributes={"inferred": True, "inverse_of": relation_key}
                    )
                    
                    inferred_relations.append({
                        "source": target,
                        "target": source,
                        "type": "has_part",
                        "confidence": relation.confidence * 0.95,
                        "pattern": "inverse_part_of"
                    })
            
            elif rel_type == "follows":
                inverse_key = (target, source, "precedes")
                if inverse_key not in self.relations:
                    # Add inverse relation
                    inverse_relation = self.add_relation(
                        source=target,
                        target=source,
                        relation_type="precedes",
                        weight=relation.weight,
                        attributes={"inferred": True, "inverse_of": relation_key}
                    )
                    
                    inferred_relations.append({
                        "source": target,
                        "target": source,
                        "type": "precedes",
                        "confidence": relation.confidence * 0.95,
                        "pattern": "inverse_follows"
                    })
        
        return {
            "success": True,
            "inferred_relations": inferred_relations,
            "count": len(inferred_relations)
        }
    
    def generate_knowledge_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the knowledge graph
        
        Returns:
            Dictionary with knowledge summary
        """
        # Basic statistics
        concept_count = len(self.concepts)
        relation_count = len(self.relations)
        
        # Concept type distribution
        concept_types = Counter()
        for concept in self.concepts.values():
            concept_types[concept.node_type] += 1
        
        # Relation type distribution
        relation_types = Counter()
        for relation_key in self.relations:
            relation_types[relation_key[2]] += 1
        
        # Most connected concepts
        concept_connections = Counter()
        for relation_key in self.relations:
            source, target, _ = relation_key
            concept_connections[source] += 1
            concept_connections[target] += 1
        
        top_concepts = concept_connections.most_common(10)
        
        # Compute graph metrics if available
        graph_metrics = {}
        if self.graph is not None:
            try:
                # Number of connected components
                graph_metrics["connected_components"] = nx.number_connected_components(
                    self.graph.to_undirected()
                )
                
                # Average clustering coefficient
                graph_metrics["avg_clustering"] = nx.average_clustering(
                    self.graph.to_undirected()
                )
                
                # Compute centrality for a sample of nodes
                sample_size = min(100, len(self.graph))
                if sample_size > 0:
                    sample_nodes = list(self.graph.nodes())[:sample_size]
                    
                    # Degree centrality
                    degree_centrality = nx.degree_centrality(self.graph)
                    graph_metrics["avg_degree_centrality"] = sum(
                        degree_centrality[n] for n in sample_nodes
                    ) / sample_size
                    
                    # Find central concepts
                    central_concepts = sorted(
                        [(n, c) for n, c in degree_centrality.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                    
                    graph_metrics["central_concepts"] = [
                        {"name": n, "centrality": c} for n, c in central_concepts
                    ]
            except Exception as e:
                logger.error(f"Error computing graph metrics: {e}")
        
        return {
            "concept_count": concept_count,
            "relation_count": relation_count,
            "concept_types": dict(concept_types),
            "relation_types": dict(relation_types),
            "top_connected_concepts": [
                {"name": name, "connections": count} for name, count in top_concepts
            ],
            "graph_metrics": graph_metrics
        }


class KnowledgeGraphModule:
    """Module for advanced knowledge management with graph-based knowledge representation"""
    
    def __init__(self, continuous_learning_module=None):
        """
        Initialize the knowledge graph module
        
        Args:
            continuous_learning_module: ContinuousLearningModule instance (optional)
        """
        # Get or create knowledge base
        if continuous_learning_module:
            self.continuous_learning = continuous_learning_module
            self.knowledge_base = continuous_learning_module.knowledge_base
        else:
            # Try to import ContinuousLearningModule
            try:
                from AgentSystem.modules.continuous_learning import ContinuousLearningModule
                self.continuous_learning = ContinuousLearningModule()
                self.knowledge_base = self.continuous_learning.knowledge_base
            except ImportError:
                # Create a standalone knowledge base
                from AgentSystem.modules.continuous_learning import KnowledgeBase
                self.continuous_learning = None
                
                # Use a persistent file in the user's data directory
                data_dir = os.path.join(os.path.expanduser("~"), ".agent_system", "knowledge")
                os.makedirs(data_dir, exist_ok=True)
                db_path = os.path.join(data_dir, "knowledge.db")
                
                self.knowledge_base = KnowledgeBase(db_path)
        
        # Create knowledge graph
        self.graph = KnowledgeGraph(self.knowledge_base)
        
        # Configure graph
        self.enable_auto_extraction = True  # Auto-extract concepts from facts
        self.enable_auto_inference = True   # Auto-infer new relations
        self.inference_interval = 3600      # 1 hour
        self.last_inference = datetime.now()
        
        # Register for fact updates if continuous learning is available
        if self.continuous_learning:
            # This is a placeholder - in a real implementation we would
            # register callbacks or event handlers
            pass
        
        logger.info(f"Initialized KnowledgeGraphModule")
    
    def get_tools(self) -> Dict[str, Any]:
        """Get tools provided by this module"""
        return {
            "add_concept": {
                "description": "Add a concept to the knowledge graph",
                "function": self.add_concept,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the concept"
                        },
                        "node_type": {
                            "type": "string",
                            "description": "Type of node (concept, entity, event, etc.)",
                            "default": "concept"
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the concept (optional)"
                        },
                        "attributes": {
                            "type": "object",
                            "description": "Additional attributes (optional)"
                        }
                    },
                    "required": ["name"]
                }
            },
            "add_relation": {
                "description": "Add a relation between concepts",
                "function": self.add_relation,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source concept name"
                        },
                        "target": {
                            "type": "string",
                            "description": "Target concept name"
                        },
                        "relation_type": {
                            "type": "string",
                            "description": "Type of relation",
                            "default": "related_to"
                        },
                        "weight": {
                            "type": "number",
                            "description": "Relation weight/strength",
                            "default": 1.0
                        },
                        "attributes": {
                            "type": "object",
                            "description": "Additional attributes (optional)"
                        }
                    },
                    "required": ["source", "target"]
                }
            },
            "get_concept": {
                "description": "Get a concept by name",
                "function": self.get_concept,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Concept name"
                        }
                    },
                    "required": ["name"]
                }
            },
            "search_concepts": {
                "description": "Search for concepts",
                "function": self.search_concepts,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            "get_concept_neighborhood": {
                "description": "Get the neighborhood of a concept",
                "function": self.get_concept_neighborhood,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept_name": {
                            "type": "string",
                            "description": "Name of the central concept"
                        },
                        "max_distance": {
                            "type": "integer",
                            "description": "Maximum distance from the central concept",
                            "default": 2
                        }
                    },
                    "required": ["concept_name"]
                }
            },
            "find_paths": {
                "description": "Find paths between two concepts",
                "function": self.find_paths,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source concept name"
                        },
                        "target": {
                            "type": "string",
                            "description": "Target concept name"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum path length",
                            "default": 4
                        }
                    },
                    "required": ["source", "target"]
                }
            },
            "get_related_concepts": {
                "description": "Get concepts related to the specified concept",
                "function": self.get_related_concepts,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept_name": {
                            "type": "string",
                            "description": "Name of the concept"
                        },
                        "relation_type": {
                            "type": "string",
                            "description": "Type of relation to filter by (optional)"
                        }
                    },
                    "required": ["concept_name"]
                }
            },
            "find_similar_concepts": {
                "description": "Find concepts similar to the specified concept",
                "function": self.find_similar_concepts,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept_name": {
                            "type": "string",
                            "description": "Name of the concept"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Similarity threshold (0.0-1.0)",
                            "default": 0.3
                        }
                    },
                    "required": ["concept_name"]
                }
            },
            "extract_concepts_from_text": {
                "description": "Extract concepts and relations from text",
                "function": self.extract_concepts_from_text,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to process"
                        },
                        "min_confidence": {
                            "type": "number",
                            "description": "Minimum confidence threshold",
                            "default": 0.5
                        }
                    },
                    "required": ["text"]
                }
            },
            "perform_memory_cleanup": {
                "description": "Perform memory cleanup (forgetting)",
                "function": self.perform_memory_cleanup,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "description": "Force cleanup even if interval hasn't elapsed",
                            "default": False
                        }
                    }
                }
            },
            "merge_similar_concepts": {
                "description": "Merge similar concepts to consolidate knowledge",
                "function": self.merge_similar_concepts,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "threshold": {
                            "type": "number",
                            "description": "Similarity threshold for merging (0.0-1.0)",
                            "default": 0.8
                        }
                    }
                }
            },
            "infer_new_relations": {
                "description": "Infer new relations based on existing knowledge",
                "function": self.infer_new_relations,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "generate_knowledge_summary": {
                "description": "Generate a summary of the knowledge graph",
                "function": self.generate_knowledge_summary,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "process_fact": {
                "description": "Process a fact to extract concepts and relations",
                "function": self.process_fact,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fact_id": {
                            "type": "integer",
                            "description": "ID of the fact to process"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the fact (optional if fact_id is provided)"
                        }
                    }
                }
            }
        }
    
    def add_concept(self, name: str, node_type: str = "concept", 
                   description: Optional[str] = None, 
                   attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a concept to the knowledge graph
        
        Args:
            name: Name of the concept
            node_type: Type of node
            description: Description of the concept
            attributes: Additional attributes
            
        Returns:
            Dictionary with result information
        """
        concept = self.graph.add_concept(
            name=name,
            node_type=node_type,
            description=description,
            attributes=attributes
        )
        
        return {
            "success": True,
            "concept": concept.to_dict()
        }
    
    def add_relation(self, source: str, target: str, relation_type: str = "related_to",
                    weight: float = 1.0, attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a relation between concepts
        
        Args:
            source: Source concept name
            target: Target concept name
            relation_type: Type of relation
            weight: Relation weight/strength
            attributes: Additional attributes
            
        Returns:
            Dictionary with result information
        """
        relation = self.graph.add_relation(
            source=source,
            target=target,
            relation_type=relation_type,
            weight=weight,
            attributes=attributes
        )
        
        return {
            "success": True,
            "relation": relation.to_dict()
        }
    
    def get_concept(self, name: str) -> Dict[str, Any]:
        """
        Get a concept by name
        
        Args:
            name: Concept name
            
        Returns:
            Dictionary with concept information
        """
        concept = self.graph.get_concept(name)
        
        if concept:
            return {
                "success": True,
                "concept": concept.to_dict()
            }
        else:
            return {
                "success": False,
                "error": f"Concept '{name}' not found"
            }
    
    def search_concepts(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for concepts
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        results = self.graph.search_concepts(query, limit)
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    
    def get_concept_neighborhood(self, concept_name: str, max_distance: int = 2) -> Dict[str, Any]:
        """
        Get the neighborhood of a concept
        
        Args:
            concept_name: Name of the central concept
            max_distance: Maximum distance from the central concept
            
        Returns:
            Dictionary with neighborhood information
        """
        neighborhood = self.graph.get_concept_neighborhood(concept_name, max_distance)
        
        if "error" in neighborhood:
            return {
                "success": False,
                "error": neighborhood["error"]
            }
        
        return {
            "success": True,
            "neighborhood": neighborhood
        }
    
    def find_paths(self, source: str, target: str, max_length: int = 4) -> Dict[str, Any]:
        """
        Find paths between two concepts
        
        Args:
            source: Source concept name
            target: Target concept name
            max_length: Maximum path length
            
        Returns:
            Dictionary with path information
        """
        paths = self.graph.find_paths(source, target, max_length)
        
        return {
            "success": True,
            "paths": paths,
            "count": len(paths)
        }
    
    def get_related_concepts(self, concept_name: str, relation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get concepts related to the specified concept
        
        Args:
            concept_name: Name of the concept
            relation_type: Type of relation to filter by (optional)
            
        Returns:
            Dictionary with related concepts
        """
        related = self.graph.get_related_concepts(concept_name, relation_type)
        
        if "error" in related:
            return {
                "success": False,
                "error": related["error"]
            }
        
        return {
            "success": True,
            "related_concepts": related
        }
    
    def find_similar_concepts(self, concept_name: str, max_results: int = 10, 
                             threshold: float = 0.3) -> Dict[str, Any]:
        """
        Find concepts similar to the specified concept
        
        Args:
            concept_name: Name of the concept
            max_results: Maximum number of results
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            Dictionary with similar concepts
        """
        similar = self.graph.find_similar_concepts(concept_name, max_results, threshold)
        
        return {
            "success": True,
            "similar_concepts": similar,
            "count": len(similar)
        }
    
    def extract_concepts_from_text(self, text: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Extract concepts and relations from text
        
        Args:
            text: Text to process
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with extracted concepts and relations
        """
        extraction_result = self.graph.extract_concepts_from_text(text, min_confidence)
        
        if "error" in extraction_result:
            return {
                "success": False,
                "error": extraction_result["error"]
            }
        
        return {
            "success": True,
            "extraction": extraction_result
        }
    
    def perform_memory_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform memory cleanup (forgetting)
        
        Args:
            force: Force cleanup even if interval hasn't elapsed
            
        Returns:
            Dictionary with cleanup results
        """
        cleanup_result = self.graph.perform_memory_cleanup(force)
        
        return {
            "success": True,
            "cleanup": cleanup_result
        }
    
    def merge_similar_concepts(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Merge similar concepts to consolidate knowledge
        
        Args:
            threshold: Similarity threshold for merging (0.0-1.0)
            
        Returns:
            Dictionary with merge results
        """
        merge_result = self.graph.merge_similar_concepts(threshold)
        
        return {
            "success": merge_result["success"],
            "merge": merge_result
        }
    
    def infer_new_relations(self) -> Dict[str, Any]:
        """
        Infer new relations based on existing knowledge
        
        Returns:
            Dictionary with inference results
        """
        inference_result = self.graph.infer_new_relations()
        
        return {
            "success": inference_result["success"],
            "inference": inference_result
        }
    
    def generate_knowledge_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the knowledge graph
        
        Returns:
            Dictionary with knowledge summary
        """
        summary = self.graph.generate_knowledge_summary()
        
        return {
            "success": True,
            "summary": summary
        }
    
    def process_fact(self, fact_id: Optional[int] = None, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a fact to extract concepts and relations
        
        Args:
            fact_id: ID of the fact to process
            content: Content of the fact (optional if fact_id is provided)
            
        Returns:
            Dictionary with processing results
        """
        # Get fact content if ID is provided
        if fact_id and not content:
            if not self.knowledge_base:
                return {
                    "success": False,
                    "error": "Knowledge base not available"
                }
            
            fact = self.knowledge_base.get_fact(fact_id)
            if not fact:
                return {
                    "success": False,
                    "error": f"Fact with ID {fact_id} not found"
                }
            
            content = fact["content"]
        
        if not content:
            return {
                "success": False,
                "error": "No content provided"
            }
        
        # Extract concepts and relations
        extraction_result = self.graph.extract_concepts_from_text(content)
        
        if "error" in extraction_result:
            return {
                "success": False,
                "error": extraction_result["error"]
            }
        
        # Add to knowledge graph
        add_result = self.graph.add_extracted_concepts(
            extraction_result,
            source_id=fact_id,
            source_type="fact" if fact_id else None
        )
        
        return {
            "success": add_result["success"],
            "processing": add_result,
            "extraction": extraction_result
        }
    
    def _check_auto_inference(self) -> None:
        """Check if automatic inference should be performed"""
        if not self.enable_auto_inference:
            return
        
        now = datetime.now()
        time_since_inference = (now - self.last_inference).total_seconds()
        
        if time_since_inference >= self.inference_interval:
            # Perform inference
            self.infer_new_relations()
            self.last_inference = now
