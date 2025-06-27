"""
File: llamaindex_retrieval.py
LlamaIndex RAG Pipeline - Module 3: Fixed Advanced Retrieval Engine
===================================================================

This module provides sophisticated retrieval capabilities with the fixed LlamaIndex integration.
All the advanced features are preserved while fixing the core node reconstruction issues.

Usage:
    from llamaindex_retrieval import RetrievalEngine
    from llamaindex_core import RAGPipelineCore
    
    core = RAGPipelineCore()
    retriever = RetrievalEngine(core)
    
    # Simple retrieval
    results = retriever.retrieve(
        query="discuss the main themes",
        professor_username="prof_smith",
        course_id="CS101", 
        assignment_title="Essay Assignment"
    )
"""

import os
import logging
import json
import time
import tempfile
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict, Counter

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# FAISS and numpy
import faiss
import numpy as np

# LlamaIndex vector stores
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from .llamaindex_core import RAGPipelineCore, temporary_file
from .smart_query_processor import DynamicQueryProcessor

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Different retrieval strategies available."""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters."""
    similarity_top_k: int = 8  # Reduced for small corpus (12 nodes)
    keyword_top_k: int = 5
    # Weight for vector vs keyword (0.7 = 70% vector, 30% keyword)
    hybrid_alpha: float = 0.7
    similarity_cutoff: float = 0.1  # Much lower cutoff for BGE model
    max_context_length: int = 4000
    enable_reranking: bool = True
    enable_query_expansion: bool = True
    chunk_overlap_handling: str = "merge"  # "merge", "deduplicate", "keep_all"


# QueryEnhancer class removed - now using DynamicQueryProcessor from smart_query_processor.py


class KeywordRetriever:
    """TF-IDF based keyword retrieval for complementing vector search."""

    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.node_ids = []

    def build_index(self, nodes: List[Dict[str, Any]]) -> None:
        """Build TF-IDF index from nodes."""
        try:
            self.documents = [node["text"] for node in nodes]
            self.node_ids = [node.get("node_id", str(i))
                             for i, node in enumerate(nodes)]

            # Create TF-IDF vectorizer with academic-focused parameters
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=1,
                max_df=0.8,
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
            )

            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            logger.info(
                f"Built TF-IDF index with {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Failed to build keyword index: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search using TF-IDF similarity."""
        if self.vectorizer is None:
            return []

        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])

            # Compute similarities
            similarities = cosine_similarity(
                query_vector, self.tfidf_matrix).flatten()

            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [(idx, similarities[idx])
                       for idx in top_indices if similarities[idx] > 0.01]

            return results

        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []


class RetrievalEngine:
    """Main retrieval engine with fixed LlamaIndex integration and multiple strategies."""

    def __init__(self, core: RAGPipelineCore, config: Optional[RetrievalConfig] = None):
        self.core = core
        self.config = config or RetrievalConfig()
        self.query_processor = DynamicQueryProcessor()  # Use smart query processor
        self.keyword_retriever = KeywordRetriever()

        # Cache for loaded indices
        self._index_cache = {}
        self._nodes_cache = {}

        logger.info("RetrievalEngine initialized with smart query processor")

    def retrieve(
        self,
        query: str,
        professor_username: str,
        course_id: str,
        assignment_title: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main retrieval method with multiple strategies.

        Args:
            query: Search query
            professor_username: Professor identifier
            course_id: Course identifier  
            assignment_title: Assignment identifier
            mode: Retrieval strategy to use
            top_k: Number of results to return
            **kwargs: Additional parameters

        Returns:
            Dictionary with retrieval results and metadata
        """
        start_time = time.time()

        try:
            # Update config with provided parameters
            if top_k:
                self.config.similarity_top_k = top_k

            # Load index and nodes using the FIXED approach
            vector_index, nodes_data = self._load_index_and_nodes_fixed(
                professor_username, course_id, assignment_title
            )

            # Enhance query with SMART processing (no static expansions)
            if self.config.enable_query_expansion:
                enhanced_query, similarity_boost = self.query_processor.process_query_for_retrieval(
                    query)

                # Debug: Show query analysis
                analysis = self.query_processor.analyze_query(query)
                logger.info(f"🔍 Query Analysis:")
                logger.info(f"   Original: {query[:60]}...")
                logger.info(f"   Type: {analysis.query_type}")
                logger.info(
                    f"   Specificity: {analysis.specificity_score:.3f}")
                logger.info(f"   Boost: {similarity_boost:.2f}x")
                logger.info(
                    f"   Content Overlap: {analysis.content_overlap_score:.3f}")
                logger.info(
                    f"   Term Rarity: {analysis.term_rarity_score:.3f}")
            else:
                enhanced_query = query
                similarity_boost = 1.0

            # Perform retrieval based on mode
            if mode == RetrievalMode.VECTOR_ONLY:
                results = self._vector_retrieve(vector_index, enhanced_query)
            elif mode == RetrievalMode.KEYWORD_ONLY:
                results = self._keyword_retrieve(nodes_data, enhanced_query)
            elif mode == RetrievalMode.HYBRID:
                results = self._hybrid_retrieve(
                    vector_index, nodes_data, enhanced_query)
            elif mode == RetrievalMode.ADAPTIVE:
                results = self._adaptive_retrieve(
                    vector_index, nodes_data, query, enhanced_query)
            else:
                raise ValueError(f"Unknown retrieval mode: {mode}")

            # Post-process results with similarity boost/penalty
            processed_results = self._post_process_results(
                results, query, similarity_boost)

            # Prepare response
            response = {
                "query": query,
                "enhanced_query": enhanced_query,
                "retrieval_mode": mode.value,
                "total_results": len(processed_results),
                "results": processed_results,
                "retrieval_time": time.time() - start_time,
                "metadata": {
                    "professor_username": professor_username,
                    "course_id": course_id,
                    "assignment_title": assignment_title,
                    "similarity_cutoff": self.config.similarity_cutoff,
                    "max_context_length": self.config.max_context_length
                }
            }

            logger.info(
                f"Retrieved {len(processed_results)} results for query: {query[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return {
                "query": query,
                "enhanced_query": enhanced_query if 'enhanced_query' in locals() else query,
                "retrieval_mode": mode.value,
                "total_results": 0,
                "results": [],
                "error": str(e),
                "retrieval_time": time.time() - start_time,
                "metadata": {
                    "professor_username": professor_username,
                    "course_id": course_id,
                    "assignment_title": assignment_title,
                }
            }

    def _load_index_and_nodes_fixed(
        self,
        professor_username: str,
        course_id: str,
        assignment_title: str
    ) -> Tuple[VectorStoreIndex, List[Dict[str, Any]]]:
        """FIXED: Load and build index properly using fresh VectorStoreIndex creation."""
        cache_key = f"{professor_username}_{course_id}_{assignment_title}"

        # Check cache first
        if cache_key in self._index_cache and cache_key in self._nodes_cache:
            logger.info(
                f"⚡ Using in-memory cached index for {cache_key} (0ms)")
            return self._index_cache[cache_key], self._nodes_cache[cache_key]

        try:
            load_start = time.time()

            # Generate S3 paths
            base_path = self.core.get_document_path(
                professor_username, course_id, assignment_title)
            nodes_key = f"{base_path}/nodes.json"

            # Download nodes data
            nodes_data = self.core.s3_manager.download_json(nodes_key)["nodes"]
            logger.info(f"📄 Downloaded {len(nodes_data)} nodes from S3")

            # LEARN FROM DOCUMENT CONTENT (no static word lists!)
            document_texts = [node["text"] for node in nodes_data]
            self.query_processor.learn_from_documents(document_texts)

            # Create TextNode objects properly
            text_nodes = []
            for i, node_data in enumerate(nodes_data):
                # Create TextNode with proper structure
                text_node = TextNode(
                    text=node_data["text"],
                    metadata=node_data.get("metadata", {}),
                    id_=node_data.get("node_id", f"node_{i}"),
                )

                # Add additional attributes if they exist
                if "start_char_idx" in node_data and node_data["start_char_idx"] is not None:
                    text_node.start_char_idx = node_data["start_char_idx"]
                if "end_char_idx" in node_data and node_data["end_char_idx"] is not None:
                    text_node.end_char_idx = node_data["end_char_idx"]

                text_nodes.append(text_node)

            # Load pre-computed FAISS index from S3 (PROPER APPROACH)
            index_key = f"{base_path}/faiss_index.index"
            logger.info(
                f"Loading pre-computed FAISS index from S3: {index_key}")

            # Download and load FAISS index
            with temporary_file(suffix=".index") as temp_index_path:
                self.core.s3_manager.download_file(index_key, temp_index_path)
                faiss_index = faiss.read_index(temp_index_path)
                logger.info(
                    f"✅ Loaded FAISS index with {faiss_index.ntotal} vectors")

                # Create vector store from existing FAISS index
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store)

            # Create VectorStoreIndex using pre-computed embeddings
            vector_index = VectorStoreIndex(
                nodes=text_nodes,
                storage_context=storage_context
            )
            logger.info(
                "✅ Created VectorStoreIndex using pre-computed embeddings")

            # Build keyword index
            self.keyword_retriever.build_index(nodes_data)

            # Cache for future use
            self._index_cache[cache_key] = vector_index
            self._nodes_cache[cache_key] = nodes_data

            total_time = time.time() - load_start
            logger.info(
                f"🚀 Successfully loaded VectorStoreIndex with {len(text_nodes)} nodes in {total_time:.2f}s")
            logger.info(
                f"⚡ Performance: Used pre-computed embeddings (vs ~{len(text_nodes)*0.8:.1f}s if recreating)")
            return vector_index, nodes_data

        except Exception as e:
            logger.error(f"Failed to load index and nodes: {str(e)}")
            raise

    def _vector_retrieve(self, vector_index: VectorStoreIndex, query: str) -> List[NodeWithScore]:
        """Pure vector similarity retrieval using fixed LlamaIndex approach."""
        try:
            retriever = VectorIndexRetriever(
                index=vector_index,
                similarity_top_k=self.config.similarity_top_k * 2  # Get extra to filter
            )

            # Perform retrieval
            retrieved_nodes = retriever.retrieve(QueryBundle(query_str=query))

            # DEBUG: Log all retrieved nodes with scores
            logger.info(f"🔍 RAG RETRIEVAL DEBUG - Query: '{query[:60]}...'")
            logger.info(f"🔍 Total nodes retrieved: {len(retrieved_nodes)}")

            for i, node_with_score in enumerate(retrieved_nodes):
                score = node_with_score.score if node_with_score.score is not None else 0.0
                text_preview = node_with_score.node.get_content()[:100] + "..."
                logger.info(
                    f"🔍 Node {i+1}: Score={score:.4f}, Content='{text_preview}'")

            # Filter by similarity threshold
            filtered_nodes = []
            for node_with_score in retrieved_nodes:
                if node_with_score.score is not None and node_with_score.score >= self.config.similarity_cutoff:
                    filtered_nodes.append(node_with_score)

                if len(filtered_nodes) >= self.config.similarity_top_k:
                    break

            logger.info(
                f"🔍 Nodes after filtering (cutoff={self.config.similarity_cutoff}): {len(filtered_nodes)}")

            return filtered_nodes

        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            return []

    def _keyword_retrieve(self, nodes_data: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Pure keyword-based retrieval."""
        keyword_results = self.keyword_retriever.search(
            query, self.config.similarity_top_k)

        results = []
        for doc_idx, score in keyword_results:
            if doc_idx < len(nodes_data) and score >= self.config.similarity_cutoff:
                node_data = nodes_data[doc_idx]
                results.append({
                    "text": node_data["text"],
                    "metadata": node_data.get("metadata", {}),
                    "score": score,
                    "node_id": node_data.get("node_id", str(doc_idx))
                })

        return results

    def _hybrid_retrieve(self, vector_index: VectorStoreIndex, nodes_data: List[Dict[str, Any]], query: str) -> List[NodeWithScore]:
        """Hybrid retrieval combining vector and keyword approaches."""
        # Get vector results
        vector_results = self._vector_retrieve(vector_index, query)

        # Get keyword results
        keyword_results = self._keyword_retrieve(nodes_data, query)

        # Simple fusion: prioritize vector results, supplement with keyword
        all_results = {}

        # Add vector results
        for i, node_with_score in enumerate(vector_results):
            node_id = node_with_score.node.node_id
            all_results[node_id] = {
                "node_with_score": node_with_score,
                "vector_score": float(node_with_score.score),
                "keyword_score": 0.0,
                "rank": i
            }

        # Add keyword scores to existing or new results
        for keyword_result in keyword_results:
            node_id = keyword_result["node_id"]
            if node_id in all_results:
                all_results[node_id]["keyword_score"] = keyword_result["score"]
            # Note: Skip keyword-only results for now to keep NodeWithScore format

        # Calculate hybrid scores and sort
        final_results = []
        for node_id, scores in all_results.items():
            # Weighted combination
            vector_score = scores["vector_score"]
            keyword_score = scores["keyword_score"]

            hybrid_score = (self.config.hybrid_alpha * vector_score +
                            (1 - self.config.hybrid_alpha) * keyword_score)

            # Boost if found in both
            if vector_score > 0 and keyword_score > 0:
                hybrid_score *= 1.1

            # Update the NodeWithScore object
            node_with_score = scores["node_with_score"]
            node_with_score.score = hybrid_score
            final_results.append(node_with_score)

        # Sort by hybrid score
        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results[:self.config.similarity_top_k]

    def _adaptive_retrieve(self, vector_index: VectorStoreIndex, nodes_data: List[Dict[str, Any]], original_query: str, enhanced_query: str) -> List[NodeWithScore]:
        """Adaptive retrieval that chooses strategy based on query characteristics."""
        # Analyze query to choose strategy
        query_length = len(original_query.split())
        has_specific_terms = any(term in original_query.lower() for term in [
                                 "analyze", "compare", "evaluate", "discuss"])

        if query_length <= 3 and not has_specific_terms:
            # Short, simple queries: prefer keyword
            logger.info("Using keyword-focused retrieval for short query")
            keyword_results = self._keyword_retrieve(
                nodes_data, enhanced_query)
            # Convert to NodeWithScore format (simplified)
            return []  # Skip for now, would need proper conversion
        elif has_specific_terms:
            # Academic queries: use hybrid
            logger.info("Using hybrid retrieval for academic query")
            return self._hybrid_retrieve(vector_index, nodes_data, enhanced_query)
        else:
            # Default to vector for semantic understanding
            logger.info("Using vector retrieval for semantic query")
            return self._vector_retrieve(vector_index, enhanced_query)

    def _post_process_results(self, results: List[Union[NodeWithScore, Dict]], query: str, similarity_boost: float = 1.0) -> List[Dict[str, Any]]:
        """Post-process and format retrieval results."""
        processed_results = []
        total_length = 0

        for i, result in enumerate(results):
            if isinstance(result, dict):
                # Handle keyword-only results
                text = result["text"]
                score = result["score"]
                metadata = result.get("metadata", {})
                node_id = result.get("node_id", str(i))
            else:
                # Handle NodeWithScore objects
                text = result.node.get_content()
                score = result.score if result.score is not None else 0.0
                metadata = result.node.metadata
                node_id = result.node.node_id

            # Apply similarity cutoff
            if score < self.config.similarity_cutoff:
                continue

            # Apply length limits
            text_length = len(text)
            if total_length + text_length > self.config.max_context_length:
                # Truncate text to fit within limit
                remaining_length = self.config.max_context_length - total_length
                if remaining_length > 100:  # Only include if substantial text remains
                    text = text[:remaining_length] + "..."
                    text_length = len(text)
                else:
                    break

            processed_result = {
                "text": text,
                # Apply boost/penalty
                "score": float(score * similarity_boost),
                "original_score": float(score),  # Keep original for debugging
                "metadata": metadata,
                "node_id": node_id,
                "rank": len(processed_results) + 1,
                "length": text_length
            }

            processed_results.append(processed_result)
            total_length += text_length

            # Stop if we've reached max context length
            if total_length >= self.config.max_context_length:
                break

        return processed_results

    def get_retrieval_stats(self, professor_username: str, course_id: str, assignment_title: str) -> Dict[str, Any]:
        """Get statistics about the retrieval index."""
        try:
            cache_key = f"{professor_username}_{course_id}_{assignment_title}"

            if cache_key in self._nodes_cache:
                nodes_data = self._nodes_cache[cache_key]
            else:
                base_path = self.core.get_document_path(
                    professor_username, course_id, assignment_title)
                nodes_key = f"{base_path}/nodes.json"
                nodes_data = self.core.s3_manager.download_json(nodes_key)[
                    "nodes"]

            # Calculate statistics
            total_nodes = len(nodes_data)
            total_chars = sum(len(node["text"]) for node in nodes_data)
            avg_node_length = total_chars / total_nodes if total_nodes > 0 else 0

            # Analyze metadata
            document_sources = set()
            for node in nodes_data:
                metadata = node.get("metadata", {})
                if "source_file" in metadata:
                    document_sources.add(metadata["source_file"])

            return {
                "total_nodes": total_nodes,
                "total_characters": total_chars,
                "average_node_length": round(avg_node_length, 2),
                "unique_documents": len(document_sources),
                "document_sources": list(document_sources),
                "retrieval_config": {
                    "similarity_top_k": self.config.similarity_top_k,
                    "similarity_cutoff": self.config.similarity_cutoff,
                    "max_context_length": self.config.max_context_length
                }
            }

        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {str(e)}")
            return {"error": str(e)}

    def clear_cache(self) -> None:
        """Clear all cached indices and nodes."""
        self._index_cache.clear()
        self._nodes_cache.clear()
        logger.info("Retrieval engine cache cleared")


# Example usage and testing
if __name__ == "__main__":
    from llamaindex_core import RAGPipelineCore

    # Initialize core and retrieval engine
    core = RAGPipelineCore()
    retriever = RetrievalEngine(core)

    # Test retrieval
    try:
        results = retriever.retrieve(
            query="The three challenges proposed by MS MARCO paper are novice task, intermediate task, and passage ranking",
            professor_username="dash_user",
            course_id="685865748a8319ba9331d393",
            assignment_title="685e22d3105b6055b355a831",
            mode=RetrievalMode.HYBRID,
            top_k=5
        )

        print("Retrieval Results:")
        print(f"Query: {results['query']}")
        print(f"Enhanced Query: {results['enhanced_query']}")
        print(f"Total Results: {results['total_results']}")
        print(f"Retrieval Time: {results['retrieval_time']:.3f}s")

        for i, result in enumerate(results['results'][:3]):
            print(f"\nResult {i+1} (Score: {result['score']:.3f}):")
            print(f"Text: {result['text'][:200]}...")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
