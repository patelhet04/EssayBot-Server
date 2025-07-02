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
        index_type: str = "course_content",
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
            index_type: Type of index to retrieve from ("course_content", "supporting_docs")
            **kwargs: Additional parameters

        Returns:
            Dictionary with retrieval results and metadata
        """
        start_time = time.time()

        try:
            # Validate index_type
            if index_type not in ["course_content", "supporting_docs"]:
                raise ValueError(
                    f"Invalid index_type: {index_type}. Must be 'course_content' or 'supporting_docs'")

            # Update config with provided parameters
            if top_k:
                self.config.similarity_top_k = top_k

            # Load index and nodes using the FIXED approach with index type
            vector_index, nodes_data = self._load_index_and_nodes_fixed(
                professor_username, course_id, assignment_title, index_type
            )

            # Enhance query with SMART processing
            if self.config.enable_query_expansion:
                enhanced_query, similarity_boost = self.query_processor.process_query_for_retrieval(
                    query)
            else:
                enhanced_query = query
                similarity_boost = 1.0

            # Perform retrieval based on mode
            if mode == RetrievalMode.VECTOR_ONLY:
                results = self._vector_retrieve(vector_index, enhanced_query)
            elif mode == RetrievalMode.KEYWORD_ONLY:
                results = self._keyword_retrieve(nodes_data, enhanced_query)
            elif mode == RetrievalMode.HYBRID:
                # Fallback to keyword-only if vector retrieval fails
                try:
                    results = self._hybrid_retrieve(
                        vector_index, nodes_data, enhanced_query)
                except Exception:
                    logger.warning(
                        "Vector retrieval failed, falling back to keyword-only")
                    results = self._keyword_retrieve(
                        nodes_data, enhanced_query)
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
        assignment_title: str,
        index_type: str = "course_content"
    ) -> Tuple[VectorStoreIndex, List[Dict[str, Any]]]:
        """
        Load both vector index and original node data for the specified index type.
        This ensures we have complete node information including metadata.
        """
        # Validate index_type
        if index_type not in ["course_content", "supporting_docs"]:
            raise ValueError(
                f"Invalid index_type: {index_type}. Must be 'course_content' or 'supporting_docs'")

        cache_key = f"{professor_username}_{course_id}_{assignment_title}_{index_type}"

        # Check cache first
        if cache_key in self._index_cache and cache_key in self._nodes_cache:
            logger.info(f"Using cached index and nodes for {cache_key}")
            return self._index_cache[cache_key], self._nodes_cache[cache_key]

        # Construct S3 paths for the specific index type
        s3_index_path = f"{professor_username}/{course_id}/{assignment_title}/{index_type}_index/faiss_index.index"
        s3_nodes_path = f"{professor_username}/{course_id}/{assignment_title}/{index_type}_index/nodes.json"

        logger.info(f"Loading {index_type} index from S3: {s3_index_path}")
        logger.info(f"Loading {index_type} nodes from S3: {s3_nodes_path}")

        try:
            load_start = time.time()

            # Download nodes data
            nodes_data = self.core.s3_manager.download_json(s3_nodes_path)[
                "nodes"]

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

            # Load pre-computed FAISS index from S3
            with temporary_file(suffix=".index") as temp_index_path:
                self.core.s3_manager.download_file(
                    s3_index_path, temp_index_path)
                faiss_index = faiss.read_index(temp_index_path)

                # Create vector store from existing FAISS index
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store)

            # Create VectorStoreIndex using pre-computed embeddings
            vector_index = VectorStoreIndex(
                nodes=text_nodes,
                storage_context=storage_context
            )

            # Build keyword index
            self.keyword_retriever.build_index(nodes_data)

            # Cache for future use
            self._index_cache[cache_key] = vector_index
            self._nodes_cache[cache_key] = nodes_data

            return vector_index, nodes_data

        except Exception as e:
            logger.error(f"Failed to load index and nodes: {str(e)}")
            raise

    def _vector_retrieve(self, vector_index: VectorStoreIndex, query: str) -> List[NodeWithScore]:
        """Pure vector similarity retrieval using fixed LlamaIndex approach."""
        try:
            # Get the number of nodes in the index to avoid requesting more than available
            total_nodes = len(vector_index.docstore.docs)
            # Use minimum of requested nodes and available nodes to prevent index errors
            safe_top_k = min(self.config.similarity_top_k * 2,
                             total_nodes, 20)  # Cap at 20 for safety

            logger.debug(
                f"Vector index has {total_nodes} nodes, requesting {safe_top_k} results")

            retriever = VectorIndexRetriever(
                index=vector_index,
                similarity_top_k=safe_top_k
            )

            # Perform retrieval
            retrieved_nodes = retriever.retrieve(QueryBundle(query_str=query))

            # Filter by similarity threshold
            filtered_nodes = []
            for node_with_score in retrieved_nodes:
                if node_with_score.score is not None and node_with_score.score >= self.config.similarity_cutoff:
                    filtered_nodes.append(node_with_score)

                if len(filtered_nodes) >= self.config.similarity_top_k:
                    break

            logger.debug(
                f"Vector retrieval returned {len(filtered_nodes)} filtered results")
            return filtered_nodes

        except Exception as e:
            logger.error(
                f"Vector retrieval failed: {str(e)} (type: {type(e).__name__})")
            # Log more details about the error
            if hasattr(e, 'args') and e.args:
                logger.error(f"Error details: {e.args}")
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

        # If vector search failed but keyword search has results, use keyword-only
        if not vector_results and keyword_results:
            logger.info("Vector search failed, using keyword-only results")
            # Convert keyword results to NodeWithScore format
            keyword_nodes = []
            for keyword_result in keyword_results[:self.config.similarity_top_k]:
                # Create a simple TextNode
                text_node = TextNode(
                    text=keyword_result["text"],
                    metadata=keyword_result.get("metadata", {}),
                    id_=keyword_result.get("node_id", "keyword_node")
                )
                # Create NodeWithScore
                node_with_score = NodeWithScore(
                    node=text_node, score=keyword_result["score"])
                keyword_nodes.append(node_with_score)
            return keyword_nodes

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

    def retrieve_dual_context(
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
        Retrieve from both course content and supporting documents indices.

        Args:
            query: Search query
            professor_username: Professor identifier
            course_id: Course identifier  
            assignment_title: Assignment identifier
            mode: Retrieval strategy to use
            top_k: Number of results to return
            **kwargs: Additional parameters

        Returns:
            Dictionary with retrieval results from both indices
        """
        start_time = time.time()
        results = {
            "course_content": {"results": [], "total_results": 0, "has_content": False},
            "supporting_docs": {"results": [], "total_results": 0, "has_content": False},
            "query": query,
            "retrieval_time": 0.0,
            "mode": mode.value
        }

        try:
            # Try to retrieve from course content index
            try:
                course_results = self.retrieve(
                    query=query,
                    professor_username=professor_username,
                    course_id=course_id,
                    assignment_title=assignment_title,
                    mode=mode,
                    top_k=top_k,
                    index_type="course_content",
                    **kwargs
                )
                results["course_content"] = course_results
                results["course_content"]["has_content"] = course_results["total_results"] > 0
                logger.info(
                    f"Retrieved {course_results['total_results']} results from course content")
            except Exception as e:
                logger.warning(f"Failed to retrieve from course content: {e}")
                results["course_content"]["error"] = str(e)

            # Try to retrieve from supporting docs index
            try:
                supporting_results = self.retrieve(
                    query=query,
                    professor_username=professor_username,
                    course_id=course_id,
                    assignment_title=assignment_title,
                    mode=mode,
                    top_k=top_k,
                    index_type="supporting_docs",
                    **kwargs
                )
                results["supporting_docs"] = supporting_results
                results["supporting_docs"]["has_content"] = supporting_results["total_results"] > 0
                logger.info(
                    f"Retrieved {supporting_results['total_results']} results from supporting docs")
            except Exception as e:
                logger.warning(f"Failed to retrieve from supporting docs: {e}")
                results["supporting_docs"]["error"] = str(e)

            results["retrieval_time"] = time.time() - start_time

            # Log summary
            total_course = results["course_content"]["total_results"]
            total_supporting = results["supporting_docs"]["total_results"]
            logger.info(
                f"Dual retrieval completed: {total_course} course content + {total_supporting} supporting docs in {results['retrieval_time']:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Dual context retrieval failed: {str(e)}")
            results["retrieval_time"] = time.time() - start_time
            results["error"] = str(e)
            return results


# Global singleton instance
_retrieval_engine_instance = None
_retrieval_engine_lock = None


def get_retrieval_engine() -> 'RetrievalEngine':
    """Get or create the global RetrievalEngine singleton instance."""
    global _retrieval_engine_instance, _retrieval_engine_lock

    if _retrieval_engine_lock is None:
        import threading
        _retrieval_engine_lock = threading.Lock()

    if _retrieval_engine_instance is None:
        with _retrieval_engine_lock:
            # Double-check locking pattern
            if _retrieval_engine_instance is None:
                from .llamaindex_core import RAGPipelineCore
                logger.info(
                    "🔄 Creating global RetrievalEngine singleton instance")
                rag_core = RAGPipelineCore()
                _retrieval_engine_instance = RetrievalEngine(rag_core)
                logger.info(
                    "✅ Global RetrievalEngine singleton instance created")

    return _retrieval_engine_instance


def clear_global_retrieval_cache():
    """Clear the global retrieval cache (for testing/debugging)."""
    global _retrieval_engine_instance
    if _retrieval_engine_instance:
        _retrieval_engine_instance.clear_cache()
        logger.info("🧹 Global retrieval cache cleared")


def warm_up_cache(assignments: List[Dict[str, str]]) -> Dict[str, Any]:
    """Pre-load cache with specified assignments at startup."""
    retrieval_engine = get_retrieval_engine()
    results = {"total_assignments": len(
        assignments), "successful": 0, "failed": 0}

    for assignment in assignments:
        try:
            retrieval_engine._load_index_and_nodes_fixed(
                assignment["professor_username"],
                assignment["course_id"],
                assignment["assignment_title"]
            )
            results["successful"] += 1
        except Exception:
            results["failed"] += 1

    if results["successful"] > 0:
        logger.info(f"Cache pre-loaded: {results['successful']} assignments")
    return results


def warm_up_from_env() -> Dict[str, Any]:
    """Load assignments from environment variables for cache warmup."""
    import os

    warmup_config = os.getenv("WARMUP_ASSIGNMENTS", "")
    if not warmup_config:
        return {"total_assignments": 0, "successful": 0, "failed": 0}

    assignments = []
    try:
        for assignment_str in warmup_config.split(";"):
            if assignment_str.strip():
                parts = assignment_str.strip().split(",")
                if len(parts) == 3:
                    assignments.append({
                        "professor_username": parts[0].strip(),
                        "course_id": parts[1].strip(),
                        "assignment_title": parts[2].strip()
                    })
    except Exception:
        return {"total_assignments": 0, "successful": 0, "failed": 1}

    return warm_up_cache(assignments) if assignments else {"total_assignments": 0, "successful": 0, "failed": 0}


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

        logger.info(
            f"Test retrieval completed: {results['total_results']} results")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
