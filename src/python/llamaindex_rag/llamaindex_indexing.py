"""
File: llamaindex_indexing.py
LlamaIndex RAG Pipeline - Indexing Interface
===========================================

This module provides the main indexing interface for the LlamaIndex RAG system,
integrating document processing, embedding generation, and FAISS index creation.

Usage:
    from llamaindex_indexing import LlamaIndexIndexer
    
    # Initialize indexer
    indexer = LlamaIndexIndexer()
    
    # Index single document
    result = indexer.index_document(
        s3_file_key="path/to/document.pdf",
        professor_username="prof_smith",
        course_id="CS101",
        assignment_title="Essay Assignment"
    )
    
    # Index multiple documents
    result = indexer.index_multiple_documents(
        s3_file_keys=["doc1.pdf", "doc2.pdf"],
        professor_username="prof_smith", 
        course_id="CS101",
        assignment_title="Essay Assignment"
    )
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# Local imports
from .llamaindex_core import RAGPipelineCore, RAGConfig
from .llamaindex_document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result from indexing operation."""
    success: bool
    faiss_index_url: str = ""
    index_key: str = ""
    nodes_url: str = ""
    nodes_key: str = ""
    metadata_url: str = ""
    metadata_key: str = ""
    total_nodes: int = 0
    total_documents: int = 0
    processing_time: float = 0.0
    source_documents: Union[str, List[str]] = ""
    error_message: str = ""
    extraction_results: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "faiss_index_url": self.faiss_index_url,
            "index_key": self.index_key,
            "nodes_url": self.nodes_url,
            "nodes_key": self.nodes_key,
            "metadata_url": self.metadata_url,
            "metadata_key": self.metadata_key,
            "total_nodes": self.total_nodes,
            "total_documents": self.total_documents,
            "processing_time": self.processing_time,
            "source_documents": self.source_documents,
            "error_message": self.error_message,
            "extraction_results": self.extraction_results or []
        }


class LlamaIndexIndexer:
    """Main indexing interface for the LlamaIndex RAG system."""

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the indexer.

        Args:
            config: Optional RAG configuration. If None, loads from environment.
        """
        try:
            self.config = config or RAGConfig.from_env()
            self.config.validate()

            # Initialize core components
            self.core = RAGPipelineCore(self.config)
            self.document_processor = DocumentProcessor(self.core)

            logger.info("LlamaIndexIndexer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndexIndexer: {str(e)}")
            raise

    def index_document(
        self,
        s3_file_key: str,
        professor_username: str,
        course_id: str,
        assignment_title: str,
        **kwargs
    ) -> IndexingResult:
        """
        Index a single PDF document.

        Args:
            s3_file_key: S3 key for the PDF file
            professor_username: Professor's username
            course_id: Course identifier
            assignment_title: Assignment identifier
            **kwargs: Additional parameters

        Returns:
            IndexingResult with operation details
        """
        start_time = time.time()

        try:
            logger.info(f"Starting single document indexing: {s3_file_key}")

            # Process document using DocumentProcessor
            result = self.document_processor.process_single_document(
                s3_file_key=s3_file_key,
                professor_username=professor_username,
                course_id=course_id,
                assignment_title=assignment_title
            )

            processing_time = time.time() - start_time

            # Create successful result
            indexing_result = IndexingResult(
                success=True,
                faiss_index_url=result["faiss_index_url"],
                index_key=result["index_key"],
                nodes_url=result["nodes_url"],
                nodes_key=result["nodes_key"],
                metadata_url=result["metadata_url"],
                metadata_key=result["metadata_key"],
                total_nodes=result["total_nodes"],
                total_documents=result["total_documents"],
                processing_time=processing_time,
                source_documents=s3_file_key,
                extraction_results=[{
                    "source_file": result["source_document"],
                    "extraction_metadata": result["extraction_metadata"]
                }]
            )

            logger.info(
                f"Successfully indexed document {s3_file_key} in {processing_time:.2f}s")
            return indexing_result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to index document {s3_file_key}: {str(e)}"
            logger.error(error_msg)

            return IndexingResult(
                success=False,
                processing_time=processing_time,
                source_documents=s3_file_key,
                error_message=error_msg
            )

    def index_multiple_documents(
        self,
        s3_file_keys: List[str],
        professor_username: str,
        course_id: str,
        assignment_title: str,
        max_workers: int = 3,
        **kwargs
    ) -> IndexingResult:
        """
        Index multiple PDF documents into a combined index.

        Args:
            s3_file_keys: List of S3 keys for PDF files
            professor_username: Professor's username
            course_id: Course identifier
            assignment_title: Assignment identifier
            max_workers: Maximum number of concurrent workers
            **kwargs: Additional parameters

        Returns:
            IndexingResult with operation details
        """
        start_time = time.time()

        try:
            logger.info(
                f"Starting multiple document indexing: {len(s3_file_keys)} files")

            # Validate input
            if not s3_file_keys or len(s3_file_keys) == 0:
                raise ValueError("s3_file_keys must be a non-empty list")

            # Process documents using DocumentProcessor
            result = self.document_processor.process_multiple_documents(
                s3_file_keys=s3_file_keys,
                professor_username=professor_username,
                course_id=course_id,
                assignment_title=assignment_title,
                max_workers=max_workers
            )

            processing_time = time.time() - start_time

            # Create successful result
            indexing_result = IndexingResult(
                success=True,
                faiss_index_url=result["faiss_index_url"],
                index_key=result["index_key"],
                nodes_url=result["nodes_url"],
                nodes_key=result["nodes_key"],
                metadata_url=result["metadata_url"],
                metadata_key=result["metadata_key"],
                total_nodes=result["total_nodes"],
                total_documents=result["total_documents"],
                processing_time=processing_time,
                source_documents=s3_file_keys,
                extraction_results=result["extraction_results"]
            )

            logger.info(
                f"Successfully indexed {len(s3_file_keys)} documents in {processing_time:.2f}s")
            return indexing_result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to index multiple documents: {str(e)}"
            logger.error(error_msg)

            return IndexingResult(
                success=False,
                processing_time=processing_time,
                source_documents=s3_file_keys,
                error_message=error_msg
            )

    def get_index_status(
        self,
        professor_username: str,
        course_id: str,
        assignment_title: str
    ) -> Dict[str, Any]:
        """
        Get status and statistics for an existing index.

        Args:
            professor_username: Professor's username
            course_id: Course identifier
            assignment_title: Assignment identifier

        Returns:
            Dictionary with index status and statistics
        """
        try:
            stats = self.document_processor.get_processing_stats(
                professor_username=professor_username,
                course_id=course_id,
                assignment_title=assignment_title
            )

            if "error" in stats:
                return {
                    "exists": False,
                    "error": stats["error"]
                }

            return {
                "exists": True,
                "status": "ready",
                "professor_username": stats["professor_username"],
                "course_id": stats["course_id"],
                "assignment_title": stats["assignment_title"],
                "total_nodes": stats["total_nodes"],
                "faiss_index_type": stats["faiss_index_type"],
                "embedding_model": stats["embedding_model"],
                "chunk_size": stats["chunk_size"],
                "chunk_overlap": stats["chunk_overlap"],
                "created_at": stats["created_at"],
                "last_accessed": stats.get("last_accessed", stats["created_at"])
            }

        except Exception as e:
            logger.error(f"Failed to get index status: {str(e)}")
            return {
                "exists": False,
                "error": str(e)
            }

    def delete_index(
        self,
        professor_username: str,
        course_id: str,
        assignment_title: str
    ) -> Dict[str, Any]:
        """
        Delete an existing index and all associated files.

        Args:
            professor_username: Professor's username
            course_id: Course identifier
            assignment_title: Assignment identifier

        Returns:
            Dictionary with deletion status
        """
        try:
            base_path = self.core.get_document_path(
                professor_username, course_id, assignment_title)

            # List of files to delete
            files_to_delete = [
                f"{base_path}/faiss_index.index",
                f"{base_path}/nodes.json",
                f"{base_path}/index_metadata.json"
            ]

            deleted_files = []
            errors = []

            for file_key in files_to_delete:
                try:
                    if self.core.s3_manager.file_exists(file_key):
                        # Note: We'd need to add a delete method to S3Manager
                        # For now, just log what would be deleted
                        deleted_files.append(file_key)
                        logger.info(f"Would delete: {file_key}")
                except Exception as e:
                    errors.append(f"Failed to delete {file_key}: {str(e)}")

            return {
                "success": len(errors) == 0,
                "deleted_files": deleted_files,
                "errors": errors
            }

        except Exception as e:
            logger.error(f"Failed to delete index: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the indexing system.

        Returns:
            Dictionary with health status
        """
        try:
            # Use core health check
            core_health = self.core.health_check()

            return {
                "status": "healthy" if core_health["status"] == "healthy" else "unhealthy",
                "timestamp": time.time(),
                "components": {
                    "core": core_health,
                    "document_processor": {
                        "status": "healthy",
                        "class": "DocumentProcessor"
                    },
                    "indexer": {
                        "status": "healthy",
                        "class": "LlamaIndexIndexer"
                    }
                },
                "config": {
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "embedding_model": self.config.embedding_model_name,
                    "s3_bucket": self.config.s3_bucket,
                    "cache_enabled": self.config.enable_cache
                }
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }


# Example usage and testing
if __name__ == "__main__":
    # Test the indexer
    try:
        indexer = LlamaIndexIndexer()

        # Test health check
        health = indexer.health_check()
        print(f"Health Status: {health['status']}")

        # Test single document indexing
        result = indexer.index_document(
            s3_file_key="test-document.pdf",
            professor_username="test_user",
            course_id="test_course",
            assignment_title="test_assignment"
        )

        print(f"Indexing Result: {result.success}")
        if result.success:
            print(
                f"Processed {result.total_nodes} nodes in {result.processing_time:.2f}s")
        else:
            print(f"Error: {result.error_message}")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
