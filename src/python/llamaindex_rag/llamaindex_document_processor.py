import os
import logging
import json
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import PDFReader
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import BaseNode, MetadataMode

# FAISS and numpy
import faiss
import numpy as np

# PDF processing
import pdfplumber
from PyPDF2 import PdfReader
import fitz  # PyMuPDF as fallback

# Local imports
from .llamaindex_core import RAGPipelineCore, RAGConfig, temporary_file

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Enhanced PDF text extraction with multiple fallback methods."""

    def __init__(self):
        self.extraction_methods = [
            self._extract_with_pdfplumber,
            self._extract_with_llamaindex,
            self._extract_with_pymupdf,
            self._extract_with_pypdf2
        ]

    def extract_text(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF with multiple fallback methods.
        Returns: (extracted_text, metadata)
        """
        metadata = {
            "extraction_method": None,
            "page_count": 0,
            "file_size": os.path.getsize(file_path),
            "extraction_time": 0,
            "extraction_success": False
        }

        start_time = time.time()

        for i, method in enumerate(self.extraction_methods):
            try:
                logger.info(
                    f"Attempting extraction method {i+1}: {method.__name__}")
                text, method_metadata = method(file_path)

                if text and len(text.strip()) > 100:  # Minimum viable text length
                    metadata.update(method_metadata)
                    metadata["extraction_method"] = method.__name__
                    metadata["extraction_time"] = time.time() - start_time
                    metadata["extraction_success"] = True

                    logger.info(
                        f"Successfully extracted {len(text)} characters using {method.__name__}")
                    return text, metadata

            except Exception as e:
                logger.warning(
                    f"Extraction method {method.__name__} failed: {str(e)}")
                continue

        # If all methods fail
        metadata["extraction_time"] = time.time() - start_time
        raise ValueError("All PDF extraction methods failed")

    def _extract_with_pdfplumber(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text using pdfplumber (most reliable for academic papers)."""
        text_parts = []
        metadata = {"page_count": 0, "tables_found": 0}

        with pdfplumber.open(file_path) as pdf:
            metadata["page_count"] = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    # Clean and add page text
                    page_text = self._clean_page_text(page_text, page_num + 1)
                    text_parts.append(page_text)

                # Count tables
                tables = page.extract_tables()
                if tables:
                    metadata["tables_found"] += len(tables)

        return "\n\n".join(text_parts), metadata

    def _extract_with_llamaindex(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text using LlamaIndex PDFReader."""
        pdf_reader = PDFReader()
        documents = pdf_reader.load_data(file=Path(file_path))

        text_parts = []
        for doc in documents:
            if doc.text:
                text_parts.append(doc.text)

        metadata = {
            "page_count": len(documents),
            "documents_created": len(documents)
        }

        return "\n\n".join(text_parts), metadata

    def _extract_with_pymupdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text using PyMuPDF (good for complex layouts)."""
        doc = fitz.open(file_path)
        text_parts = []
        metadata = {"page_count": doc.page_count}

        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text:
                page_text = self._clean_page_text(page_text, page_num + 1)
                text_parts.append(page_text)

        doc.close()
        return "\n\n".join(text_parts), metadata

    def _extract_with_pypdf2(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text using PyPDF2 (fallback method)."""
        reader = PdfReader(file_path)
        text_parts = []
        metadata = {"page_count": len(reader.pages)}

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_text = self._clean_page_text(page_text, page_num + 1)
                text_parts.append(page_text)

        return "\n\n".join(text_parts), metadata

    def _clean_page_text(self, text: str, page_num: int) -> str:
        """Clean extracted page text."""
        import re

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers at start/end
        text = re.sub(r'^Page \d+\s*', '', text)
        text = re.sub(r'\s*Page \d+$', '', text)

        # Remove common headers/footers
        text = re.sub(
            r'^.*?(?:Abstract|Introduction|ABSTRACT|INTRODUCTION)', r'\1', text)

        # Add page marker for reference
        text = f"[Page {page_num}] {text.strip()}"

        return text


class DocumentProcessor:
    """Main document processing class for LlamaIndex RAG pipeline."""

    def __init__(self, core: RAGPipelineCore):
        self.core = core
        self.config = core.config
        self.s3_manager = core.s3_manager
        self.service_manager = core.service_manager
        self.cache_manager = core.cache_manager
        self.pdf_extractor = PDFExtractor()

        logger.info("DocumentProcessor initialized")

    def process_single_document(
        self,
        s3_file_key: str,
        professor_username: str,
        course_id: str,
        assignment_title: str
    ) -> Dict[str, Any]:
        """
        Process a single PDF document and create FAISS index.

        Args:
            s3_file_key: S3 key for the PDF file
            professor_username: Professor's username
            course_id: Course identifier
            assignment_title: Assignment title

        Returns:
            Dictionary with processing results and S3 URLs
        """
        try:
            logger.info(f"Processing single document: {s3_file_key}")

            # Generate cache key
            cache_key = self.cache_manager.get_cache_key(
                professor_username, course_id, assignment_title
            )

            # Download PDF from S3
            with temporary_file(suffix=".pdf") as temp_pdf_path:
                self.s3_manager.download_file(s3_file_key, temp_pdf_path)

                # Extract text with metadata
                text, extraction_metadata = self.pdf_extractor.extract_text(
                    temp_pdf_path)

                # Create LlamaIndex document
                document = Document(
                    text=text,
                    metadata={
                        "source_file": s3_file_key,
                        "professor_username": professor_username,
                        "course_id": course_id,
                        "assignment_title": assignment_title,
                        "document_type": "pdf",
                        **extraction_metadata
                    }
                )

                # Process document and create index
                result = self._create_index_from_documents(
                    [document], cache_key, professor_username, course_id, assignment_title
                )

                result.update({
                    "source_document": s3_file_key,
                    "extraction_metadata": extraction_metadata
                })

                logger.info(
                    f"Successfully processed single document: {s3_file_key}")
                return result

        except Exception as e:
            logger.error(
                f"Error processing single document {s3_file_key}: {str(e)}")
            raise

    def process_multiple_documents(
        self,
        s3_file_keys: List[str],
        professor_username: str,
        course_id: str,
        assignment_title: str,
        max_workers: int = 3
    ) -> Dict[str, Any]:
        """
        Process multiple PDF documents and create combined FAISS index.

        Args:
            s3_file_keys: List of S3 keys for PDF files
            professor_username: Professor's username
            course_id: Course identifier
            assignment_title: Assignment title
            max_workers: Maximum number of concurrent workers

        Returns:
            Dictionary with processing results and S3 URLs
        """
        try:
            logger.info(f"Processing {len(s3_file_keys)} documents")

            # Generate cache key
            cache_key = self.cache_manager.get_cache_key(
                professor_username, course_id, assignment_title
            )

            documents = []
            extraction_results = []

            # Process documents in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all download and extraction tasks
                future_to_key = {
                    executor.submit(self._download_and_extract, s3_key): s3_key
                    for s3_key in s3_file_keys
                }

                for future in as_completed(future_to_key):
                    s3_key = future_to_key[future]
                    try:
                        text, metadata = future.result()

                        # Create LlamaIndex document
                        document = Document(
                            text=text,
                            metadata={
                                "source_file": s3_key,
                                "professor_username": professor_username,
                                "course_id": course_id,
                                "assignment_title": assignment_title,
                                "document_type": "pdf",
                                **metadata
                            }
                        )

                        documents.append(document)
                        extraction_results.append({
                            "source_file": s3_key,
                            "extraction_metadata": metadata
                        })

                        logger.info(
                            f"Successfully extracted text from {s3_key}")

                    except Exception as e:
                        logger.error(f"Failed to process {s3_key}: {str(e)}")
                        extraction_results.append({
                            "source_file": s3_key,
                            "error": str(e)
                        })

            if not documents:
                raise ValueError("No documents were successfully processed")

            # Create combined index
            result = self._create_index_from_documents(
                documents, cache_key, professor_username, course_id, assignment_title
            )

            result.update({
                "source_documents": s3_file_keys,
                "processed_documents": len(documents),
                "extraction_results": extraction_results
            })

            logger.info(f"Successfully processed {len(documents)} documents")
            return result

        except Exception as e:
            logger.error(f"Error processing multiple documents: {str(e)}")
            raise

    def _download_and_extract(self, s3_key: str) -> Tuple[str, Dict[str, Any]]:
        """Download PDF from S3 and extract text."""
        with temporary_file(suffix=".pdf") as temp_path:
            self.s3_manager.download_file(s3_key, temp_path)
            return self.pdf_extractor.extract_text(temp_path)

    def _create_index_from_documents(
        self,
        documents: List[Document],
        cache_key: str,
        professor_username: str,
        course_id: str,
        assignment_title: str
    ) -> Dict[str, Any]:
        """Create FAISS index from processed documents."""
        try:
            logger.info(f"Creating index from {len(documents)} documents")

            # Parse documents into nodes using SemanticSplitterNodeParser
            node_parser = self.service_manager.node_parser
            logger.info(
                f"🧠 Using {type(node_parser).__name__} for semantic chunking...")
            nodes = node_parser.get_nodes_from_documents(documents)

            # Debug: Log node sizes to see semantic chunking improvement
            node_sizes = [len(node.get_content(
                metadata_mode=MetadataMode.NONE)) for node in nodes]
            avg_size = sum(node_sizes) / len(node_sizes) if node_sizes else 0
            logger.info(
                f"📊 Semantic chunking results: {len(nodes)} nodes, avg size: {avg_size:.0f} chars")
            logger.info(
                f"📊 Node size range: {min(node_sizes) if node_sizes else 0}-{max(node_sizes) if node_sizes else 0} chars")

            # Filter nodes by minimum size
            valid_nodes = [
                node for node in nodes
                if len(node.get_content(metadata_mode=MetadataMode.NONE)) >= self.config.min_chunk_size
            ]

            # Debug: Show sample node previews to verify semantic coherence
            for i, node in enumerate(valid_nodes[:3]):
                content_preview = node.get_content(metadata_mode=MetadataMode.NONE)[
                    :100].replace('\n', ' ')
                logger.info(
                    f"📄 Sample Node {i+1}: '{content_preview}...' ({len(node.get_content(metadata_mode=MetadataMode.NONE))} chars)")

            logger.info(
                f"✅ Created {len(valid_nodes)} semantically coherent nodes from {len(nodes)} total nodes")

            if not valid_nodes:
                raise ValueError("No valid nodes created from documents")

            # Create FAISS vector store
            faiss_index = self._create_optimized_faiss_index(len(valid_nodes))
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store)

            # Create vector index
            vector_index = VectorStoreIndex(
                nodes=valid_nodes,
                storage_context=storage_context,
                embed_model=self.service_manager.embedding_model
            )

            logger.info("Vector index created successfully")

            # Save to S3
            result = self._save_index_to_s3(
                vector_index, valid_nodes, professor_username, course_id, assignment_title
            )

            # Cache locally if enabled
            if self.config.enable_cache:
                self._cache_index_locally(vector_index, cache_key)

            result.update({
                "total_nodes": len(valid_nodes),
                "total_documents": len(documents),
                "cache_key": cache_key
            })

            return result

        except Exception as e:
            logger.error(f"Error creating index from documents: {str(e)}")
            raise

    def _create_optimized_faiss_index(self, num_vectors: int) -> faiss.Index:
        """Create optimized FAISS index based on vector count."""
        dimension = self.config.faiss_dimension

        # Choose index type based on number of vectors
        if num_vectors < 1000:
            # Use flat index for small datasets
            # Inner product for normalized vectors
            index = faiss.IndexFlatIP(dimension)
            logger.info(f"Created IndexFlatIP for {num_vectors} vectors")

        elif num_vectors < 10000:
            # Use IVF index for medium datasets
            nlist = min(100, int(np.sqrt(num_vectors)))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            logger.info(
                f"Created IndexIVFFlat with nlist={nlist} for {num_vectors} vectors")

        else:
            # Use IVF-PQ for large datasets
            nlist = min(500, int(np.sqrt(num_vectors) * 2))
            m = min(16, max(4, dimension // 32))
            nbits = 8

            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
            logger.info(
                f"Created IndexIVFPQ with nlist={nlist}, m={m} for {num_vectors} vectors")

        return index

    def _save_index_to_s3(
        self,
        vector_index: VectorStoreIndex,
        nodes: List[BaseNode],
        professor_username: str,
        course_id: str,
        assignment_title: str
    ) -> Dict[str, Any]:
        """Save FAISS index and nodes to S3."""
        try:
            # Generate S3 paths
            base_path = self.core.get_document_path(
                professor_username, course_id, assignment_title)
            index_key = f"{base_path}/faiss_index.index"
            nodes_key = f"{base_path}/nodes.json"
            metadata_key = f"{base_path}/index_metadata.json"

            # Save FAISS index
            with temporary_file(suffix=".index") as temp_index_path:
                faiss.write_index(
                    vector_index._vector_store._faiss_index, temp_index_path)

                # Validate index
                test_index = faiss.read_index(temp_index_path)
                if test_index.ntotal != len(nodes):
                    raise ValueError("FAISS index validation failed")

                index_url = self.s3_manager.upload_file(
                    temp_index_path, index_key, "application/octet-stream"
                )

            # Prepare nodes data for JSON serialization
            nodes_data = {
                "nodes": [
                    {
                        "text": node.get_content(metadata_mode=MetadataMode.NONE),
                        "metadata": node.metadata,
                        "node_id": node.node_id,
                        "start_char_idx": getattr(node, 'start_char_idx', None),
                        "end_char_idx": getattr(node, 'end_char_idx', None),
                    }
                    for node in nodes
                ],
                "index_key": index_key,
                "total_nodes": len(nodes)
            }

            # Save nodes
            nodes_url = self.s3_manager.upload_json(nodes_data, nodes_key)

            # Save metadata
            metadata = {
                "professor_username": professor_username,
                "course_id": course_id,
                "assignment_title": assignment_title,
                "index_key": index_key,
                "nodes_key": nodes_key,
                "total_nodes": len(nodes),
                "faiss_index_type": type(vector_index._vector_store._faiss_index).__name__,
                "embedding_model": self.config.embedding_model_name,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "created_at": time.time()
            }

            metadata_url = self.s3_manager.upload_json(metadata, metadata_key)

            logger.info(f"Successfully saved index and nodes to S3")

            return {
                "faiss_index_url": index_url,
                "index_key": index_key,
                "nodes_url": nodes_url,
                "nodes_key": nodes_key,
                "metadata_url": metadata_url,
                "metadata_key": metadata_key,
                "course_id": course_id,
                "assignment_title": assignment_title
            }

        except Exception as e:
            logger.error(f"Error saving index to S3: {str(e)}")
            raise

    def _cache_index_locally(self, vector_index: VectorStoreIndex, cache_key: str) -> None:
        """Cache index locally for faster access."""
        try:
            if not self.config.enable_cache:
                return

            cache_path = self.cache_manager.get_index_cache_path(cache_key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            faiss.write_index(
                vector_index._vector_store._faiss_index, str(cache_path))
            logger.info(f"Cached index locally at {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to cache index locally: {str(e)}")

    def get_processing_stats(
        self,
        professor_username: str,
        course_id: str,
        assignment_title: str
    ) -> Dict[str, Any]:
        """Get processing statistics for a given assignment."""
        try:
            metadata_key = f"{self.core.get_document_path(professor_username, course_id, assignment_title)}/index_metadata.json"

            if not self.s3_manager.file_exists(metadata_key):
                return {"error": "No processing data found"}

            metadata = self.s3_manager.download_json(metadata_key)

            # Add current status
            metadata["status"] = "ready"
            metadata["last_accessed"] = time.time()

            return metadata

        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {"error": str(e)}
