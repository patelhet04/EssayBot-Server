"""
File: llamaindex_core.py
LlamaIndex RAG Pipeline - Module 1: Core Setup and Configuration
================================================================

This module provides the foundational setup for the LlamaIndex-based RAG pipeline,
including configuration management, service context setup, and base utilities.

Usage:
    from llamaindex_core import RAGPipelineCore, RAGConfig
    
    # Initialize with environment variables
    pipeline = RAGPipelineCore()
    
    # Or with custom config
    config = RAGConfig(chunk_size=1500, similarity_top_k=15)
    pipeline = RAGPipelineCore(config)
"""

import os
import logging
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import tempfile
from contextlib import contextmanager
from datetime import datetime

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.core.service_context import ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext

# AWS/S3 imports
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import faiss

# Environment and logging setup
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llamaindex_rag.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration class for RAG pipeline settings."""

    # Text processing - Optimized for academic papers
    chunk_size: int = 800  # Smaller chunks for better retrieval
    chunk_overlap: int = 200  # Proportional overlap
    min_chunk_size: int = 100  # Higher minimum for meaningful content
    max_token_length: int = 512

    # Embedding settings
    embedding_model_name: str = "BAAI/bge-large-en"
    embedding_batch_size: int = 32

    # Retrieval settings
    similarity_top_k: int = 10
    distance_threshold: float = 0.5
    max_total_length: int = 4000

    # FAISS settings
    faiss_dimension: int = 1024  # BGE-large-en dimension
    faiss_nlist: Optional[int] = None
    faiss_m: int = 8
    faiss_nbits: int = 8

    # S3/MinIO settings
    s3_endpoint: str = "http://127.0.0.1:9000"
    s3_bucket: str = "essaybot"
    s3_region: str = "us-east-1"

    # Cache settings
    enable_cache: bool = True
    cache_dir: str = "./cache"

    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables."""
        return cls(
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", 1200)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", 240)),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", 50)),
            max_token_length=int(os.getenv("MAX_TOKEN_LENGTH", 512)),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL", "BAAI/bge-large-en"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", 32)),
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", 10)),
            distance_threshold=float(os.getenv("DISTANCE_THRESHOLD", 0.5)),
            max_total_length=int(os.getenv("MAX_TOTAL_LENGTH", 4000)),
            s3_endpoint=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
            s3_bucket=os.getenv("MINIO_BUCKET", "essaybot"),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            enable_cache=os.getenv("ENABLE_CACHE", "true").lower() == "true",
            cache_dir=os.getenv("CACHE_DIR", "./cache")
        )

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.similarity_top_k <= 0:
            raise ValueError("similarity_top_k must be positive")
        if not (0.0 <= self.distance_threshold <= 2.0):
            raise ValueError("distance_threshold must be between 0.0 and 2.0")


class S3Manager:
    """Manages S3/MinIO operations with comprehensive error handling."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = self._create_client()
        self._validate_bucket()

    def _create_client(self) -> boto3.client:
        """Create S3 client with proper configuration."""
        try:
            return boto3.client(
                "s3",
                endpoint_url=self.config.s3_endpoint,
                aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
                region_name=self.config.s3_region,
                config=boto3.session.Config(signature_version='s3v4')
            )
        except NoCredentialsError:
            logger.error("S3 credentials not found in environment variables")
            raise
        except Exception as e:
            logger.error(f"Failed to create S3 client: {str(e)}")
            raise

    def _validate_bucket(self) -> None:
        """Validate that the S3 bucket exists and is accessible."""
        try:
            self.client.head_bucket(Bucket=self.config.s3_bucket)
            logger.info(
                f"Successfully connected to S3 bucket: {self.config.s3_bucket}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(
                    f"S3 bucket '{self.config.s3_bucket}' does not exist")
            elif error_code == '403':
                logger.error(
                    f"Access denied to S3 bucket '{self.config.s3_bucket}'")
            else:
                logger.error(f"Error accessing S3 bucket: {str(e)}")
            raise

    def download_file(self, s3_key: str, local_path: Optional[str] = None) -> str:
        """Download file from S3 to local path."""
        if local_path is None:
            local_path = tempfile.mktemp()

        try:
            self.client.download_file(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Filename=local_path
            )
            logger.info(f"Downloaded {s3_key} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {str(e)}")
            raise

    def upload_file(self, local_path: str, s3_key: str, content_type: str = "application/octet-stream") -> str:
        """Upload file from local path to S3."""
        try:
            with open(local_path, "rb") as f:
                self.client.put_object(
                    Bucket=self.config.s3_bucket,
                    Key=s3_key,
                    Body=f,
                    ContentType=content_type
                )
            # Return MinIO URL format instead of AWS S3 format
            url = f"{self.config.s3_endpoint}/{self.config.s3_bucket}/{s3_key}"
            logger.info(f"Uploaded {local_path} to {s3_key}")
            return url
        except Exception as e:
            logger.error(
                f"Failed to upload {local_path} to {s3_key}: {str(e)}")
            raise

    def upload_json(self, data: Dict[str, Any], s3_key: str) -> str:
        """Upload JSON data to S3."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name

        try:
            return self.upload_file(temp_path, s3_key, "application/json")
        finally:
            os.unlink(temp_path)

    def download_json(self, s3_key: str) -> Dict[str, Any]:
        """Download and parse JSON from S3."""
        temp_path = self.download_file(s3_key)
        try:
            with open(temp_path, 'r') as f:
                return json.load(f)
        finally:
            os.unlink(temp_path)

    def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3."""
        try:
            self.client.head_object(Bucket=self.config.s3_bucket, Key=s3_key)
            return True
        except ClientError:
            return False


class LlamaIndexServiceManager:
    """Manages LlamaIndex service context and global settings."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = None
        self.node_parser = None
        self.callback_manager = None
        self._setup_service_context()

    def _setup_service_context(self) -> None:
        """Setup LlamaIndex service context with optimized settings."""
        try:
            # Setup embedding model
            self.embedding_model = HuggingFaceEmbedding(
                model_name=self.config.embedding_model_name,
                max_length=self.config.max_token_length,
                cache_folder=self.config.cache_dir if self.config.enable_cache else None
            )
            logger.info(
                f"Initialized embedding model: {self.config.embedding_model_name}")

            # Setup semantic splitter for academic papers - groups semantically related content
            from llama_index.core.node_parser.text.semantic_splitter import SemanticSplitterNodeParser
            self.node_parser = SemanticSplitterNodeParser.from_defaults(
                embed_model=self.embedding_model,
                buffer_size=2,  # Group 2 sentences for balanced context vs granularity
                breakpoint_percentile_threshold=80,  # Even lower for academic section detection
                include_metadata=True,
                include_prev_next_rel=True,
                # Keep default sentence splitter for academic content
            )
            logger.info(
                f"✅ Initialized SemanticSplitterNodeParser - buffer_size: 2, threshold: 80% (academic paper optimized)")

            # Setup callback manager for debugging
            llama_debug = LlamaDebugHandler(print_trace_on_end=True)
            self.callback_manager = CallbackManager([llama_debug])

            # Configure global settings
            Settings.embed_model = self.embedding_model
            Settings.node_parser = self.node_parser
            Settings.callback_manager = self.callback_manager
            # Note: SemanticSplitterNodeParser doesn't use chunk_size/chunk_overlap
            # It uses buffer_size and breakpoint_percentile_threshold instead

            logger.info("LlamaIndex service context configured successfully")

        except Exception as e:
            logger.error(
                f"Failed to setup LlamaIndex service context: {str(e)}")
            raise

    def create_storage_context(self, faiss_index: Optional[faiss.Index] = None) -> StorageContext:
        """Create storage context with optional FAISS index."""
        try:
            if faiss_index is not None:
                # Create vector store from existing FAISS index
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store)
                logger.info(
                    "Created storage context with existing FAISS index")
            else:
                # Create default storage context
                storage_context = StorageContext.from_defaults()
                logger.info("Created default storage context")

            return storage_context

        except Exception as e:
            logger.error(f"Failed to create storage context: {str(e)}")
            raise

    def get_service_context(self) -> ServiceContext:
        """Get configured service context."""
        return ServiceContext.from_defaults(
            embed_model=self.embedding_model,
            node_parser=self.node_parser,
            callback_manager=self.callback_manager
        )


class CacheManager:
    """Manages caching for embeddings and indices."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        if config.enable_cache:
            self._setup_cache_directory()

    def _setup_cache_directory(self) -> None:
        """Setup cache directory structure."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "embeddings").mkdir(exist_ok=True)
            (self.cache_dir / "indices").mkdir(exist_ok=True)
            (self.cache_dir / "documents").mkdir(exist_ok=True)
            logger.info(f"Cache directory setup at: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to setup cache directory: {str(e)}")
            raise

    def get_cache_key(self, professor_username: str, course_id: str, assignment_title: str) -> str:
        """Generate cache key for given parameters."""
        return f"{professor_username}_{course_id}_{assignment_title}".replace("/", "_").replace(" ", "_")

    def get_embedding_cache_path(self, cache_key: str) -> Path:
        """Get path for embedding cache."""
        return self.cache_dir / "embeddings" / f"{cache_key}_embeddings.npy"

    def get_index_cache_path(self, cache_key: str) -> Path:
        """Get path for index cache."""
        return self.cache_dir / "indices" / f"{cache_key}_index.faiss"

    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """Clear cache for specific key or all cache."""
        try:
            if cache_key:
                # Clear specific cache
                embedding_path = self.get_embedding_cache_path(cache_key)
                index_path = self.get_index_cache_path(cache_key)

                if embedding_path.exists():
                    embedding_path.unlink()
                if index_path.exists():
                    index_path.unlink()

                logger.info(f"Cleared cache for key: {cache_key}")
            else:
                # Clear all cache
                import shutil
                shutil.rmtree(self.cache_dir)
                self._setup_cache_directory()
                logger.info("Cleared all cache")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")


@contextmanager
def temporary_file(suffix: str = "", delete: bool = True):
    """Context manager for temporary files with proper cleanup."""
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file.close()
    try:
        yield temp_file.name
    finally:
        if delete and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


class RAGPipelineCore:
    """Core RAG pipeline manager that orchestrates all components."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig.from_env()
        self.config.validate()

        # Initialize managers
        self.s3_manager = S3Manager(self.config)
        self.service_manager = LlamaIndexServiceManager(self.config)
        self.cache_manager = CacheManager(self.config)

        logger.info("RAG Pipeline Core initialized successfully")

    def get_document_path(self, professor_username: str, course_id: str, assignment_title: str) -> str:
        """Get S3 path for document storage."""
        return f"{professor_username}/{course_id}/{assignment_title}"

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            "timestamp": str(datetime.now()),
            "config_valid": True,
            "s3_accessible": False,
            "embedding_model_loaded": False,
            "cache_accessible": False
        }

        try:
            # Check S3 connectivity
            self.s3_manager._validate_bucket()
            health_status["s3_accessible"] = True
        except Exception as e:
            logger.error(f"S3 health check failed: {str(e)}")

        try:
            # Check embedding model
            test_embedding = self.service_manager.embedding_model.get_text_embedding(
                "test")
            health_status["embedding_model_loaded"] = len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Embedding model health check failed: {str(e)}")

        try:
            # Check cache
            if self.config.enable_cache:
                self.cache_manager._setup_cache_directory()
            health_status["cache_accessible"] = True
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")

        return health_status


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG pipeline core
    try:
        config = RAGConfig.from_env()
        pipeline_core = RAGPipelineCore(config)

        # Perform health check
        health = pipeline_core.health_check()
        logger.info("Health Check Results:")
        for key, value in health.items():
            logger.info(f"  {key}: {value}")

        logger.info("RAG Pipeline Core setup completed successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline Core: {str(e)}")
        raise
