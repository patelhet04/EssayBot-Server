from dotenv import load_dotenv
import os
import logging
from flask import request, jsonify, Blueprint
import sys
from pathlib import Path

# Add the python directory to the path for importing llamaindex modules
sys.path.append(str(Path(__file__).parent.parent))

# LlamaIndex RAG imports
try:
    from llamaindex_rag.llamaindex_indexing import LlamaIndexIndexer
    from llamaindex_rag.llamaindex_core import RAGPipelineCore
    LLAMAINDEX_AVAILABLE = True
    print("✅ LlamaIndex RAG system loaded successfully")
except ImportError as e:
    LLAMAINDEX_AVAILABLE = False
    print(f"⚠️ LlamaIndex RAG system not available: {e}")

rag_bp = Blueprint("rag", __name__)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize LlamaIndex RAG system
llamaindex_indexer = None

if LLAMAINDEX_AVAILABLE:
    try:
        llamaindex_indexer = LlamaIndexIndexer()
        logger.info("🚀 LlamaIndex RAG system initialized successfully")
        logger.info(
            "🚀 RetrievalEngine will use singleton pattern for optimal caching")

    except Exception as e:
        logger.error(f"❌ Failed to initialize LlamaIndex RAG system: {e}")
        LLAMAINDEX_AVAILABLE = False


@rag_bp.route("/index", methods=["POST"])
def index_documents():
    """Simple single document indexing - kept for compatibility."""
    if not LLAMAINDEX_AVAILABLE or not llamaindex_indexer:
        return jsonify({"error": "LlamaIndex RAG system not available"}), 503

    try:
        data = request.get_json()
        s3_file_key = data.get("s3_file_key")
        professor_username = data.get("username")
        course_id = data.get("courseId")
        assignment_title = data.get("assignmentTitle")

        if not all([s3_file_key, professor_username, course_id, assignment_title]):
            return jsonify({
                "error": "s3_file_key, username, courseId, and assignmentTitle are required"
            }), 400

        result = llamaindex_indexer.index_document(
            s3_file_key=s3_file_key,
            professor_username=professor_username,
            course_id=course_id,
            assignment_title=assignment_title
        )

        if result.success:
            response_data = result.to_dict()
            response_data.update({
                "course_id": course_id,
                "assignment_title": assignment_title,
                "chunks_url": result.nodes_url,
                "chunks_key": result.nodes_key,
            })
            return jsonify(response_data), 200
        else:
            return jsonify({
                "error": f"Failed to index document: {result.error_message}"
            }), 500

    except Exception as e:
        logger.error(f"❌ Indexing failed: {str(e)}")
        return jsonify({"error": f"Indexing failed: {str(e)}"}), 500


@rag_bp.route("/index-multiple", methods=["POST"])
def index_multiple_documents():
    """Primary route for indexing multiple documents - main route used by Node.js."""
    if not LLAMAINDEX_AVAILABLE or not llamaindex_indexer:
        return jsonify({"error": "LlamaIndex RAG system not available"}), 503

    try:
        data = request.get_json()
        s3_file_keys = data.get("s3_file_keys")
        professor_username = data.get("username")
        course_id = data.get("courseId")
        assignment_title = data.get("assignmentTitle")
        max_workers = data.get("max_workers", 3)

        if not all([s3_file_keys, professor_username, course_id, assignment_title]):
            return jsonify({
                "error": "s3_file_keys, username, courseId, and assignmentTitle are required"
            }), 400

        if not isinstance(s3_file_keys, list) or len(s3_file_keys) == 0:
            return jsonify({"error": "s3_file_keys must be a non-empty array"}), 400

        logger.info(
            f"🚀 Indexing {len(s3_file_keys)} documents for {professor_username}")

        result = llamaindex_indexer.index_multiple_documents(
            s3_file_keys=s3_file_keys,
            professor_username=professor_username,
            course_id=course_id,
            assignment_title=assignment_title,
            max_workers=max_workers
        )

        if result.success:
            logger.info(
                f"✅ Successfully indexed {len(s3_file_keys)} documents in {result.processing_time:.2f}s")
            response_data = result.to_dict()
            response_data.update({
                "course_id": course_id,
                "assignment_title": assignment_title,
                "chunks_url": result.nodes_url,
                "chunks_key": result.nodes_key,
            })
            return jsonify(response_data), 200
        else:
            logger.error(f"❌ Indexing failed: {result.error_message}")
            return jsonify({
                "error": f"Failed to index documents: {result.error_message}"
            }), 500

    except Exception as e:
        logger.error(f"❌ Multiple document indexing failed: {str(e)}")
        return jsonify({"error": f"Indexing failed: {str(e)}"}), 500


# Legacy compatibility function for existing routes
def retrieve_relevant_text(query: str, k: int = 10, professor_username: str = None,
                           course_id: str = None, assignmentTitle: str = None,
                           distance_threshold: float = 0.5, max_total_length: int = 4000):
    """
    Legacy compatibility function for existing routes.
    Uses the new LlamaIndex retrieval system.
    """
    if not LLAMAINDEX_AVAILABLE:
        raise ValueError("LlamaIndex RAG system not available")

    if not all([professor_username, course_id, assignmentTitle]):
        raise ValueError(
            "professor_username, course_id, and assignmentTitle are required")

    try:
        from llamaindex_rag.llamaindex_retrieval import get_retrieval_engine, RetrievalMode

        # ⚡ PERMANENT FIX: Use global singleton - no more downloads every request!
        retriever = get_retrieval_engine()

        # Perform retrieval using hybrid mode (best for general use)
        results = retriever.retrieve(
            query=query,
            professor_username=professor_username,
            course_id=course_id,
            assignment_title=assignmentTitle,
            mode=RetrievalMode.HYBRID,
            top_k=k
        )

        # Extract text chunks for legacy compatibility
        if results['total_results'] > 0:
            text_chunks = []
            total_length = 0

            for result in results['results']:
                chunk_text = result['text']
                chunk_length = len(chunk_text)

                # Respect max_total_length limit
                if total_length + chunk_length > max_total_length:
                    break

                text_chunks.append(chunk_text)
                total_length += chunk_length

            logger.info(
                f"📖 Legacy retrieval: {len(text_chunks)} chunks, {total_length} chars")
            return text_chunks
        else:
            logger.warning(
                f"📖 Legacy retrieval: No results for query: {query[:50]}...")
            return []

    except Exception as e:
        logger.error(f"❌ Legacy retrieval failed: {str(e)}")
        raise ValueError(f"Failed to retrieve relevant text: {str(e)}")


# Legacy compatibility function - not needed anymore but kept for compatibility
def get_faiss_index_from_s3(index_key: str):
    """
    Legacy compatibility function - no longer needed with LlamaIndex system.
    Raises an error to indicate this function should not be used.
    """
    raise ValueError(
        "get_faiss_index_from_s3 is deprecated. Use LlamaIndex retrieval system instead."
    )
