from dotenv import load_dotenv
import os
import logging
import json
from io import BytesIO
from flask import Flask, request, jsonify, Blueprint
import boto3
import faiss
import numpy as np
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
from typing import List, Generator, Optional
from transformers import AutoTokenizer
import re

rag_bp = Blueprint("rag", __name__)

load_dotenv()

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 240))
BATCH_SIZE = int(os.getenv("RAG_BATCH_SIZE", 32))
MIN_CHUNK_SIZE = 50  # Minimum chunk size to ensure meaningful context
DISTANCE_THRESHOLD = float(
    os.getenv("DISTANCE_THRESHOLD", 0.5))  # Configurable
MAX_TOTAL_LENGTH = int(os.getenv("MAX_TOTAL_LENGTH", 4000))      # Configurable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en")
MAX_TOKEN_LENGTH = 512

s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name="us-east-1",
    config=boto3.session.Config(signature_version='s3v4')
)
S3_BUCKET = os.getenv("MINIO_BUCKET", "essaybot")


def download_file_from_s3(s3_key: str) -> BytesIO:
    logger.info(f"Downloading file from S3: {s3_key}")
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return BytesIO(response["Body"].read())
    except Exception as e:
        logger.error(f"Failed to download from S3: {str(e)}")
        raise


def upload_file_to_s3(file_path: str, s3_key: str, content_type: str = "application/octet-stream") -> str:
    try:
        with open(file_path, "rb") as f:
            s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key,
                                 Body=f, ContentType=content_type)
        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
        logger.info(f"Uploaded to S3: {url}")
        return url
    except boto3.exceptions.S3UploadFailedError as e:
        logger.error(f"S3 upload failed: {str(e)}")
        raise


def upload_json_to_s3(data: dict, s3_key: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as temp_file:
        json.dump(data, temp_file)
        temp_file.flush()
        url = upload_file_to_s3(temp_file.name, s3_key,
                                content_type="application/json")
    os.remove(temp_file.name)
    return url


def extract_text_from_pdf(file_obj: BytesIO) -> Generator[str, None, None]:
    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    yield page_text.strip()
                else:
                    logger.warning(
                        f"Page {page.page_number} has no extractable text")
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def clean_chunk(chunk: str) -> str:
    """Cleans a text chunk by removing excessive whitespace and special characters."""
    chunk = re.sub(r'\s+', ' ', chunk.strip())
    chunk = re.sub(r'Page \d+', '', chunk)
    return chunk


def truncate_chunk(chunk: str, max_tokens: int = MAX_TOKEN_LENGTH) -> str:
    """Truncates a chunk to the maximum token length, respecting sentence boundaries."""
    tokens = tokenizer.tokenize(chunk)
    if len(tokens) <= max_tokens:
        return chunk
    truncated = tokens[:max_tokens]
    text = tokenizer.convert_tokens_to_string(truncated)
    last_period = text.rfind('.')
    if last_period > len(text) // 2:  # Avoid cutting too early
        return text[:last_period + 1]
    return text


def embed_in_batches(text_chunks: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    embeddings = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch = [truncate_chunk(chunk) for chunk in batch]
        try:
            batch_embeddings = embeddings_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Failed to embed batch {i//batch_size}: {str(e)}")
            continue
    if not embeddings:
        raise ValueError("No embeddings generated for any chunks")
    embeddings = np.array(embeddings).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)
    return embeddings


def create_quantized_index(embeddings: np.ndarray, nlist: Optional[int] = None, m: int = 8, nbits: int = 8) -> faiss.Index:
    d = embeddings.shape[1]
    num_embeddings = len(embeddings)
    min_points_per_centroid = 39

    if nlist is None:
        nlist = max(1, min(500, int(np.sqrt(num_embeddings) * 2)))

    min_required_points = nlist * min_points_per_centroid
    if num_embeddings < min_required_points:
        logger.warning(
            f"Not enough training data ({num_embeddings}) for clustering with nlist={nlist}. "
            f"Requires at least {min_required_points} points. Using IndexFlatL2.")
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        return index

    adjusted_nbits = min(nbits, max(4, int(np.log2(num_embeddings) - 1)))
    if num_embeddings < (1 << adjusted_nbits):
        logger.warning(
            f"Not enough training data ({num_embeddings}) for quantization with adjusted_nbits={adjusted_nbits}. Using IndexFlatL2.")
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        return index

    m = min(16, max(4, d // 32))
    logger.info(
        f"Creating quantized FAISS index with nlist={nlist}, m={m}, nbits={adjusted_nbits}, dimension={d}")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, adjusted_nbits)

    logger.info("Training FAISS index")
    try:
        index.train(embeddings)
    except Exception as e:
        logger.warning(
            f"Training failed: {str(e)}. Falling back to IndexFlatL2.")
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        return index

    logger.info("Validating FAISS index")
    test_query = embeddings[0:1]
    distances, indices = index.search(test_query, 1)
    if indices[0][0] == -1:
        logger.warning(
            "Index validation failed: no results returned. Falling back to IndexFlatL2.")
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        return index

    logger.info("Adding embeddings to FAISS index")
    index.add(embeddings)
    return index


@rag_bp.route("/index", methods=["POST"])
def index_pdf():
    logger.info("Entered index_pdf function")
    data = request.get_json()
    logger.debug(f"Received data: {data}")
    s3_file_key = data.get("s3_file_key")
    professor_username = data.get("username")
    course_id = data.get("courseId")
    assignment_title = data.get("assignmentTitle")
    logger.debug(
        f"s3_file_key: {s3_file_key}, username: {professor_username}, courseId: {course_id}, assignmentTitle: {assignment_title}")
    if not all([s3_file_key, professor_username, course_id, assignment_title]):
        logger.error("Missing required fields in request data")
        return jsonify({"error": "s3_file_key, username, courseId, and assignmentTitle are required"}), 400

    try:
        file_obj = download_file_from_s3(s3_file_key)
        logger.info("File downloaded successfully from S3")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        all_text_chunks = []
        for page_text in extract_text_from_pdf(file_obj):
            chunks = text_splitter.split_text(page_text)
            chunks = [clean_chunk(chunk)
                      for chunk in chunks if len(chunk) >= MIN_CHUNK_SIZE]
            all_text_chunks.extend(chunks)

        if not all_text_chunks:
            logger.warning("No text chunks generated from PDF")
            return jsonify({"error": "No text chunks generated from PDF"}), 400

        if all_text_chunks:
            avg_chunk_size = sum(len(chunk)
                                 for chunk in all_text_chunks) / len(all_text_chunks)
            logger.info(
                f"Generated {len(all_text_chunks)} chunks with average size {avg_chunk_size:.2f} characters")

        all_text_chunks = list(dict.fromkeys(all_text_chunks))
        embeddings = embed_in_batches(all_text_chunks)
        logger.info("Embeddings generated successfully")
        optimized_index = create_quantized_index(embeddings)
        logger.info("FAISS index created successfully")

        professor_dir = f"{professor_username}/{course_id}/{assignment_title}"
        index_key = f"{professor_dir}/faiss_index.index"
        chunks_key = f"{professor_dir}/chunks.json"

        chunks_data = {"chunks": all_text_chunks, "index_key": index_key}
        chunks_url = upload_json_to_s3(chunks_data, chunks_key)
        logger.info(f"Chunks uploaded to S3: {chunks_url}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as temp_file:
            faiss.write_index(optimized_index, temp_file.name)
            try:
                test_index = faiss.read_index(temp_file.name)
                if test_index.ntotal != len(all_text_chunks):
                    raise ValueError(
                        "FAISS index validation failed: incorrect number of embeddings")
            except Exception as e:
                logger.error(f"FAISS index validation failed: {str(e)}")
                raise
            index_url = upload_file_to_s3(
                temp_file.name, index_key, "application/octet-stream")
        os.remove(temp_file.name)

        logger.info(f"FAISS index uploaded to S3: {index_url}")
        return jsonify({
            "faiss_index_url": index_url,
            "index_key": index_key,
            "chunks_url": chunks_url,
            "chunks_key": chunks_key,
            "course_id": course_id,
            "assignment_title": assignment_title
        })
    except Exception as e:
        logger.exception(f"Error indexing PDF: {str(e)}")
        return jsonify({"error": f"Failed to index PDF: {str(e)}"}), 500
    finally:
        logger.info("Exiting index_pdf function")


@rag_bp.route("/index-multiple", methods=["POST"])
def index_multiple_pdfs():
    data = request.get_json()
    s3_file_keys = data.get("s3_file_keys")
    professor_username = data.get("username")
    course_id = data.get("courseId")
    assignment_title = data.get("assignmentTitle")
    if not all([s3_file_keys, professor_username, course_id, assignment_title]):
        return jsonify({"error": "s3_file_keys, username, courseId, and assignmentTitle are required"}), 400

    if not isinstance(s3_file_keys, list) or len(s3_file_keys) == 0:
        return jsonify({"error": "s3_file_keys must be a non-empty array"}), 400

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        all_text_chunks = []
        for s3_file_key in s3_file_keys:
            logger.info(f"Processing file: {s3_file_key}")
            file_obj = download_file_from_s3(s3_file_key)
            for page_text in extract_text_from_pdf(file_obj):
                chunks = text_splitter.split_text(page_text)
                chunks = [clean_chunk(chunk) for chunk in chunks if len(
                    chunk) >= MIN_CHUNK_SIZE]
                all_text_chunks.extend(chunks)

        if not all_text_chunks:
            return jsonify({"error": "No text chunks generated from PDFs"}), 400

        if all_text_chunks:
            avg_chunk_size = sum(len(chunk)
                                 for chunk in all_text_chunks) / len(all_text_chunks)
            logger.info(
                f"Generated {len(all_text_chunks)} chunks with average size {avg_chunk_size:.2f} characters")

        all_text_chunks = list(dict.fromkeys(all_text_chunks))
        embeddings = embed_in_batches(all_text_chunks)
        optimized_index = create_quantized_index(embeddings)

        professor_dir = f"{professor_username}/{course_id}/{assignment_title}"
        index_key = f"{professor_dir}/faiss_index.index"
        chunks_key = f"{professor_dir}/chunks.json"

        chunks_data = {"chunks": all_text_chunks, "index_key": index_key}
        chunks_url = upload_json_to_s3(chunks_data, chunks_key)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as temp_file:
            faiss.write_index(optimized_index, temp_file.name)
            try:
                test_index = faiss.read_index(temp_file.name)
                if test_index.ntotal != len(all_text_chunks):
                    raise ValueError(
                        "FAISS index validation failed: incorrect number of embeddings")
            except Exception as e:
                logger.error(f"FAISS index validation failed: {str(e)}")
                raise
            index_url = upload_file_to_s3(
                temp_file.name, index_key, "application/octet-stream")
        os.remove(temp_file.name)

        return jsonify({
            "faiss_index_url": index_url,
            "index_key": index_key,
            "chunks_url": chunks_url,
            "chunks_key": chunks_key,
            "course_id": course_id,
            "assignment_title": assignment_title
        })
    except Exception as e:
        logger.exception(f"Error indexing multiple PDFs: {str(e)}")
        return jsonify({"error": f"Failed to index PDFs: {str(e)}"}), 500


def get_faiss_index_from_s3(index_key: str) -> faiss.Index:
    logger.info(f"Attempting to download FAISS index from key: {index_key}")
    try:
        # HEAD check
        s3_client.head_object(Bucket=S3_BUCKET, Key=index_key)
    except Exception as e:
        logger.error(f"FAISS index does not exist or is inaccessible: {e}")
        raise ValueError(f"Index file not found at key: {index_key}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as temp_file:
        try:
            s3_client.download_file(
                Bucket=S3_BUCKET, Key=index_key, Filename=temp_file.name)
            index = faiss.read_index(temp_file.name)
            logger.info("FAISS index downloaded and loaded successfully")
            return index
        except Exception as e:
            logger.error(f"Error downloading or loading FAISS index: {e}")
            raise
        finally:
            os.remove(temp_file.name)


def download_json_from_s3(json_key: str) -> dict:
    logger.info(f"Downloading JSON from S3: {json_key}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        try:
            s3_client.download_file(
                Bucket=S3_BUCKET, Key=json_key, Filename=temp_file.name)
            with open(temp_file.name, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error downloading JSON from S3: {str(e)}")
            raise
        finally:
            os.remove(temp_file.name)


def download_chunks_from_s3(chunks_key: str) -> List[str]:
    data = download_json_from_s3(chunks_key)
    return data["chunks"]


def expand_query(query: str) -> str:
    """No expansion for precision in retrieval."""
    return query


def retrieve_relevant_text(query: str, faiss_index: Optional[faiss.Index] = None, k: int = 10,
                           professor_username: Optional[str] = None, course_id: Optional[str] = None,
                           assignmentTitle: Optional[str] = None,
                           distance_threshold: float = DISTANCE_THRESHOLD,
                           max_total_length: int = MAX_TOTAL_LENGTH) -> List[str]:
    if not all([professor_username, course_id, assignmentTitle]):
        raise ValueError(
            "professor_username, course_id, and assignmentTitle are required")

    index_key = f"{professor_username}/{course_id}/{assignmentTitle}/faiss_index.index"
    chunks_key = f"{professor_username}/{course_id}/{assignmentTitle}/chunks.json"
    logger.info(f"Index key: {index_key}")
    logger.info(f"Chunks key: {chunks_key}")

    try:
        faiss_index = faiss_index or get_faiss_index_from_s3(index_key)
        text_chunks = download_chunks_from_s3(chunks_key)
    except Exception as e:
        logger.error(f"Error loading FAISS index or chunks: {str(e)}")
        raise ValueError(f"Failed to load FAISS index or chunks: {str(e)}")

    expanded_query = expand_query(query)
    expanded_query = truncate_chunk(expanded_query)
    logger.info(f"Query: {expanded_query[:100]}...")

    query_embedding = embeddings_model.embed_query(expanded_query)
    query_embedding = np.array([query_embedding]).astype("float32")
    norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_embedding = query_embedding / np.maximum(norms, 1e-10)

    distances, indices = faiss_index.search(query_embedding, k * 2)
    relevant_chunks = []
    total_length = 0
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= len(text_chunks):
            continue
        if dist > distance_threshold:
            break  # Stop at first irrelevant chunk (distances are sorted)
        chunk = text_chunks[idx]
        chunk_length = len(chunk)
        if total_length + chunk_length > max_total_length:
            break
        relevant_chunks.append(chunk)
        total_length += chunk_length

    total_length = sum(len(chunk) for chunk in relevant_chunks)
    if total_length < 500 and len(text_chunks) > len(relevant_chunks):
        logger.info("Initial retrieval insufficient, increasing k...")
        more_distances, more_indices = faiss_index.search(
            query_embedding, k * 4)
        for dist, idx in zip(more_distances[0], more_indices[0]):
            if idx >= len(text_chunks) or dist > distance_threshold:
                continue
            chunk = text_chunks[idx]
            if chunk in relevant_chunks:
                continue
            chunk_length = len(chunk)
            if total_length + chunk_length > max_total_length:
                break
            relevant_chunks.append(chunk)
            total_length += chunk_length

    relevant_chunks = list(dict.fromkeys(relevant_chunks))
    if not relevant_chunks:
        logger.warning(
            f"No relevant chunks retrieved for query: {query[:50]}...")
    logger.info(
        f"Retrieved {len(relevant_chunks)} chunks for query: {query[:50]}...")
    return relevant_chunks
