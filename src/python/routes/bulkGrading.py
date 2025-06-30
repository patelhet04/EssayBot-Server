from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from flask import Flask, request, jsonify, Blueprint
import logging
import pandas as pd
from io import BytesIO
import os
from datetime import datetime
import sys
import boto3
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc  # Add garbage collection
from urllib.parse import urlparse, unquote

# Assuming these are defined elsewhere
from .rag_pipeline import retrieve_relevant_text
from agents import get_prompt

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
MAX_CONCURRENT_REQUESTS = 4
DEFAULT_MODEL = "llama3.1:8b"
RAG_K = 10
RAG_DISTANCE_THRESHOLD = 0.5
RAG_MAX_TOTAL_LENGTH = 6000


@dataclass
class GradingProgress:
    total_essays: int
    completed_essays: int = 0
    failed_essays: int = 0
    current_essay_index: int = 0


bulkGrading_bp = Blueprint("bulkGrading", __name__)

# LLM API settings
LLM_API_URL = "http://localhost:5001/api/generate"
s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name="us-east-1",
    config=boto3.session.Config(signature_version='s3v4')
)
S3_BUCKET = os.getenv("MINIO_BUCKET", "essaybot")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")


def send_post_request_sync(
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    model: str = DEFAULT_MODEL
) -> Optional[Dict[str, Any]]:
    """Send a request to the remote LLM API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "format": "json"
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return None


def grade_essay_sync(
    essay: str,
    question: str,
    config_prompt: Dict[str, Any],
    professor_username: str,
    course_id: str,
    assignment_title: str,
    model: str,
    progress: GradingProgress,
    tone: str = "moderate"
) -> Dict[str, Dict[str, Any]]:
    """
    Grades a single essay using the stored config_prompt from the Assignment.
    Returns feedback and scores for each agent (criterion).
    """
    try:
        # Step 1 & 2: PERMANENT FIX - Use global singleton for caching
        retrieval_engine = None
        try:
            from llamaindex_rag.llamaindex_retrieval import get_retrieval_engine

            # ⚡ PERMANENT FIX: Use global singleton - no more downloads every request!
            retrieval_engine = get_retrieval_engine()

            # Load documents and train query processor (ONCE)
            vector_index, nodes_data = retrieval_engine._load_index_and_nodes_fixed(
                professor_username, course_id, assignment_title
            )
            document_texts = [node["text"] for node in nodes_data]
            retrieval_engine.query_processor.learn_from_documents(
                document_texts)

            # Analyze student's essay for quality/relevance
            essay_analysis = retrieval_engine.query_processor.analyze_query(
                essay)
            specificity_score = essay_analysis.specificity_score
            quality_multiplier = essay_analysis.similarity_boost

            logger.info(f"🎯 BULK ESSAY ANALYSIS - Text: '{essay[:50]}...'")
            logger.info(
                f"🎯 BULK ESSAY ANALYSIS - Specificity: {specificity_score:.3f}, Multiplier: {quality_multiplier:.2f}")

            # Conditional RAG retrieval using SAME retrieval engine (no duplicate initialization!)
            if specificity_score < 0.1:  # Skip RAG for gibberish responses
                rag_context = "No context provided due to irrelevant response."
                logger.info(
                    f"⚠️ SKIPPING RAG RETRIEVAL - Essay {progress.current_essay_index + 1} too irrelevant (specificity: {specificity_score:.3f})")
            else:
                # Use the ALREADY INITIALIZED retrieval engine for RAG
                rag_results = retrieval_engine.retrieve(
                    query=question,
                    professor_username=professor_username,
                    course_id=course_id,
                    assignment_title=assignment_title,
                    top_k=RAG_K
                )

                if rag_results['total_results'] > 0:
                    # Limit total context length
                    rag_chunks = []
                    total_length = 0
                    for result in rag_results['results']:
                        chunk_text = result['text']
                        if total_length + len(chunk_text) > RAG_MAX_TOTAL_LENGTH:
                            break
                        rag_chunks.append(chunk_text)
                        total_length += len(chunk_text)
                    rag_context = "\n".join(rag_chunks)
                else:
                    rag_context = "No relevant context available."

                logger.info(
                    f"📖 RAG context retrieved for essay {progress.current_essay_index + 1}/{progress.total_essays}")

        except Exception as e:
            logger.warning(
                f"Smart analysis/RAG failed, using default scoring: {e}")
            specificity_score = 0.5
            quality_multiplier = 1.0
            rag_context = "No context available due to analysis error."

        # Step 3: Assemble the prompts using get_prompt from agents.py (with quality awareness)
        assembled_prompts = get_prompt(
            config_prompt,
            tone=tone,
            quality_multiplier=quality_multiplier,
            specificity_score=specificity_score
        )
        if not assembled_prompts or "criteria_prompts" not in assembled_prompts:
            raise ValueError("Failed to assemble prompts")

        # Step 4: Grade the essay for each criterion (agent)
        grading_results = {}
        for criterion_name, prompt_data in assembled_prompts["criteria_prompts"].items():
            # Replace placeholders in the prompt
            full_prompt = prompt_data["prompt"]
            full_prompt = full_prompt.replace("{{question}}", question)
            full_prompt = full_prompt.replace("{{essay}}", essay)
            full_prompt = full_prompt.replace("{{rag_context}}", rag_context)

            # Send the prompt to the LLM for grading
            response = send_post_request_sync(full_prompt, model=model)
            if response and "response" in response:
                try:
                    result = json.loads(response["response"])
                    if "score" not in result or "feedback" not in result:
                        logger.error(
                            f"Invalid grading response format for criterion: {criterion_name}")
                        grading_results[criterion_name] = {
                            "score": 0,
                            "feedback": "Invalid grading response format"
                        }
                        continue

                    # Apply quality multiplier to the score
                    original_score = result["score"]
                    adjusted_score = max(
                        0, min(100, original_score * quality_multiplier))

                    # Add quality analysis to feedback if significantly different
                    feedback = result["feedback"]
                    if quality_multiplier < 0.9:
                        feedback += f" [Note: Response quality suggests limited engagement with course material. Score adjusted for relevance.]"
                    elif quality_multiplier > 1.1:
                        feedback += f" [Note: Response demonstrates strong engagement with course concepts.]"

                    grading_results[criterion_name] = {
                        "score": round(adjusted_score, 1),
                        "feedback": feedback,
                        "original_score": original_score,
                        "quality_multiplier": quality_multiplier,
                        "specificity_score": specificity_score
                    }

                    logger.info(
                        f"Criterion {criterion_name}: {original_score} -> {adjusted_score:.1f} (x{quality_multiplier:.2f})")

                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse LLM grading response for criterion: {criterion_name}, error: {str(e)}")
                    grading_results[criterion_name] = {
                        "score": 0,
                        "feedback": f"Failed to parse grading response: {str(e)}"
                    }
            else:
                logger.error(
                    f"Failed to grade essay for criterion: {criterion_name}")
                grading_results[criterion_name] = {
                    "score": 0,
                    "feedback": "Failed to get response from LLM"
                }

        progress.completed_essays += 1
        progress.current_essay_index += 1
        logger.info(
            f"Completed essay {progress.current_essay_index}/{progress.total_essays}")
        return grading_results

    except Exception as e:
        logger.error(f"Error grading essay: {e}")
        progress.failed_essays += 1
        progress.current_essay_index += 1

        # Get criteria names from config_prompt to ensure we return a result for all criteria
        criteria = []
        if isinstance(config_prompt, dict) and "criteria_prompts" in config_prompt:
            criteria = list(config_prompt["criteria_prompts"].keys())

        return {
            criterion: {"score": 0, "feedback": f"Error grading: {str(e)}"}
            for criterion in criteria
        }


def run_threaded_grading(
    essays: List[str],
    question: str,
    config_prompt: Dict[str, Any],
    professor_username: str,
    course_id: str,
    assignment_title: str,
    model: str,
    tone: str = "moderate"
) -> List[Dict[str, Dict[str, Any]]]:
    """
    Runs grading on multiple essays in parallel using threading.
    """
    logger.info(f"Starting threaded grading of {len(essays)} essays...")
    # Pre-allocate results list to maintain order
    grading_results = [None] * len(essays)
    progress = GradingProgress(total_essays=len(essays))

    # Use context manager to ensure proper cleanup
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        try:
            # Submit all tasks and map them to their original indices
            future_to_index = {
                executor.submit(
                    grade_essay_sync,
                    essay, question, config_prompt,
                    professor_username, course_id, assignment_title, model, progress, tone
                ): i
                for i, essay in enumerate(essays)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    grading_results[idx] = result
                except Exception as e:
                    logger.error(f"Grading failed for essay index {idx}: {e}")

                    # Get criteria names to ensure we return a result for all criteria
                    criteria = []
                    if isinstance(config_prompt, dict) and "criteria_prompts" in config_prompt:
                        criteria = list(
                            config_prompt["criteria_prompts"].keys())

                    grading_results[idx] = {
                        criterion: {"score": 0, "feedback": f"Error: {str(e)}"}
                        for criterion in criteria
                    }
        finally:
            # Explicitly shutdown the executor and clear futures
            executor.shutdown(wait=True)
            future_to_index.clear()
            # Force garbage collection to ensure resources are released
            gc.collect()

    logger.info(
        f"Completed grading {progress.completed_essays} essays. Failed: {progress.failed_essays}")
    return grading_results


def download_file_from_s3(s3_key: str) -> BytesIO:
    """Download a file from S3 bucket."""
    logger.info(f"Downloading file from S3: {s3_key}")
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return BytesIO(response["Body"].read())
    except Exception as e:
        logger.error(f"Failed to download from S3: {str(e)}")
        raise


def upload_file_to_s3(file_obj: BytesIO, s3_key: str) -> str:
    """Upload a file to S3 bucket and return the URL."""
    try:
        s3_client.upload_fileobj(file_obj, S3_BUCKET, s3_key)
        # ✅ this works with MinIO
        url = f"{MINIO_ENDPOINT}/{S3_BUCKET}/{s3_key}"

        logger.info(f"Uploaded to S3: {url}")
        return url
    except Exception as e:
        logger.error(f"S3 upload failed: {str(e)}")
        raise


@bulkGrading_bp.route('/grade_bulk_essays', methods=['POST'])
def grade_bulk_essays() -> Tuple[Dict[str, Any], int]:
    """
    Grades multiple essays from an Excel file.
    Expects a POST request with courseId, assignmentTitle, config_prompt, question, username, s3_excel_link, and optional tone.
    Returns a link to a graded Excel file with feedback and scores for each criterion.
    """
    try:
        # Expect JSON data
        data = request.get_json()
        required_fields = ["courseId", "assignmentTitle",
                           "config_prompt", "question", "username", "s3_excel_link"]
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {', '.join(required_fields)}"}), 400

        course_id = data["courseId"]
        assignment_title = data["assignmentTitle"]
        config_prompt = data["config_prompt"]
        question = data["question"]
        professor_username = data["username"]
        s3_excel_link = data["s3_excel_link"]
        model = data.get("model", DEFAULT_MODEL)
        tone = data.get("tone", "moderate")  # Get tone or use default

        # Parse S3 path from link
        # Parse S3 path from link
        parsed_url = urlparse(s3_excel_link)
        s3_key = unquote(parsed_url.path.lstrip("/"))  # Removes leading '/'

        # If S3 key includes the bucket name as prefix, strip it
        if s3_key.startswith(f"{S3_BUCKET}/"):
            s3_key = s3_key[len(S3_BUCKET) + 1:]

        logger.info(f"Parsed S3 key: {s3_key}")
        folder = "/".join(s3_key.split("/")[:-1])
        logger.info(f"Parsed folder: {folder}")

        logger.info(f"Parsed S3 key: {s3_key}")
        logger.info(f"Parsed folder: {folder}")

        # Download and read the file
        file_obj = download_file_from_s3(s3_key)

        # Check file extension and read accordingly
        if s3_key.lower().endswith('.csv'):
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_excel(file_obj, engine='openpyxl')

        if "ID" not in df.columns or "Response" not in df.columns:
            return jsonify({"error": "Excel must contain 'ID' and 'Response' columns"}), 400

        # Grade all essays
        grading_results = run_threaded_grading(
            df["Response"].tolist(),
            question, config_prompt, professor_username, course_id, assignment_title, model, tone
        )

        # Additional cleanup to prevent resource leaks
        gc.collect()

        # Prepare output data
        # Get criteria names from the config_prompt
        if isinstance(config_prompt, dict) and "criteria_prompts" in config_prompt:
            criteria = list(config_prompt["criteria_prompts"].keys())
        else:
            # If we can't determine criteria from config_prompt, try to get them from the first result
            if grading_results and grading_results[0]:
                criteria = list(grading_results[0].keys())
            else:
                return jsonify({"error": "Failed to determine grading criteria"}), 500

        # Create output DataFrame
        output_data = {"ID": df["ID"], "Response": df["Response"]}

        # Add feedback and scores for each criterion
        for criterion in criteria:
            output_data[f"{criterion}_feedback"] = [
                result.get(criterion, {}).get("feedback", "No feedback")
                for result in grading_results
            ]
            output_data[f"{criterion}_score"] = [
                result.get(criterion, {}).get("score", 0)
                for result in grading_results
            ]

        # Calculate total score
        output_data["Total_Score"] = [
            sum(result.get(c, {}).get("score", 0) for c in criteria)
            for result in grading_results
        ]

        # Create output Excel file
        output_df = pd.DataFrame(output_data)
        # Use a Minio-compatible timestamp format (YYYY-MM-DD-HH-mm-ss)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_key = f"{folder}/gradedFiles/graded_response_{timestamp}.xlsx"
        output_buffer = BytesIO()
        output_df.to_excel(output_buffer, index=False)
        output_buffer.seek(0)

        # Upload to S3
        s3_url = upload_file_to_s3(output_buffer, output_key)

        return jsonify({
            "message": "Bulk essays graded successfully",
            "s3_graded_link": s3_url,
            "total_essays": len(df),
            "completed_essays": len([r for r in grading_results if r is not None]),
            "failed_essays": len([r for r in grading_results if r is None])
        }), 200

    except Exception as e:
        logger.error(f"Error grading bulk essays: {str(e)}")
        return jsonify({"error": str(e)}), 500
