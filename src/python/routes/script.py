from agents import get_prompt  # Import get_prompt from agents.py
import json
from flask import Flask, request, jsonify, Blueprint
import logging
import requests
from .rag_pipeline import retrieve_relevant_text
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

singleGrading_bp = Blueprint("singleGrading", __name__)

# LLM API settings
LLM_API_URL = "http://localhost:5001/api/generate"


def send_post_request(prompt, temperature=0.2, top_p=0.1, max_tokens=2048, model="llama3.1:8b"):
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
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get response from LLM server: {e}")
        return None


@singleGrading_bp.route('/grade_single_essay', methods=['POST'])
def grade_single_essay():
    """
    Grades a single essay using the stored config_prompt from the Assignment.
    Expects a POST request with courseId, assignmentTitle, essay, config_prompt, question, and username.
    Returns feedback and scores for each agent (criterion).
    """
    try:
        # Expect JSON from Node.js: {courseId, assignmentTitle, essay, config_prompt, question, username}
        data = request.get_json()
        required_fields = ["courseId", "assignmentTitle",
                           "essay", "config_prompt", "question", "username"]
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {', '.join(required_fields)}"}), 400

        course_id = data["courseId"]
        assignment_title = data["assignmentTitle"]
        essay = data["essay"]
        config_prompt = data["config_prompt"]
        question = data["question"]
        professor_username = data["username"]

        # Step 1 & 2: Optimized single initialization for both essay analysis and RAG
        retrieval_engine = None
        try:
            from llamaindex_rag.llamaindex_retrieval import RetrievalEngine
            from llamaindex_rag.llamaindex_core import RAGPipelineCore

            # Debug: Log what we're actually analyzing
            logger.info(
                f"🔍 DEBUG: About to analyze student essay: '{essay[:100]}...'")
            logger.info(f"🔍 DEBUG: Assignment question: '{question[:100]}...'")

            # ⚡ OPTIMIZED: Initialize retrieval engine ONCE for both essay analysis and RAG
            rag_core = RAGPipelineCore()
            retrieval_engine = RetrievalEngine(rag_core)

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

            logger.info(f"🎯 ESSAY ANALYSIS - Text: '{essay[:50]}...'")
            logger.info(
                f"🎯 ESSAY ANALYSIS - Specificity: {specificity_score:.3f}, Multiplier: {quality_multiplier:.2f}")

            # Conditional RAG retrieval using SAME retrieval engine (no duplicate initialization!)
            if specificity_score < 0.1:  # Skip RAG for gibberish responses
                rag_context = "No context provided due to irrelevant response."
                logger.info(
                    f"⚠️ SKIPPING RAG RETRIEVAL - Essay too irrelevant (specificity: {specificity_score:.3f})")
            else:
                # Use the ALREADY INITIALIZED retrieval engine for RAG
                rag_results = retrieval_engine.retrieve(
                    query=question,
                    professor_username=professor_username,
                    course_id=course_id,
                    assignment_title=assignment_title,
                    top_k=10
                )

                if rag_results['total_results'] > 0:
                    # Limit total context length
                    rag_chunks = []
                    total_length = 0
                    for result in rag_results['results']:
                        chunk_text = result['text']
                        if total_length + len(chunk_text) > 6000:
                            break
                        rag_chunks.append(chunk_text)
                        total_length += len(chunk_text)
                    rag_context = "\n".join(rag_chunks)
                else:
                    rag_context = "No relevant context available."

                logger.info(
                    f"📖 RAG context retrieved for grading: {rag_context[:200]}...")

        except Exception as e:
            logger.warning(
                f"Smart analysis/RAG failed, using default scoring: {e}")
            specificity_score = 0.5
            quality_multiplier = 1.0
            rag_context = "No context available due to analysis error."

        # Step 3: Assemble the prompts using get_prompt from agents.py (with quality awareness)
        assembled_prompts = get_prompt(
            config_prompt,
            tone=data["tone"],
            quality_multiplier=quality_multiplier,
            specificity_score=specificity_score
        )
        if not assembled_prompts or "criteria_prompts" not in assembled_prompts:
            return jsonify({"error": "Failed to assemble prompts"}), 500

        # Step 4: Grade the essay for each criterion (agent)
        grading_results = {}
        for criterion_name, prompt_data in assembled_prompts["criteria_prompts"].items():
            # Replace placeholders in the prompt
            full_prompt = prompt_data["prompt"]
            full_prompt = full_prompt.replace("{{question}}", question)
            full_prompt = full_prompt.replace("{{essay}}", essay)
            full_prompt = full_prompt.replace("{{rag_context}}", rag_context)

            # DEBUG: Log the actual prompt being sent to LLM
            logger.info(f"🔍 SENDING TO LLM - Criterion: {criterion_name}")
            logger.info(f"🔍 RAG Context in prompt: {rag_context[:300]}...")
            logger.info(f"🔍 Essay in prompt: {essay[:100]}...")
            logger.info(f"🔍 FULL PROMPT BEING SENT TO LLM:")
            logger.info(f"{full_prompt}")
            logger.info(f"🔍 END OF FULL PROMPT")

            # Send the prompt to the LLM for grading
            response = send_post_request(
                full_prompt, temperature=0.2, top_p=0.1, max_tokens=2048, model=data["model"])
            if response and "response" in response:
                try:
                    result = json.loads(response["response"])
                    if "score" not in result or "feedback" not in result:
                        logger.error(
                            f"Invalid grading response format for criterion: {criterion_name}")
                        grading_results[criterion_name] = {
                            "error": "Invalid grading response format"
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
                        f"Single essay - Criterion {criterion_name}: {original_score} -> {adjusted_score:.1f} (x{quality_multiplier:.2f})")

                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse LLM grading response for criterion: {criterion_name}, error: {str(e)}")
                    grading_results[criterion_name] = {
                        "error": "Failed to parse grading response"
                    }
            else:
                logger.error(
                    f"Failed to grade essay for criterion: {criterion_name}")
                grading_results[criterion_name] = {
                    "error": "Failed to grade essay"
                }

        # Step 4: Return the grading results
        return jsonify({
            "message": "Essay graded successfully",
            "grading_results": grading_results
        }), 200
    except Exception as e:
        logger.error(f"Error grading essay: {e}")
        return jsonify({"error": str(e)}), 500
