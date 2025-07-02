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
            from llamaindex_rag.llamaindex_retrieval import get_retrieval_engine

            # Debug: Log what we're actually analyzing
            logger.info(
                f"🔍 DEBUG: About to analyze student essay: '{essay[:100]}...'")
            logger.info(f"🔍 DEBUG: Assignment question: '{question[:100]}...'")

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

            logger.info(f"🎯 ESSAY ANALYSIS - Text: '{essay[:50]}...'")
            logger.info(
                f"🎯 ESSAY ANALYSIS - Specificity: {specificity_score:.3f}, Multiplier: {quality_multiplier:.2f}")

            # Conditional RAG retrieval using SAME retrieval engine (no duplicate initialization!)
            if specificity_score < 0.1:  # Skip RAG for gibberish responses
                course_context = "No context provided due to irrelevant response."
                supporting_context = ""
                has_supporting_docs = False
                logger.info(
                    f"⚠️ SKIPPING RAG RETRIEVAL - Essay too irrelevant (specificity: {specificity_score:.3f})")
            else:
                # Use the ALREADY INITIALIZED retrieval engine for dual RAG
                dual_results = retrieval_engine.retrieve_dual_context(
                    query=question,
                    professor_username=professor_username,
                    course_id=course_id,
                    assignment_title=assignment_title,
                    top_k=10
                )

                # Process course content results
                if dual_results['course_content']['total_results'] > 0:
                    course_chunks = []
                    total_length = 0
                    for result in dual_results['course_content']['results']:
                        chunk_text = result['text']
                        if total_length + len(chunk_text) > 4000:
                            break
                        course_chunks.append(chunk_text)
                        total_length += len(chunk_text)
                    course_context = "\n".join(course_chunks)
                else:
                    course_context = "No relevant course content available."

                # Process supporting docs results
                supporting_context = ""
                has_supporting_docs = dual_results['supporting_docs']['has_content']
                if has_supporting_docs and dual_results['supporting_docs']['total_results'] > 0:
                    supporting_chunks = []
                    total_length = 0
                    for result in dual_results['supporting_docs']['results']:
                        chunk_text = result['text']
                        # Smaller limit for supporting docs
                        if total_length + len(chunk_text) > 2000:
                            break
                        supporting_chunks.append(chunk_text)
                        total_length += len(chunk_text)
                    supporting_context = "\n".join(supporting_chunks)

                logger.info(
                    f"📖 Course content: {len(course_context)} chars, Supporting docs: {len(supporting_context)} chars")

        except Exception as e:
            logger.warning(
                f"Smart analysis/RAG failed, using default scoring: {e}")
            specificity_score = 0.5
            quality_multiplier = 1.0
            course_context = "No context available due to analysis error."
            supporting_context = ""
            has_supporting_docs = False

        # Step 3: Assemble the prompts using get_prompt from agents.py (with quality awareness)
        assembled_prompts = get_prompt(
            config_prompt,
            tone=data["tone"],
            quality_multiplier=quality_multiplier,
            specificity_score=specificity_score,
            has_supporting_docs=has_supporting_docs
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
            full_prompt = full_prompt.replace(
                "{{course_context}}", course_context)

            # Conditionally replace supporting context
            if has_supporting_docs:
                full_prompt = full_prompt.replace(
                    "{{supporting_context}}", supporting_context)

            # DEBUG: Log the actual prompt being sent to LLM
            logger.info(f"🔍 SENDING TO LLM - Criterion: {criterion_name}")
            logger.info(f"🔍 Course Context: {course_context[:200]}...")
            if has_supporting_docs:
                logger.info(
                    f"🔍 Supporting Context: {supporting_context[:200]}...")
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
