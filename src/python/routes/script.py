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


def send_post_request(prompt, temperature=0.7, top_p=0.9, max_tokens=2048, model="llama3.1:8b"):
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

        # Step 1: Retrieve RAG context for grading
        try:
            rag_chunks = retrieve_relevant_text(
                query=question,  # Use the question as the query for RAG
                professor_username=professor_username,
                course_id=course_id,
                assignmentTitle=assignment_title,
                k=10, distance_threshold=0.5,  # Default value, adjust based on your embedding model
                max_total_length=6000
            )
            rag_context = "\n".join(
                rag_chunks) if rag_chunks else "No relevant context available."
        except Exception as e:
            logger.error(f"Failed to retrieve RAG context: {str(e)}")
            rag_context = "No relevant context available due to retrieval error."
        logger.info(
            f"RAG context retrieved for grading: {rag_context[:200]}...")

        # Step 2: Assemble the prompts using get_prompt from agents.py
        assembled_prompts = get_prompt(config_prompt, tone=data["tone"])
        if not assembled_prompts or "criteria_prompts" not in assembled_prompts:
            return jsonify({"error": "Failed to assemble prompts"}), 500

        # Step 3: Grade the essay for each criterion (agent)
        grading_results = {}
        for criterion_name, prompt_data in assembled_prompts["criteria_prompts"].items():
            # Replace placeholders in the prompt
            full_prompt = prompt_data["prompt"]
            full_prompt = full_prompt.replace("{{question}}", question)
            full_prompt = full_prompt.replace("{{essay}}", essay)
            full_prompt = full_prompt.replace("{{rag_context}}", rag_context)

            # Send the prompt to the LLM for grading
            response = send_post_request(
                full_prompt, temperature=0.7, top_p=0.9, max_tokens=2048, model=data["model"])
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
                    grading_results[criterion_name] = {
                        "score": result["score"],
                        "feedback": result["feedback"]
                    }
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
