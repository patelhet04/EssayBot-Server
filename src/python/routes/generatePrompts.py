from flask import Flask, request, jsonify, Blueprint
import json
import logging
import requests
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

prompt_bp = Blueprint("prompt", __name__)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# LLM API settings
LLM_API_URL = "http://localhost:5001/api/generate"


def send_post_request(prompt, temperature=0.3, top_p=0.1, max_tokens=2048, model="llama3.1:8b"):
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


def generate_criterion_prompt(criterion, agent_index, model):
    """
    Uses LLM to generate the evaluation instructions for a single criterion in JSON format.

    Args:
        criterion (dict): The criterion details.
        agent_index (int): The index of the agent (e.g., 2 for "Agent 2").

    Returns:
        str: The generated prompt as a JSON string, or None if failed.
    """
    llm_instruction = f"""
You are an expert prompt engineer. Generate 2-3 focused evaluation points for the criterion '{criterion['name']}'. 

Requirements:
- Each point should be ONE specific thing to check in the essay
- Use simple, direct language (avoid complex phrasing)
- Focus ONLY on this criterion, not general essay quality
- Keep each instruction under 15 words

Criterion: {json.dumps(criterion)}

Return JSON format:
{{
  "instructions": [
    "<check 1>",
    "<check 2>",
    "<check 3>"
  ]
}}
"""

    response = send_post_request(
        llm_instruction, temperature=0.3, top_p=0.1, max_tokens=2048, model=model)
    if response and "response" in response:
        try:
            result = json.loads(response["response"])
            if "instructions" not in result or not isinstance(result["instructions"], list):
                logger.error(
                    f"Invalid response format from LLM for criterion: {criterion['name']}")
                return None

            # Ensure the instructions are limited to 2-3 (simplified)
            if len(result["instructions"]) > 3:
                result["instructions"] = result["instructions"][:3]
            elif len(result["instructions"]) < 2:
                logger.warning(
                    f"LLM generated fewer than 2 instructions for criterion: {criterion['name']}")
                # Accept what the LLM provides, even if fewer than expected

            # Construct the prompt as a JSON object
            prompt_data = {
                "header": f"**{criterion['name']} (Max: {criterion['weight']} points)**",
                "introduction": f"Check if the essay meets these requirements:",
                "instructions": result["instructions"]
            }

            # Return the prompt as a JSON string
            return json.dumps(prompt_data)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM response for criterion: {criterion['name']}, error: {str(e)}")
            return None
    else:
        logger.error(
            f"Failed to generate prompt for criterion: {criterion['name']}")
        return None


@prompt_bp.route('/generate_prompt', methods=['POST'])
def generate_prompt():
    """
    Generates prompts for a rubric and returns them to be stored in Node.js.
    Expects a POST request with criteria JSON, username, courseId, and assignmentTitle.
    """
    try:
        data = request.get_json()
        required_fields = ["criteria", "username",
                           "courseId", "assignmentTitle", "model"]
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {', '.join(required_fields)}"}), 400

        rubric_json = data["criteria"]

        # Validate rubric_json
        if not isinstance(rubric_json, list):
            return jsonify({"error": "Criteria must be a list of criterion objects"}), 400

        # Generate prompts for each criterion
        criteria_prompts = {}
        # Start agent index at 1
        for idx, criterion in enumerate(rubric_json, start=1):
            prompt = generate_criterion_prompt(
                criterion, agent_index=idx, model=data["model"])
            if not prompt:
                return jsonify({"error": f"Failed to generate prompt for criterion: {criterion['name']}"}), 500
            criteria_prompts[criterion["name"]] = prompt
            logger.info(f"Generated prompt for criterion: {criterion['name']}")

        # Prepare the response in a flatter format
        return jsonify({
            "message": "Prompts generated successfully",
            "criteria_prompts": criteria_prompts
        }), 200
    except Exception as e:
        logger.error(f"Error generating prompts: {e}")
        return jsonify({"error": str(e)}), 500
