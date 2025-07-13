import os
import json
import logging
import random
import requests
import re
from typing import List, Dict, Any
from flask import Flask, request, jsonify, Blueprint

from .rag_pipeline import retrieve_relevant_text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create a Flask blueprint for rubric generation
rubric_bp = Blueprint("rubric", __name__)

# Local LLM API configuration
API_URL = "http://localhost:5001/api/generate"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 4000


def send_post_request(prompt: str, temperature=DEFAULT_TEMPERATURE,
                      top_p=DEFAULT_TOP_P, max_tokens=DEFAULT_MAX_TOKENS,
                      model="llama3.3:70b") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    headers = {"Content-Type": "application/json"}
    logger.info(f"Sending request to local model: {model}")
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling local model API: {str(e)}")
        raise


def generate_sample_rubric(question: str, context: List[str], model: str = "llama3.3:70b") -> Dict[str, Any]:
    """Generate a single sample rubric for the given question and context."""
    logger.info(f"Generating a sample rubric using model: {model}...")
    try:
        context_text = ' '.join(context)
        prompt = f"""
        You are an expert educational assessment designer. Your task is to create a grading rubric based ONLY on the assignment question and course content provided below.

        **ASSIGNMENT QUESTION:**
        {question}

        **COURSE CONTENT (Textbooks, Lectures, Course Materials):**
        {context_text}

        **RUBRIC CREATION INSTRUCTIONS:**
        Create grading criteria based EXCLUSIVELY on:
        1. **The assignment question requirements** - what specific knowledge/skills it's testing
        2. **The course content provided** - the main learning materials for this course
        
        Do NOT use external knowledge or general essay writing criteria. Base your rubric entirely on what students should demonstrate based on the course materials and question requirements.

        **CRITERIA REQUIREMENTS:**
        Each criterion should be:
        1. **Specific to this assignment question** - directly assess what the question is asking
        2. **Grounded in course content** - evaluate understanding of the provided course materials
        3. **Measurable and distinct** - clearly define different aspects of student performance
        4. **Keep criteria separate** - do not combine multiple concepts into one criterion
        5. **Concise** - criterion descriptions should be 40-50 words maximum

        Create a grading rubric with 3-5 relevant criteria. Each criterion should include:
        1. A clear name
        2. A detailed description based on course content expectations
        3. A weight (numerical value where all weights add up to 100)
        4. Scoring levels with descriptions for full, partial, and minimal performance
        5. An empty subCriteria array
        
        Return the rubric as a valid JSON object with the following structure:
        
        {{
          "criteria": [
            {{
              "name": "Criterion Name",
              "description": "Detailed description grounded in course content and question requirements",
              "weight": number,
              "scoringLevels": {{
                "full": "Description of full points performance based on course expectations",
                "partial": "Description of partial points performance based on course expectations",
                "minimal": "Description of minimal points performance based on course expectations"
              }},
              "subCriteria": []
            }}
          ]
        }}

        **IMPORTANT:**
        - Return ONLY the JSON object with no additional text before or after it
        - Do NOT include labels like "Full Points:" or "Partial Points:" in the scoringLevels descriptions
        - The descriptions should be the actual expectation text only
        - Keep scoring level descriptions concise and specific
        """
        response = send_post_request(prompt=prompt, temperature=0.3, top_p=0.1,
                                     max_tokens=1000, model=model)
        response = response.strip()
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            try:
                rubric_json = json.loads(json_str)
                if "criteria" not in rubric_json:
                    rubric_json = {"criteria": rubric_json}
                for criterion in rubric_json["criteria"]:
                    if "subCriteria" not in criterion:
                        criterion["subCriteria"] = []
                    if "scoringLevels" not in criterion:
                        criterion["scoringLevels"] = {
                            "full": "Excellent performance in this criterion.",
                            "partial": "Satisfactory performance in this criterion.",
                            "minimal": "Minimal performance in this criterion."
                        }
                    
                    # Clean scoring levels to remove any labels
                    if "scoringLevels" in criterion and isinstance(criterion["scoringLevels"], dict):
                        def clean_text(text):
                            if not text:
                                return ""
                            # Remove common labels that might be included
                            return re.sub(r'^(Full Points?|Partial Points?|Minimal Points?):\s*', '', text, flags=re.IGNORECASE).strip()
                        
                        for level in ["full", "partial", "minimal"]:
                            if level in criterion["scoringLevels"]:
                                criterion["scoringLevels"][level] = clean_text(criterion["scoringLevels"][level])
                    
                    if "weight" not in criterion or not isinstance(criterion["weight"], (int, float)):
                        criterion["weight"] = 100 // len(
                            rubric_json["criteria"])
                total_weight = sum(c["weight"]
                                   for c in rubric_json["criteria"])
                if total_weight != 100:
                    scale_factor = 100 / total_weight
                    for criterion in rubric_json["criteria"]:
                        criterion["weight"] = round(
                            criterion["weight"] * scale_factor)
                    diff = 100 - sum(c["weight"]
                                     for c in rubric_json["criteria"])
                    if diff != 0:
                        rubric_json["criteria"][0]["weight"] += diff
                logger.info("Successfully generated sample rubric")
                return rubric_json
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error parsing JSON from model response: {str(e)}")
                raise
        else:
            logger.error("Could not find valid JSON in model response")
            raise ValueError("No valid JSON found in response")
    except Exception as e:
        logger.error(f"Error generating rubric: {str(e)}")
        return {
            "criteria": [
                {
                    "name": "Criterion 1",
                    "description": "Auto-generated placeholder criterion",
                    "weight": 100,
                    "scoringLevels": {
                        "full": "Excellent performance in this criterion.",
                        "partial": "Satisfactory performance in this criterion.",
                        "minimal": "Minimal performance in this criterion."
                    },
                    "subCriteria": []
                }
            ]
        }


def generate_criteria_expectations(criteria_name: str, criteria_description: str, question: str, context: List[str], bracket_labels: List[str], model: str = "llama3.3:70b") -> Dict[str, str]:
    """Generate expectations for a single criterion based on its name and description, for arbitrary bracket labels."""
    logger.info(f"Generating expectations for criterion: {criteria_name} using model: {model}...")
    try:
        context_text = ' '.join(context)
        # Dynamically build the bracket instructions
        bracket_instructions = "\n".join([
            f"{i+1}. **{label}**: What constitutes {label.lower()} performance for this criterion" for i, label in enumerate(bracket_labels)
        ])
        bracket_json = ",\n".join([f'  "{label}": "Description for {label} performance"' for label in bracket_labels])
        prompt = f"""
        You are an expert educational assessment designer. Your task is to generate specific expectations for a grading criterion based on the assignment question and course content.

        **ASSIGNMENT QUESTION:**
        {question}

        **CRITERION DETAILS:**
        Name: {criteria_name}
        Description: {criteria_description}

        **COURSE CONTENT (Textbooks, Lectures, Course Materials):**
        {context_text}

        **EXPECTATIONS GENERATION INSTRUCTIONS:**
        Based on the criterion name, description, assignment question, and course content, generate expectations for each of the following marks brackets:
{bracket_instructions}

        **REQUIREMENTS:**
        - Base expectations EXCLUSIVELY on the course content and assignment requirements
        - Be specific and measurable
        - Use clear, academic language
        - **Keep expectations SHORT and PRECISE - maximum 10-15 words each**
        - Focus on key performance indicators only
        - Avoid verbose descriptions
        - Ensure expectations align with the criterion's focus and the assignment question

        Return ONLY a JSON object with the following structure:
        {{
{bracket_json}
        }}

        **IMPORTANT:**
        - Return ONLY the JSON object with no additional text before or after it
        - Do NOT include labels like "Full Points:" or similar in the descriptions
        - The descriptions should be the actual expectation text only
        - Keep each description short and precise (10-15 words maximum)
        """
        response = send_post_request(prompt=prompt, temperature=0.3, top_p=0.1,
                                     max_tokens=800, model=model)
        response = response.strip()
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            try:
                expectations_json = json.loads(json_str)
                # Ensure all required fields are present
                for label in bracket_labels:
                    if label not in expectations_json or not expectations_json[label]:
                        expectations_json[label] = f"Default {label} expectation for {criteria_name}"
                logger.info(f"Successfully generated expectations for criterion: {criteria_name}")
                return expectations_json
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from model response: {str(e)}")
                raise
        else:
            logger.error("Could not find valid JSON in model response")
            raise ValueError("No valid JSON found in response")
    except Exception as e:
        logger.error(f"Error generating expectations for criterion {criteria_name}: {str(e)}")
        return {label: f"Default {label} expectation for {criteria_name}" for label in bracket_labels}


@rubric_bp.route("/generate_rubric", methods=["POST"])
def generate_rubric():
    data = request.get_json()
    print(data)
    question = data.get("question")
    professor = data.get("username")
    title = data.get("title")
    course_id = data.get("courseId")  # e.g., "67dd03b10804dc82ad45da1d"
    model = data.get("model", "llama3.3:70b")

    if not all([question, professor, course_id, title]):
        return jsonify({"error": "question, username, and courseId are required"}), 400

    try:
        # Retrieve comprehensive context from the FAISS index for rubric generation
        context = retrieve_relevant_text(
            query=question,
            k=30,  # Get many more chunks for comprehensive rubric coverage
            professor_username=professor,
            course_id=course_id,
            assignmentTitle=title,
            distance_threshold=0.3,  # Lower threshold to include more content
            max_total_length=12000  # Double the context length for rubrics
        )

        # Generate a single rubric
        rubric = generate_sample_rubric(question, context, model=model)
        print(rubric)
        result = {
            "success": True,
            "message": "Generated a sample rubric",
            "rubric": rubric
        }
        return jsonify(result)

    except Exception as e:
        logger.exception(f"Error generating sample rubric: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error generating sample rubric: {str(e)}",
            "error": str(e)
        }), 500


@rubric_bp.route("/fill_expectations", methods=["POST"])
def fill_expectations():
    data = request.get_json()
    print(data)
    criteria_name = data.get("criteriaName")
    criteria_description = data.get("criteriaDescription")
    question = data.get("question")
    professor = data.get("username")
    title = data.get("title")
    course_id = data.get("courseId")
    model = data.get("model", "llama3.3:70b")
    bracket_labels = data.get("bracketLabels")
    if not all([criteria_name, criteria_description, question, professor, course_id, title, bracket_labels]):
        return jsonify({"error": "criteriaName, criteriaDescription, question, username, courseId, and bracketLabels are required"}), 400
    try:
        # Retrieve relevant context from the FAISS index
        context = retrieve_relevant_text(
            query=f"{criteria_name} {criteria_description}",
            k=20,
            professor_username=professor,
            course_id=course_id,
            assignmentTitle=title,
            distance_threshold=0.4,
            max_total_length=8000
        )
        # Generate expectations for the criterion and all brackets
        expectations = generate_criteria_expectations(
            criteria_name, criteria_description, question, context, bracket_labels, model=model
        )
        result = {
            "success": True,
            "message": "Generated expectations successfully",
            "expectations": expectations
        }
        return jsonify(result)
    except Exception as e:
        logger.exception(f"Error generating expectations: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error generating expectations: {str(e)}",
            "error": str(e)
        }), 500
