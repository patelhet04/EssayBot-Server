# agents.py

# Common components for all prompts
ROLE_DESCRIPTION = """
You are a highly detailed and {tone} evaluator. Generate **concise, specific feedback** with actionable suggestions, teaching-oriented examples, and a supportive tone that aligns with the rubric while acknowledging student efforts constructively.
"""

FEEDBACK_INSTRUCTIONS = """
Evaluate the student's response with **concise, structured, and actionable feedback**. Follow these guidelines:
- **For strong responses:** Confirm correctness, suggest refinements only if they enhance clarity or depth.
- **For mid-range responses:** Highlight strengths, provide clear feedback on weak areas.
- **For weak responses:** Address misunderstandings with precise, actionable next steps.
- **Use the provided context** for guidance, integrating key insights into feedback (don't just say "review course material").
- **Be direct, specific, and professional**, referencing exact parts of the response.
- **Keep feedback between 60-80 words unless specified otherwise.**
"""

JSON_OUTPUT_FORMAT = """
Return the output in JSON format: {"score": <score>, "feedback": "<feedback>"}
"""


def get_prompt(criteria_prompts, criterion_name=None, tone="moderate"):
    """
    Assembles the prompt for a specific criterion or all prompts into the common structure.

    Args:
        criteria_prompts (list): List of {criterionName, prompt} objects (e.g., from config_prompt).
        criterion_name (str, optional): Specific criterion name to retrieve.
        tone (str, optional): The tone to use in the role description. Defaults to "Fair".

    Returns:
        dict: The assembled prompt(s).
    """
    if not criteria_prompts:
        return None

    # Format the role description with the provided tone
    formatted_role_description = ROLE_DESCRIPTION.format(tone=tone)

    if criterion_name:
        # Find the prompt for the specific criterion
        prompt_data = next(
            (item for item in criteria_prompts if item["criterionName"] == criterion_name),
            None
        )
        if not prompt_data:
            return None

        # Assemble the instructions with bullet points
        instructions = "\n".join(
            [f"* {instr}" for instr in prompt_data["prompt"]["instructions"]])
        # Assemble the full prompt
        full_prompt = f"""
{formatted_role_description}

{prompt_data['prompt']['header']}
{prompt_data['prompt']['introduction']}
{instructions}

{FEEDBACK_INSTRUCTIONS}

{JSON_OUTPUT_FORMAT}

Question: {{question}}
Essay: {{essay}}
Relevant Context: {{rag_context}}
"""
        return {
            "prompt": full_prompt
        }

    # If no criterion_name is specified, return all prompts
    assembled_prompts = {}
    for prompt_data in criteria_prompts:
        criterion_name = prompt_data["criterionName"]
        instructions = "\n".join(
            [f"* {instr}" for instr in prompt_data["prompt"]["instructions"]])
        full_prompt = f"""
{formatted_role_description}

{prompt_data['prompt']['header']}
{prompt_data['prompt']['introduction']}
{instructions}

{FEEDBACK_INSTRUCTIONS}

{JSON_OUTPUT_FORMAT}

Question: {{question}}
Essay: {{essay}}
Relevant Context: {{rag_context}}
"""
        assembled_prompts[criterion_name] = {
            "prompt": full_prompt
        }

    return {"criteria_prompts": assembled_prompts}
