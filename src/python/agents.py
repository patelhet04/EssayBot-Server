# agents.py - Simplified for clarity and anti-hallucination

# Common components for all prompts
ROLE_DESCRIPTION = """
You are a {tone} academic essay grader with expertise in evaluating student work. 
Provide fair, consistent scoring and constructive feedback based on the rubric.
Do NOT use external knowledge or make up quotes.
"""

GRADING_PRINCIPLES = """
GRADING PRINCIPLES:
- Grade based ONLY on those exact phrases from the student's writing
- Reference specific parts of their essay in your feedback only if necessary
- Be consistent with the rubric criteria and point allocation
- Apply quality adjustments for response relevance and engagement
- For responses that don't address the assignment: assign 0 points
"""


def get_feedback_instructions(quality_multiplier=1.0, specificity_score=0.5):
    """Generate quality-aware feedback instructions based on essay analysis."""
    if quality_multiplier < 0.7:  # Low quality / gibberish responses (specificity < 0.3)
        return """
**For Low-Quality/Irrelevant Responses (Gibberish Detection):**
- This response shows minimal or zero engagement with the assignment (quality score: {:.2f})
- State clearly that the response doesn't address the assignment requirements
- Be direct: "This response does not address the assignment question and lacks relevant academic content."
""".format(quality_multiplier)
    elif quality_multiplier < 0.9:  # Moderate quality (specificity 0.3-0.7)
        return """
**For Moderate-Quality Responses:**
- This response shows moderate engagement (quality score: {:.2f})
- Reference specific parts of their essay when giving feedback only if necessary
- Identify what they did well and provide constructive feedback
- Provide 1-2 concrete suggestions based on their actual content if required
- Keep feedback between 50-60 words
""".format(quality_multiplier)
    else:  # High quality responses (specificity > 0.7)
        return """
**For High-Quality Responses:**
- This response shows strong engagement with course material (quality score: {:.2f})
- Quote or reference specific strengths in their essay only if necessary
- Provide nuanced feedback that pushes them to the next level
- Suggest ways to deepen their analysis based on what they've written
- Keep feedback between 50-60 words
""".format(quality_multiplier)


def get_quality_analysis_note(quality_multiplier, specificity_score):
    """Generate a note about the quality analysis for transparency."""
    if quality_multiplier < 0.7:
        return f"\n[Analysis: Response specificity: {specificity_score:.3f}, Quality multiplier: {quality_multiplier:.2f} - Low engagement detected]"
    elif quality_multiplier > 1.1:
        return f"\n[Analysis: Response specificity: {specificity_score:.3f}, Quality multiplier: {quality_multiplier:.2f} - Strong engagement detected]"
    else:
        return ""


def _create_prompt_template(prompt_data, instructions, feedback_instructions, quality_multiplier, quality_note, has_supporting_docs=False):
    """Create the standardized prompt template to eliminate duplication."""

    # Base template with course content
    template = f"""
{ROLE_DESCRIPTION}

========== ASSIGNMENT QUESTION ==========
{{{{question}}}}
========== END OF QUESTION ==========   

========== STUDENT ESSAY (ONLY SOURCE FOR QUOTES) ==========
{{{{essay}}}}
========== END OF STUDENT ESSAY ==========

========== COURSE CONTENT (PRIMARY GRADING SOURCE) ==========
{{{{course_context}}}}
========== END OF COURSE CONTENT =========="""

    # Conditionally add supporting docs section
    if has_supporting_docs:
        template += """

========== SUPPORTING DOCUMENTS (FACT-CHECKING & REFERENCE) ==========
{{{{supporting_context}}}}
========== END OF SUPPORTING DOCUMENTS =========="""

    # Continue with the rest of the template
    template += f"""

GRADING TASK: {prompt_data['prompt']['header']}

GRADING CRITERIA:
{instructions}

{GRADING_PRINCIPLES}    

CONTEXT USAGE INSTRUCTIONS:
- Use COURSE CONTENT as your PRIMARY SOURCE for grading criteria and standards
- Course content contains the main materials (textbooks, lectures) for this assignment"""

    if has_supporting_docs:
        template += """
- Use SUPPORTING DOCUMENTS for fact-checking and additional context only
- Supporting documents provide supplementary information but should not override course content"""

    template += f"""

{feedback_instructions}

ABSOLUTE REQUIREMENTS:
- Do NOT paraphrase or rewrite what the student said
- Do NOT reference content not written by the student
- If student essay doesn't address the criteria, say so directly
- Base grading primarily on course content alignment

Quality Level: {quality_multiplier:.2f} (affects final score){quality_note}

Return ONLY this JSON format:
{{"score": <number>, "feedback": <string>}}
"""

    return template


def get_prompt(criteria_prompts, criterion_name=None, tone="moderate", quality_multiplier=1.0, specificity_score=0.5, has_supporting_docs=False):
    """
    Assembles prompts with smart gibberish detection and quality-aware scoring.

    Args:
        quality_multiplier (float): 0.6 for gibberish, 1.0 neutral, 1.2 for excellent (from LlamaIndex analysis)
        specificity_score (float): 0.0-1.0 specificity score from DynamicQueryProcessor
        has_supporting_docs (bool): Whether supporting documents are available
    """
    if not criteria_prompts:
        return None

    # Format the role description with the provided tone
    formatted_role_description = ROLE_DESCRIPTION.format(tone=tone)

    # Get quality-aware feedback instructions
    feedback_instructions = get_feedback_instructions(
        quality_multiplier, specificity_score)

    # Get quality analysis note for transparency
    quality_note = get_quality_analysis_note(
        quality_multiplier, specificity_score)

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
            [f"• {instr}" for instr in prompt_data["prompt"]["instructions"]])

        # Create standardized prompt using reusable template
        full_prompt = _create_prompt_template(
            prompt_data=prompt_data,
            instructions=instructions,
            feedback_instructions=feedback_instructions,
            quality_multiplier=quality_multiplier,
            quality_note=quality_note,
            has_supporting_docs=has_supporting_docs
        )
        return {
            "prompt": full_prompt
        }

    # If no criterion_name is specified, return all prompts
    assembled_prompts = {}
    for prompt_data in criteria_prompts:
        criterion_name = prompt_data["criterionName"]
        instructions = "\n".join(
            [f"• {instr}" for instr in prompt_data["prompt"]["instructions"]])

        full_prompt = _create_prompt_template(
            prompt_data=prompt_data,
            instructions=instructions,
            feedback_instructions=feedback_instructions,
            quality_multiplier=quality_multiplier,
            quality_note=quality_note,
            has_supporting_docs=has_supporting_docs
        )
        assembled_prompts[criterion_name] = {
            "prompt": full_prompt
        }

    return {"criteria_prompts": assembled_prompts}
