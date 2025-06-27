# agents.py - Simplified for clarity and anti-hallucination

# Common components for all prompts
ROLE_DESCRIPTION = """
You are a {tone} academic essay grader with expertise in evaluating student work. 
Provide fair, consistent scoring and constructive feedback based on the rubric.
"""

GRADING_PRINCIPLES = """
**Grading Principles:**
- Score based solely on what the student actually wrote
- Reference specific parts of their essay in your feedback
- Be consistent with the rubric criteria and point allocation
- Apply quality adjustments for response relevance and engagement
"""


def get_feedback_instructions(quality_multiplier=1.0, specificity_score=0.5):
    """Generate quality-aware feedback instructions based on essay analysis."""
    if quality_multiplier < 0.7:  # Low quality / gibberish responses (specificity < 0.3)
        return """
**For Low-Quality/Irrelevant Responses (Gibberish Detection):**
- This response shows minimal engagement with the assignment (quality score: {:.2f})
- State clearly that the response doesn't address the assignment requirements
- Keep feedback under 40 words - do NOT provide detailed suggestions for gibberish
- Be direct: "This response does not address the assignment question and lacks relevant academic content."
""".format(quality_multiplier)
    elif quality_multiplier < 0.9:  # Moderate quality (specificity 0.3-0.7)
        return """
**For Moderate-Quality Responses:**
- This response shows moderate engagement (quality score: {:.2f})
- Reference specific parts of their essay when giving feedback
- Identify what they did well and what needs improvement
- Provide 1-2 concrete suggestions based on their actual content
- Keep feedback between 50-80 words
""".format(quality_multiplier)
    else:  # High quality responses (specificity > 0.7)
        return """
**For High-Quality Responses:**
- This response shows strong engagement with course material (quality score: {:.2f})
- Quote or reference specific strengths in their essay
- Provide nuanced feedback that pushes them to the next level
- Suggest ways to deepen their analysis based on what they've written
- Keep feedback between 80-120 words
""".format(quality_multiplier)


def get_quality_analysis_note(quality_multiplier, specificity_score):
    """Generate a note about the quality analysis for transparency."""
    if quality_multiplier < 0.7:
        return f"\n[Analysis: Response specificity: {specificity_score:.3f}, Quality multiplier: {quality_multiplier:.2f} - Low engagement detected]"
    elif quality_multiplier > 1.1:
        return f"\n[Analysis: Response specificity: {specificity_score:.3f}, Quality multiplier: {quality_multiplier:.2f} - Strong engagement detected]"
    else:
        return ""


def get_prompt(criteria_prompts, criterion_name=None, tone="moderate", quality_multiplier=1.0, specificity_score=0.5):
    """
    Assembles prompts with smart gibberish detection and quality-aware scoring.

    Args:
        quality_multiplier (float): 0.6 for gibberish, 1.0 neutral, 1.2 for excellent (from LlamaIndex analysis)
        specificity_score (float): 0.0-1.0 specificity score from DynamicQueryProcessor
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

        # EXTREME anti-hallucination prompt with context and feedback instructions
        full_prompt = f"""
You must evaluate this student essay using ONLY the content provided below. Do NOT use external knowledge or make up quotes.

========== ASSIGNMENT QUESTION ==========
{{{{question}}}}
========== END OF QUESTION ==========   

========== COURSE MATERIAL (REFERENCE ONLY) ==========
{{{{rag_context}}}}
========== END OF COURSE MATERIAL ==========

========== STUDENT ESSAY (ONLY SOURCE FOR QUOTES) ==========
{{{{essay}}}}
========== END OF STUDENT ESSAY ==========

GRADING TASK: {prompt_data['prompt']['header']}

GRADING CRITERIA:
{instructions}

{feedback_instructions}

ABSOLUTE REQUIREMENTS:
- Quote EXACT words from the student essay (copy-paste only)
- Do NOT paraphrase or rewrite what the student said
- Do NOT reference content not written by the student
- If student essay doesn't address the criteria, say so directly
- Use course material only for context, NOT as source of quotes about the student

STEP 1: Find 2-3 exact phrases from the STUDENT ESSAY above that relate to the grading criteria
STEP 2: Grade based ONLY on those exact phrases from the student's writing

Quality Level: {quality_multiplier:.2f} (affects final score)

Return ONLY this JSON format:
{{"score": <number>, "feedback": "Based on your essay content: '[exact quote from student]'. [brief evaluation based on that exact quote]"}}
"""
        return {
            "prompt": full_prompt
        }

    # If no criterion_name is specified, return all prompts
    assembled_prompts = {}
    for prompt_data in criteria_prompts:
        criterion_name = prompt_data["criterionName"]
        instructions = "\n".join(
            [f"• {instr}" for instr in prompt_data["prompt"]["instructions"]])

        full_prompt = f"""
You must evaluate this student essay using ONLY the content provided below. Do NOT use external knowledge or make up quotes.

========== ASSIGNMENT QUESTION ==========
{{{{question}}}}
========== END OF QUESTION ==========

========== COURSE MATERIAL (REFERENCE ONLY) ==========
{{{{rag_context}}}}
========== END OF COURSE MATERIAL ==========

========== STUDENT ESSAY (ONLY SOURCE FOR QUOTES) ==========
{{{{essay}}}}
========== END OF STUDENT ESSAY ==========

GRADING TASK: {prompt_data['prompt']['header']}

GRADING CRITERIA:
{instructions}

{feedback_instructions}

ABSOLUTE REQUIREMENTS:
- Quote EXACT words from the student essay (copy-paste only)
- Do NOT paraphrase or rewrite what the student said
- Do NOT reference content not written by the student
- If student essay doesn't address the criteria, say so directly
- Use course material only for context, NOT as source of quotes about the student

STEP 1: Find 2-3 exact phrases from the STUDENT ESSAY above that relate to the grading criteria
STEP 2: Grade based ONLY on those exact phrases from the student's writing

Quality Level: {quality_multiplier:.2f} (affects final score)

Return ONLY this JSON format:
{{"score": <number>, "feedback": "Based on your essay content: '[exact quote from student]'. [brief evaluation based on that exact quote]"}}
"""
        assembled_prompts[criterion_name] = {
            "prompt": full_prompt
        }

    return {"criteria_prompts": assembled_prompts}
