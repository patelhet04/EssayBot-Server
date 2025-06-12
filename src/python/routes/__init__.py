from flask import Blueprint
from .rag_pipeline import rag_bp  # Import your blueprints
from .generateRubric import rubric_bp  # Import other blueprints if applicable
from .generatePrompts import prompt_bp  # Import other blueprints if applicable
from .script import singleGrading_bp
# Import other blueprints if applicable
from .bulkGrading import bulkGrading_bp
from .reports import reports_bp  # Import reports blueprint


def create_blueprints():
    """Registers all Flask blueprints."""
    return [rag_bp, rubric_bp, prompt_bp, singleGrading_bp, bulkGrading_bp, reports_bp]
