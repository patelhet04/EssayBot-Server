#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import json
import logging
import boto3
from io import BytesIO
from flask import Blueprint, request, jsonify
import os
from urllib.parse import urlparse, unquote

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create blueprint
reports_bp = Blueprint("reports", __name__)

# S3 configuration
s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name="us-east-1",
    config=boto3.session.Config(signature_version='s3v4')
)
S3_BUCKET = os.getenv("MINIO_BUCKET", "essaybot")


def download_file_from_s3(s3_key: str) -> BytesIO:
    """Download a file from S3 and return it as a BytesIO object."""
    logger.info(f"Downloading file from S3: {s3_key}")
    try:
        # If full URL is passed, extract key from path
        if s3_key.startswith("http"):
            parsed_url = urlparse(s3_key)
            s3_key = unquote(parsed_url.path.lstrip("/"))

        # ✅ Strip leading bucket prefix if accidentally present
        if s3_key.startswith(f"{S3_BUCKET}/"):
            s3_key = s3_key[len(f"{S3_BUCKET}/"):]

        logger.info(f"Final S3 key used for download: {s3_key}")
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return BytesIO(response["Body"].read())

    except Exception as e:
        logger.error(f"Failed to download from S3: {str(e)}")
        raise


def get_feedback_and_score_columns(df, criteria_names):
    """
    Extract feedback and score columns from DataFrame based on criteria names.
    Returns a mapping of criteria to their feedback and score columns.
    """
    column_mapping = {}

    # Print columns for debugging
    logger.info(f"Available columns in Excel: {df.columns.tolist()}")

    for criterion in criteria_names:
        # Create exact patterns to match feedback and score columns
        feedback_col = f"{criterion}_feedback"
        score_col = f"{criterion}_score"

        # Also try with spaces removed
        feedback_pattern = criterion.replace(" ", "").upper() + ".*FEEDBACK"
        score_pattern = criterion.replace(" ", "").upper() + ".*SCORE"

        # Find matching columns
        feedback_cols = [col for col in df.columns if feedback_col.upper(
        ) in col.upper() or re.search(feedback_pattern, col.upper())]
        score_cols = [col for col in df.columns if score_col.upper(
        ) in col.upper() or re.search(score_pattern, col.upper())]

        if feedback_cols and score_cols:
            column_mapping[criterion] = {
                "feedback": feedback_cols[0],
                "score": score_cols[0]
            }
        else:
            logger.warning(
                f"Could not find feedback or score columns for criterion: {criterion}")
            logger.warning(f"Looked for patterns: {feedback_col}, {score_col}")

    return column_mapping


def get_score_distribution_data(detailed_stats):
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
              "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    criteria_names = list(detailed_stats.keys())
    matrix = []

    for criterion in criteria_names:
        data = detailed_stats[criterion]
        weight = data["weight"]
        raw_scores = data["scores"]

        # Convert raw scores to percentage scores
        percentages = [(s / weight) * 100 for s in raw_scores]

        # Bin the percentages
        bin_counts = [0] * (len(bins) - 1)
        for p in percentages:
            for i in range(len(bins) - 1):
                if i == len(bins) - 2:
                    if bins[i] <= p <= bins[i+1]:
                        bin_counts[i] += 1
                else:
                    if bins[i] <= p < bins[i+1]:
                        bin_counts[i] += 1
        matrix.append(bin_counts)

    return {
        "matrix": matrix,
        "labels": labels,
        "criteria_names": criteria_names
    }


def get_radar_chart_data(df, columns_mapping, criteria_data, total_scores):
    """
    Calculate radar chart data showing average performance percentage for top 25% and bottom 25% students.

    Args:
        df: DataFrame with score data
        columns_mapping: Mapping of criteria to their column names
        criteria_data: Information about criteria weights
        total_scores: Series of total scores used to identify top/bottom students

    Returns:
        Dictionary with radar chart data for top and bottom 25% of students
    """
    # Identify the cut-off scores for the top 25% and bottom 25%
    top_25_cutoff = total_scores.quantile(0.75)
    bottom_25_cutoff = total_scores.quantile(0.25)

    # Create masks for the top and bottom students
    top_students_mask = total_scores >= top_25_cutoff
    bottom_students_mask = total_scores <= bottom_25_cutoff

    # Filter the dataframes
    top_df = df[top_students_mask]
    bottom_df = df[bottom_students_mask]

    # Calculate average scores as percentages for each criterion
    top_percentages = []
    bottom_percentages = []
    criteria_names = []

    for criterion, details in criteria_data.items():
        if criterion in columns_mapping:
            criteria_names.append(criterion)
            weight = details["weight"]
            score_col = columns_mapping[criterion]["score"]

            # Calculate average percentage for top students (score / max possible score * 100)
            if len(top_df) > 0:
                top_percentage = (top_df[score_col].mean() / weight) * 100
            else:
                top_percentage = 0

            # Calculate average percentage for bottom students
            if len(bottom_df) > 0:
                bottom_percentage = (
                    bottom_df[score_col].mean() / weight) * 100
            else:
                bottom_percentage = 0

            top_percentages.append(float(top_percentage))
            bottom_percentages.append(float(bottom_percentage))

    return {
        "criteria": criteria_names,
        "top_25_percent": top_percentages,
        "bottom_25_percent": bottom_percentages
    }


def analyze_grading_performance(file_path, config_rubric):
    """
    Analyze grading performance for AI grading.

    Args:
        file_path: S3 path to the Excel file
        config_rubric: Rubric configuration containing criteria details
    """
    # Load Data from S3
    file_obj = download_file_from_s3(file_path)

    # Read the Excel file
    df_ai = pd.read_excel(file_obj)

    # Extract criteria names and weights
    criteria_data = {}
    for criterion in config_rubric["criteria"]:
        weight = criterion["weight"]
        # Convert weight to int if it's a string
        if isinstance(weight, str):
            weight = int(weight)
        # Handle MongoDB extended JSON format
        elif isinstance(weight, dict) and "$numberInt" in weight:
            weight = int(weight["$numberInt"])

        criteria_data[criterion["name"]] = {
            "weight": weight,
            "description": criterion["description"]
        }

    # Get column mappings for AI dataframe
    ai_columns = get_feedback_and_score_columns(df_ai, criteria_data.keys())

    # Calculate weighted total scores
    def calculate_total_score(df, columns_mapping):
        total_scores = pd.Series(0, index=df.index)
        total_percentages = pd.Series(0, index=df.index)
        max_possible_score = 0

        # First pass to calculate max_possible_score
        for criterion, details in criteria_data.items():
            if criterion in columns_mapping:
                max_possible_score += details["weight"]

        # Second pass to calculate scores and percentages
        for criterion, details in criteria_data.items():
            if criterion in columns_mapping:
                weight = details["weight"]
                score_col = columns_mapping[criterion]["score"]

                # Raw scores
                weighted_score = df[score_col]
                total_scores += weighted_score

                # Calculate percentage achievement for this criterion
                criterion_percentage = (df[score_col] / weight) * 100
                # Weight the percentage by the criterion's weight relative to total
                weighted_percentage = criterion_percentage * \
                    (weight / max_possible_score)
                total_percentages += weighted_percentage

        return total_scores, total_percentages, max_possible_score

    # Calculate total scores and percentages for AI
    df_ai["TOTAL_SCORE"], df_ai["TOTAL_PERCENTAGE"], ai_max_score = calculate_total_score(
        df_ai, ai_columns)

    # Generate detailed histogram data with specific bins
    def generate_detailed_histogram(scores, max_score, is_percentage=False):
        if is_percentage:
            # Use 20% intervals for percentage histogram
            percentages = [0, 20, 40, 60, 80, 100]
            labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        else:
            # Use 5% intervals for raw score histogram
            percentages = [0, 5, 10, 15, 20, 25, 30, 35, 40,
                           45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            labels = [f"{i}-{i+5}%" for i in range(0, 100, 5)]

        bins = [max_score * (p/100) for p in percentages]
        counts = []

        logger.info(f"Generating histogram with max_score: {max_score}")
        logger.info(f"Score distribution: {scores.describe()}")
        logger.info(f"Bin edges: {bins}")

        for i in range(len(bins)-1):
            # For the last bin, include the upper bound
            if i == len(bins)-2:
                count = len(scores[(scores >= bins[i])
                            & (scores <= bins[i+1])])
            else:
                count = len(scores[(scores >= bins[i]) & (scores < bins[i+1])])
            counts.append(count)
            logger.info(
                f"Bin {labels[i]}: {count} scores between {bins[i]:.1f} and {bins[i+1]:.1f}")

        return {
            "labels": labels,
            "counts": counts,
            "bins": [float(b) for b in bins],
            "max_score": float(max_score)
        }

    # Calculate statistics for each criterion
    def compute_detailed_stats(df, columns_mapping, criteria_data):
        stats = {}
        for criterion, columns in columns_mapping.items():
            score_col = columns["score"]
            if score_col in df.columns:
                stats[criterion] = {
                    "min": float(df[score_col].min()),
                    "max": float(df[score_col].max()),
                    "mean": float(df[score_col].mean()),
                    "weight": criteria_data[criterion]["weight"],
                    "description": criteria_data[criterion]["description"],
                    # Individual scores for detailed analysis
                    "scores": df[score_col].tolist()
                }
        return stats

    # Construct the response data
    response_data = {
        "total_students": int(df_ai["TOTAL_SCORE"].count()),
        "histogram": generate_detailed_histogram(df_ai["TOTAL_SCORE"], ai_max_score, is_percentage=False),
        "percentage_histogram": generate_detailed_histogram(df_ai["TOTAL_PERCENTAGE"], 100, is_percentage=True),
        "statistics": {
            "count": float(df_ai["TOTAL_SCORE"].count()),
            "mean": float(df_ai["TOTAL_SCORE"].mean()),
            "std": float(df_ai["TOTAL_SCORE"].std()),
            "min": float(df_ai["TOTAL_SCORE"].min()),
            "25%": float(df_ai["TOTAL_SCORE"].quantile(0.25)),
            "50%": float(df_ai["TOTAL_SCORE"].quantile(0.50)),
            "75%": float(df_ai["TOTAL_SCORE"].quantile(0.75)),
            "max": float(df_ai["TOTAL_SCORE"].max()),
            "percentage_stats": {
                "mean": float(df_ai["TOTAL_PERCENTAGE"].mean()),
                "std": float(df_ai["TOTAL_PERCENTAGE"].std()),
                "min": float(df_ai["TOTAL_PERCENTAGE"].min()),
                "max": float(df_ai["TOTAL_PERCENTAGE"].max()),
            }
        },
        "rubric_evaluation": {
            "criteria": list(criteria_data.keys()),
            "weights": [data["weight"] for data in criteria_data.values()],
            "ai": {
                "means": [float(df_ai[ai_columns[c]["score"]].mean()) if c in ai_columns else 0
                          for c in criteria_data.keys()],
                "detailed_stats": compute_detailed_stats(df_ai, ai_columns, criteria_data)
            }
        }
    }

    # Add the score distribution data
    detailed_stats = response_data["rubric_evaluation"]["ai"]["detailed_stats"]
    response_data["score_distribution_data"] = get_score_distribution_data(
        detailed_stats)

    # Add radar chart data
    response_data["radar_chart_data"] = get_radar_chart_data(
        df_ai, ai_columns, criteria_data, df_ai["TOTAL_SCORE"])

    return response_data


@reports_bp.route('/analyze_grading', methods=['POST'])
def analyze_grading():
    """
    Analyze grading performance from an Excel file in S3.
    Expects a POST request with s3_file_path and config_rubric.
    """
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")
        required_fields = ["s3_file_path", "config_rubric"]

        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {', '.join(required_fields)}"}), 400

        s3_file_path = data["s3_file_path"]
        config_rubric = data["config_rubric"]

        # Validate config_rubric
        if not isinstance(config_rubric, dict) or "criteria" not in config_rubric:
            return jsonify({"error": "config_rubric must be an object with a criteria array"}), 400

        # Analyze the grading performance
        result = analyze_grading_performance(
            file_path=s3_file_path,
            config_rubric=config_rubric
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error analyzing grading performance: {str(e)}")
        return jsonify({"error": str(e)}), 500
