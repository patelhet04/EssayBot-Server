import { Request, Response } from "express";
import axios from "axios";
import { Schema } from "mongoose";
import { Course } from "../../models/Course";
import { Assignment } from "../../models/Assignment";
import { GradingStats } from "../../models/GradingStats";
import { Criterion } from "../assignments";
import { GradingHistory } from "../../models/GradingHistory";

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

interface AnalyzeGradingRequest {
  courseId: string;
  assignmentId: string;
  gradingHistoryId?: string; // Optional parameter to specify which grading history to analyze
  config_rubric: {
    criteria: Criterion[];
  };
}

export const analyzeGradingPerformance = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  try {
    const { courseId, assignmentId, gradingHistoryId } =
      req.body as AnalyzeGradingRequest;
    const username = req.user.username;

    // Validate required fields
    if (!courseId || !assignmentId) {
      return res.status(400).json({
        message: "Missing required fields: courseId, assignmentId",
      });
    }

    // Check if the course exists
    const course = await Course.findById(courseId);
    if (!course) {
      return res.status(404).json({ message: "Course not found" });
    }

    // Check if the assignment exists and belongs to the course
    const assignment = await Assignment.findOne({
      _id: assignmentId,
      course: courseId,
    });
    if (!assignment) {
      return res.status(404).json({
        message: "Assignment not found or does not belong to this course",
      });
    }

    // Find the GradingHistory record based on the provided ID or get the latest one
    const gradingHistory = await GradingHistory.findById(gradingHistoryId)
      

    if (!gradingHistory) {
      return res.status(400).json({
        message: "No grading history found for this assignment",
      });
    }

    // Get the GradingStats record using the gradingStatsId
    const gradingStats = await GradingStats.findById(
      gradingHistory.gradingStatsId
    );
    if (!gradingStats || !gradingStats.gradeFile?.url) {
      return res.status(400).json({
        message: "No grading file found for this grading attempt",
      });
    }

    console.log(gradingStats.gradeFile.url);

    // Call the Python backend service
    const pythonServiceUrl =
      process.env.PYTHON_SERVICE_URL || "http://localhost:6000";
    const response = await axios.post(`${pythonServiceUrl}/analyze_grading`, {
      s3_file_path: gradingStats.gradeFile.url,
      config_rubric: gradingHistory.config_rubric, // Use the rubric from GradingHistory
    });

    // Return the analysis results
    res.status(200).json(response.data);
  } catch (error: any) {
    console.error("Error analyzing grading performance:", error);
    res.status(500).json({
      message: "Failed to analyze grading performance",
      error: error.message,
    });
  }
};
