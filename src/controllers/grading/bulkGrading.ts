import { Request, Response } from "express";
import { Course } from "../../models/Course";
import { Assignment, IAssignment } from "../../models/Assignment";
import { GradingStats } from "../../models/GradingStats";
import axios from "axios";
import { Types } from "mongoose";
import { GradingHistory } from "../../models/GradingHistory";

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

interface BulkGradingRequest {
  courseId: string;
  assignmentId: string;
  s3_excel_link: string;
  model?: string;
  tone?: string;
}

// Function to sanitize file names for S3/Minio compatibility
const sanitizeFileName = (fileName: string): string => {
  return (
    fileName
      // Replace spaces with underscores
      .replace(/\s+/g, "_")
      // Replace special characters with underscores
      .replace(/[!@#$%^&*()+\=\[\]{}|\\:;"'<>,?\/]/g, "_")
      // Replace any remaining non-alphanumeric characters (except -_.)
      .replace(/[^a-zA-Z0-9-_\.]/g, "_")
      // Replace multiple consecutive underscores with a single one
      .replace(/_{2,}/g, "_")
      // Remove leading/trailing underscores
      .replace(/^_+|_+$/g, "")
      // Convert to lowercase
      .toLowerCase()
  );
};

// Function to generate a Minio-compatible timestamp
const generateMinioSafeTimestamp = (): string => {
  const now = new Date();
  return now
    .toISOString()
    .replace(/T/g, "-") // Replace T with hyphen
    .replace(/:/g, "-") // Replace colons with hyphens
    .replace(/\./g, "-") // Replace dots with hyphens
    .replace(/Z/g, "") // Remove Z
    .slice(0, 19); // Take only YYYY-MM-DD-HH-mm-ss part
};

export const gradeBulkEssays = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  try {
    const {
      courseId,
      assignmentId,
      s3_excel_link,
      model,
      tone,
    }: BulkGradingRequest = req.body;
    const username = req?.user?.username;
    const userId = req?.user?.id;

    // Validate required fields
    if (!courseId || !assignmentId || !s3_excel_link) {
      return res.status(400).json({
        message:
          "Missing required fields: courseId, assignmentId, s3_excel_link",
      });
    }

    // Find the assignment
    const assignment: IAssignment | null = await Assignment.findOne({
      _id: assignmentId,
      course: courseId,
    });

    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }

    // Prepare request data for Flask API
    const requestData = {
      courseId,
      assignmentTitle: assignment._id,
      config_prompt: assignment.config_prompt,
      question: assignment.question,
      username,
      s3_excel_link,
      model: model || "llama3.1:8b", // Use default model if not specified
      tone: tone || "moderate",
    };
    console.log(requestData);

    // Call Flask API for bulk grading
    const response: any = await axios.post(
      "http://localhost:6000/grade_bulk_essays",
      requestData
    );

    if (response.data.s3_graded_link) {
      // Use the new Minio-safe timestamp function
      const timestamp = generateMinioSafeTimestamp();
      const gradedFileName = `graded_${assignment.title}_${timestamp}.xlsx`;
      console.log("Generated file name:", gradedFileName);

      // Create a new GradingStats record
      const gradingStats = await GradingStats.create({
        courseId: new Types.ObjectId(courseId),
        assignmentId: new Types.ObjectId(assignmentId),
        modelName: model || "llama3.1:8b",
        gradeFile: {
          url: response.data.s3_graded_link,
          originalName: gradedFileName,
          uploadedAt: new Date(),
        },
        totalEssays: response.data.total_essays || 0,
        completedAt: new Date(),
        createdBy: new Types.ObjectId(userId),
      });

      await GradingHistory.findOneAndUpdate(
        {
          assignmentId: new Types.ObjectId(assignmentId),
          courseId: new Types.ObjectId(courseId),
          gradingStatsId: { $exists: false },
        },
        {
          $set: {
            gradingStatsId: gradingStats._id,
            config_rubric: assignment.config_rubric,
            createdBy: new Types.ObjectId(userId),
          },
        },
        {
          sort: { createdAt: -1 },
          new: true,
          upsert: true, // This is the key change
        }
      );
    }

    console.log(response.data);
    // Return the response from Flask API
    return res.status(200).json(response.data);
  } catch (error: any) {
    console.error("Error in bulk grading:", error);

    // Log the full error details
    if (error.response) {
      console.error("Error response data:", error.response.data);
      console.error("Error response status:", error.response.status);
    }

    // Handle specific error cases
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      return res.status(error.response.status).json({
        message: error.response.data.error || "Error in bulk grading",
        details: error.response.data,
      });
    } else if (error.request) {
      // The request was made but no response was received
      return res.status(503).json({
        message: "No response received from grading service",
        details: error.message,
      });
    } else {
      // Something happened in setting up the request that triggered an Error
      return res.status(500).json({
        message: "Error setting up bulk grading request",
        details: error.message,
      });
    }
  }
};
