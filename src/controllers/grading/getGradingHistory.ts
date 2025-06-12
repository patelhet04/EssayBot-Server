import { Request, Response } from "express";
import { GradingHistory } from "../../models/GradingHistory";
import { Types } from "mongoose";

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

interface GradingStatsDetails {
  modelName: string;
  completedAt: Date;
  gradeFile: {
    url: string;
    originalName: string;
    uploadedAt: Date;
  };
}

interface GradingHistoryResponse {
  _id: string;
  config_rubric: {
    criteria: Array<{
      name: string;
      description: string;
      weight: number;
      scoringLevels: {
        full: string;
        partial: string;
        minimal: string;
      };
      subCriteria: any[];
    }>;
  };
  createdAt: Date;
  gradingDetails?: GradingStatsDetails;
}

interface PopulatedGradingHistory {
  _id: Types.ObjectId;
  config_rubric: GradingHistoryResponse["config_rubric"];
  createdAt: Date;
  gradingStatsId: GradingStatsDetails | null;
}

export const getGradingHistory = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  try {
    const { courseId, assignmentId } = req.params;

    // Validate required fields
    if (!courseId || !assignmentId) {
      return res.status(400).json({
        message: "Missing required fields: courseId, assignmentId",
      });
    }

    // Find all grading history records with populated grading details in a single query
    const historyRecords = await GradingHistory.find({
      createdBy: new Types.ObjectId(req.user.id),
      courseId: new Types.ObjectId(courseId),
      assignmentId: new Types.ObjectId(assignmentId),
    })
      .populate({
        path: "gradingStatsId",
        select: "modelName completedAt gradeFile",
      })
      .sort({ createdAt: -1 })
      .lean();

    // Transform the data to match our response format
    const history = (
      historyRecords as unknown as PopulatedGradingHistory[]
    ).map((record) => {
      // Handle case where gradingStatsId is null or undefined
      const gradingDetails = record.gradingStatsId
        ? {
            modelName: record.gradingStatsId.modelName || "Unknown",
            completedAt: record.gradingStatsId.completedAt || new Date(),
            gradeFile: record.gradingStatsId.gradeFile || {
              url: "",
              originalName: "No file available",
              uploadedAt: new Date(),
            },
          }
        : undefined;

      return {
        _id: record._id.toString(),
        config_rubric: record.config_rubric,
        createdAt: record.createdAt,
        gradingDetails,
      };
    });

    return res.status(200).json({
      message: "Grading history retrieved successfully",
      history,
    });
  } catch (error) {
    console.error("Error fetching grading history:", error);
    return res.status(500).json({
      message: "Error fetching grading history",
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
};
