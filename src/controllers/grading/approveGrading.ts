import { Request, Response } from "express";
import { Assignment } from "../../models/Assignment";
import { GradingStats } from "../../models/GradingStats";
import { User } from "../../models/User";
import { Types } from "mongoose";

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

export const approveGrading = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  try {
    const { assignmentId, courseId } = req.params;
    const { totalEssays } = req.body;
    const userId = req.user?.id;

    // Find the assignment
    const assignment = await Assignment.findById(assignmentId);
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }

    // Check if the user is the instructor of the course
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }

    // Check if user has access to this course
    const hasAccess = user.courses.some(
      (course) => course.toString() === courseId
    );

    if (!hasAccess) {
      return res
        .status(403)
        .json({ message: "Not authorized to approve grading" });
    }

    // Find the latest GradingStats record for this assignment
    const latestGradingStats = await GradingStats.findOne({
      assignmentId: assignmentId,
      courseId: courseId,
    }).sort({ completedAt: -1 });

    if (!latestGradingStats) {
      return res
        .status(400)
        .json({ message: "No grading records found for this assignment" });
    }

    // Update the GradingStats record with approval information
    latestGradingStats.gradingApproved = true;
    latestGradingStats.approvedOn = new Date();
    if (totalEssays !== undefined) {
      latestGradingStats.totalEssays = totalEssays;
    }
    await latestGradingStats.save();

    return res.status(200).json({
      message: "Grading approved successfully",
    });
  } catch (error) {
    console.error("Error approving grading:", error);
    return res.status(500).json({ message: "Internal server error" });
  }
};
