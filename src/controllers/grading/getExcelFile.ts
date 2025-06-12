import { Request, Response } from "express";
import { Course } from "../../models/Course";
import { Assignment } from "../../models/Assignment";
import mongoose from "mongoose";

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

export const getExcelFile = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  try {
    const { courseId, assignmentId } = req.params;

    if (!courseId || !assignmentId) {
      return res.status(400).json({
        message: "courseId and assignmentId are required",
      });
    }

    // Check if the course exists
    const course = await Course.findById(courseId);
    if (!course) {
      return res.status(404).json({ message: "Course not found" });
    }

    // Check if the assignment exists and belongs to the course
    const assignment = await Assignment.findById(assignmentId);
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }

    // Check if assignmentId exists in course.assignments using string comparison
    const assignmentExists = course.assignments.some(
      (id) => id.toString() === assignmentId
    );

    if (!assignmentExists) {
      return res
        .status(400)
        .json({ message: "Assignment does not belong to this course" });
    }

    // Get all Excel files for the assignment
    if (!assignment.excelFiles || assignment.excelFiles.length === 0) {
      return res
        .status(404)
        .json({ message: "No Excel files found for this assignment" });
    }

    // Process all Excel files and format their dates
    const formattedExcelFiles = assignment.excelFiles.map((excelFile) => {
      const fileName = excelFile.url.split("/").pop() || "";
      let formattedDate = "";

      // Try to extract timestamp from filename
      const timestampMatch = fileName.match(/student_responses_(.+)\./);

      if (timestampMatch && timestampMatch[1]) {
        // Convert the filename timestamp format back to a Date object
        const timestampStr = timestampMatch[1].replace(
          /-/g,
          (match, offset) => {
            // Replace only the hyphens that should be colons or dots in ISO format
            if (offset > 10) {
              // After YYYY-MM-DD part
              if (offset === 13 || offset === 16) return ":"; // For hours and minutes
              if (offset === 19) return "."; // For seconds and milliseconds
            }
            return match; // Keep other hyphens
          }
        );

        try {
          const extractedDate = new Date(timestampStr);
          formattedDate = formatDate(extractedDate);
        } catch (err) {
          // Fallback to uploadedAt if timestamp extraction fails
          formattedDate = formatDate(excelFile.uploadedAt);
        }
      } else {
        // Fallback to uploadedAt if filename doesn't match expected pattern
        formattedDate = formatDate(excelFile.uploadedAt);
      }

      return {
        fileUrl: excelFile.url,
        originalName: excelFile.originalName,
        uploadedAt: formattedDate,
        fileName: fileName,
      };
    });

    res.status(200).json({
      message: "Excel files retrieved successfully",
      files: formattedExcelFiles,
    });
  } catch (error: any) {
    console.error("Error fetching Excel files:", error);
    res.status(500).json({
      message: "Internal server error",
      error: error.message,
    });
  }
};

// Helper function to format date consistently
function formatDate(date: Date): string {
  return new Date(date).toLocaleString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}
