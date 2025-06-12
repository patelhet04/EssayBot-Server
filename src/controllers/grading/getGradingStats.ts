import { Request, Response } from "express";
import { Course } from "../../models/Course";
import { Assignment, IAssignment } from "../../models/Assignment";
import { Types } from "mongoose";

interface AssignmentStats {
  title: string;
  studentCount: number;
}

interface CourseStats {
  title: string;
  assignments: Map<string, AssignmentStats>;
}

interface PopulatedCourse {
  _id: Types.ObjectId;
  title: string;
}

interface PopulatedAssignment extends Omit<IAssignment, "course"> {
  course: PopulatedCourse;
  _id: Types.ObjectId;
}

export const getGradingStats = async (req: Request, res: Response) => {
  try {
    // Get all approved assignments
    const approvedAssignments = (await Assignment.find({
      gradingApproved: true,
    }).populate("course", "title")) as unknown as PopulatedAssignment[];

    // Create a map to store course and assignment statistics
    const statsMap = new Map<string, CourseStats>();

    // Process each approved assignment
    for (const assignment of approvedAssignments) {
      const courseId = assignment.course._id.toString();
      const assignmentId = assignment._id.toString();

      if (!statsMap.has(courseId)) {
        statsMap.set(courseId, {
          title: assignment.course.title,
          assignments: new Map(),
        });
      }

      const courseStats = statsMap.get(courseId);
      if (!courseStats) continue;

      courseStats.assignments.set(assignmentId, {
        title: assignment.title,
        studentCount: 1, // Each approved assignment represents one student
      });
    }

    // Convert the map to the desired response format
    const response = Array.from(statsMap.values()).map((course) => ({
      courseName: course.title,
      assignments: Array.from(course.assignments.values()).map(
        (assignment) => ({
          assignmentName: assignment.title,
          studentCount: assignment.studentCount,
        })
      ),
    }));

    res.status(200).json(response);
  } catch (error) {
    console.error("Error getting grading statistics:", error);
    res.status(500).json({ error: "Failed to get grading statistics" });
  }
};
