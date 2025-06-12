import { Request, Response } from "express";
import { Course } from "../../models/Course";

export const updateCourse = async (req: Request, res: Response) => {
  const { courseId } = req.params;
  const { assignments, attachments } = req.body; // Fields to update

  try {
    const course = await Course.findById(courseId);

    if (!course) {
      return res.status(404).json({ message: "Course not found" });
    }

    if (assignments) {
      course.assignments = assignments;
    }
    if (attachments) {
      course.attachments = attachments;
    }

    console.log("Looks good!");

    const updatedCourse = await course.save();

    res
      .status(200)
      .json({ message: "Course updated successfully", updatedCourse });
  } catch (error) {
    console.error("Error updating course:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};
