import { Request, Response } from "express";
import { Course } from "../../models/Course";
import { Assignment, ICreateAssignment } from "../../models/Assignment";
import { Schema } from "mongoose";

export const createAssignment = async (req: Request, res: Response) => {
  try {
    const { courseId } = req.params;
    const { title, question, config_rubric, config_prompt } = req.body;

    // Find the course by ID
    const course = await Course.findById(courseId);
    if (!course) {
      return res.status(404).json({ message: "Course not found" });
    }

    // Create assignment data
    const assignmentData: ICreateAssignment = {
      title,
      question,
      course: course._id as Schema.Types.ObjectId,
      config_rubric,
      config_prompt,
    };

    // Create and save the assignment
    const assignment = new Assignment(assignmentData);
    await assignment.save();

    // Add assignment to course's assignments array
    course.assignments.push(assignment._id as Schema.Types.ObjectId);
    await course.save();

    return res.status(201).json(assignment);
  } catch (error) {
    console.error("Error creating assignment:", error);
    return res.status(500).json({ message: "Error creating assignment" });
  }
};
