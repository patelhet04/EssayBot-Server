import { Request, Response } from "express";
import { Course } from "../../models/Course";
import {
  Assignment,
  IAssignment,
  ICreateAssignment,
} from "../../models/Assignment";
import { Schema } from "mongoose";
import axios from "axios";
interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

export const gradeSingleEssay = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  try {
    const { courseId, assignmentId, essay, model, tone } = req.body;
    const username = req?.user?.username;

    const assignment: IAssignment | null = await Assignment.findOne({
      _id: assignmentId,
      course: courseId,
    });
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }
    const requestData = {
      courseId,
      assignmentTitle: assignment._id,
      essay,
      config_prompt: assignment.config_prompt,
      question: assignment.question,
      username,
      model,
      tone,
    };
    const response: any = await axios.post(
      "http://localhost:6000/grade_single_essay",
      requestData
    );
    console.log(response.data);

    return res.status(201).json(response.data);
  } catch (error) {
    console.error("Error grading essay:", error);
    return res.status(500).json({ message: "Error grading essay" });
  }
};
