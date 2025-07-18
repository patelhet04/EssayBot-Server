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
    const { courseId, assignmentId, essay, model, tone, currentRubric } = req.body;
    const username = req?.user?.username;

    const assignment: IAssignment | null = await Assignment.findOne({
      _id: assignmentId,
      course: courseId,
    });
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }

    // If currentRubric is provided, generate prompts on-the-fly
    let config_prompt;
    if (currentRubric && currentRubric.mainCriteria && currentRubric.mainCriteria.length > 0) {
      try {
        // Generate prompts for the current rubric
        const promptResponse = await axios.post(
          "http://localhost:6000/generate_prompt",
          {
            criteria: currentRubric.mainCriteria,
            courseId,
            assignmentTitle: assignment._id,
            username,
            model,
          }
        );
        
        // Type check the response
        if (promptResponse.data && typeof promptResponse.data === 'object' && 'criteria_prompts' in promptResponse.data) {
          config_prompt = (promptResponse.data as any).criteria_prompts;
          console.log("Generated prompts for current rubric:", config_prompt);
        } else {
          throw new Error("Invalid response format from prompt generation service");
        }
      } catch (promptError) {
        console.error("Error generating prompts for current rubric:", promptError);
        // Fallback to stored config_prompt
        config_prompt = assignment.config_prompt;
      }
    } else {
      // Use stored config_prompt if no current rubric provided
      config_prompt = assignment.config_prompt;
    }

    const requestData = {
      courseId,
      assignmentTitle: assignment._id,
      essay,
      config_prompt: config_prompt,
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
