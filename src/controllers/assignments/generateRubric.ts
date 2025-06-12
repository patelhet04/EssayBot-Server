import { Request, Response } from "express";
import axios from "axios";
import { Assignment, IAssignment } from "../../models/Assignment";

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

export const createRubric = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  try {
    const { courseId, question, assignmentId, model } = req.body;
    const username = req.user.username;

    if (!courseId || !question || !assignmentId) {
      return res.status(400).json({
        message: "courseId and question are required",
      });
    }

    // Prepare request payload for Flask
    const payload = {
      courseId,
      question,
      username,
      title: assignmentId,
      model,
    };

    // Call Flask /generate_rubric endpoint
    const flaskResponse = await axios.post(
      "http://localhost:6000/generate_rubric",
      payload
    );

    const { data }: any = flaskResponse;
    console.log(data?.rubric);
    if ((data as any).success) {
      return res.status(200).json({
        message: "Rubric generated successfully",
        rubric: (data as any).rubric,
      });
    } else {
      return res.status(500).json({
        message: "Failed to generate rubric",
        error: (data as any).error || "Unknown error",
      });
    }
  } catch (error: any) {
    console.error("Error creating rubric:", error);
    return res.status(500).json({
      message: "Internal server error",
      error: error?.response?.data?.error || error.message,
    });
  }
};
