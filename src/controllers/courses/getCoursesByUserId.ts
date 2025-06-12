import { Request, Response } from "express";
import { Course } from "../../models/Course";

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

export const getCoursesByUserId = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  try {
    const userId = req.user.id;

    const courses = await Course.find({ createdBy: userId }).populate({
      path: "assignments",
      select: "title question config_rubric config_prompt",
    });

    if (!courses) {
      return res
        .status(404)
        .json({ message: "No courses found for this user" });
    }

    res.status(200).json(courses);
  } catch (error) {
    console.error("Error fetching courses:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};
