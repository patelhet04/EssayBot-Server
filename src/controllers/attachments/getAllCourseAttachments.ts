import { Request, Response } from "express";
import { Attachment } from "../../models/Attachment";

export const getAllCourseAttachments = async (req: Request, res: Response) => {
  const { courseId } = req.params;

  try {
    const attachments = await Attachment.find({ courseId });

    res.status(200).json(attachments);
  } catch (error) {
    console.error("Error fetching attachments:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};
