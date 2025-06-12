import { Request, Response } from "express";
import mongoose from "mongoose";
import { Attachment } from "../../models/Attachment";

export const getAttachments = async (req: Request, res: Response) => {
  const { courseId, assignmentId } = req.params;

  // Validate courseId and assignmentId
  if (!mongoose.Types.ObjectId.isValid(courseId)) {
    return res.status(400).json({ message: "Invalid course ID" });
  }
  if (!mongoose.Types.ObjectId.isValid(assignmentId)) {
    return res.status(400).json({ message: "Invalid assignment ID" });
  }

  try {
    // Find attachments matching the courseId and assignmentId
    const attachments = await Attachment.find({
      courseId: new mongoose.Types.ObjectId(courseId),
      assignmentId: new mongoose.Types.ObjectId(assignmentId),
    });

    // Return the attachments (will be an empty array if none are found)
    res.status(200).json(attachments);
  } catch (error) {
    console.error("Error fetching attachments:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};
