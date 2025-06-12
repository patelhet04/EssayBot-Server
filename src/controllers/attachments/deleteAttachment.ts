import { Request, Response } from "express";
import mongoose from "mongoose";
import { Attachment } from "../../models/Attachment";

// Extend the Request type to include user (from authentication middleware)
interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

export const deleteAttachment = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  const { attachmentId } = req.params;

  // Validate attachmentId
  if (!mongoose.Types.ObjectId.isValid(attachmentId)) {
    return res.status(400).json({ message: "Invalid attachment ID" });
  }

  try {
    // Find the attachment and populate courseId
    const attachment = await Attachment.findById(attachmentId).populate(
      "courseId"
    );

    if (!attachment) {
      return res.status(404).json({ message: "Attachment not found" });
    }

    // Check if the user has permission to delete the attachment
    const course = attachment.courseId as any; // Simplified type handling
    if (!course) {
      return res.status(404).json({ message: "Associated course not found" });
    }

    // Assuming createdBy is an ObjectId; adjust if it's a username string
    if (course.createdBy.toString() !== req.user.id) {
      return res.status(403).json({
        message: "You do not have permission to delete this attachment",
      });
    }

    // Delete the attachment (S3 cleanup is handled by the Attachment schema middleware)
    await Attachment.findByIdAndDelete(attachmentId);

    res.status(200).json({ message: "Attachment deleted successfully" });
  } catch (error) {
    console.error("Error deleting attachment:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};
