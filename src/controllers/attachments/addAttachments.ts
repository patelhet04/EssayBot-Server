import { Request, Response, NextFunction } from "express";
import multer from "multer";
import axios from "axios";
import { Course } from "../../models/Course";
import { Assignment } from "../../models/Assignment";
import { Attachment } from "../../models/Attachment";
import { fileUploadService } from "../../utils/awsS3"; // Rename if needed

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedMimeTypes = [
      "application/pdf",
      "application/msword",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ];
    if (!allowedMimeTypes.includes(file.mimetype)) return cb(null, false);
    cb(null, true);
  },
}).array("files", 10);

const retry = async (
  fn: () => Promise<any>,
  retries: number = 3,
  delay: number = 1000
) => {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise((res) => setTimeout(res, delay));
    }
  }
};

export const addAttachments = [
  (req: Request, res: Response, next: NextFunction) => {
    upload(req, res, (err) => {
      if (err instanceof multer.MulterError || err) {
        return res
          .status(400)
          .json({ message: "Upload error", error: err.message });
      }
      next();
    });
  },
  async (req: AuthenticatedRequest, res: Response) => {
    const { courseId, assignmentId } = req.body;
    console.log(req.body);
    const username = req.user.username;
    const files = req.files as Express.Multer.File[];

    if (!courseId || !assignmentId || !files?.length) {
      return res
        .status(400)
        .json({ message: "Missing fields or no files uploaded" });
    }

    try {
      const course = await Course.findById(courseId);
      const assignment = await Assignment.findById(assignmentId);
      if (
        !course ||
        !assignment ||
        !course.assignments.includes(assignmentId)
      ) {
        return res
          .status(404)
          .json({ message: "Invalid course or assignment" });
      }

      const savedAttachments = [];
      const errors = [];
      const fileKeys: string[] = [];

      for (const file of files) {
        try {
          const fileUrl = await fileUploadService.uploadFile(
            file,
            courseId,
            assignmentId,
            username,
            false
          );

          const url = new URL(fileUrl);
          const key = url.pathname
            .replace(`/${process.env.MINIO_BUCKET}/`, "")
            .replace(/^\//, "");
          if (!key) throw new Error("Failed to extract file key");

          fileKeys.push(key);

          const newAttachment = new Attachment({
            fileName: file.originalname,
            fileUrl,
            fileType: file.mimetype,
            fileSize: file.size,
            courseId,
            assignmentId,
          });

          const saved = await newAttachment.save();
          savedAttachments.push(saved);
        } catch (error: any) {
          errors.push({ fileName: file.originalname, error: error.message });
        }
      }

      let faissIndexUrl = "";
      let chunksFileUrl = "";
      if (fileKeys.length > 0) {
        try {
          const response = await retry(async () =>
            axios.post("http://localhost:6000/index-multiple", {
              username,
              s3_file_keys: fileKeys,
              courseId,
              assignmentTitle: assignment._id,
            })
          );

          faissIndexUrl = response.data.faiss_index_url;
          chunksFileUrl = response.data.chunks_key;

          for (const attachment of savedAttachments) {
            attachment.faissIndexUrl = faissIndexUrl;
            attachment.chunksFileUrl = chunksFileUrl;
            await attachment.save();
          }
        } catch (indexError: any) {
          for (const attachment of savedAttachments) {
            await fileUploadService.deleteFile(attachment.fileUrl);
            await Attachment.deleteOne({ _id: attachment._id });
          }
          return res.status(500).json({
            message: "Failed to index files",
            error: indexError.message,
          });
        }
      }

      if (errors.length > 0) {
        return res.status(207).json({
          message: "Partial success",
          success: savedAttachments,
          errors,
        });
      }

      res.status(201).json(savedAttachments);
    } catch (error: any) {
      res.status(500).json({ message: "Internal error", error: error.message });
    }
  },
];
