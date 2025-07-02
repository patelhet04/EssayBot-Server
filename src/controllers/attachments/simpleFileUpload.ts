import { Request, Response, NextFunction } from "express";
import multer from "multer";
import {
  S3Client,
  PutObjectCommand,
  ListObjectsV2Command,
} from "@aws-sdk/client-s3";

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

// Configure MinIO client
const s3Client = new S3Client({
  region: "us-east-1",
  endpoint: process.env.MINIO_ENDPOINT,
  forcePathStyle: true,
  credentials: {
    accessKeyId: process.env.MINIO_ACCESS_KEY || "",
    secretAccessKey: process.env.MINIO_SECRET_KEY || "",
  },
});

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    const allowedMimeTypes = [
      "application/pdf",
      "application/msword",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/plain", // TXT files
    ];
    if (!allowedMimeTypes.includes(file.mimetype)) {
      return cb(null, false); // Reject file without error
    }
    cb(null, true); // Accept file
  },
}).array("files", 15);

const sanitizeFileName = (fileName: string): string => {
  const extension = fileName.slice(fileName.lastIndexOf("."));
  const name = fileName.slice(0, fileName.lastIndexOf("."));
  const sanitized = name.replace(/[^a-zA-Z0-9]/g, "-");
  return `${sanitized}${extension}`;
};

export const uploadCourseContent = [
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
    const username = req.user.username;
    const files = req.files as Express.Multer.File[];

    if (!courseId || !assignmentId || !files?.length) {
      return res.status(400).json({
        message: "Missing courseId, assignmentId, or no files uploaded",
      });
    }

    try {
      const bucket = process.env.MINIO_BUCKET || "essaybot";
      const uploadedFiles = [];
      const errors = [];

      for (const file of files) {
        try {
          // Validate file type
          const allowedMimeTypes = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain", // TXT files
          ];

          if (!allowedMimeTypes.includes(file.mimetype)) {
            errors.push({
              fileName: file.originalname,
              error: "Only PDF, DOC, DOCX, and TXT files are allowed",
            });
            continue;
          }

          const sanitizedFileName = sanitizeFileName(file.originalname);
          // Create folder structure: username/courseId/assignmentId/course_content/
          const fileKey = `${username}/${courseId}/${assignmentId}/course_content/${sanitizedFileName}`;

          const uploadParams = {
            Bucket: bucket,
            Key: fileKey,
            Body: file.buffer,
            ContentType: file.mimetype,
            CacheControl: "max-age=31536000",
          };

          const command = new PutObjectCommand(uploadParams);
          await s3Client.send(command);

          const fileUrl = `${process.env.MINIO_ENDPOINT}/${bucket}/${fileKey}`;

          uploadedFiles.push({
            originalName: file.originalname,
            sanitizedName: sanitizedFileName,
            fileUrl: fileUrl,
            fileKey: fileKey,
            fileSize: file.size,
            fileType: file.mimetype,
            folder: "course_content",
          });
        } catch (error: any) {
          errors.push({
            fileName: file.originalname,
            error: error.message,
          });
        }
      }

      if (errors.length > 0 && uploadedFiles.length === 0) {
        return res.status(500).json({
          message: "All uploads failed",
          errors: errors,
        });
      }

      res.status(201).json({
        message: "Course content uploaded successfully",
        uploadedFiles: uploadedFiles,
        errors: errors.length > 0 ? errors : undefined,
        totalUploaded: uploadedFiles.length,
        totalErrors: errors.length,
      });
    } catch (error: any) {
      console.error("Error uploading course content:", error);
      res.status(500).json({
        message: "Internal server error",
        error: error.message,
      });
    }
  },
];

export const uploadSupportingDocs = [
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
    const username = req.user.username;
    const files = req.files as Express.Multer.File[];

    if (!courseId || !assignmentId || !files?.length) {
      return res.status(400).json({
        message: "Missing courseId, assignmentId, or no files uploaded",
      });
    }

    try {
      const bucket = process.env.MINIO_BUCKET || "essaybot";
      const uploadedFiles = [];
      const errors = [];

      for (const file of files) {
        try {
          // Validate file type
          const allowedMimeTypes = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain", // TXT files
          ];

          if (!allowedMimeTypes.includes(file.mimetype)) {
            errors.push({
              fileName: file.originalname,
              error: "Only PDF, DOC, DOCX, and TXT files are allowed",
            });
            continue;
          }

          const sanitizedFileName = sanitizeFileName(file.originalname);
          // Create folder structure: username/courseId/assignmentId/supporting_docs/
          const fileKey = `${username}/${courseId}/${assignmentId}/supporting_docs/${sanitizedFileName}`;

          const uploadParams = {
            Bucket: bucket,
            Key: fileKey,
            Body: file.buffer,
            ContentType: file.mimetype,
            CacheControl: "max-age=31536000",
          };

          const command = new PutObjectCommand(uploadParams);
          await s3Client.send(command);

          const fileUrl = `${process.env.MINIO_ENDPOINT}/${bucket}/${fileKey}`;

          uploadedFiles.push({
            originalName: file.originalname,
            sanitizedName: sanitizedFileName,
            fileUrl: fileUrl,
            fileKey: fileKey,
            fileSize: file.size,
            fileType: file.mimetype,
            folder: "supporting_docs",
          });
        } catch (error: any) {
          errors.push({
            fileName: file.originalname,
            error: error.message,
          });
        }
      }

      if (errors.length > 0 && uploadedFiles.length === 0) {
        return res.status(500).json({
          message: "All uploads failed",
          errors: errors,
        });
      }

      res.status(201).json({
        message: "Supporting documents uploaded successfully",
        uploadedFiles: uploadedFiles,
        errors: errors.length > 0 ? errors : undefined,
        totalUploaded: uploadedFiles.length,
        totalErrors: errors.length,
      });
    } catch (error: any) {
      console.error("Error uploading supporting docs:", error);
      res.status(500).json({
        message: "Internal server error",
        error: error.message,
      });
    }
  },
];

export const listFiles = async (req: AuthenticatedRequest, res: Response) => {
  const { courseId, assignmentId } = req.params;
  const username = req.user.username;

  if (!courseId || !assignmentId) {
    return res
      .status(400)
      .json({ message: "Missing courseId or assignmentId" });
  }

  try {
    const bucket = process.env.MINIO_BUCKET || "essaybot";

    // List course content files
    const courseContentPrefix = `${username}/${courseId}/${assignmentId}/course_content/`;
    const courseContentCommand = new ListObjectsV2Command({
      Bucket: bucket,
      Prefix: courseContentPrefix,
    });

    // List supporting docs files
    const supportingDocsPrefix = `${username}/${courseId}/${assignmentId}/supporting_docs/`;
    const supportingDocsCommand = new ListObjectsV2Command({
      Bucket: bucket,
      Prefix: supportingDocsPrefix,
    });

    const [courseContentResponse, supportingDocsResponse] = await Promise.all([
      s3Client.send(courseContentCommand),
      s3Client.send(supportingDocsCommand),
    ]);

    const formatFiles = (objects: any[], folderType: string) => {
      return (objects || [])
        .filter((obj) => obj.Key && !obj.Key.endsWith("/")) // Filter out folder entries
        .map((obj) => ({
          fileName: obj.Key.split("/").pop(), // Extract just the filename
          fileKey: obj.Key,
          fileUrl: `${process.env.MINIO_ENDPOINT}/${bucket}/${obj.Key}`,
          fileSize: obj.Size,
          lastModified: obj.LastModified,
          folder: folderType,
        }));
    };

    const courseContentFiles = formatFiles(
      courseContentResponse.Contents || [],
      "course_content"
    );
    const supportingDocsFiles = formatFiles(
      supportingDocsResponse.Contents || [],
      "supporting_docs"
    );

    res.status(200).json({
      courseId,
      assignmentId,
      files: {
        course_content: courseContentFiles,
        supporting_docs: supportingDocsFiles,
      },
      summary: {
        total_course_content: courseContentFiles.length,
        total_supporting_docs: supportingDocsFiles.length,
        total_files: courseContentFiles.length + supportingDocsFiles.length,
      },
    });
  } catch (error: any) {
    console.error("Error listing files:", error);
    res.status(500).json({
      message: "Internal server error",
      error: error.message,
    });
  }
};
