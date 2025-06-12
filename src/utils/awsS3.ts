import {
  S3Client,
  PutObjectCommand,
  DeleteObjectCommand,
  HeadObjectCommand,
} from "@aws-sdk/client-s3";
import { Attachment, IAttachment } from "../models/Attachment";
import { Assignment } from "../models/Assignment";
import { MulterFile } from "../types/multer";

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

const ALLOWED_FILE_TYPES = {
  "application/pdf": ".pdf",
  "application/msword": ".doc",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    ".docx",
  "application/vnd.ms-excel": ".xls",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
  "text/csv": ".csv",
};

const MAX_FILE_SIZE = 10 * 1024 * 1024;

export class FileUploadService {
  private bucket: string;

  constructor() {
    this.bucket = process.env.MINIO_BUCKET || "essaybotbucket";
  }

  private validateFileType(mimetype: string): void {
    if (!Object.keys(ALLOWED_FILE_TYPES).includes(mimetype)) {
      throw new Error(
        `Invalid file type. Only ${Object.values(ALLOWED_FILE_TYPES).join(
          ", "
        )} files are allowed.`
      );
    }
  }

  private validateFileSize(size: number): void {
    if (size > MAX_FILE_SIZE) {
      throw new Error(
        `File size exceeds limit. Maximum file size allowed is ${
          MAX_FILE_SIZE / (1024 * 1024)
        }MB`
      );
    }
  }

  private sanitizeFileName(fileName: string): string {
    const extension = fileName.slice(fileName.lastIndexOf("."));
    const name = fileName.slice(0, fileName.lastIndexOf("."));
    const sanitized = name.replace(/[^a-zA-Z0-9]/g, "-");
    return `${sanitized}${extension}`;
  }

  private async checkFileExistsInS3(fileKey: string): Promise<boolean> {
    try {
      const command = new HeadObjectCommand({
        Bucket: this.bucket,
        Key: fileKey,
      });
      await s3Client.send(command);
      return true;
    } catch (error: any) {
      if (error.$metadata?.httpStatusCode === 404) {
        return false;
      }
      throw new Error("Error checking file existence in MinIO");
    }
  }

  private async checkAttachmentExists(
    fileName: string,
    courseId: string,
    assignmentId: string
  ): Promise<IAttachment | null> {
    return await Attachment.findOne({
      fileName,
      courseId,
      assignmentId,
    });
  }

  async uploadFile(
    file: MulterFile,
    courseId: string,
    assignmentId: string,
    professorUsername: string,
    overwrite: boolean = false
  ): Promise<string> {
    this.validateFileType(file.mimetype);
    this.validateFileSize(file.size);

    const sanitizedFileName = this.sanitizeFileName(file.originalname);
    const fileKey = `${professorUsername}/${courseId}/${assignmentId}/${sanitizedFileName}`;

    const existingAttachment = await this.checkAttachmentExists(
      file.originalname,
      courseId,
      assignmentId
    );

    if (existingAttachment && !overwrite) {
      throw new Error(
        `File '${file.originalname}' already exists for this course and assignment.`
      );
    }

    const fileExistsInS3 = await this.checkFileExistsInS3(fileKey);
    if (fileExistsInS3 && !overwrite) {
      throw new Error(
        `File '${file.originalname}' already exists in MinIO for this course and assignment.`
      );
    }

    const uploadParams = {
      Bucket: this.bucket,
      Key: fileKey,
      Body: file.buffer,
      ContentType: file.mimetype,
      CacheControl: "max-age=31536000",
    };

    try {
      const command = new PutObjectCommand(uploadParams);
      await s3Client.send(command);
      const host = process.env.MINIO_ENDPOINT?.replace(/^https?:\/\//, "");
      const fileUrl = `${process.env.MINIO_ENDPOINT}/${this.bucket}/${fileKey}`;
      return fileUrl;
    } catch (error) {
      console.error("Error uploading file to MinIO:", error);
      throw new Error("Failed to upload file");
    }
  }

  async deleteFile(fileUrl: string): Promise<void> {
    const url = new URL(fileUrl);
    const key = url.pathname.replace(`/${this.bucket}/`, "").replace(/^\//, "");

    const deleteParams = {
      Bucket: this.bucket,
      Key: key,
    };

    try {
      const command = new DeleteObjectCommand(deleteParams);
      await s3Client.send(command);
    } catch (error) {
      console.error("Error deleting file from MinIO:", error);
      throw new Error("Failed to delete file");
    }
  }
}

export const fileUploadService = new FileUploadService();
