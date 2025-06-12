import { Schema, model, Document, CallbackError } from "mongoose";
import { Course } from "./Course";
import { S3Client, DeleteObjectCommand } from "@aws-sdk/client-s3";

// Validate env
if (
  !process.env.MINIO_ACCESS_KEY ||
  !process.env.MINIO_SECRET_KEY ||
  !process.env.MINIO_ENDPOINT
) {
  throw new Error(
    "MinIO credentials are not provided in environment variables"
  );
}

const s3Client = new S3Client({
  region: "us-east-1",
  endpoint: process.env.MINIO_ENDPOINT,
  forcePathStyle: true,
  credentials: {
    accessKeyId: process.env.MINIO_ACCESS_KEY,
    secretAccessKey: process.env.MINIO_SECRET_KEY,
  },
});

const ALLOWED_MIME_TYPES = [
  "application/pdf",
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
];

export interface IAttachment extends Document {
  fileName: string;
  fileUrl: string;
  fileType: string;
  fileSize: number;
  courseId: Schema.Types.ObjectId;
  assignmentId: Schema.Types.ObjectId;
  faissIndexUrl: string;
  chunksFileUrl: string;
  uploadedAt: Date;
}

const attachmentSchema = new Schema<IAttachment>(
  {
    fileName: { type: String, required: true },
    fileUrl: { type: String, required: true },
    fileType: { type: String, required: true, enum: ALLOWED_MIME_TYPES },
    fileSize: { type: Number, required: true, min: 0 },
    courseId: { type: Schema.Types.ObjectId, ref: "Course", required: true },
    assignmentId: {
      type: Schema.Types.ObjectId,
      ref: "Assignment",
      required: true,
    },
    faissIndexUrl: { type: String, default: "" },
    chunksFileUrl: { type: String, default: "" },
    uploadedAt: { type: Date, default: Date.now },
  },
  {
    timestamps: true,
  }
);

attachmentSchema.index({ courseId: 1 });
attachmentSchema.index({ assignmentId: 1 });
attachmentSchema.index(
  { fileName: 1, courseId: 1, assignmentId: 1 },
  { unique: true }
);

const retry = async (fn: () => Promise<any>, retries = 3, delay = 1000) => {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === retries - 1) throw error;
      console.log(`Retry ${i + 1} failed, retrying in ${delay}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
};

attachmentSchema.pre("findOneAndDelete", async function (next) {
  try {
    const doc = await this.model.findOne(this.getQuery());
    if (doc) {
      const bucket = process.env.MINIO_BUCKET;
      if (!bucket)
        throw new Error("MINIO_BUCKET environment variable is not set");

      const host = process.env.MINIO_ENDPOINT?.replace(/^https?:\/\//, "");

      const extractKey = (url: string) => {
        const pathname = new URL(url).pathname;
        return pathname.replace(`/${bucket}/`, "").replace(/^\//, "");
      };

      const deleteIfValid = async (url: string, label: string) => {
        if (url && url.includes(`${host}/${bucket}/`)) {
          const key = extractKey(url);
          const params = { Bucket: bucket, Key: key };
          await retry(async () => {
            await s3Client.send(new DeleteObjectCommand(params));
          });
          console.log(`Deleted ${label} from MinIO: ${url}`);
        }
      };

      await deleteIfValid(doc.fileUrl, "file");
      await deleteIfValid(doc.chunksFileUrl, "chunks file");
      await deleteIfValid(doc.faissIndexUrl, "FAISS index file");
    }
    next();
  } catch (error) {
    console.error(
      "Error in pre-findOneAndDelete middleware for Attachment:",
      error
    );
    next(error as CallbackError);
  }
});

attachmentSchema.post("save", async function (doc: IAttachment) {
  try {
    const courseId = doc.courseId;
    const attachmentId = doc._id;

    const updatedCourse = await Course.findByIdAndUpdate(
      courseId,
      { $push: { attachments: attachmentId } },
      { new: true }
    );

    if (!updatedCourse) {
      throw new Error(`Course with ID ${courseId} not found`);
    }
  } catch (error) {
    console.error("Error in post-save middleware for Attachment:", error);
    throw error;
  }
});

attachmentSchema.post(
  "findOneAndDelete",
  async function (doc: IAttachment | null) {
    if (doc) {
      try {
        const courseId = doc.courseId;
        const attachmentId = doc._id;

        const updatedCourse = await Course.findByIdAndUpdate(
          courseId,
          { $pull: { attachments: attachmentId } },
          { new: true }
        );

        if (!updatedCourse) {
          throw new Error(`Course with ID ${courseId} not found`);
        }
      } catch (error) {
        console.error(
          "Error in post-findOneAndDelete middleware for Attachment:",
          error
        );
        throw error;
      }
    }
  }
);

export const Attachment = model<IAttachment>("Attachment", attachmentSchema);
