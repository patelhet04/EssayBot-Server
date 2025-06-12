import { Schema, model, Document } from "mongoose";
import { Types } from "mongoose";
import { Assignment } from "./Assignment";

export interface IGradingStats extends Document {
  courseId: Types.ObjectId;
  assignmentId: Types.ObjectId;
  modelName: string;
  gradeFile: {
    url: string;
    originalName: string;
    uploadedAt: Date;
  };
  gradingApproved: boolean;
  approvedOn: Date;
  totalEssays?: number;
  completedAt: Date;
  createdBy: Types.ObjectId;
  createdAt: Date;
  updatedAt: Date;
}

const gradingStatsSchema = new Schema<IGradingStats>(
  {
    courseId: {
      type: Schema.Types.ObjectId,
      ref: "Course",
      required: true,
    },
    assignmentId: {
      type: Schema.Types.ObjectId,
      ref: "Assignment",
      required: true,
    },
    modelName: {
      type: String,
      required: true,
    },
    gradeFile: {
      url: { type: String, required: true },
      originalName: { type: String, required: true },
      uploadedAt: { type: Date, default: Date.now },
    },
    gradingApproved: { type: Boolean, default: false },
    approvedOn: { type: Date, required: false },
    totalEssays: {
      type: Number,
      required: false,
    },
    completedAt: {
      type: Date,
      default: Date.now,
    },
    createdBy: {
      type: Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    createdAt: { type: Date, default: Date.now },
    updatedAt: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

// Add indexes for efficient querying
gradingStatsSchema.index({ courseId: 1, assignmentId: 1 });
gradingStatsSchema.index({ completedAt: -1 });
gradingStatsSchema.index({ modelName: 1 });
gradingStatsSchema.index({ createdBy: 1 });

// Add post-save middleware to automatically update Assignment's gradingStats array
gradingStatsSchema.post("save", async function (doc) {
  try {
    await Assignment.findByIdAndUpdate(doc.assignmentId, {
      $addToSet: { gradingStats: doc._id },
    });
  } catch (error) {
    console.error("Error updating Assignment gradingStats:", error);
  }
});

export const GradingStats = model<IGradingStats>(
  "GradingStats",
  gradingStatsSchema
);
