import { Schema, model, Document, Types } from "mongoose";

// Interface for the rubric criteria
interface Criterion {
  name: string;
  description: string;
  weight: number;
  scoringLevels: {
    full: string;
    partial: string;
    minimal: string;
  };
  subCriteria: Criterion[];
}

// Interface for the rubric configuration
interface RubricConfig {
  criteria: Criterion[];
}

// Interface for the GradingHistory document
export interface IGradingHistory extends Document {
  courseId: Types.ObjectId;
  assignmentId: Types.ObjectId;
  config_rubric: RubricConfig;
  gradingStatsId?: Types.ObjectId;
  createdAt: Date;
  createdBy: Types.ObjectId;
}

const gradingHistorySchema = new Schema<IGradingHistory>(
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
    config_rubric: {
      criteria: {
        type: [Schema.Types.Mixed],
        required: true,
      },
    },
    gradingStatsId: {
      type: Schema.Types.ObjectId,
      ref: "GradingStats",
      required: false,
    },
    createdBy: {
      type: Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    createdAt: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

// Add indexes for efficient querying
gradingHistorySchema.index({ courseId: 1, assignmentId: 1 });
gradingHistorySchema.index({ gradingStatsId: 1 });
gradingHistorySchema.index({ createdAt: -1 });

export const GradingHistory = model<IGradingHistory>(
  "GradingHistory",
  gradingHistorySchema
);
