import { Schema, model, Document } from "mongoose";
import { User } from "./User"; // Import the User model

export interface ICourse extends Document {
  title: string;
  description?: string;
  createdBy: Schema.Types.ObjectId;
  assignments: Schema.Types.ObjectId[];
  attachments: Schema.Types.ObjectId[];
  createdAt: Date;
  updatedAt: Date;
}

const courseSchema = new Schema<ICourse>(
  {
    title: { type: String, required: true, unique: true },
    description: { type: String },
    createdBy: { type: Schema.Types.ObjectId, ref: "User", required: true },
    assignments: [{ type: Schema.Types.ObjectId, ref: "Assignment" }],
    attachments: [{ type: Schema.Types.ObjectId, ref: "Attachment" }],
    createdAt: { type: Date, default: Date.now },
    updatedAt: { type: Date, default: Date.now },
  },
  { timestamps: true } // Automatically manage createdAt and updatedAt
);

courseSchema.post("save", async function (doc) {
  const userId = doc.createdBy; // The user who created the course
  const courseId = doc._id; // The newly created course's ID

  // Update the user's courses array
  await User.findByIdAndUpdate(userId, { $push: { courses: courseId } });
});

courseSchema.post("findOneAndDelete", async function (doc) {
  if (doc) {
    const userId = doc.createdBy; // The user who created the course
    const courseId = doc._id; // The deleted course's ID

    // Remove the course ID from the user's courses array
    await User.findByIdAndUpdate(userId, { $pull: { courses: courseId } });
  }
});

export const Course = model<ICourse>("Course", courseSchema);
