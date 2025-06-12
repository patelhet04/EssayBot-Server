import { Schema, model, Document } from "mongoose";
import { User } from "./User";

export interface ISection extends Document {
  name: string;
  course: Schema.Types.ObjectId;
  instructor: Schema.Types.ObjectId;
  tasks: Schema.Types.ObjectId[];
  students?: Schema.Types.ObjectId[];
  createdAt: Date;
  updatedAt: Date;
}

const sectionSchema = new Schema<ISection>(
  {
    name: { type: String, required: true },
    course: { type: Schema.Types.ObjectId, ref: "Course", required: true },
    instructor: { type: Schema.Types.ObjectId, ref: "User", required: true },
    tasks: [{ type: Schema.Types.ObjectId, ref: "Task" }],
    students: [{ type: Schema.Types.ObjectId, ref: "User" }],
    createdAt: { type: Date, default: Date.now },
    updatedAt: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

// Update the course's sections array when a new section is created
sectionSchema.post("save", async function (doc) {
  const courseId = doc.course;
  const sectionId = doc._id;

  // Update the course's sections array
  await model("Course").findByIdAndUpdate(courseId, {
    $push: { sections: sectionId },
  });

  // Update the instructor's sections array
  await User.findByIdAndUpdate(doc.instructor, {
    $push: { sections: sectionId },
  });
});

// Remove section from course and instructor when deleted
sectionSchema.post("findOneAndDelete", async function (doc) {
  if (doc) {
    const courseId = doc.course;
    const sectionId = doc._id;
    const instructorId = doc.instructor;

    // Remove section from course
    await model("Course").findByIdAndUpdate(courseId, {
      $pull: { sections: sectionId },
    });

    // Remove section from instructor
    await User.findByIdAndUpdate(instructorId, {
      $pull: { sections: sectionId },
    });
  }
});

export const Section = model<ISection>("Section", sectionSchema);
