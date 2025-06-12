import mongoose, { Document, Schema, Model } from "mongoose";
import bcrypt from "bcryptjs";

// Define an interface for the User document
interface IUser extends Document {
  name: string;
  username: string;
  password: string;
  courses: mongoose.Types.ObjectId[];
  lastLogin: Date;
  matchPassword(enteredPassword: string): Promise<boolean>;
}

// Define the User Schema
const userSchema: Schema<IUser> = new mongoose.Schema({
  name: { type: String, required: true },
  username: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  courses: [{ type: Schema.Types.ObjectId, ref: "Course" }],
  lastLogin: { type: Date, default: null },
});

// Hash password before saving
userSchema.pre<IUser>("save", async function (next) {
  if (!this.isModified("password")) return next();
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

// Method to compare passwords
userSchema.methods.matchPassword = async function (
  enteredPassword: string
): Promise<boolean> {
  return await bcrypt.compare(enteredPassword, this.password);
};

// Define and export the User model
const User: Model<IUser> = mongoose.model<IUser>("User", userSchema);
export { User, IUser };
