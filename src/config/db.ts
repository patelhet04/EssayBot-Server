import mongoose from "mongoose";
import dotenv from "dotenv";

dotenv.config();

const MONGO_URI = process.env.MONGO_URI as string;

export const connectDB = async () => {
  try {
    if (!MONGO_URI) {
      throw new Error("MONGO_URI is not defined in the environment variables");
    }
    console.log("Connecting to MongoDB...");
    await mongoose.connect(
      "mongodb://admin:BotBox010825@localhost:27017/essaybot?authSource=admin"
    );

    console.log("MongoDB Connected Successfully!");
  } catch (error) {
    console.error("MongoDB Connection Error:", error);
    process.exit(1);
  }
};
