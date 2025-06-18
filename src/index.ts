import express, { Request, Response } from "express";
import { connectDB } from "./config/db";
import routes from "./routes/routes";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";
import cors from "cors";

dotenv.config();

const app = express();
const PORT = process.env.PORT;
console.log("Using PORT:", PORT);

app.use(express.json());

app.use(cookieParser());

// Allow frontend to send cookies
app.use(
  cors({
    origin: process.env.FRONTEND_BASE_URL,
    credentials: true,
  })
);

app.use("/api", routes);

// Health check endpoint
app.get("/health", (req: Request, res: Response) => {
  res.status(200).json({
    status: "OK",
    message: "Express.js server is running",
    timestamp: new Date().toISOString(),
    port: PORT,
  });
});

connectDB();

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
