import express, { Request, Response } from "express";
import { connectDB } from "./config/db";
import routes from "./routes/routes";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";
import cors from "cors";
import axios from "axios";

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

app.get("/list-models", (req: Request, res: Response, next: Function) => {
  axios
    .get("http://localhost:5001/api/tags")
    .then((response: any) => {
      res
        .status(200)
        .json(
          response.data.models
            .map((model: any) => model.name)
            .filter((name: string) => !name.toLowerCase().includes("deepseek"))
        );
    })
    .catch((error) => {
      res.status(500).json({
        message: "Failed to list models",
        error: error.message,
      });
    });
});

connectDB();

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
