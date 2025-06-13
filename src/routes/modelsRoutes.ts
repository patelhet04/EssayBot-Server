import express, { Request, Response } from "express";
import axios from "axios";

const router = express.Router();

// GET /api/list-models - Get list of available models from Ollama (filtered by parameter size ≤ 16B)
router.get("/list-models", (req: Request, res: Response) => {
  const ollamaUrl = process.env.OLLAMA_URL || "http://localhost:5000";

  axios
    .get(`${ollamaUrl}/api/tags`)
    .then((response: any) => {
      const filteredModels = response.data.models
        .filter((model: any) => {
          // Filter out deepseek models
          if (model.name.toLowerCase().includes("deepseek")) {
            return false;
          }

          // Filter by parameter size (≤ 16B)
          if (model.details && model.details.parameter_size) {
            const parameterSize = model.details.parameter_size;
            const numericPart = parseInt(parameterSize); // Get number before decimal point
            return numericPart <= 16;
          }

          // If no parameter size info, include the model (for backwards compatibility)
          return true;
        })
        .map((model: any) => model.name);

      res.status(200).json(filteredModels);
    })
    .catch((error) => {
      res.status(500).json({
        message: "Failed to list models",
        error: error.message,
      });
    });
});

export default router;
