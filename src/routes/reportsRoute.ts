import express, { RequestHandler } from "express";
import { authenticateToken } from "../middleware/authenticateToken";
import { analyzeGradingPerformance } from "../controllers/reports/analyzeGradingPerformance";

const router = express.Router();

// Apply authentication middleware to all routes
router.use(authenticateToken as RequestHandler);

// Route for analyzing grading performance
router.post(
  "/analyze-grading",
  analyzeGradingPerformance as unknown as RequestHandler
);

export default router;
