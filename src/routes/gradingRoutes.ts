import express, { RequestHandler } from "express";
import { authenticateToken } from "../middleware/authenticateToken";
import { gradeSingleEssay } from "../controllers/grading/gradeSingleEssay";
import { uploadExcelGrades } from "../controllers/grading/uploadExcelGrades";
import { getExcelFile } from "../controllers/grading/getExcelFile";
import { gradeBulkEssays } from "../controllers/grading/bulkGrading";
import { getGradingStats } from "../controllers/grading/getGradingStats";
import { approveGrading } from "../controllers/grading/approveGrading";
import { getGradingHistory } from "../controllers/grading/getGradingHistory";

const router = express.Router();

router.use(authenticateToken as RequestHandler);

router.post(
  "/grade-single-essay",
  gradeSingleEssay as unknown as RequestHandler
);

router.post("/bulk-grade", gradeBulkEssays as unknown as RequestHandler);

router.post(
  "/upload-excel-grades",
  uploadExcelGrades as unknown as RequestHandler[]
);

router.get(
  "/excel-files/:courseId/:assignmentId",
  getExcelFile as unknown as RequestHandler
);

router.get("/stats", getGradingStats as unknown as RequestHandler);

router.post(
  "/approve/:courseId/:assignmentId",
  approveGrading as unknown as RequestHandler
);

router.get(
  "/:courseId/:assignmentId/history",
  getGradingHistory as unknown as RequestHandler
);

export default router;
