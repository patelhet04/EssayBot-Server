import express, { RequestHandler } from "express";
import { authenticateToken } from "../middleware/authenticateToken";
import { addAttachments } from "../controllers/attachments/addAttachments";
import { deleteAttachment } from "../controllers/attachments/deleteAttachment";
import { getAllCourseAttachments } from "../controllers/attachments/getAllCourseAttachments";
import { getAttachments } from "../controllers/attachments/getFilesUsingAssignmentId";
import {
  uploadCourseContent,
  uploadSupportingDocs,
  listFiles,
} from "../controllers/attachments/simpleFileUpload";
import { indexContentSpecifications } from "../controllers/attachments/indexContentSpecifications";

const router = express.Router();

router.use(authenticateToken as RequestHandler);

router.post("/addAttachments", addAttachments as unknown as RequestHandler);
router.get(
  "/getAllCourseAttachments/:courseId",
  getAllCourseAttachments as unknown as RequestHandler
);
router.delete(
  "/deleteAttachment/:attachmentId",
  deleteAttachment as unknown as RequestHandler
);

router.get(
  "/:courseId/:assignmentId",
  getAttachments as unknown as RequestHandler
);

// New simple upload routes (no RAG processing)
router.post(
  "/upload/course-content",
  uploadCourseContent as unknown as RequestHandler
);
router.post(
  "/upload/supporting-docs",
  uploadSupportingDocs as unknown as RequestHandler
);
router.get(
  "/list/:courseId/:assignmentId",
  listFiles as unknown as RequestHandler
);

// New dual indexing route for content specifications
router.post(
  "/index-content-specifications",
  indexContentSpecifications as unknown as RequestHandler
);

export default router;
