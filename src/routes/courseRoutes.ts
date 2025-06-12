import express, { RequestHandler } from "express";
import { authenticateToken } from "../middleware/authenticateToken";
import { createCourse } from "../controllers/courses/createCourse";
import { deleteCourse } from "../controllers/courses/deleteCourse";
import { updateCourse } from "../controllers/courses/updateCourse";
import { getCoursesByUserId } from "../controllers/courses/getCoursesByUserId";

const router = express.Router();

router.use(authenticateToken as RequestHandler);

router.get("/", getCoursesByUserId as unknown as RequestHandler);

router.post("/", createCourse as unknown as RequestHandler);

router.patch("/:courseId", updateCourse as unknown as RequestHandler);

router.delete("/:courseId", deleteCourse as unknown as RequestHandler);

export default router;
