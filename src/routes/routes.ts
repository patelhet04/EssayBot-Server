import express from "express";
import authRoutes from "./authRoutes";
import courseRoutes from "./courseRoutes";
// import fileRoutes from "./fileRoutes";
import assignmentRoutes from "./assignmentRoutes";
import userRoutes from "./userRoutes";
import attachmentRoutes from "./attachmentRoutes";
import gradingRoutes from "./gradingRoutes";
import reportsRoutes from "./reportsRoute";

const router = express.Router();

// Group all API routes
router.use("/auth", authRoutes);
router.use("/courses", courseRoutes);
router.use("/courses", assignmentRoutes);
router.use("/users", userRoutes);
router.use("/attachments", attachmentRoutes);
router.use("/grading", gradingRoutes);
router.use("/reports", reportsRoutes);

export default router;
