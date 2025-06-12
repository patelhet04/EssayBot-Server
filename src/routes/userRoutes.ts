import express, { RequestHandler } from "express";
import { authenticateToken } from "../middleware/authenticateToken";
import { getUser } from "../controllers/users/getUser";
import { updateUser } from "../controllers/users/updateUser";

const router = express.Router();

// Apply authentication middleware to all routes
router.use(authenticateToken as RequestHandler);

// Get assignment details
router.get("/getUser/:userId", getUser as RequestHandler);
router.patch("/updateUser/:userId", updateUser as RequestHandler);

export default router;
