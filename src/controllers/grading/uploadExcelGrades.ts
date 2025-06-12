import { Request, Response, NextFunction } from "express";
import multer from "multer";
import { Course } from "../../models/Course";
import { Assignment } from "../../models/Assignment";
import { fileUploadService } from "../../utils/awsS3";

// Extend the Request type to include user
interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

// Multer configuration specifically for Excel files
const uploadExcel = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    console.log(
      `Uploading Excel file: ${file.originalname}, type: ${file.mimetype}`
    );
    const allowedMimeTypes = [
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "text/csv",
    ];
    if (!allowedMimeTypes.includes(file.mimetype)) {
      console.error(
        `Invalid file type: ${file.mimetype}. Only Excel files (.xls, .xlsx, .csv) are allowed.`
      );
      return cb(null, false); // Reject the file without throwing an error
    }
    cb(null, true);
  },
}).single("file"); // Only accept a single file

export const uploadExcelGrades = [
  // Middleware to handle file upload
  (req: Request, res: Response, next: NextFunction) => {
    uploadExcel(req, res, (err) => {
      if (err instanceof multer.MulterError) {
        console.error("Multer error:", err);
        return res
          .status(400)
          .json({ message: "File upload error", error: err.message });
      } else if (err) {
        console.error("Excel file upload error:", err);
        return res
          .status(400)
          .json({ message: "Invalid Excel file upload", error: err.message });
      }

      // Check if file was rejected due to invalid file type
      if (!req.file) {
        // If there's no error but also no file, it might have been rejected by fileFilter
        return res.status(400).json({
          message: "Invalid file type",
          error: "Only Excel files (.xls, .xlsx, .csv) are allowed.",
        });
      }

      next();
    });
  },
  // Controller to process the upload
  async (req: AuthenticatedRequest, res: Response) => {
    console.log("Received Excel Upload Request Body:", req.body);

    const { courseId, assignmentId } = req.body;
    const username = req.user.username;
    const file = req.file as Express.Multer.File;

    if (!courseId || !assignmentId) {
      return res
        .status(400)
        .json({ message: "Missing required fields: courseId, assignmentId" });
    }

    if (!file) {
      return res.status(400).json({ message: "No Excel file uploaded" });
    }

    try {
      // Check if the course exists
      const course = await Course.findById(courseId);
      if (!course) {
        return res.status(404).json({ message: "Course not found" });
      }

      // Check if the assignment exists and belongs to the course
      const assignment = await Assignment.findById(assignmentId);
      if (!assignment) {
        return res.status(404).json({ message: "Assignment not found" });
      }
      if (!course.assignments.includes(assignmentId)) {
        return res
          .status(400)
          .json({ message: "Assignment does not belong to this course" });
      }

      // Upload the Excel file to S3 using the same structure as course materials
      try {
        // Create a standardized filename with timestamp
        const originalExt = file.originalname.split(".").pop() || "xlsx";
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        const standardizedFileName = `student_responses_${timestamp}.${originalExt}`;

        // Create a new file object with the standardized name
        const fileWithNewName = {
          ...file,
          originalname: standardizedFileName,
        };

        // The fileUploadService will automatically use the assignment title for folder structure
        const fileUrl = await fileUploadService.uploadFile(
          fileWithNewName,
          courseId,
          assignmentId,
          username,
          false // Don't overwrite if file exists
        );

        console.log(`Excel file uploaded to S3: ${fileUrl}`);

        // Update the assignment with the new Excel file information
        await Assignment.findByIdAndUpdate(
          assignmentId,
          {
            $push: {
              excelFiles: {
                url: fileUrl,
                originalName: file.originalname,
                uploadedAt: new Date(),
              },
            },
          },
          { new: true }
        );

        // Return the file URL
        res.status(201).json({
          message: "Excel file uploaded successfully",
          fileUrl,
          originalFilename: file.originalname,
          savedAsFilename: standardizedFileName,
        });
      } catch (uploadError: any) {
        console.error("Error uploading Excel file to Minio:", uploadError);
        return res.status(500).json({
          message: "Failed to upload Excel file",
          error: uploadError.message,
        });
      }
    } catch (error: any) {
      console.error("Error processing Excel upload:", error);
      res.status(500).json({
        message: "Internal server error",
        error: error.message,
      });
    }
  },
];
