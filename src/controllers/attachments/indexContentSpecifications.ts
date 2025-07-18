import { Request, Response } from "express";
import axios from "axios";

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

interface ContentSpecification {
  fileKey: string;
  fileName: string;
  fileType: "course_content" | "supporting_docs";
  useEntireDocument: boolean;
  relevantPages?: string | null;
}

const retry = async (
  fn: () => Promise<any>,
  retries: number = 3,
  delay: number = 1000
) => {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise((res) => setTimeout(res, delay));
    }
  }
};

export const indexContentSpecifications = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  const { courseId, assignmentId, contentSpecifications } = req.body;
  const username = req.user.username;

  // Validate required fields
  if (!courseId || !assignmentId || !contentSpecifications) {
    return res.status(400).json({
      message:
        "Missing required fields: courseId, assignmentId, or contentSpecifications",
    });
  }

  // Validate contentSpecifications array
  if (
    !Array.isArray(contentSpecifications) ||
    contentSpecifications.length === 0
  ) {
    return res.status(400).json({
      message: "contentSpecifications must be a non-empty array",
    });
  }

  // Validate each content specification
  for (const spec of contentSpecifications) {
    if (!spec.fileKey || !spec.fileName || !spec.fileType) {
      return res.status(400).json({
        message:
          "Each content specification must have fileKey, fileName, and fileType",
      });
    }

    if (!["course_content", "supporting_docs"].includes(spec.fileType)) {
      return res.status(400).json({
        message:
          "fileType must be either 'course_content' or 'supporting_docs'",
      });
    }

    if (typeof spec.useEntireDocument !== "boolean") {
      return res.status(400).json({
        message: "useEntireDocument must be a boolean",
      });
    }
  }

  try {
    console.log(
      `🚀 Processing ${contentSpecifications.length} content specifications for ${username}`
    );

    // Call Python endpoint for dual indexing
    const response = await retry(async () =>
      axios.post("http://localhost:6000/index-content-specifications", {
        username,
        courseId,
        assignmentId, // This will be used as assignment_title in Python
        contentSpecifications,
      })
    );

    console.log(
      `✅ Successfully processed content specifications in ${response.data.processing_time?.toFixed(
        2
      )}s`
    );

    // Return the Python response directly
    res.status(200).json({
      message: "Content specifications indexed successfully",
      ...response.data,
    });
  } catch (error: any) {
    console.error("❌ Content specifications indexing failed:", error.message);

    // Handle axios errors specifically
    if (error.response) {
      // Python endpoint returned an error
      return res.status(error.response.status || 500).json({
        message: "Indexing failed",
        error: error.response.data?.error || error.message,
        details: error.response.data,
      });
    } else if (error.request) {
      // Network error - Python service not reachable
      return res.status(503).json({
        message: "Python indexing service unavailable",
        error: "Unable to connect to indexing service",
      });
    } else {
      // Other error
      return res.status(500).json({
        message: "Internal server error",
        error: error.message,
      });
    }
  }
};
