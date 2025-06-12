import { Request, Response } from "express";
import { Course } from "../../models/Course";
import { Assignment } from "../../models/Assignment";
import { GradingHistory } from "../../models/GradingHistory";
import axios from "axios";
import { AssignmentUpdatePayload } from ".";

interface AuthenticatedRequest extends Request {
  user: {
    id: string;
    username: string;
  };
}

interface PromptResponse {
  message: string;
  criteria_prompts: Record<string, string>;
}

interface CleanedPromptResponse {
  message: string;
  criteria_prompts: Array<{ criterionName: string; prompt: any }>;
}

// Function to extract the agent number from the header
const getAgentNumber = (header: string): number => {
  const match = header.match(/Agent (\d+)/);
  return match ? parseInt(match[1], 10) : 0;
};

// Function to clean and sort the prompts response
const cleanPromptsResponse = (
  responseData: PromptResponse
): CleanedPromptResponse => {
  const { message, criteria_prompts } = responseData;
  const cleanedCriteriaPrompts: Record<string, any> = {};

  for (const [criterionName, promptString] of Object.entries(
    criteria_prompts
  )) {
    try {
      let cleanedString = promptString
        .replace(/\n/g, "")
        .replace(/\s+/g, " ")
        .trim();

      cleanedString = cleanedString
        .replace(/,\s*}/g, "}")
        .replace(/,\s*]/g, "]");

      const parsedPrompt = JSON.parse(cleanedString);
      cleanedCriteriaPrompts[criterionName] = parsedPrompt;
    } catch (error) {
      console.error(
        `Error parsing prompt for criterion "${criterionName}":`,
        error
      );
      cleanedCriteriaPrompts[criterionName] = {
        error: `Failed to parse prompt: ${error}`,
      };
    }
  }

  const sortedCriteriaPrompts = Object.entries(cleanedCriteriaPrompts)
    .sort(([, promptA], [, promptB]) => {
      const agentNumberA = getAgentNumber(promptA.header);
      const agentNumberB = getAgentNumber(promptB.header);
      return agentNumberA - agentNumberB;
    })
    .map(([criterionName, prompt]) => ({
      criterionName,
      prompt,
    }));

  return {
    message,
    criteria_prompts: sortedCriteriaPrompts,
  };
};

// Update Assignment Endpoint
export const updateAssignment = async (req: Request, res: Response) => {
  try {
    const { courseId, assignmentId } = req.params;
    const updates: AssignmentUpdatePayload = req.body;

    if (!assignmentId) {
      return res.status(400).json({ message: "Assignment ID is required" });
    }

    if (!Object.keys(updates).length) {
      return res.status(400).json({ message: "No updates provided" });
    }

    const course = await Course.findById(courseId);
    if (!course) {
      return res.status(404).json({ message: "Course not found" });
    }

    const assignment = await Assignment.findOne({
      _id: assignmentId,
      course: courseId,
    });

    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }

    if (updates?.question) {
      assignment.question = updates.question;
    }
    if (updates?.config_rubric) {
      assignment.config_rubric = updates.config_rubric;
    }

    await assignment.save();

    return res.status(200).json(assignment);
  } catch (error) {
    console.error("Error updating assignment:", error);
    return res.status(500).json({ message: "Error updating assignment" });
  }
};

// Combined Endpoint: Finalize Rubric and Generate Prompt
export const finalizeRubricAndGeneratePrompt = async (
  req: AuthenticatedRequest,
  res: Response
) => {
  try {
    const { courseId, assignmentId } = req.params;
    const { config_rubric, question, model } =
      req.body as AssignmentUpdatePayload;
    const username = req?.user?.username;

    // Validate inputs
    if (!courseId || !assignmentId) {
      return res
        .status(400)
        .json({ message: "Course ID and Assignment ID are required" });
    }

    if (!config_rubric || !config_rubric.criteria) {
      return res.status(400).json({ message: "Rubric criteria are required" });
    }

    // Step 1: Find the course and assignment
    const course = await Course.findById(courseId);
    if (!course) {
      return res.status(404).json({ message: "Course not found" });
    }

    const assignment = await Assignment.findOne({
      _id: assignmentId,
      course: courseId,
    });

    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }

    // Step 2: Update the assignment with the new rubric
    assignment.config_rubric = config_rubric;
    await assignment.save();

    // Step 3: Create a GradingHistory record for this rubric
    // Note: gradingStatsId will be null initially since grading happens after rubric creation
    await GradingHistory.create({
      courseId: course._id,
      assignmentId: assignment._id,
      config_rubric,
      createdBy: req.user.id,
      createdAt: new Date(),
    });

    // Step 4: Generate prompts using the Python service
    const requestData = {
      criteria: config_rubric.criteria,
      courseId,
      assignmentTitle: assignment.title,
      username,
      model,
    };

    const response: any = await axios.post(
      "http://localhost:6000/generate_prompt",
      requestData
    );

    console.log("Raw Grading Result:", response.data);

    const cleanedData = cleanPromptsResponse(response.data);

    console.log("Cleaned Grading Result:", cleanedData);

    // Step 5: Update the assignment with the generated prompts
    assignment.config_prompt = cleanedData.criteria_prompts;
    await assignment.save();

    // Step 6: Return the updated assignment
    return res.status(200).json({
      message: "Rubric updated and prompts generated successfully",
      assignment,
    });
  } catch (error) {
    console.error("Error finalizing rubric and generating prompt:", error);
    return res.status(500).json({ message: "Internal server error" });
  }
};
