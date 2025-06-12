import { Request, Response } from "express";
import { User } from "../../models/User";

// Define the getUser controller
export const getUser = async (req: Request, res: Response) => {
  try {
    // Extract the user ID from the request parameters
    const userId = req.params.userId;

    // Find the user by ID, exclude the password, and populate courses and assignments
    const user = await User.findById(userId)
      .select("-password")
      .populate({
        path: "courses",
        options: { distinct: true }, // Add distinct option to prevent duplicates
        populate: {
          path: "assignments",
        },
      })
      .lean() // Convert to plain JavaScript object
      .exec();

    // If user is not found, return a 404 error
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }

    // Ensure courses array has unique entries based on _id
    if (user.courses) {
      user.courses = Array.from(
        new Map(
          user.courses.map((course) => [course._id.toString(), course])
        ).values()
      );
    }

    // Return the user data with populated courses and assignments
    res.status(200).json(user);
  } catch (error) {
    // Handle any errors that occur during the process
    console.error(error);
    res.status(500).json({ message: "Server error" });
  }
};
