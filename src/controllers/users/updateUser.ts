import { Request, Response } from "express";
import { User } from "../../models/User";

// Define the updateUser controller
export const updateUser = async (req: Request, res: Response) => {
  try {
    // Extract the user ID from the request parameters
    const userId = req.params.userId;

    // Extract the update data from the request body
    const updatedData = req.body;

    // Find the user by ID
    const user = await User.findById(userId);

    // If user is not found, return a 404 error
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }

    // Merge the existing user data with the updated data
    const mergedData = { ...user.toObject(), ...updatedData };

    // Update the user with the merged data
    Object.assign(user, mergedData);

    // Save the updated user to the database
    const updatedUser = await user.save();

    // Return the updated user data
    res.status(200).json(updatedUser);
  } catch (error) {
    // Handle any errors that occur during the process
    console.error(error);
    res.status(500).json({ message: "Server error" });
  }
};
