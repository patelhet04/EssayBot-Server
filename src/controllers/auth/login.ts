import { Request, Response } from "express";
import jwt from "jsonwebtoken";
import { User, IUser } from "../../models/User";

export const loginUser = async (req: Request, res: Response): Promise<void> => {
  try {
    const { username, password } = req.body;

    // Find user by username
    const user: IUser | null = await User.findOne({ username });

    if (!user || !(await user.matchPassword(password))) {
      res.status(400).json({ message: "Invalid credentials" });
      return;
    }

    // Update lastLogin timestamp
    await User.findByIdAndUpdate(user._id, { lastLogin: new Date() });

    // Generate JWT token
    const token = jwt.sign(
      { id: user._id, username: user.username },
      process.env.JWT_SECRET as string,
      {
        expiresIn: "1d",
      }
    );

    // Set token in HttpOnly, Secure cookie
    res.cookie("authToken", token, {
      httpOnly: true, // Prevents JavaScript access (XSS protection)
      secure: process.env.NODE_ENV === "production", // Ensures cookie is sent only over HTTPS in production
      sameSite: "strict", // Prevents CSRF attacks
      maxAge: 24 * 60 * 60 * 1000, // 1 day expiration
      path: "/",
    });

    res.json({ message: "Login successful", userId: user._id });
  } catch (error) {
    console.log({ error });
    res.status(500).json({ message: "Error logging in", error });
  }
};
