import { Request, Response } from "express";

export const logoutUser = async (
  req: Request,
  res: Response
): Promise<void> => {
  try {
    // Clear the JWT token cookie - MUST match login cookie settings exactly
    res.clearCookie("authToken", {
      httpOnly: true,
      secure: false, // Match login: secure is false
      sameSite: "lax", // Match login: sameSite is lax
      path: "/", // Match login: path is "/"
    });

    res.status(200).json({ message: "Logout successful" });
  } catch (error) {
    console.error("Error during logout:", error);
    res.status(500).json({ message: "Error logging out", error });
  }
};
