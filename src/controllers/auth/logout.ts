import { Request, Response } from "express";

export const logoutUser = async (
  req: Request,
  res: Response
): Promise<void> => {
  try {
    // Clear the JWT token cookie
    res.clearCookie("authToken", {
      httpOnly: true,
      secure: true, // Ensure this matches the cookie settings used in login
      sameSite: "strict",
    });

    res.status(200).json({ message: "Logout successful" });
  } catch (error) {
    console.error("Error during logout:", error);
    res.status(500).json({ message: "Error logging out", error });
  }
};
