import { Request, Response } from "express";
import jwt from "jsonwebtoken";
import { User, IUser } from "../../models/User";

export const validateToken = async (
  req: Request,
  res: Response
): Promise<void> => {
  try {
    const authToken = req.cookies.authToken;

    if (!authToken) {
      res.status(401).json({ message: "No token provided" });
      return;
    }

    const decoded = jwt.verify(authToken, process.env.JWT_SECRET as string) as {
      id: string;
    };

    const user: IUser | null = await User.findById(decoded.id);

    if (!user) {
      res.status(401).json({ message: "Invalid token: User not found" });
      return;
    }

    res.status(200).json({
      message: "Token is valid",
      user: { id: user._id, username: user.username },
    });
  } catch (error) {
    console.log({ error });

    if (error instanceof jwt.TokenExpiredError) {
      res.status(401).json({ message: "Token expired" });
    } else if (error instanceof jwt.JsonWebTokenError) {
      res.status(401).json({ message: "Invalid token" });
    } else {
      res.status(500).json({ message: "Error validating token", error });
    }
  }
};
