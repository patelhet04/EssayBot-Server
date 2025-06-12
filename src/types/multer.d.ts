import { Request } from "express";

export interface MulterFile {
  fieldname: string;
  originalname: string;
  encoding: string;
  mimetype: string;
  buffer: Buffer;
  size: number;
}

declare global {
  namespace Express {
    interface Multer {
      File: MulterFile;
    }
  }
}
