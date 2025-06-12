import { ParamsDictionary } from "express-serve-static-core";

export interface AssignmentRouteParams {
  assignmentId?: string;
}

export type AssignmentParams = AssignmentRouteParams & ParamsDictionary;

export interface ScoringLevels {
  full: string;
  partial: string;
  minimal: string;
}

export interface Criterion {
  name: string;
  description: string;
  weight: number;
  scoringLevels: ScoringLevels;
  subCriteria: Criterion[];
}

// Type for the update payload
export interface AssignmentUpdatePayload {
  question?: string;
  config_rubric?: {
    criteria: Criterion[];
  };
  config_prompt?: Record<string, any>;
  model?: string;
}
