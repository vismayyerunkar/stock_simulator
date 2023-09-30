import { IGenericResponse } from 'src/models/genericModels';

export interface Comment {
  id: number;
  comment: string;
  user: string;
  created_on: Date;
}

export interface IComment {
  id: number;
  comment: string;
  user: string;
  support_request: number;
  created_on: Date;
}

export interface SupportRequestComment {
  support_request_id: number;
  comment: string;
}
export interface IAddCommentResponse extends IGenericResponse {
  data: SupportRequestComment;
}

export interface ICommentListResponse extends IGenericResponse {
  data: {
    count: number;
    comments: Array<IComment>;
  };
}