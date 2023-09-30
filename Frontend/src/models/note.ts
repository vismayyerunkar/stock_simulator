import { Comment } from './comment';
import { IGenericResponse } from 'src/models/genericModels';

export interface Note {
  id: number;
  note: string;
  user: string;
  created_on: Date;
}
export interface INote {
  id: number;
  note: string;
  user: string;
  support_request: number;
  created_on: Date;
}

export interface SupportRequestNote {
  support_request_id: number;
  note: string;
}
export interface IAddNoteResponse extends IGenericResponse {
  data: SupportRequestNote;
}
export interface IFetchNoteListResponse extends IGenericResponse {
  data: {
    count: number;
    notes: Array<INote>;
  };
}