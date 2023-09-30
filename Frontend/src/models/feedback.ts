import { IGenericResponse } from 'src/models/genericModels';

export interface Feedback {
  id: number;
  category: Category;
  is_deleted: Date;
  deleted_at: Date;
  created_on: Date;
  updated_on: Date;
  email: string;
  name: string;
  phone_number: string;
  title: string;
  description: string;
  user: number;
}

export interface Category {
  id: number;
  category: string;
}

export interface IFeedbackListResponse extends IGenericResponse {
  data: {
    count: number;
    feedback_list: Array<Feedback>;
  }
}

export interface IFeedbackListCategoryResponse extends IGenericResponse {
  data: Array<Category>;
}
