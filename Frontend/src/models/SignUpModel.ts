import { IGenericResponse } from './genericModels';

export interface ISignupResponse extends IGenericResponse {
  data: {
    token: string;
  };
}
