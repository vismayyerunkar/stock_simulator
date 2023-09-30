import { IGenericResponse } from './genericModels';

export interface Transactions {
  price:Number;
  type: String;
  symbol: String;
  date:Date;
}

export interface ITransactionsResponse extends IGenericResponse {
  data: any[];
}
