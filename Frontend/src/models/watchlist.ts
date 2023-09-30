import { IGenericResponse } from './genericModels';

export interface Watchlist {
  _id: string;
  userId: string;
  stockSymbol: string;
  stockName: string;
  Type: string;
  __v: number;
}
export interface IWatchListResponse extends IGenericResponse {
  data: any[];
}
