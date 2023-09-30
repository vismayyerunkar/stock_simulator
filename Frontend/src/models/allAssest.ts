import { IGenericResponse } from './genericModels';

export interface GetAllAssest {
    assetSymbol:string;
    assetName: string;
    assetPrice: Number;
    purchasedDate:Date;
    assetType:string;
    assetQuantity:Number;
    //userId: 
}
export interface IGetAllAssest extends IGenericResponse {
  data: any[];
}
