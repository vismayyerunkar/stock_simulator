import { IGenericResponse } from './genericModels';

export interface Property {
  key: string;
  value: any;
}

export interface IName {
  id: number;
  name: string;
}
export interface AssetInvestment {
  investment_id: string;
  amount: string;
  type: string;
  user_id: number;
  created_on: Date;
  status: string;
}

export interface IAsset {
  id: number;
  accepted_investment_types: Array<string>;
  asset_class: {
    name: string;
    properties_to_display: Array<string>;
  };
  created_on: Date;
  updated_on: Date;
  asset_id: string;
  amount: string;
  subscribed_amount: string;
  image: string;
  name: string;
  short_description: string;
  description: string;
  interest_rate: string;
  tenure: number;
  tenure_type: string;
  minimum_investment: string;
  is_secured: boolean;
  is_listed: boolean;
  repayment_frequency: string;
  risk_category: string;
  maturity_date: string;
  maturity_date_buffer: number;
  status: string;
  additional_properties: any;
  partner: IName;
  account: number;
  properties: any;
  tag: string;
}

export interface InvestableAsset {
  id: number;
  asset_summary: any;
  created_on: Date;
  updated_on: Date;
  asset_id: string;
  amount: string;
  subscribed_amount: string;
  image: string;
  name: string;
  short_description: string;
  description: string;
  interest_rate: string;
  tenure: number;
  tenure_type: string;
  minimum_investment: string;
  is_secured: boolean;
  is_listed: boolean;
  repayment_frequency: string;
  maturity_date: Date;
  status: string;
  additional_properties: any;
  partner: number;
  account: number;
  asset_class: number;
}
export interface InvestmentUtilization {
  id: number;
  type: string;
  created_on: Date;
  updated_on: Date;
  utilization_id: string;
  amount: string;
  status: string;
  status_dtm: Date;
  investment: number;
  asset: number;
  transaction: number;
}

export interface ByAssestModel{
  assetSymbol: string,
  assetName: string,
  assetPrice: Number,
  assetType:string,
  assetQuantity:Number
}

export interface IAssetDetail extends IAsset {
  invested_amount: number;
  investor_count: number;
}


export interface IAssetClassesResponse extends IGenericResponse {
  data: Array<IName>;
}

export interface IAssetPartnerListResponse extends IGenericResponse {
  data: {
    count: number;
    partners: Array<IName>;
  };
}
export interface IInvestableAssetResponse extends IGenericResponse {
  data: {
    count: number;
    investable_assets: Array<InvestableAsset>;
  };
}
export interface IInvestmentUtilizationResponse extends IGenericResponse {
  data: {
    count: number;
    results: Array<InvestmentUtilization>;
  };
}

export interface IAssetListResponse extends IGenericResponse {
  data: {
    total_count: number;
    pagination_count: number;
    list: Array<IAsset>;
  };
}

export interface IAssetDetailResponse extends IGenericResponse {
  data: IAssetDetail;
}

export interface IAssetInvestmentResponse extends IGenericResponse {
  data: Array<AssetInvestment>;
}

export interface IAddOrUpdateAssetResponse extends IGenericResponse {
  data: IAsset;
}

export interface IBuyAssest extends IGenericResponse{
  data: ByAssestModel;
}
