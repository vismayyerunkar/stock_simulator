import { IGenericResponse } from 'src/models/genericModels';

export interface Investor {
  id: number;
  user_id: string;
  email: string;
  name: string;
  profile_url: string;
  mobile_number: string;
  gender: string;
  lead_status: string;
  profile_completion_status: string;
  last_comment: string;
  manager: string;
}

export interface InvestmentDetails {
  id: number;
  investment_summary: {
    id: number;
    created_on: Date;
    updated_on: Date;
    committed_principal_sum: string;
    expected_minimum_interest_sum: string;
    expected_minimum_repayment: string;
    paid_repayment_sum: string;
    repayment_status: string;
    repayment_status_date: Date;
    amortization_schedule_generated_on: Date;
    sanity_check_failed: boolean;
    investment: number;
  };
  created_on: Date;
  updated_on: Date;
  investment_id: String;
  amount: string;
  unutilized_amount: string;
  current_returns: string;
  expected_returns: string;
  current_roi: string;
  expected_roi: string;
  lock_in_period: string;
  tenure: number;
  tenure_type: string;
  interest_rate: string;
  interest_rate_type: string;
  status: string;
  status_dtm: Date;
  type: string;
  maturity_date: Date;
  user: number;
  asset: string;
  account: string;
}
export interface InvestorInfo {
  news_letter: any;
  hot_deals_update: boolean;
  sms_notification: boolean;
  id: number;
  is_deleted: boolean;
  deleted_at: Date;
  created_on: Date;
  updated_on: Date;
  dob: string;
  gender: string;
  pan: string;
  aadhaar: string;
  tnc_accepted: boolean;
  pp_accepted: boolean;
  sp_accepted: boolean;
  gdpr_accepted: boolean;
  rf_paid: boolean;
  rf_amount: string;
  user: number;
}

export interface Address {
  id: number;
  location: string;
  landmark: string;
  pin_code: number;
  district: string;
  city: string;
  state: string;
  country: string;
}
export interface Setting {
  id: number;
  sms_notification: boolean;
  email_notification: boolean;
  call_notification: boolean;
  news_letter: boolean;
  hot_deals_update: boolean;
}
export interface Document {
  id: number;
  type: {
    id: number;
    is_deleted: boolean;
    deleted_at: string;
    created_on: string;
    updated_on: string;
    type: string;
  };
  verified_by: any;
  is_deleted: boolean;
  deleted_at: string;
  created_on: string;
  updated_on: string;
  file: string;
  password: any;
  is_valid: any;
  user: number;
}

export interface Bank {
  user_id: number;
  account_id: string;
  account_number: string;
  holder_name: string;
  ifsc: string;
  bank_name: string;
  is_verified: false;
}

export interface InvestorDetail {
  user: Investor;
  info: InvestorInfo;
  address: Address;
  bank: Bank;
}

export interface InvestmentSummary {
  expected_minimum_repayment: string;
  paid_repayment_sum: string;
  repayment_status: string;
  repayment_status_date: Date;
  sanity_check_failed: boolean;
}

export interface SmartInvestment {
  id: number;
  investment_summary: InvestmentSummary;
  created_on: Date;
  updated_on: Date;
  investment_id: string;
  amount: string;
  unutilized_amount: string;
  current_returns: string;
  expected_returns: string;
  current_roi: string;
  expected_roi: string;
  lock_in_period: string;
  tenure: string;
  tenure_type: string;
  interest_rate: string;
  interest_rate_type: string;
  status: string;
  status_dtm: Date;
  type: string;
  maturity_date: string;
  user: number;
  asset: number;
  account: number;
}

export interface IReport {
  investment_id: string;
  created_on: Date;
  amount: string;
  type: string;
  expected_roi: string;
  tenure: number;
  tenure_type: string;
  maturity_date: string;
  purpose: string;
  transaction_dtm: Date;
  transaction_id: string;
}

export interface InvestmentSummary {
  total_investment_amount: number;
  smart_investment_amount: number;
  fd_investment_amount: number;
  sip_investment_amount: number;
  expected_roi: number;
  fd_unutilized_funds: number;
  sip_unutilized_funds: number;
  balance: number;
}

export interface InvestBody {
  investment_id: number;
  asset_id: number;
  amount: number;
  error?: string;
  status?: string;
}
export interface UploadDocument {
  id: number;
  type: string;
  file: string
}

export interface IInvestorListResponse extends IGenericResponse {
  data: {
    count: number;
    users: Array<Investor>;
  };
}
export interface ISettingResponse extends IGenericResponse {
  data: Setting;
}
export interface IDocumentResponse extends IGenericResponse {
  data: { documents: Array<Document> };
}

export interface IInvestorDetailResponse extends IGenericResponse {
  data: InvestorDetail;
}

export interface ISmartInvestmentListResponse extends IGenericResponse {
  data: {
    count: number;
    investments: Array<SmartInvestment>;
  };
}

export interface IFDInvestmentListResponse
  extends ISmartInvestmentListResponse { }

export interface ISIPInvestmentsListResponse
  extends ISmartInvestmentListResponse { }

export interface IReportListResponse extends IGenericResponse {
  data: Array<IReport>;
}

export interface InvestmentSummaryResponse extends IGenericResponse {
  data: InvestmentSummary;
}

export interface InvestMultippleResponse extends IGenericResponse {
  data: Array<InvestBody>;
}
export interface IUploadDocumentResponse extends IGenericResponse {
  data: {
    count: number;
    document_types: Array<UploadDocument>;
  };
}
export interface InvestmentDetailsResponse extends IGenericResponse {
  data: InvestmentDetails;
}
