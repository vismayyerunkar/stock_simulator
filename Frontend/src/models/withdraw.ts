import { IGenericResponse } from './genericModels';

export interface WithdrawRequest {
  id: number;
  created_on: Date;
  updated_on: Date;
  transaction_id: string;
  amount: string;
  payment_reference_id: string;
  payment_source: string;
  purpose: string;
  transaction_date: string;
  transaction_dtm: Date;
  status: string;
  additional_details: any;
  previous_balance_debit_account: string;
  previous_balance_credit_account: string;
  debit_account: number;
  credit_account: number;
}
export interface IWithdrawListResponse extends IGenericResponse {
  data: {
    count: number;
    withdraw_requests: Array<WithdrawRequest>;
  };
}
