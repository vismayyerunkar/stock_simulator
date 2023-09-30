import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/environment';
import { IInvestmentUtilizationResponse } from 'src/models/asset';
import { IGenericResponse } from 'src/models/genericModels';
import {
  IFDInvestmentListResponse,
  IInvestorDetailResponse,
  IInvestorListResponse,
  ISettingResponse,
  IReportListResponse,
  ISIPInvestmentsListResponse,
  ISmartInvestmentListResponse,
  IDocumentResponse,
  InvestmentSummaryResponse,
  InvestMultippleResponse,
  InvestBody,
  IUploadDocumentResponse,
  InvestmentDetailsResponse,
} from 'src/models/investor';

@Injectable({
  providedIn: 'root',
})
export class InvestorService {
  constructor(private http: HttpClient) { }

  fetchInvestorList(
    limit: number,
    offset: number,
    searchTerm: string
  ): Observable<IInvestorListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/staff/v1/list/?search_query=${searchTerm}&limit=${limit}&offset=${offset}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IInvestorListResponse>(url, httpOptions);
  }

  fetchInvestorDetail(userId: number): Observable<IInvestorDetailResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/staff/v1/details/?user_id=${userId}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IInvestorDetailResponse>(url, httpOptions);
  }

  fetchInvestorSettings(userId: number): Observable<ISettingResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/setting/staff/v1/?user_id=${userId}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<ISettingResponse>(url, httpOptions);
  }

  fetchInvestmentDetails(userId: string): Observable<InvestmentDetailsResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/investor/staff/investment/v1/details/?investment_id=${userId}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<InvestmentDetailsResponse>(url, httpOptions);
  }

  fetchInvestorDocuments(userId: number): Observable<IDocumentResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/staff/v1/documents/?user_id=${userId}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IDocumentResponse>(url, httpOptions);
  }

  fetchSmartInvestmentList(
    offset: number,
    limit: number,
    status: string,
    startDate: any,
    endDate: any,
    assetClasses: string,
    userid: string
  ): Observable<ISmartInvestmentListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/investor/staff/smart-investment/v1/list/?offset=${offset}&limit=${limit}&status=${status}&start_date=${startDate}&end_date=${endDate}&asset_class_ids=${assetClasses}&user_id=${userid}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<ISmartInvestmentListResponse>(url, httpOptions);
  }

  fetchFDInvestmentList(
    offset: number,
    limit: number,
    status: string,
    startDate: any,
    endDate: any,
    sort_by: string,
    userid: string
  ): Observable<IFDInvestmentListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/investor/staff/fixed-deposits/investments/v1/list/?offset=${offset}&limit=${limit}&status=${status}&start_date=${startDate}&end_date=${endDate}&sort_by=${sort_by}&user_id=${userid}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IFDInvestmentListResponse>(url, httpOptions);
  }

  fetchSIPInvestmentList(
    offset: number,
    limit: number,
    status: string,
    startDate: any,
    endDate: any,
    sort_by: string,
    userid: string
  ): Observable<ISIPInvestmentsListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/investor/staff/sip/investments/v1/list/?offset=${offset}&limit=${limit}&status=${status}&start_date=${startDate}&end_date=${endDate}&sort_by=${sort_by}&user_id=${userid}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<ISIPInvestmentsListResponse>(url, httpOptions);
  }

  fetchReportList(
    type: string,
    emailReports: boolean,
    startDate: any,
    endDate: any,
    showData: boolean,
    userid: string
  ): Observable<IReportListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/investor/v1/staff/reports/?type=${type}&email_reports=${emailReports}&start_date=${startDate}&end_date=${endDate}&show_data=${showData}&user_id=${userid}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IReportListResponse>(url, httpOptions);
  }

  verifyDocument(body: any): Observable<IGenericResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/staff/v1/documents/verification/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.post<IGenericResponse>(url, body, httpOptions);
  }

  uploadDocuments(body: FormData): Observable<IGenericResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/staff/v1/documents/`;
    let httpOptions = {
      headers: new HttpHeaders({
        Authorization: token,
      }),
    };
    return this.http.post<IGenericResponse>(url, body, httpOptions);
  }

  fetchInvestmentSummary(
    user_id: string
  ): Observable<InvestmentSummaryResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/investor/staff/v1/investment/summary/?user_id=${user_id}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<InvestmentSummaryResponse>(url, httpOptions);
  }

  fetchInvestmentUtilization(
    investmentId: string
  ): Observable<IInvestmentUtilizationResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/investor/staff/fixed-deposits/investment-utilization/v1/list/?investment_id=${investmentId}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IInvestmentUtilizationResponse>(url, httpOptions);
  }

  fetchUploadDocuments(): Observable<IUploadDocumentResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/v1/document-types/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IUploadDocumentResponse>(url, httpOptions);
  }

  investFDAmountInAssetMultiple(
    body: Array<InvestBody>
  ): Observable<InvestMultippleResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/investor/staff/fixed-deposits/investments/v1/multiple/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.post<InvestMultippleResponse>(url, body, httpOptions);
  }
}
