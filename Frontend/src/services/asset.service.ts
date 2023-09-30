import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/environment';
import {
  IAssetClassesResponse,
  IAssetPartnerListResponse,
  IAssetListResponse,
  IAssetDetailResponse,
  IAssetInvestmentResponse,
  IAddOrUpdateAssetResponse,
  IInvestableAssetResponse,
  IBuyAssest
 
} from 'src/models/asset';
import {IGetAllAssest} from 'src/models/allAssest'
import { IGenericResponse } from 'src/models/genericModels';

@Injectable({
  providedIn: 'root',
})
export class AssetService {
  constructor(private http: HttpClient) {}

  fetchAssetClasses(): Observable<IAssetClassesResponse> {
    const url = `${environment.baseUrl}apis/asset/v1/classes/`;
    let httpOptions = {};
    return this.http.get<IAssetClassesResponse>(url, httpOptions);
  }

  fetchPartnerList(): Observable<IAssetPartnerListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/staff/v1/partners/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IAssetPartnerListResponse>(url, httpOptions);
  }

  fetchAssetList(
    offset: number,
    limit: number,
    startDate: any,
    endDate: any,
    statuses: string,
    assetClasses: string,
    partnerIds: string,
    sort_by: string
  ): Observable<any> {
    const url = `${environment.baseUrl}apis/asset/v1/staff/?offset=${offset}&limit=${limit}&statuses=${statuses}&start_date=${startDate}&end_date=${endDate}&asset_classes=${assetClasses}&partner_ids=${partnerIds}&sort_by=${sort_by}`;
    let httpOptions = {};
    return this.http.get<IAssetListResponse>(url, httpOptions);
  }

  fetchAssetDetails(assetId: string): Observable<IAssetDetailResponse> {
    const url = `${environment.baseUrl}apis/asset/v1/details/?asset_id=${assetId}`;
    let httpOptions = {};
    return this.http.get<IAssetDetailResponse>(url, httpOptions);
  }
//Buy Assest 
  BuyAsset(body: FormData): Observable<IBuyAssest> {
    const token = `Bearer ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}/api/assets/purchaseAsset`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.post<IBuyAssest>(url, body, httpOptions);
  }

  //requirenment for this api req buy asset
  // assetSymbol: string,
  // assetName: string,
  // assetPrice: Number,
  // assetType:string,
  // assetQuantity:Number

// Sell Assest
  SellAssest(): Observable<IAssetInvestmentResponse> {
  const token = `Bearer ${localStorage.getItem('authToken')}`;
  const url = `${environment.baseUrl}/api/sellAsset`;
  let httpOptions = {
    headers: new HttpHeaders({ Authorization: token }),
  };
  return this.http.get<IAssetInvestmentResponse>(url, httpOptions);
}

//goal
getNewGoalDetails(data: any): Observable<any> {
  const url = `${environment.flaskURL}/get_portfolio`;
  return this.http.post<any>(url,data,{});
}

createGoal(data:any):Observable<any>{
  const url = `${environment.baseUrl}/api/goal/create`;
  const token = `Bearer ${localStorage.getItem('authToken')}`;
  let httpOptions = {
    headers: new HttpHeaders({ Authorization: token }),
  };
  return this.http.post<any>(url,data,httpOptions);
}

getCreatedGoals():Observable<any>{
  const token = `Bearer ${localStorage.getItem('authToken')}`;
  const url = `${environment.baseUrl}/api/goals`;
  let httpOptions = {
    headers: new HttpHeaders({ Authorization: token }),
  };
  return this.http.get<any>(url, httpOptions);
}

getGoalDetails(goalId:string):Observable<any>{
  const token = `Bearer ${localStorage.getItem('authToken')}`;
  const url = `${environment.baseUrl}/api/goal/getGoalDetails/${goalId}`;
  let httpOptions = {
    headers: new HttpHeaders({ Authorization: token }),
  };
  return this.http.get<any>(url, httpOptions);
}

//Get All assest for portfolio page
GetAssest(): Observable<IGetAllAssest> {
  const token = `Bearer ${localStorage.getItem('authToken')}`;
  const url = `${environment.baseUrl}/api/assets`;
  let httpOptions = {
    headers: new HttpHeaders({ Authorization: token }),
  };
  return this.http.get<IGetAllAssest>(url, httpOptions);
}

  fetchInvestableAsset(
    investmentId: string
  ): Observable<IInvestableAssetResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/asset/v1/staff/investable/?investment_id=${investmentId}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IInvestableAssetResponse>(url, httpOptions);
  }

  createOrUpdateAsset(body: FormData): Observable<IAddOrUpdateAssetResponse> {
    const url = `${environment.baseUrl}apis/asset/v1/staff/`;
    let httpOptions = {};
    return this.http.post<IAddOrUpdateAssetResponse>(url, body, httpOptions);
  }

  approveAsset(assetId: string): Observable<IGenericResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/asset/v1/staff/approve/?asset_id=${assetId}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.post<IGenericResponse>(url, {}, httpOptions);
  }
  cancelAsset(assetId: string): Observable<IGenericResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/asset/v1/staff/cancel/?asset_id=${assetId}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.post<IGenericResponse>(url, {}, httpOptions);
  }
}
