import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/environment';
import { IWithdrawListResponse } from 'src/models/withdraw';

@Injectable({
  providedIn: 'root',
})
export class WithdrawRequestService {
  constructor(private http: HttpClient) {}

  fetchWithdrawList(
    offset: number,
    limit: number,
    status: string
  ): Observable<IWithdrawListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/accounting/staff/v1/withdraw-money/?status=${status}&offset=${offset}&limit=${limit}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IWithdrawListResponse>(url, httpOptions);
  }
}
