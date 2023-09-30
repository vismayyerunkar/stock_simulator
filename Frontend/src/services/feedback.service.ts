import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { IFeedbackListCategoryResponse, IFeedbackListResponse } from 'src/models/feedback';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root',
})
export class FeedbackService {
  constructor(private http: HttpClient) {}

  fetchFeedbackList(offset: number, limit: number, sort_by: string, category_id: string, startDate: any, endDate: any): Observable<IFeedbackListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/feedback/staff/v1/?offset=${offset}&limit=${limit}&sort_by=${sort_by}&category_id=${category_id}&start_date=${startDate}&end_date=${endDate}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IFeedbackListResponse>(url, httpOptions);
  }
  
  fetchFeedbackCategoryList(): Observable<IFeedbackListCategoryResponse> {
    const url = `${environment.baseUrl}apis/users/feedback/v1/categories/`;
    let httpOptions = {};
    return this.http.get<IFeedbackListCategoryResponse>(url, httpOptions);
  }
}
