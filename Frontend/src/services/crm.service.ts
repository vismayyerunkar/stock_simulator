import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import {
  IFeedbackListCategoryResponse,
  IFeedbackListResponse,
} from 'src/models/feedback';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root',
})
export class CrmService {
  constructor(private http: HttpClient) {}

  updateLeadStatus(userId: number, lead_status: string): Observable<any> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/crm/statuses/v1/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    let data = {
      user_id: userId,
      status: lead_status,
    };
    return this.http.post<any>(url, data, httpOptions);
  }

  updateProfileCompletionStatus(
    userId: number,
    status: string
  ): Observable<any> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/crm/profile-completion-statuses/v1/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    let data = {
      user_id: userId,
      profile_completion_status: status,
    };
    return this.http.post<any>(url, data, httpOptions);
  }

  addComment(userId: number, comment: string): Observable<any> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/crm/comments/v1/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    let data = {
      user_id: userId,
      comment: comment,
    };
    return this.http.post<any>(url, data, httpOptions);
  }
}
