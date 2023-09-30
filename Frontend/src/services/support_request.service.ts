import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { IStaffMemberResponse } from 'src/models/staff-member';
import { IAddNoteResponse, IFetchNoteListResponse } from 'src/models/note';
import {
  IAddCommentResponse,
  ICommentListResponse,
} from 'src/models/comment';
import { environment } from 'src/environments/environment';
import { IGenericResponse } from 'src/models/genericModels';
import { ISupportRequestDetailResponse, ISupportRequestListResponse } from 'src/models/support-request';

@Injectable({
  providedIn: 'root',
})
export class SupportRequestService {
  constructor(private http: HttpClient) {}

  fetchStaffMemberList(): Observable<IStaffMemberResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/staff/v1/staff-dropdown/list/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IStaffMemberResponse>(url, httpOptions);
  }

  fetchNoteList(id: string): Observable<IFetchNoteListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/support-request/staff/v1/notes/?support_request_id=${id}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IFetchNoteListResponse>(url, httpOptions);
  }

  fetchCommentList(id: string): Observable<ICommentListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/support-request/staff/v1/comments/?support_request_id=${id}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<ICommentListResponse>(url, httpOptions);
  }

  addSupportRequestNote(body: any): Observable<IAddNoteResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/support-request/staff/v1/notes/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.post<IAddNoteResponse>(url, body, httpOptions);
  }

  addSupportRequestComment(body: any): Observable<IAddCommentResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/support-request/staff/v1/comments/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.post<IAddCommentResponse>(url, body, httpOptions);
  }

  updateSupportRequestStatus(body: any): Observable<IGenericResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/support-request/staff/v1/status/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.post<IGenericResponse>(url, body, httpOptions);
  }

  updateSupportRequestAssignee(body: any): Observable<IGenericResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/support-request/staff/v1/assignee/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.post<IGenericResponse>(url, body, httpOptions);
  }

  fetchSupportRequestList(
    offset: number,
    limit: number,
    status: string,
    startDate: any,
    endDate: any,
    sort_by: string,
    categoryId: string,
    AssignedToId: string,
    userid: string
  ): Observable<ISupportRequestListResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/support-request/staff/v1/?offset=${offset}&limit=${limit}&status=${status}&start_date=${startDate}&end_date=${endDate}&sort_by=${sort_by}&category_id=${categoryId}&assigned_to_id=${AssignedToId}&user_id=${userid}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<ISupportRequestListResponse>(url, httpOptions);
  }

  fetchSupportRequestDetail(id: string): Observable<ISupportRequestDetailResponse> {
    const token = `Token ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}apis/users/support-request/staff/v1/details/?support_request_id=${id}`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<ISupportRequestDetailResponse>(url, httpOptions);
  }
}
