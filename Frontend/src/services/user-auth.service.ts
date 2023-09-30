import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Router } from '@angular/router';
import { Observable, of } from 'rxjs';
import { environment } from 'src/environments/environment';
import { ISignupResponse } from 'src/models/SignUpModel';
import { ILoginResponse } from 'src/models/UserModels';

@Injectable({
  providedIn: 'root',
})
export class UserAuthService {
  constructor(private http: HttpClient, private router: Router) {}

  set accessToken(token: string) {
    localStorage.setItem('authToken', token);
  }

  get accessToken(): string {
    return localStorage.getItem('authToken') ?? '';
  }

  login(data: any): Observable<ILoginResponse> {
    const url = `${environment.baseUrl}/api/user/login`;
    const httpOptions = {};
    return this.http.post<ILoginResponse>(url, data, httpOptions);
  }
  signup(data: any): Observable<ISignupResponse> {
    const url = `${environment.baseUrl}/api/user/register`;
    const httpOptions = {};
    return this.http.post<ISignupResponse>(url, data, httpOptions);
  }

  logout() {
    console.log('In logout');
    localStorage.clear();
    this.router.navigate(['login']);
  }

  check(): Observable<boolean> {
    if (this.accessToken) {
      return of(true);
    } else {
      return of(false);
    }
  }
}
