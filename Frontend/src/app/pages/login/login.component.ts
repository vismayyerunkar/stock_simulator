import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { MessageService } from 'primeng/api';
import { ILoginResponse } from 'src/models/UserModels';
import { UserAuthService } from 'src/services/user-auth.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss'],
})
export class LoginComponent implements OnInit, OnDestroy {
  loginForm = new FormGroup({
    email: new FormControl('', [Validators.required, Validators.email]),
    password: new FormControl('', [
      Validators.required,
      Validators.minLength(6),
    ]),
  });

  constructor(
    private userAuthService: UserAuthService,
    private router: Router,
    public messageService: MessageService
  ) {}

  ngOnInit(): void {}

  login() {
    var body = this.loginForm.value;

    this.userAuthService.login(body).subscribe({
      next: (res: ILoginResponse) => {
        console.log('Login Response:', res);

        this.loginForm.reset();

        const token = (res as any).token;

        if(token != 'undefined' && typeof token != typeof undefined){
          this.userAuthService.accessToken = token;
          localStorage.setItem("user",JSON.stringify((res as any)._doc))
          this.router.navigate(['/dashboard']);
        }else{
          alert("Invalid credentials");
        }
      },
      error: (err: any) => {
        this.messageService.add({
          severity: 'error',
          summary: 'Failed',
          detail: err.error.message,
        });
        console.error('Login Error:', err);
      },
    });
  }

  navsignup(){
    this.router.navigate(["/signup"])
  }

  ngOnDestroy() {
    this.messageService.clear();
  }
}
